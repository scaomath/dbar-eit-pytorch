"""Approximate continuum D-bar utilities.

Port status versus the MATLAB KIT4 scripts:

- comp02_ND_buildFromKIT4.m:
  not exactly ported; replaced here by an algebraic ND estimate from current/voltage data.
- comp03_DN_build.m:
  not exactly ported; replaced here by a projected pseudoinverse ND->DN conversion.
- comp04_psi_BIE.m:
  not ported; no boundary integral equation for CGO solutions is solved here.
- comp05_tBIE_psi.m:
  not ported; the scattering transform is approximated directly from delta_lambda.
- comp06_Dbarsolve.m:
  only the interpolation of precomputed scattering data are ported here;
  the final solve in this file remains a truncated Born-style approximation,
  not the MATLAB real-linear GMRES D-bar solve.

This module also does not use MATLAB-style intermediate files such as
`data/KIT4_measurement.mat` or `data/reconstruction.mat`; it keeps the
relevant quantities in memory.

The current PyTorch path reconstructs on the full Cartesian square grid.
No unit-disk masking is applied.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "make_trig_basis",
    "make_reference_dn_map",
    "make_mean_free_projector",
    "make_adjacent_current_patterns",
    "estimate_nd_map",
    "nd_to_dn_map",
    "estimate_dn_map",
    "DbarReconstruction2D",
]


def _complex_dtype_from(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float64, torch.complex128):
        return torch.complex128
    if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.complex64):
        return torch.complex64
    raise TypeError(f"Unsupported dtype for complex conversion: {dtype}.")


def _as_batch_square_matrix(x: torch.Tensor, name: str) -> tuple[torch.Tensor, bool]:
    if x.dim() == 2:
        if x.shape[0] != x.shape[1]:
            raise ValueError(f"{name} must be square, got {tuple(x.shape)}.")
        return x.unsqueeze(0), True
    if x.dim() == 3:
        if x.shape[-1] != x.shape[-2]:
            raise ValueError(f"{name} must be square on the last two axes, got {tuple(x.shape)}.")
        return x, False
    raise ValueError(f"{name} must have shape (N, N) or (B, N, N), got {tuple(x.shape)}.")


def _as_batch_measurements(x: torch.Tensor, name: str) -> tuple[torch.Tensor, bool]:
    if x.dim() == 2:
        return x.unsqueeze(0), True
    if x.dim() == 3:
        return x, False
    raise ValueError(
        f"{name} must have shape (n_boundary, n_patterns) or (B, n_boundary, n_patterns), got {tuple(x.shape)}."
    )


def _as_batch_scattering_samples(
    x: torch.Tensor,
    name: str,
    *,
    grid_shape: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, bool]:
    if x.dim() == 1:
        return x.unsqueeze(0), True
    if x.dim() == 2:
        if grid_shape is not None and tuple(x.shape) == tuple(grid_shape):
            return x.unsqueeze(0), True
        return x, False
    if x.dim() == 3:
        return x, False
    raise ValueError(
        f"{name} must have shape (N,), (B, N), (H, W), or (B, H, W), got {tuple(x.shape)}."
    )


def _as_batch_scattering_grid(
    x: torch.Tensor,
    name: str,
    expected_size: int,
) -> tuple[torch.Tensor, bool]:
    if x.dim() == 2:
        if x.shape != (expected_size, expected_size):
            raise ValueError(
                f"{name} must have shape ({expected_size}, {expected_size}), got {tuple(x.shape)}."
            )
        return x.unsqueeze(0), True
    if x.dim() == 3:
        if x.shape[-2:] != (expected_size, expected_size):
            raise ValueError(
                f"{name} must have shape (B, {expected_size}, {expected_size}), got {tuple(x.shape)}."
            )
        return x, False
    raise ValueError(
        f"{name} must have shape ({expected_size}, {expected_size}) or "
        f"(B, {expected_size}, {expected_size}), got {tuple(x.shape)}."
    )


def _interp2_bicubic_rect_grid(
    K1: np.ndarray,
    K2: np.ndarray,
    values: np.ndarray,
    x_query: np.ndarray,
    y_query: np.ndarray,
) -> np.ndarray:
    from scipy.interpolate import RectBivariateSpline

    x_coords, y_coords, values = _rect_grid_coords(K1, K2, values)
    spline_re = RectBivariateSpline(y_coords, x_coords, np.real(values), kx=3, ky=3)
    spline_im = RectBivariateSpline(y_coords, x_coords, np.imag(values), kx=3, ky=3)
    return spline_re.ev(y_query, x_query) + 1j * spline_im.ev(y_query, x_query)


def _interp_cubic_scattered(
    Kvec: np.ndarray,
    values: np.ndarray,
    x_query: np.ndarray,
    y_query: np.ndarray,
) -> np.ndarray:
    from scipy.interpolate import griddata

    points = np.column_stack((np.imag(Kvec.reshape(-1)), np.real(Kvec.reshape(-1))))
    values = values.reshape(-1)
    real_part = griddata(points, np.real(values), (y_query, x_query), method="cubic", fill_value=0.0)
    imag_part = griddata(points, np.imag(values), (y_query, x_query), method="cubic", fill_value=0.0)
    return real_part + 1j * imag_part


def make_trig_basis(
    n_boundary_nodes: int,
    n_modes: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1.")
    max_modes = (n_boundary_nodes - 1) // 2
    if n_modes > max_modes:
        raise ValueError(
            f"n_modes={n_modes} is too large for n_boundary_nodes={n_boundary_nodes}; "
            f"expected n_modes <= {max_modes}."
        )

    theta = torch.linspace(0.0, 2.0 * math.pi, n_boundary_nodes + 1, device=device, dtype=dtype)[:-1]
    modes = torch.cat(
        (
            torch.arange(-n_modes, 0, device=device),
            torch.arange(1, n_modes + 1, device=device),
        ),
        dim=0,
    )
    phase = theta[:, None] * modes[None, :].to(dtype)
    basis = torch.polar(torch.ones_like(phase), phase) / math.sqrt(n_boundary_nodes)
    return basis, modes, theta


def make_reference_dn_map(
    n_boundary_nodes: int,
    n_modes: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    basis, modes, _ = make_trig_basis(
        n_boundary_nodes=n_boundary_nodes,
        n_modes=n_modes,
        device=device,
        dtype=dtype,
    )
    eigenvalues = modes.abs().to(dtype)
    return (basis * eigenvalues.unsqueeze(0).to(basis.dtype)) @ basis.conj().transpose(0, 1)


def make_mean_free_projector(
    size: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    eye = torch.eye(size, device=device, dtype=dtype)
    return eye - torch.ones((size, size), device=device, dtype=dtype) / float(size)


def make_adjacent_current_patterns(
    n_electrodes: int,
    *,
    amplitude: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    patterns = torch.zeros((n_electrodes, n_electrodes), device=device, dtype=dtype)
    idx = torch.arange(n_electrodes, device=device)
    patterns[idx, idx] = amplitude
    patterns[(idx + 1) % n_electrodes, idx] = -amplitude
    return patterns


def estimate_nd_map(
    currents: torch.Tensor,
    voltages: torch.Tensor,
    regularization: float = 1e-6,
) -> torch.Tensor:
    currents, squeeze = _as_batch_measurements(currents, "currents")
    voltages, _ = _as_batch_measurements(voltages, "voltages")

    if currents.shape != voltages.shape:
        raise ValueError(
            f"currents and voltages must have the same shape, got {tuple(currents.shape)} and {tuple(voltages.shape)}."
        )

    complex_dtype = _complex_dtype_from(torch.promote_types(currents.dtype, voltages.dtype))
    currents = currents.to(dtype=complex_dtype)
    voltages = voltages.to(dtype=complex_dtype)

    gram = currents @ currents.conj().transpose(-1, -2)
    cross = voltages @ currents.conj().transpose(-1, -2)
    eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype).expand_as(gram)
    nd_map = cross @ torch.linalg.inv(gram + regularization * eye)
    return nd_map[0] if squeeze else nd_map


def nd_to_dn_map(
    nd_map: torch.Tensor,
    regularization: float = 1e-6,
    *,
    rcond: float | None = None,
) -> torch.Tensor:
    nd_map, squeeze = _as_batch_square_matrix(nd_map, "nd_map")
    complex_dtype = _complex_dtype_from(nd_map.dtype)
    nd_map = nd_map.to(dtype=complex_dtype)

    n = nd_map.shape[-1]
    projector = make_mean_free_projector(n, device=nd_map.device, dtype=nd_map.dtype).expand(nd_map.shape[0], -1, -1)
    nd_proj = projector @ nd_map @ projector
    nd_reg = nd_proj + regularization * projector

    if rcond is None:
        dn_map = projector @ torch.linalg.pinv(nd_reg) @ projector
    else:
        dn_map = projector @ torch.linalg.pinv(nd_reg, rcond=rcond) @ projector

    dn_map = 0.5 * (dn_map + dn_map.conj().transpose(-1, -2))
    return dn_map[0] if squeeze else dn_map


def estimate_dn_map(
    currents: torch.Tensor,
    voltages: torch.Tensor,
    regularization: float = 1e-6,
) -> torch.Tensor:
    nd_map = estimate_nd_map(currents=currents, voltages=voltages, regularization=regularization)
    return nd_to_dn_map(nd_map, regularization=regularization)


class DbarReconstruction2D(nn.Module):
    r"""
    Approximate continuum D-bar preprocessor for square-grid outputs.

    Notes:
        - `forward_from_measurements(...)` is a high-level surrogate for MATLAB
          `comp02_*` to `comp05_*` examples.
        - `forward_from_precomputed_scattering(...)` mirrors the interpolation
          stage of `comp06_Dbarsolve.m`, but `solve_sigma(...)` is still not
          the MATLAB GMRES D-bar solve.
        - no unit-disk masking is applied anywhere in this module.
    """

    def __init__(
        self,
        image_size: int = 64,
        n_boundary_nodes: int = 64,
        n_modes: int = 31,
        k_grid_size: int = 64,
        k_radius: float = 6.0,
        k_extent: float | None = None,
        conductivity_min: float = 0.0,
        conductivity_max: float | None = 5.0,
        scattering_scale: float = 1.0,
        z_chunk_size: int = 1024,
        eps: float = 1e-6,
        background_conductivity: float = 1.0,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.n_boundary_nodes = n_boundary_nodes
        self.n_modes = n_modes
        self.k_grid_size = k_grid_size
        self.k_radius = k_radius
        self.k_extent = k_radius if k_extent is None else k_extent
        if self.k_extent < self.k_radius:
            raise ValueError(f"k_extent must be >= k_radius, got k_extent={self.k_extent}, k_radius={self.k_radius}.")
        self.conductivity_min = conductivity_min
        self.conductivity_max = conductivity_max
        self.scattering_scale = scattering_scale
        self.z_chunk_size = z_chunk_size
        self.eps = eps
        self.background_conductivity = background_conductivity  # retained for API compatibility
        self.boundary_weight = 2.0 * math.pi / float(n_boundary_nodes)

        _, _, theta = make_trig_basis(n_boundary_nodes, n_modes)
        boundary_points = torch.polar(torch.ones_like(theta), theta)

        xy = torch.linspace(-1.0, 1.0, image_size)
        yy, xx = torch.meshgrid(xy, xy, indexing="ij")
        z_grid = xx + 1j * yy

        kx = torch.linspace(-self.k_extent, self.k_extent, k_grid_size)
        ky = torch.linspace(-self.k_extent, self.k_extent, k_grid_size)
        kyy, kxx = torch.meshgrid(ky, kx, indexing="ij")
        k_grid = kxx + 1j * kyy
        k_mask = (kxx.square() + kyy.square()) <= (k_radius ** 2)

        dk = float(kx[1] - kx[0]) if k_grid_size > 1 else 1.0
        lambda_ref = make_reference_dn_map(n_boundary_nodes=n_boundary_nodes, n_modes=n_modes)

        self.register_buffer("theta", theta)
        self.register_buffer("boundary_points", boundary_points)
        self.register_buffer("z_grid", z_grid)
        self.register_buffer("k_grid", k_grid)
        self.register_buffer("k_mask", k_mask)
        self.register_buffer("dk", torch.tensor(dk, dtype=torch.float32))
        self.register_buffer("lambda_ref", lambda_ref)

    def extra_repr(self) -> str:
        return (
            f"image_size={self.image_size}, n_boundary_nodes={self.n_boundary_nodes}, "
            f"n_modes={self.n_modes}, k_grid_size={self.k_grid_size}, "
            f"k_radius={self.k_radius}, k_extent={self.k_extent}"
        )

    def compute_scattering_transform(self, delta_lambda: torch.Tensor) -> torch.Tensor:
        if delta_lambda.shape[-2:] != (self.n_boundary_nodes, self.n_boundary_nodes):
            raise ValueError(
                f"Expected delta_lambda with shape (*, {self.n_boundary_nodes}, {self.n_boundary_nodes}), "
                f"got {tuple(delta_lambda.shape)}."
            )

        k = self.k_grid.reshape(-1)
        boundary = self.boundary_points.to(dtype=delta_lambda.dtype)

        psi_in = torch.exp(1j * (k[:, None] * boundary[None, :]))
        psi_out = torch.exp(1j * (k.conj()[:, None] * boundary.conj()[None, :]))

        psi_in = psi_in - psi_in.mean(dim=1, keepdim=True)
        psi_out = psi_out - psi_out.mean(dim=1, keepdim=True)

        delta_psi = torch.einsum("bij,kj->bik", delta_lambda, psi_in)
        t_exp = self.boundary_weight * torch.einsum("ki,bik->bk", psi_out.conj(), delta_psi)
        t_exp = self.scattering_scale * t_exp
        t_exp = t_exp * self.k_mask.reshape(1, -1).to(dtype=t_exp.real.dtype)
        return t_exp.reshape(delta_lambda.shape[0], self.k_grid_size, self.k_grid_size)

    def interpolate_precomputed_scattering(
        self,
        tBIE: torch.Tensor,
        *,
        K1: torch.Tensor | None = None,
        K2: torch.Tensor | None = None,
        Kvec: torch.Tensor | None = None,
        t_max: float | None = None,
        cutoff: float | None = 25.0,
    ) -> torch.Tensor:
        """Interpolate precomputed scattering samples onto the internal k-grid.

        This mirrors the `interp2(K1,K2,scatBIE,...)` part of `comp06_Dbarsolve.m`,
        but not the subsequent MATLAB GMRES D-bar solve.
        """
        if (K1 is None) != (K2 is None):
            raise ValueError("K1 and K2 must be provided together.")
        if K1 is None and Kvec is None:
            raise ValueError("Provide either (K1, K2) or Kvec.")

        grid_shape = tuple(K1.shape) if K1 is not None else None
        tBIE, squeeze = _as_batch_scattering_samples(tBIE, "tBIE", grid_shape=grid_shape)
        complex_dtype = _complex_dtype_from(tBIE.dtype)
        out = torch.zeros(
            (tBIE.shape[0], self.k_grid_size, self.k_grid_size),
            device=self.k_grid.device,
            dtype=complex_dtype,
        )

        query_mask = self.k_mask.reshape(-1).detach().cpu().numpy().astype(bool)
        query_x = self.k_grid.real.reshape(-1)[self.k_mask.reshape(-1)].detach().cpu().numpy()
        query_y = self.k_grid.imag.reshape(-1)[self.k_mask.reshape(-1)].detach().cpu().numpy()

        K1_np = None if K1 is None else torch.as_tensor(K1).detach().cpu().numpy()
        K2_np = None if K2 is None else torch.as_tensor(K2).detach().cpu().numpy()
        Kvec_np = None if Kvec is None else torch.as_tensor(Kvec).detach().cpu().numpy()

        for b in range(tBIE.shape[0]):
            sample = tBIE[b].to(dtype=complex_dtype).detach().cpu().numpy()

            if K1_np is not None:
                if sample.ndim == 1:
                    if t_max is None:
                        raise ValueError("t_max is required when tBIE is provided as vector samples.")
                    scat_grid = np.zeros_like(K1_np, dtype=sample.dtype)
                    inside = np.abs(K1_np + 1j * K2_np) < float(t_max)
                    if sample.size != int(inside.sum()):
                        raise ValueError(
                            f"Vector tBIE length {sample.size} does not match the number of |K|<t_max points {int(inside.sum())}."
                        )
                    scat_grid[inside] = sample.reshape(-1)
                else:
                    scat_grid = sample

                if cutoff is not None:
                    scat_grid = scat_grid.copy()
                    scat_grid[np.abs(np.real(scat_grid)) > cutoff] = 0
                    scat_grid[np.abs(np.imag(scat_grid)) > cutoff] = 0

                interp_vals = _interp2_bicubic_rect_grid(K1_np, K2_np, scat_grid, query_x, query_y)
            else:
                if cutoff is not None:
                    sample = sample.copy()
                    sample[np.abs(np.real(sample)) > cutoff] = 0
                    sample[np.abs(np.imag(sample)) > cutoff] = 0
                interp_vals = _interp_cubic_scattered(Kvec_np, sample, query_x, query_y)

            flat = np.zeros(self.k_grid.numel(), dtype=interp_vals.dtype)
            flat[query_mask] = interp_vals
            out[b] = torch.from_numpy(flat.reshape(self.k_grid_size, self.k_grid_size)).to(
                device=self.k_grid.device,
                dtype=complex_dtype,
            )

        return out[0] if squeeze else out

    def solve_sigma(self, scattering_transform: torch.Tensor) -> torch.Tensor:
        """Approximate inverse step from scattering data to conductivity."""
        scattering_transform, squeeze = _as_batch_scattering_grid(
            scattering_transform,
            "scattering_transform",
            self.k_grid_size,
        )

        batch_size = scattering_transform.shape[0]
        k_mask = self.k_mask.reshape(-1)
        k = self.k_grid.reshape(-1)[k_mask]
        t = scattering_transform.reshape(batch_size, -1)[:, k_mask]

        denom = 4.0 * math.pi * torch.where(k.abs() > self.eps, k.conj(), torch.ones_like(k))
        coeff = t / denom

        z_flat = self.z_grid.reshape(-1).to(dtype=coeff.dtype)
        born_terms = []

        for start in range(0, z_flat.numel(), self.z_chunk_size):
            stop = min(start + self.z_chunk_size, z_flat.numel())
            z_chunk = z_flat[start:stop]
            phase = torch.exp(-1j * (k[:, None] * z_chunk[None, :] + k.conj()[:, None] * z_chunk.conj()[None, :]))
            born_terms.append(torch.einsum("bk,kn->bn", coeff, phase))

        born = torch.cat(born_terms, dim=1)
        mu0 = 1.0 + (self.dk.to(dtype=born.real.dtype) ** 2 / math.pi) * born
        sigma = mu0.abs().square().reshape(batch_size, self.image_size, self.image_size)

        if self.conductivity_min is not None:
            sigma = sigma.clamp_min(self.conductivity_min)
        if self.conductivity_max is not None:
            sigma = sigma.clamp_max(self.conductivity_max)

        sigma = sigma.unsqueeze(1).to(dtype=self.z_grid.real.dtype)
        return sigma[0] if squeeze else sigma

    def forward(
        self,
        lambda_sigma: torch.Tensor,
        lambda_ref: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            lambda_sigma: DN map on the boundary, shape (N, N) or (B, N, N).
            lambda_ref: Optional reference DN map.

        Returns:
            Square Cartesian image with shape (1, H, W) or (B, 1, H, W).
        """
        lambda_sigma, squeeze = _as_batch_square_matrix(lambda_sigma, "lambda_sigma")
        complex_dtype = _complex_dtype_from(lambda_sigma.dtype)
        device = self.z_grid.device

        lambda_sigma = lambda_sigma.to(device=device, dtype=complex_dtype)

        if lambda_ref is None:
            lambda_ref_batch = self.lambda_ref.to(device=device, dtype=complex_dtype).unsqueeze(0)
        else:
            lambda_ref_batch, _ = _as_batch_square_matrix(lambda_ref, "lambda_ref")
            lambda_ref_batch = lambda_ref_batch.to(device=device, dtype=complex_dtype)

        if lambda_ref_batch.shape[0] == 1 and lambda_sigma.shape[0] > 1:
            lambda_ref_batch = lambda_ref_batch.expand(lambda_sigma.shape[0], -1, -1)
        elif lambda_ref_batch.shape[0] != lambda_sigma.shape[0]:
            raise ValueError(
                f"lambda_ref batch dimension must be 1 or {lambda_sigma.shape[0]}, got {lambda_ref_batch.shape[0]}."
            )

        delta_lambda = lambda_sigma - lambda_ref_batch
        t_exp = self.compute_scattering_transform(delta_lambda)
        sigma = self.solve_sigma(t_exp)
        return sigma[0] if squeeze else sigma

    @torch.inference_mode()
    def forward_from_measurements(
        self,
        currents: torch.Tensor,
        voltages: torch.Tensor,
        reference_currents: torch.Tensor | None = None,
        reference_voltages: torch.Tensor | None = None,
        regularization: float = 1e-6,
    ) -> torch.Tensor:
        warnings.warn(
            "forward_from_measurements() is not an exact port of MATLAB comp02-comp05; "
            "it uses approximate ND/DN/scattering surrogates.",
            stacklevel=2,
        )
        lambda_sigma = estimate_dn_map(currents=currents, voltages=voltages, regularization=regularization)

        lambda_ref = None
        if reference_voltages is not None:
            if reference_currents is None:
                reference_currents = currents
            lambda_ref = estimate_dn_map(
                currents=reference_currents,
                voltages=reference_voltages,
                regularization=regularization,
            )

        return self.forward(lambda_sigma=lambda_sigma, lambda_ref=lambda_ref)
