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
    "make_square_trig_mode_indices",
    "arc_length_params_square",
    "square_boundary_points_from_angles",
    "transform_adjacent_to_square_trig",
    "build_nd_map_from_electrode_data",
    "build_dn_map_from_electrode_data",
    "compute_psi_BIE_square",
    "compute_tBIE_square",
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


def _normalize_square_domain_size(domain_size: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(domain_size, (float, int)):
        return float(domain_size), float(domain_size)
    if len(domain_size) != 2:
        raise ValueError(f"domain_size must be a scalar or length-2 tuple, got {domain_size}.")
    return float(domain_size[0]), float(domain_size[1])


def make_square_trig_mode_indices(
    n_electrodes: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    if n_electrodes < 2:
        raise ValueError(f"n_electrodes must be at least 2, got {n_electrodes}.")
    if n_electrodes % 2 != 0:
        raise ValueError("Square-boundary trig projection currently expects an even number of electrodes.")

    half = n_electrodes // 2
    ascending = torch.arange(1, half + 1, device=device, dtype=dtype)
    descending = torch.arange(half - 1, 0, -1, device=device, dtype=dtype)
    return torch.cat((ascending, descending), dim=0)


def arc_length_params_square(
    n_electrodes: int,
    domain_size: float | tuple[float, float] = 2.0,
    *,
    n_boundary_samples: int = 512,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    if n_boundary_samples < n_electrodes:
        raise ValueError(
            f"n_boundary_samples must be >= n_electrodes, got {n_boundary_samples} and {n_electrodes}."
        )

    Lx, Ly = _normalize_square_domain_size(domain_size)
    perimeter = 2.0 * (Lx + Ly)
    electrode_edges = torch.linspace(0.0, perimeter, n_electrodes + 1, device=device, dtype=dtype)
    electrode_mid = 0.5 * (electrode_edges[:-1] + electrode_edges[1:])

    fii = torch.linspace(0.0, 2.0 * math.pi, n_boundary_samples + 1, device=device, dtype=dtype)[:-1]
    Dfii = float(2.0 * math.pi / n_boundary_samples)
    gamma_mid = (2.0 * math.pi / perimeter) * electrode_mid
    return fii, Dfii, gamma_mid


def square_boundary_points_from_angles(
    theta_arc: torch.Tensor,
    *,
    domain_size: float | tuple[float, float] = 2.0,
) -> torch.Tensor:
    theta_arc = torch.as_tensor(theta_arc)
    Lx, Ly = _normalize_square_domain_size(domain_size)
    perimeter = 2.0 * (Lx + Ly)
    s = (theta_arc.to(dtype=torch.float64) / (2.0 * math.pi)) * perimeter
    s = torch.remainder(s, perimeter)

    z = torch.empty_like(s, dtype=torch.complex128)

    bottom = s < Lx
    right = (s >= Lx) & (s < Lx + Ly)
    top = (s >= Lx + Ly) & (s < 2.0 * Lx + Ly)
    left = ~(bottom | right | top)

    x = torch.empty_like(s)
    y = torch.empty_like(s)

    x[bottom] = s[bottom]
    y[bottom] = 0.0

    x[right] = Lx
    y[right] = s[right] - Lx

    x[top] = 2.0 * Lx + Ly - s[top]
    y[top] = Ly

    x[left] = 0.0
    y[left] = perimeter - s[left]

    z.real = x
    z.imag = y
    return z.to(device=theta_arc.device, dtype=_complex_dtype_from(theta_arc.dtype))


def _square_trig_current_basis(
    n_electrodes: int,
    *,
    domain_size: float | tuple[float, float] = 2.0,
    trig_mode_indices: torch.Tensor | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if trig_mode_indices is None:
        trig_mode_indices = make_square_trig_mode_indices(n_electrodes, device=device)
    else:
        trig_mode_indices = torch.as_tensor(trig_mode_indices, device=device)

    trig_mode_indices = trig_mode_indices.to(device=device, dtype=dtype)
    Lx, Ly = _normalize_square_domain_size(domain_size)
    perimeter = 2.0 * (Lx + Ly)
    electrode_starts = torch.linspace(0.0, perimeter, n_electrodes + 1, device=device, dtype=dtype)[:-1]
    theta_start = (2.0 * math.pi / perimeter) * electrode_starts

    aad = torch.eye(n_electrodes, dtype=dtype, device=device)
    aad[torch.arange(1, n_electrodes, device=device), torch.arange(n_electrodes - 1, device=device)] -= 1.0
    aad[0, n_electrodes - 1] = -1.0
    aad /= math.sqrt(2.0)

    n_basis = trig_mode_indices.numel()
    atrig = torch.zeros((n_electrodes, n_basis), dtype=dtype, device=device)
    norm_trig = math.sqrt(2.0 / n_electrodes)
    split = n_electrodes // 2

    for k in range(split - 1):
        atrig[:, k] = norm_trig * torch.cos(trig_mode_indices[k] * theta_start)
    atrig[:, split - 1] = (1.0 / math.sqrt(float(n_electrodes))) * torch.cos(
        trig_mode_indices[split - 1] * theta_start
    )
    for k in range(split, n_basis):
        atrig[:, k] = norm_trig * torch.sin(trig_mode_indices[k] * theta_start)

    coeff = atrig.transpose(0, 1) @ aad
    return coeff, atrig, trig_mode_indices.to(dtype=torch.int64)


def transform_adjacent_to_square_trig(
    electrode_voltages: torch.Tensor,
    *,
    domain_size: float | tuple[float, float] = 2.0,
    trig_mode_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    voltages, squeeze = _as_batch_square_matrix(torch.as_tensor(electrode_voltages), "electrode_voltages")
    n_electrodes = voltages.shape[-1]
    real_dtype = torch.float64 if voltages.dtype in (torch.float64, torch.complex128) else torch.float32
    coeff, _, _ = _square_trig_current_basis(
        n_electrodes,
        domain_size=domain_size,
        trig_mode_indices=trig_mode_indices,
        device=voltages.device,
        dtype=real_dtype,
    )

    transformed = []
    for batch_idx in range(voltages.shape[0]):
        U = voltages[batch_idx].to(dtype=_complex_dtype_from(voltages.dtype))
        U = U - U.mean(dim=0, keepdim=True)
        solution = torch.linalg.lstsq(
            coeff.transpose(0, 1).to(dtype=U.dtype),
            U.transpose(0, 1),
        ).solution.transpose(0, 1)
        transformed.append(solution)

    out = torch.stack(transformed, dim=0)
    return out[0] if squeeze else out


def _square_trace_from_trig_voltages(
    trig_voltages: torch.Tensor,
    *,
    fii: torch.Tensor,
    domain_size: float | tuple[float, float] = 2.0,
) -> torch.Tensor:
    n_electrodes = trig_voltages.shape[0]
    electrode_idx = torch.floor(fii / (2.0 * math.pi / n_electrodes)).to(dtype=torch.int64)
    electrode_idx = electrode_idx.clamp(max=n_electrodes - 1)
    trace = trig_voltages[electrode_idx, :]
    return trace - trace.mean(dim=0, keepdim=True)


def _square_trig_boundary_matrix(
    trig_mode_indices: torch.Tensor,
    fii: torch.Tensor,
) -> torch.Tensor:
    n_basis = trig_mode_indices.numel()
    B = torch.zeros((n_basis, fii.numel()), dtype=fii.dtype, device=fii.device)
    split = (n_basis + 1) // 2
    trig_mode_indices = trig_mode_indices.to(device=fii.device, dtype=fii.dtype)

    for j in range(n_basis):
        if j < split:
            B[j, :] = torch.cos(trig_mode_indices[j] * fii)
        else:
            B[j, :] = torch.sin(trig_mode_indices[j] * fii)
    return B


def build_nd_map_from_electrode_data(
    electrode_voltages: torch.Tensor,
    *,
    domain_size: float | tuple[float, float] = 2.0,
    trig_mode_indices: torch.Tensor | None = None,
    n_boundary_samples: int = 512,
) -> torch.Tensor:
    trig_voltages = transform_adjacent_to_square_trig(
        electrode_voltages,
        domain_size=domain_size,
        trig_mode_indices=trig_mode_indices,
    )
    if trig_voltages.dim() == 2:
        trig_voltages = trig_voltages.unsqueeze(0)
        squeeze = True
    elif trig_voltages.dim() == 3:
        squeeze = False
    else:
        raise ValueError(
            f"trig_voltages must have shape (L, nBasis) or (B, L, nBasis), got {tuple(trig_voltages.shape)}."
        )

    n_electrodes = trig_voltages.shape[-2]
    fii, Dfii, _ = arc_length_params_square(
        n_electrodes,
        domain_size=domain_size,
        n_boundary_samples=n_boundary_samples,
        device=trig_voltages.device,
        dtype=trig_voltages.real.dtype,
    )
    if trig_mode_indices is None:
        trig_mode_indices = make_square_trig_mode_indices(n_electrodes, device=trig_voltages.device)
    else:
        trig_mode_indices = torch.as_tensor(trig_mode_indices, device=trig_voltages.device)

    B = _square_trig_boundary_matrix(trig_mode_indices, fii).to(dtype=trig_voltages.dtype)
    nd_maps = []
    for batch_idx in range(trig_voltages.shape[0]):
        trace = _square_trace_from_trig_voltages(
            trig_voltages[batch_idx],
            fii=fii,
            domain_size=domain_size,
        ).to(dtype=trig_voltages.dtype)
        nd_map = Dfii * (B @ trace)
        nd_maps.append(0.5 * (nd_map + nd_map.transpose(-1, -2).conj()))

    out = torch.stack(nd_maps, dim=0)
    return out[0] if squeeze else out


def build_dn_map_from_electrode_data(
    electrode_voltages: torch.Tensor,
    *,
    domain_size: float | tuple[float, float] = 2.0,
    trig_mode_indices: torch.Tensor | None = None,
    n_boundary_samples: int = 512,
    regularization: float = 1e-6,
    rcond: float | None = None,
) -> torch.Tensor:
    nd_map = build_nd_map_from_electrode_data(
        electrode_voltages,
        domain_size=domain_size,
        trig_mode_indices=trig_mode_indices,
        n_boundary_samples=n_boundary_samples,
    )
    return nd_to_dn_map(nd_map, regularization=regularization, rcond=rcond)


def _square_trig_synthesis_matrix(
    trig_mode_indices: torch.Tensor,
    theta_arc: torch.Tensor,
) -> torch.Tensor:
    n_basis = trig_mode_indices.numel()
    split = (n_basis + 1) // 2
    theta_arc = theta_arc.to(dtype=torch.float64)
    trig_mode_indices = trig_mode_indices.to(device=theta_arc.device, dtype=theta_arc.dtype)
    basis = torch.zeros((theta_arc.numel(), n_basis), dtype=theta_arc.dtype, device=theta_arc.device)
    for j in range(n_basis):
        if j < split:
            basis[:, j] = (1.0 / math.sqrt(math.pi)) * torch.cos(trig_mode_indices[j] * theta_arc)
        else:
            basis[:, j] = (1.0 / math.sqrt(math.pi)) * torch.sin(trig_mode_indices[j] * theta_arc)
    return basis


def compute_psi_BIE_square(
    Kvec: torch.Tensor,
    theta_arc: torch.Tensor,
    trig_mode_indices: torch.Tensor,
    *,
    domain_size: float | tuple[float, float] = 2.0,
    Dtheta: float | None = None,
) -> torch.Tensor:
    Kvec = torch.as_tensor(Kvec)
    theta_arc = torch.as_tensor(theta_arc, device=Kvec.device)
    trig_mode_indices = torch.as_tensor(trig_mode_indices, device=Kvec.device)
    complex_dtype = _complex_dtype_from(Kvec.dtype)
    real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32

    theta_arc = theta_arc.to(dtype=real_dtype)
    trig_mode_indices = trig_mode_indices.to(dtype=torch.int64)
    Kvec = Kvec.to(dtype=complex_dtype)
    if Dtheta is None:
        if theta_arc.numel() < 2:
            raise ValueError("Dtheta is required when theta_arc has fewer than two samples.")
        Dtheta = float(theta_arc[1] - theta_arc[0])

    boundary_points = square_boundary_points_from_angles(theta_arc, domain_size=domain_size).to(dtype=complex_dtype)
    phase = torch.exp(1j * boundary_points[:, None] * Kvec[None, :])

    n_basis = trig_mode_indices.numel()
    split = (n_basis + 1) // 2
    basis = torch.zeros((n_basis, theta_arc.numel()), dtype=real_dtype, device=theta_arc.device)
    trig_mode_indices_real = trig_mode_indices.to(dtype=real_dtype)
    for j in range(n_basis):
        if j < split:
            basis[j, :] = (1.0 / math.sqrt(math.pi)) * torch.cos(trig_mode_indices_real[j] * theta_arc)
        else:
            basis[j, :] = (1.0 / math.sqrt(math.pi)) * torch.sin(trig_mode_indices_real[j] * theta_arc)

    return Dtheta * (basis.to(dtype=complex_dtype) @ phase)


def compute_tBIE_square(
    *,
    Kvec: torch.Tensor,
    DN: torch.Tensor,
    DN1: torch.Tensor,
    Fpsi_BIE: torch.Tensor,
    theta_arc: torch.Tensor,
    domain_size: float | tuple[float, float] = 2.0,
    trig_mode_indices: torch.Tensor | None = None,
    Dtheta: float | None = None,
) -> torch.Tensor:
    Kvec = torch.as_tensor(Kvec)
    DN = torch.as_tensor(DN)
    DN1 = torch.as_tensor(DN1, device=DN.device)
    Fpsi_BIE = torch.as_tensor(Fpsi_BIE, device=DN.device)
    theta_arc = torch.as_tensor(theta_arc, device=DN.device)

    complex_dtype = _complex_dtype_from(torch.promote_types(Kvec.dtype, DN.dtype))
    real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32
    if trig_mode_indices is None:
        trig_mode_indices = make_square_trig_mode_indices(DN.shape[-1] + 1, device=DN.device)
    else:
        trig_mode_indices = torch.as_tensor(trig_mode_indices, device=DN.device)

    Kvec = Kvec.to(dtype=complex_dtype, device=DN.device)
    DN = DN.to(dtype=complex_dtype)
    DN1 = DN1.to(dtype=complex_dtype)
    Fpsi_BIE = Fpsi_BIE.to(dtype=complex_dtype)
    theta_arc = theta_arc.to(dtype=real_dtype)
    if Dtheta is None:
        if theta_arc.numel() < 2:
            raise ValueError("Dtheta is required when theta_arc has fewer than two samples.")
        Dtheta = float(theta_arc[1] - theta_arc[0])

    T_basis = _square_trig_synthesis_matrix(trig_mode_indices, theta_arc).to(dtype=complex_dtype)
    FLLpsi = (DN - DN1) @ Fpsi_BIE
    LLpsi = T_basis @ FLLpsi

    boundary_points = square_boundary_points_from_angles(theta_arc, domain_size=domain_size).to(dtype=complex_dtype)
    exp_phase = torch.exp(1j * torch.conj(Kvec)[None, :] * torch.conj(boundary_points)[:, None])
    return Dtheta * torch.einsum("ij,ij->j", exp_phase, LLpsi)


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


def _rect_grid_coords(
    K1: np.ndarray,
    K2: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_coords = np.asarray(K1[0, :], dtype=np.float64)
    y_coords = np.asarray(K2[:, 0], dtype=np.float64)
    vals = np.asarray(values)
    if x_coords[0] > x_coords[-1]:
        x_coords = x_coords[::-1]
        vals = vals[:, ::-1]
    if y_coords[0] > y_coords[-1]:
        y_coords = y_coords[::-1]
        vals = vals[::-1, :]
    return x_coords, y_coords, vals


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
        n_electrodes: int | None = None,
        k_grid_size: int = 64,
        k_radius: float = 6.0,
        k_extent: float | None = None,
        conductivity_min: float = 0.0,
        conductivity_max: float | None = 5.0,
        scattering_scale: float = 1.0,
        z_chunk_size: int = 1024,
        eps: float = 1e-6,
        background_conductivity: float = 1.0,
        domain_shape: str = "circle",
        domain_size: float | tuple[float, float] = 2.0,
    ) -> None:
        super().__init__()
        domain_shape = domain_shape.lower()
        if domain_shape not in {"circle", "square"}:
            raise ValueError(f"domain_shape must be 'circle' or 'square', got {domain_shape}.")

        self.image_size = image_size
        self.n_boundary_nodes = n_boundary_nodes
        self.n_modes = n_modes
        self.n_electrodes = n_modes + 1 if n_electrodes is None else n_electrodes
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
        self.domain_shape = domain_shape
        self.domain_size = domain_size

        if self.domain_shape == "square":
            if self.n_electrodes % 2 != 0:
                raise ValueError("Square-domain D-bar requires an even number of electrodes.")
            if self.n_modes != self.n_electrodes - 1:
                raise ValueError(
                    f"For domain_shape='square', expected n_modes == n_electrodes - 1, got {self.n_modes} and {self.n_electrodes}."
                )
            theta, boundary_weight, _ = arc_length_params_square(
                self.n_electrodes,
                domain_size=domain_size,
                n_boundary_samples=n_boundary_nodes,
            )
            boundary_points = square_boundary_points_from_angles(theta, domain_size=domain_size)
            trig_mode_indices = make_square_trig_mode_indices(self.n_electrodes)
            lambda_ref = torch.zeros((self.n_modes, self.n_modes), dtype=torch.float32)
        else:
            boundary_weight = 2.0 * math.pi / float(n_boundary_nodes)
            _, _, theta = make_trig_basis(n_boundary_nodes, n_modes)
            boundary_points = torch.polar(torch.ones_like(theta), theta)
            trig_mode_indices = torch.empty(0, dtype=torch.int64)
            lambda_ref = make_reference_dn_map(n_boundary_nodes=n_boundary_nodes, n_modes=n_modes)

        xy = torch.linspace(-1.0, 1.0, image_size)
        yy, xx = torch.meshgrid(xy, xy, indexing="ij")
        z_grid = xx + 1j * yy

        kx = torch.linspace(-self.k_extent, self.k_extent, k_grid_size)
        ky = torch.linspace(-self.k_extent, self.k_extent, k_grid_size)
        kyy, kxx = torch.meshgrid(ky, kx, indexing="ij")
        k_grid = kxx + 1j * kyy
        k_mask = (kxx.square() + kyy.square()) <= (k_radius ** 2)

        dk = float(kx[1] - kx[0]) if k_grid_size > 1 else 1.0

        self.register_buffer("theta", theta)
        self.register_buffer("boundary_points", boundary_points)
        self.register_buffer("trig_mode_indices", trig_mode_indices)
        self.register_buffer("z_grid", z_grid)
        self.register_buffer("k_grid", k_grid)
        self.register_buffer("k_mask", k_mask)
        self.register_buffer("dk", torch.tensor(dk, dtype=torch.float32))
        self.register_buffer("boundary_weight", torch.tensor(boundary_weight, dtype=torch.float32))
        self.register_buffer("lambda_ref", lambda_ref)

    def extra_repr(self) -> str:
        return (
            f"domain_shape={self.domain_shape}, image_size={self.image_size}, n_boundary_nodes={self.n_boundary_nodes}, "
            f"n_modes={self.n_modes}, k_grid_size={self.k_grid_size}, "
            f"k_radius={self.k_radius}, k_extent={self.k_extent}"
        )

    def compute_scattering_transform_square(
        self,
        lambda_sigma: torch.Tensor,
        lambda_ref: torch.Tensor,
    ) -> torch.Tensor:
        expected_shape = (self.n_modes, self.n_modes)
        if lambda_sigma.shape[-2:] != expected_shape:
            raise ValueError(
                f"Expected lambda_sigma with shape (*, {expected_shape[0]}, {expected_shape[1]}), got {tuple(lambda_sigma.shape)}."
            )
        if lambda_ref.shape[-2:] != expected_shape:
            raise ValueError(
                f"Expected lambda_ref with shape (*, {expected_shape[0]}, {expected_shape[1]}), got {tuple(lambda_ref.shape)}."
            )

        complex_dtype = _complex_dtype_from(torch.promote_types(lambda_sigma.dtype, lambda_ref.dtype))
        k_mask = self.k_mask.reshape(-1)
        kvec = self.k_grid.reshape(-1)[k_mask].to(dtype=complex_dtype)
        theta_arc = self.theta.to(dtype=self.boundary_points.real.dtype)
        Dtheta = float(self.boundary_weight.item())
        Fpsi_BIE = compute_psi_BIE_square(
            kvec,
            theta_arc,
            self.trig_mode_indices,
            domain_size=self.domain_size,
            Dtheta=Dtheta,
        ).to(device=self.k_grid.device, dtype=complex_dtype)

        out = torch.zeros(
            (lambda_sigma.shape[0], self.k_grid_size, self.k_grid_size),
            dtype=complex_dtype,
            device=self.k_grid.device,
        )
        for batch_idx in range(lambda_sigma.shape[0]):
            tbie = compute_tBIE_square(
                Kvec=kvec,
                DN=lambda_sigma[batch_idx],
                DN1=lambda_ref[batch_idx],
                Fpsi_BIE=Fpsi_BIE,
                theta_arc=theta_arc,
                domain_size=self.domain_size,
                trig_mode_indices=self.trig_mode_indices,
                Dtheta=Dtheta,
            ).to(dtype=complex_dtype, device=self.k_grid.device)
            flat = torch.zeros(self.k_grid.numel(), dtype=complex_dtype, device=self.k_grid.device)
            flat[k_mask] = self.scattering_scale * tbie
            out[batch_idx] = flat.reshape(self.k_grid_size, self.k_grid_size)
        return out

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

        grid_shape = (int(K1.shape[0]), int(K1.shape[1])) if K1 is not None else None
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
            if self.domain_shape == "square":
                raise ValueError("Square-domain D-bar reconstruction requires an explicit reference DN map.")
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

        if self.domain_shape == "square":
            t_exp = self.compute_scattering_transform_square(lambda_sigma, lambda_ref_batch)
        else:
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
        if self.domain_shape == "square":
            if reference_voltages is None:
                raise ValueError("Square-domain D-bar reconstruction requires reference_voltages.")

            lambda_sigma = build_dn_map_from_electrode_data(
                voltages,
                domain_size=self.domain_size,
                trig_mode_indices=self.trig_mode_indices,
                n_boundary_samples=self.n_boundary_nodes,
            )
            lambda_ref = build_dn_map_from_electrode_data(
                reference_voltages,
                domain_size=self.domain_size,
                trig_mode_indices=self.trig_mode_indices,
                n_boundary_samples=self.n_boundary_nodes,
            )
            return self.forward(lambda_sigma=lambda_sigma, lambda_ref=lambda_ref)

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
