"""Porting D-bar method to Python

The current PyTorch code reconstructs on the full Cartesian square grid.
No unit-disk masking is applied.

"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from linear_operator import LinearOperator as _BaseLinearOperator

__all__ = [
    "DBOperator",
    "make_trig_mode_indices",
    "arc_length_params_square",
    "boundary_points_from_angles",
    "transform_adjacent_to_square_trig",
    "estimate_electrode_nd_map",
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


def get_domain_size(domain_size: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(domain_size, (float, int)):
        return float(domain_size), float(domain_size)
    if len(domain_size) != 2:
        raise ValueError(f"domain_size must be a scalar or length-2 tuple, got {domain_size}.")
    return float(domain_size[0]), float(domain_size[1])


def _default_electrode_data_scale(n_electrodes: int) -> float:
    if n_electrodes < 1:
        raise ValueError(f"n_electrodes must be positive, got {n_electrodes}.")
    return math.pi / float(n_electrodes)


def make_trig_mode_indices(
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
    r"""
    fii: n_boundary_samples sample points around the boundary
    Dfii = 2\pi / n_boundary_samples: the quadrature step size
    """
    if n_boundary_samples < n_electrodes:
        raise ValueError(
            f"n_boundary_samples must be >= n_electrodes, got {n_boundary_samples} and {n_electrodes}."
        )

    Lx, Ly = get_domain_size(domain_size)
    perimeter = 2.0 * (Lx + Ly)
    electrode_edges = torch.linspace(0.0, perimeter, n_electrodes + 1, device=device, dtype=dtype)
    electrode_mid = 0.5 * (electrode_edges[:-1] + electrode_edges[1:])

    fii = torch.linspace(0.0, 2.0 * math.pi, n_boundary_samples + 1, device=device, dtype=dtype)[:-1]
    Dfii = float(2.0 * math.pi / n_boundary_samples)
    gamma_mid = (2.0 * math.pi / perimeter) * electrode_mid
    return fii, Dfii, gamma_mid


def boundary_points_from_angles(
    theta_arc: torch.Tensor,
    *,
    domain_size: float | tuple[float, float] = 2.0,
) -> torch.Tensor:
    theta_arc = torch.as_tensor(theta_arc)
    Lx, Ly = get_domain_size(domain_size)
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
    return z.to(device=theta_arc.device, dtype=torch.complex128)


def trig_current_basis(
    n_electrodes: int,
    *,
    domain_size: float | tuple[float, float] = 2.0,
    trig_mode_indices: torch.Tensor | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if trig_mode_indices is None:
        trig_mode_indices = make_trig_mode_indices(n_electrodes, device=device)
    else:
        trig_mode_indices = torch.as_tensor(trig_mode_indices, device=device)

    trig_mode_indices = trig_mode_indices.to(device=device, dtype=dtype)
    Lx, Ly = get_domain_size(domain_size)
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


def coerce_current_patterns(
    current_patterns: torch.Tensor,
    *,
    n_electrodes: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    patterns = torch.as_tensor(current_patterns, device=device, dtype=dtype)
    if patterns.dim() == 2:
        if patterns.shape[0] == n_electrodes:
            patterns = patterns.unsqueeze(0)
        elif patterns.shape[1] == n_electrodes:
            patterns = patterns.transpose(0, 1).unsqueeze(0)
        else:
            raise ValueError(
                f"current_patterns must have one axis of length {n_electrodes}, got {tuple(patterns.shape)}."
            )
    elif patterns.dim() == 3:
        if patterns.shape[-2] == n_electrodes:
            pass
        elif patterns.shape[-1] == n_electrodes:
            patterns = patterns.transpose(-1, -2)
        else:
            raise ValueError(
                f"current_patterns must have one trailing axis of length {n_electrodes}, got {tuple(patterns.shape)}."
            )
    else:
        raise ValueError(
            f"current_patterns must have shape (L, P), (P, L), (B, L, P), or (B, P, L), got {tuple(patterns.shape)}."
        )

    if patterns.shape[0] == 1 and batch_size > 1:
        patterns = patterns.expand(batch_size, -1, -1)
    elif patterns.shape[0] != batch_size:
        raise ValueError(
            f"current_patterns batch dimension must be 1 or {batch_size}, got {patterns.shape[0]}."
        )

    return patterns


def estimate_electrode_nd_map(
    electrode_voltages: torch.Tensor,
    *,
    current_patterns: torch.Tensor,
    regularization: float = 1e-6,
    rcond: float | None = None,
) -> torch.Tensor:
    voltages, squeeze = _as_batch_measurements(torch.as_tensor(electrode_voltages), "electrode_voltages")
    n_electrodes = voltages.shape[-2]
    currents = coerce_current_patterns(
        current_patterns,
        n_electrodes=n_electrodes,
        batch_size=voltages.shape[0],
        device=voltages.device,
        dtype=voltages.real.dtype if torch.is_complex(voltages) else voltages.dtype,
    )

    
    voltages = voltages.to(dtype=torch.complex64)
    currents = currents.to(dtype=torch.complex64)

    nd_maps = []
    for batch_idx in range(voltages.shape[0]):
        I = currents[batch_idx]
        V = voltages[batch_idx]

        if I.shape != V.shape:
            raise ValueError(
                f"current_patterns and electrode_voltages must have the same electrode/pattern shape, got {tuple(I.shape)} and {tuple(V.shape)}."
            )

        V = V - V.mean(dim=0, keepdim=True)
        norms = torch.linalg.vector_norm(I, dim=0)
        if torch.any(norms <= 0):
            raise ValueError("Each current pattern must have nonzero Euclidean norm.")

        I_norm = I / norms.unsqueeze(0)
        V_norm = V / norms.unsqueeze(0)

        if rcond is None:
            nd_map = V_norm @ torch.linalg.pinv(I_norm)
        else:
            nd_map = V_norm @ torch.linalg.pinv(I_norm, rcond=rcond)

        projector = make_mean_free_projector(
            n_electrodes,
            device=nd_map.device,
            dtype=nd_map.dtype,
        )
        nd_map = projector @ nd_map @ projector
        nd_maps.append(0.5 * (nd_map + nd_map.transpose(-1, -2).conj()))

    out = torch.stack(nd_maps, dim=0)
    return out[0] if squeeze else out


def transform_adjacent_to_square_trig(
    electrode_voltages: torch.Tensor,
    *,
    current_patterns: torch.Tensor | None = None,
    domain_size: float | tuple[float, float] = 2.0,
    trig_mode_indices: torch.Tensor | None = None,
    electrode_data_scale: float | None = None,
    regularization: float = 1e-6,
    rcond: float | None = None,
) -> torch.Tensor:
    voltages, squeeze = _as_batch_measurements(torch.as_tensor(electrode_voltages), "electrode_voltages")
    n_electrodes = voltages.shape[-2]
    _, atrig, _ = trig_current_basis(
        n_electrodes,
        domain_size=domain_size,
        trig_mode_indices=trig_mode_indices,
        device=voltages.device,
        dtype=torch.float32,
    )

    if current_patterns is not None:
        nd_map = estimate_electrode_nd_map(
            voltages,
            current_patterns=current_patterns,
            regularization=regularization,
            rcond=rcond,
        )
        nd_map, _ = _as_batch_square_matrix(nd_map, "nd_map")
        atrig = atrig.to(device=nd_map.device, dtype=nd_map.dtype)
        transformed = nd_map @ atrig.unsqueeze(0).expand(nd_map.shape[0], -1, -1)
        return transformed[0] if squeeze else transformed

    if electrode_data_scale is None:
        electrode_data_scale = _default_electrode_data_scale(n_electrodes)
    coeff, _, _ = trig_current_basis(
        n_electrodes,
        domain_size=domain_size,
        trig_mode_indices=trig_mode_indices,
        device=voltages.device,
        dtype=torch.float32,
    )

    transformed = []
    for batch_idx in range(voltages.shape[0]):
        U = (electrode_data_scale * voltages[batch_idx]).to(dtype=torch.complex128)
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


def trig_boundary_matrix(
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
    current_patterns: torch.Tensor | None = None,
    domain_size: float | tuple[float, float] = 2.0,
    trig_mode_indices: torch.Tensor | None = None,
    n_boundary_samples: int = 512,
    electrode_data_scale: float | None = None,
    regularization: float = 1e-6,
    rcond: float | None = None,
) -> torch.Tensor:
    trig_voltages = transform_adjacent_to_square_trig(
        electrode_voltages,
        current_patterns=current_patterns,
        domain_size=domain_size,
        trig_mode_indices=trig_mode_indices,
        electrode_data_scale=electrode_data_scale,
        regularization=regularization,
        rcond=rcond,
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
        trig_mode_indices = make_trig_mode_indices(n_electrodes, device=trig_voltages.device)
    else:
        trig_mode_indices = torch.as_tensor(trig_mode_indices, device=trig_voltages.device)

    B = trig_boundary_matrix(trig_mode_indices, fii).to(dtype=trig_voltages.dtype)
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
    current_patterns: torch.Tensor | None = None,
    domain_size: float | tuple[float, float] = 2.0,
    trig_mode_indices: torch.Tensor | None = None,
    n_boundary_samples: int = 512,
    regularization: float = 1e-6,
    rcond: float | None = None,
    electrode_data_scale: float | None = None,
) -> torch.Tensor:
    nd_map = build_nd_map_from_electrode_data(
        electrode_voltages,
        current_patterns=current_patterns,
        domain_size=domain_size,
        trig_mode_indices=trig_mode_indices,
        n_boundary_samples=n_boundary_samples,
        electrode_data_scale=electrode_data_scale,
        regularization=regularization,
        rcond=rcond,
    )
    return nd_to_dn_map(nd_map, regularization=regularization, rcond=rcond)


def trig_synthesis_matrix(
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

    theta_arc = theta_arc.to(dtype=torch.float32)
    trig_mode_indices = trig_mode_indices.to(dtype=torch.int64)
    Kvec = Kvec.to(dtype=torch.complex64)
    if Dtheta is None:
        if theta_arc.numel() < 2:
            raise ValueError("Dtheta is required when theta_arc has fewer than two samples.")
        Dtheta = float(theta_arc[1] - theta_arc[0])

    boundary_points = boundary_points_from_angles(theta_arc, domain_size=domain_size).to(dtype=torch.complex64)
    phase = torch.exp(1j * boundary_points[:, None] * Kvec[None, :])

    n_basis = trig_mode_indices.numel()
    split = (n_basis + 1) // 2
    basis = torch.zeros((n_basis, theta_arc.numel()), dtype=torch.float32, device=theta_arc.device)
    trig_mode_indices_real = trig_mode_indices.to(dtype=torch.float32)
    for j in range(n_basis):
        if j < split:
            basis[j, :] = (1.0 / math.sqrt(math.pi)) * torch.cos(trig_mode_indices_real[j] * theta_arc)
        else:
            basis[j, :] = (1.0 / math.sqrt(math.pi)) * torch.sin(trig_mode_indices_real[j] * theta_arc)

    return Dtheta * (basis.to(dtype=torch.complex64) @ phase)


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

    
    if trig_mode_indices is None:
        trig_mode_indices = make_trig_mode_indices(DN.shape[-1] + 1, device=DN.device)
    else:
        trig_mode_indices = torch.as_tensor(trig_mode_indices, device=DN.device)

    Kvec = Kvec.to(dtype=torch.complex64, device=DN.device)
    DN = DN.to(dtype=torch.complex64)
    DN1 = DN1.to(dtype=torch.complex64)
    Fpsi_BIE = Fpsi_BIE.to(dtype=torch.complex64)
    theta_arc = theta_arc.to(dtype=torch.float32)
    if Dtheta is None:
        if theta_arc.numel() < 2:
            raise ValueError("Dtheta is required when theta_arc has fewer than two samples.")
        Dtheta = float(theta_arc[1] - theta_arc[0])

    T_basis = trig_synthesis_matrix(trig_mode_indices, theta_arc).to(dtype=torch.complex64)
    FLLpsi = (DN - DN1) @ Fpsi_BIE
    LLpsi = T_basis @ FLLpsi

    boundary_points = boundary_points_from_angles(theta_arc, domain_size=domain_size).to(dtype=torch.complex64)
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
    K1: torch.Tensor,
    K2: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    K1 = torch.as_tensor(K1)
    K2 = torch.as_tensor(K2, device=K1.device)
    vals = torch.as_tensor(values, device=K1.device)
    x_coords = K1[0, :].to(dtype=torch.float64)
    y_coords = K2[:, 0].to(dtype=torch.float64)
    if x_coords[0] > x_coords[-1]:
        x_coords = torch.flip(x_coords, dims=(0,))
        vals = torch.flip(vals, dims=(1,))
    if y_coords[0] > y_coords[-1]:
        y_coords = torch.flip(y_coords, dims=(0,))
        vals = torch.flip(vals, dims=(0,))
    return x_coords, y_coords, vals


def _interp2_bicubic_rect_grid(
    K1: torch.Tensor,
    K2: torch.Tensor,
    values: torch.Tensor,
    x_query: torch.Tensor,
    y_query: torch.Tensor,
) -> torch.Tensor:
    x_coords, y_coords, values = _rect_grid_coords(K1, K2, values)
    values = values.to(device=x_coords.device)
    
    values = values.to(dtype=torch.complex64)
    x_query = torch.as_tensor(x_query, device=x_coords.device, dtype=x_coords.dtype)
    y_query = torch.as_tensor(y_query, device=x_coords.device, dtype=y_coords.dtype)

    if x_coords.numel() < 2 or y_coords.numel() < 2:
        raise ValueError("Rectangular-grid interpolation requires at least two samples per axis.")

    def _normalize(query: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        if coords.numel() == 1:
            return torch.zeros_like(query)
        span = coords[-1] - coords[0]
        if torch.abs(span) < torch.finfo(coords.dtype).eps:
            return torch.zeros_like(query)
        return 2.0 * (query - coords[0]) / span - 1.0

    grid_x = _normalize(x_query, x_coords)
    grid_y = _normalize(y_query, y_coords)
    sample_grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, -1, 2)
    image = torch.stack((values.real, values.imag), dim=0).unsqueeze(0).to(dtype=x_coords.dtype)
    sampled = F.grid_sample(
        image,
        sample_grid,
        mode="bicubic",
        padding_mode="zeros",
        align_corners=True,
    )
    sampled = sampled.view(2, -1)
    return torch.complex(sampled[0], sampled[1]).to(dtype=torch.complex64)


def _interp_cubic_scattered(
    Kvec: torch.Tensor,
    values: torch.Tensor,
    x_query: torch.Tensor,
    y_query: torch.Tensor,
    *,
    chunk_size: int = 4096,
) -> torch.Tensor:
    Kvec = torch.as_tensor(Kvec)
    device = Kvec.device
    sample_points = torch.stack(
        (
            torch.imag(Kvec.reshape(-1)).to(dtype=torch.float64),
            torch.real(Kvec.reshape(-1)).to(dtype=torch.float64),
        ),
        dim=-1,
    )
    values = torch.as_tensor(values, device=device).reshape(-1)
    
    values = values.to(dtype=torch.complex64)
    query_points = torch.stack(
        (
            torch.as_tensor(y_query, device=device, dtype=torch.float64).reshape(-1),
            torch.as_tensor(x_query, device=device, dtype=torch.float64).reshape(-1),
        ),
        dim=-1,
    )

    if sample_points.shape[0] != values.numel():
        raise ValueError(
            f"Scattered interpolation expects the same number of points and values, got {sample_points.shape[0]} and {values.numel()}."
        )

    if sample_points.shape[0] == 0:
        return torch.zeros(query_points.shape[0], device=device, dtype=torch.complex64)

    sample_extent = sample_points.max(dim=0).values - sample_points.min(dim=0).values
    diagonal = torch.linalg.vector_norm(sample_extent)
    bandwidth = diagonal / max(math.sqrt(float(sample_points.shape[0])), 1.0)
    bandwidth = torch.clamp(bandwidth, min=1e-6)
    inside_bbox = (
        (query_points[:, 0] >= sample_points[:, 0].min())
        & (query_points[:, 0] <= sample_points[:, 0].max())
        & (query_points[:, 1] >= sample_points[:, 1].min())
        & (query_points[:, 1] <= sample_points[:, 1].max())
    )

    output = torch.zeros(query_points.shape[0], device=device, dtype=torch.complex64)
    values_real = values.real.to(dtype=torch.float64)
    values_imag = values.imag.to(dtype=torch.float64)
    log_cutoff = math.log(torch.finfo(torch.float64).tiny)
    bandwidth_sq = bandwidth.square()

    for start in range(0, query_points.shape[0], chunk_size):
        stop = min(start + chunk_size, query_points.shape[0])
        query_chunk = query_points[start:stop]
        d2 = torch.cdist(query_chunk, sample_points).square()
        log_weights = -0.5 * d2 / bandwidth_sq
        weights = torch.exp(torch.clamp(log_weights, min=log_cutoff))
        denom = weights.sum(dim=1)
        valid = (denom > 1e-12) & inside_bbox[start:stop]
        if torch.any(valid):
            numer_real = weights[valid] @ values_real
            numer_imag = weights[valid] @ values_imag
            output_chunk = torch.zeros(stop - start, device=device, dtype=torch.complex64)
            output_chunk[valid] = torch.complex(numer_real / denom[valid], numer_imag / denom[valid]).to(
                dtype=torch.complex64
            )
            output[start:stop] = output_chunk

    return output


def _require_torch_dbar() -> None:
    return None


class DBOperator(_BaseLinearOperator):
    def _init_state(
        self,
        *,
        fundfft: torch.Tensor,
        tr: torch.Tensor,
        rind: torch.Tensor,
        nind: int,
        h: float | torch.Tensor,
    ) -> None:
        self.fundfft = torch.as_tensor(fundfft)
        self.tr = torch.as_tensor(tr, device=self.fundfft.device, dtype=self.fundfft.dtype)
        self.rind = torch.as_tensor(rind, device=self.fundfft.device, dtype=torch.bool)
        self.nind = int(nind)
        self.h = torch.as_tensor(h, device=self.fundfft.device, dtype=self.fundfft.real.dtype)
        self._shape = torch.Size((2 * self.nind, 2 * self.nind))

    def _size(self) -> torch.Size:
        return self._shape

    def _transpose_nonbatch(self) -> "DBOperator":
        return self

    def centered_convolution(self, values: torch.Tensor) -> torch.Tensor:
        shifted = torch.fft.fftshift(values, dim=(-2, -1))
        transformed = torch.fft.fft2(shifted, dim=(-2, -1))
        conv = torch.fft.ifft2(self.fundfft * transformed, dim=(-2, -1))
        return (self.h.square()) * torch.fft.ifftshift(conv, dim=(-2, -1))

    def db_oper_real(self, w_vec: torch.Tensor) -> torch.Tensor:
        real_vec = torch.as_tensor(w_vec, device=self.fundfft.device, dtype=self.fundfft.real.dtype)
        if real_vec.dim() != 1:
            raise ValueError(f"w_vec must be one-dimensional, got shape {tuple(real_vec.shape)}.")
        if real_vec.numel() != 2 * self.nind:
            raise ValueError(
                f"w_vec must have length {2 * self.nind}, got {real_vec.numel()}."
            )

        w = torch.zeros_like(self.tr)
        w[self.rind] = torch.complex(real_vec[: self.nind], real_vec[self.nind :])
        conv = self.centered_convolution(self.tr * torch.conj(w))
        out = w - conv
        return torch.cat((out.real[self.rind], out.imag[self.rind]), dim=0)

    def _matmul(self, rhs: torch.Tensor) -> torch.Tensor:
        rhs = torch.as_tensor(rhs, device=self.fundfft.device, dtype=self.fundfft.real.dtype)
        if rhs.dim() == 1:
            return self.db_oper_real(rhs)
        if rhs.dim() == 2:
            return torch.stack([self.db_oper_real(rhs[:, idx]) for idx in range(rhs.shape[-1])], dim=-1)
        raise ValueError(f"rhs must have shape ({self._shape[-1]},) or ({self._shape[-1]}, C), got {tuple(rhs.shape)}.")

    def matvec(self, rhs: torch.Tensor) -> torch.Tensor:
        return self._matmul(rhs)

    def __init__(
        self,
        *,
        fundfft: torch.Tensor,
        tr: torch.Tensor,
        rind: torch.Tensor,
        nind: int,
        h: float | torch.Tensor,
    ) -> None:
        self._init_state(fundfft=fundfft, tr=tr, rind=rind, nind=nind, h=h)
        super().__init__(fundfft=self.fundfft, tr=self.tr, rind=self.rind, nind=self.nind, h=self.h)


def gmres(
    operator: DBOperator,
    rhs: torch.Tensor,
    *,
    x0: torch.Tensor | None = None,
    restart: int = 50,
    rtol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int = 500,
) -> tuple[torch.Tensor, int]:
    if restart < 1:
        raise ValueError(f"restart must be positive, got {restart}.")
    if rtol <= 0:
        raise ValueError(f"rtol must be positive, got {rtol}.")
    if atol < 0:
        raise ValueError(f"atol must be non-negative, got {atol}.")
    if maxiter < 1:
        raise ValueError(f"maxiter must be positive, got {maxiter}.")

    rhs = torch.as_tensor(rhs, device=operator.fundfft.device, dtype=operator.fundfft.real.dtype)
    if rhs.dim() != 1:
        raise ValueError(f"rhs must be one-dimensional, got shape {tuple(rhs.shape)}.")

    x = torch.zeros_like(rhs) if x0 is None else torch.as_tensor(x0, device=rhs.device, dtype=rhs.dtype).clone()
    krylov_dim = min(restart, rhs.numel())
    rhs_norm = torch.linalg.vector_norm(rhs)
    tolerance = max(rtol * float(rhs_norm.item()), atol)
    breakdown_tol = torch.finfo(rhs.dtype).eps

    for _ in range(maxiter):
        residual = rhs - operator.matvec(x)
        beta = torch.linalg.vector_norm(residual)
        if float(beta.item()) <= tolerance:
            return x, 0

        V = torch.zeros((rhs.numel(), krylov_dim + 1), device=rhs.device, dtype=rhs.dtype)
        H = torch.zeros((krylov_dim + 1, krylov_dim), device=rhs.device, dtype=rhs.dtype)
        V[:, 0] = residual / beta

        best_x = x
        best_residual = float(beta.item())
        for j in range(krylov_dim):
            w = operator.matvec(V[:, j])
            for i in range(j + 1):
                hij = torch.dot(V[:, i], w)
                H[i, j] = hij
                w = w - hij * V[:, i]

            h_next = torch.linalg.vector_norm(w)
            H[j + 1, j] = h_next

            Hj = H[: j + 2, : j + 1]
            e1 = torch.zeros(j + 2, device=rhs.device, dtype=rhs.dtype)
            e1[0] = beta
            y = torch.linalg.lstsq(Hj, e1).solution
            x_candidate = x + V[:, : j + 1] @ y

            candidate_residual = rhs - operator.matvec(x_candidate)
            candidate_norm = torch.linalg.vector_norm(candidate_residual)
            candidate_norm_value = float(candidate_norm.item())
            if candidate_norm_value < best_residual:
                best_residual = candidate_norm_value
                best_x = x_candidate

            if candidate_norm_value <= tolerance:
                return x_candidate, 0

            if float(h_next.item()) <= breakdown_tol:
                x = best_x
                break

            V[:, j + 1] = w / h_next
        else:
            x = best_x
            continue

    return x, maxiter


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
    """
    n_electrodes: total number of electrodes on the boundary (evenly spaced around the perimeter).
    """
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

    
    currents = currents.to(dtype=torch.complex64)
    voltages = voltages.to(dtype=torch.complex64)

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
    
    nd_map = nd_map.to(dtype=torch.complex64)

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
        domain_size: float | tuple[float, float] = 2.0,
        inverse_method: str = "born",
        gmres_restart: int = 50,
        gmres_rtol: float = 1e-5,
        gmres_maxiter: int = 500,
    ) -> None:
        super().__init__()
        inverse_method = inverse_method.lower()
        if inverse_method not in {"born", "dbar"}:
            raise ValueError(
                f"inverse_method must be 'born' or 'dbar', got {inverse_method}."
            )
        if gmres_restart < 1:
            raise ValueError(f"gmres_restart must be positive, got {gmres_restart}.")
        if gmres_rtol <= 0:
            raise ValueError(f"gmres_rtol must be positive, got {gmres_rtol}.")
        if gmres_maxiter < 1:
            raise ValueError(f"gmres_maxiter must be positive, got {gmres_maxiter}.")
        if inverse_method == "dbar":
            _require_torch_dbar()

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
        self.domain_size = domain_size
        self.inverse_method = inverse_method
        self.gmres_restart = gmres_restart
        self.gmres_rtol = gmres_rtol
        self.gmres_maxiter = gmres_maxiter

        if self.n_electrodes % 2 != 0:
            raise ValueError("Square-domain D-bar requires an even number of electrodes.")
        if self.n_modes != self.n_electrodes - 1:
            raise ValueError(
                f"For square domain, expected n_modes == n_electrodes - 1, got {self.n_modes} and {self.n_electrodes}."
            )
        theta, boundary_weight, _ = arc_length_params_square(
            self.n_electrodes,
            domain_size=domain_size,
            n_boundary_samples=n_boundary_nodes,
        )
        boundary_points = boundary_points_from_angles(theta, domain_size=domain_size)
        trig_mode_indices = make_trig_mode_indices(self.n_electrodes)
        lambda_ref = torch.zeros((self.n_modes, self.n_modes), dtype=torch.float32)

        Lx, Ly = get_domain_size(domain_size)
        x_coords = torch.linspace(0.0, Lx, image_size)
        y_coords = torch.linspace(0.0, Ly, image_size)

        # matching forward_solver's grid
        xx, yy = torch.meshgrid(x_coords, y_coords, indexing="ij")
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
            f"inverse_method={self.inverse_method}, image_size={self.image_size}, "
            f"n_boundary_nodes={self.n_boundary_nodes}, n_modes={self.n_modes}, domain_size={self.domain_size}, "
            f"k_grid_size={self.k_grid_size}, k_radius={self.k_radius}, k_extent={self.k_extent}"
        )

    def compute_scattering_transform(
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

        
        k_mask = self.k_mask.reshape(-1)
        kvec = self.k_grid.reshape(-1)[k_mask].to(dtype=torch.complex64)
        theta_arc = self.theta.to(dtype=self.boundary_points.real.dtype)
        Dtheta = float(self.boundary_weight.item())
        Fpsi_BIE = compute_psi_BIE_square(
            kvec,
            theta_arc,
            self.trig_mode_indices,
            domain_size=self.domain_size,
            Dtheta=Dtheta,
        ).to(device=self.k_grid.device, dtype=torch.complex64)

        out = torch.zeros(
            (lambda_sigma.shape[0], self.k_grid_size, self.k_grid_size),
            dtype=torch.complex64,
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
            ).to(dtype=torch.complex64, device=self.k_grid.device)
            flat = torch.zeros(self.k_grid.numel(), dtype=torch.complex64, device=self.k_grid.device)
            flat[k_mask] = self.scattering_scale * tbie
            out[batch_idx] = flat.reshape(self.k_grid_size, self.k_grid_size)
        return out

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
        
        device = self.k_grid.device
        out = torch.zeros(
            (tBIE.shape[0], self.k_grid_size, self.k_grid_size),
            device=device,
            dtype=torch.complex64,
        )

        query_mask = self.k_mask.reshape(-1)
        query_x = self.k_grid.real.reshape(-1)[query_mask].to(device=device)
        query_y = self.k_grid.imag.reshape(-1)[query_mask].to(device=device)

        K1_t = None if K1 is None else torch.as_tensor(K1, device=device)
        K2_t = None if K2 is None else torch.as_tensor(K2, device=device)
        Kvec_t = None if Kvec is None else torch.as_tensor(Kvec, device=device)

        for b in range(tBIE.shape[0]):
            sample = tBIE[b].to(device=device, dtype=torch.complex64)

            if K1_t is not None:
                assert K2_t is not None
                if sample.ndim == 1:
                    if t_max is None:
                        raise ValueError("t_max is required when tBIE is provided as vector samples.")
                    scat_grid = torch.zeros_like(K1_t, dtype=torch.complex64)
                    inside = torch.abs(K1_t.to(dtype=torch.float64) + 1j * K2_t.to(dtype=torch.float64)) < float(t_max)
                    if sample.numel() != int(inside.sum().item()):
                        raise ValueError(
                            f"Vector tBIE length {sample.numel()} does not match the number of |K|<t_max points {int(inside.sum().item())}."
                        )
                    scat_grid[inside] = sample.reshape(-1)
                else:
                    scat_grid = sample

                if cutoff is not None:
                    scat_grid = scat_grid.clone()
                    scat_grid[torch.abs(torch.real(scat_grid)) > cutoff] = 0
                    scat_grid[torch.abs(torch.imag(scat_grid)) > cutoff] = 0

                interp_vals = _interp2_bicubic_rect_grid(K1_t, K2_t, scat_grid, query_x, query_y)
            else:
                assert Kvec_t is not None
                if cutoff is not None:
                    sample = sample.clone()
                    sample[torch.abs(torch.real(sample)) > cutoff] = 0
                    sample[torch.abs(torch.imag(sample)) > cutoff] = 0
                interp_vals = _interp_cubic_scattered(Kvec_t, sample, query_x, query_y)

            flat = torch.zeros(self.k_grid.numel(), device=device, dtype=torch.complex64)
            flat[query_mask] = interp_vals
            out[b] = flat.reshape(self.k_grid_size, self.k_grid_size)

        return out[0] if squeeze else out

    def _clamp_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        if self.conductivity_min is not None:
            sigma = sigma.clamp_min(self.conductivity_min)
        if self.conductivity_max is not None:
            sigma = sigma.clamp_max(self.conductivity_max)
        return sigma.unsqueeze(1).to(dtype=self.z_grid.real.dtype)

    def _solve_sigma_born(self, scattering_transform: torch.Tensor) -> torch.Tensor:
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

        sigma = self._clamp_sigma(sigma)
        return sigma[0] if squeeze else sigma

    def _solve_sigma_dbar(self, scattering_transform: torch.Tensor) -> torch.Tensor:
        """Solve the real-linear D-bar equation for mu(z, k) and reconstruct sigma(z)=|mu(z,0)|^2."""
        _require_torch_dbar()

        scattering_transform, squeeze = _as_batch_scattering_grid(
            scattering_transform,
            "scattering_transform",
            self.k_grid_size,
        )

        device = scattering_transform.device

        k_grid = self.k_grid.to(device=device, dtype=torch.complex64)
        rind = self.k_mask.to(device=device, dtype=torch.bool)
        nind = int(rind.sum().item())
        if nind == 0:
            raise ValueError("The D-bar solve requires at least one k-grid point inside the truncation mask.")

        ktmp = k_grid.clone()
        ind0 = k_grid.abs() < 1e-14
        if not torch.any(ind0):
            ind0 = k_grid.abs() == k_grid.abs().min()
        ktmp[ind0] = torch.ones_like(ktmp[ind0])

        scatk_scale = torch.zeros_like(k_grid)
        scatk_scale[rind] = 1.0 / torch.conj(ktmp[rind])
        scatk_scale[ind0] = 0.0

        fund = torch.zeros_like(k_grid)
        fund[rind] = 1.0 / (math.pi * ktmp[rind])
        fund[ind0] = 0.0

        s = float(torch.abs(k_grid.real.min()).item())
        ep = s / 10.0 if s > 0 else 0.0
        rr = (s - ep) / 2.0 if s > 0 else 0.0
        radius = k_grid.abs()
        bigind = radius >= s
        fund[bigind] = 0.0
        if ep > 0:
            medind = (radius < s) & (radius > 2.0 * rr)
            fund[medind] *= 1.0 - (radius[medind] - 2.0 * rr) / ep

        fundfft = torch.fft.fft2(torch.fft.fftshift(fund, dim=(-2, -1)), dim=(-2, -1))
        rhs = torch.cat(
            (
                torch.ones(nind, device=device, dtype=torch.float32),
                torch.zeros(nind, device=device, dtype=torch.float32),
            )
        )
        z_flat = self.z_grid.to(device=device, dtype=torch.complex64).reshape(-1)
        recon = torch.empty(
            (scattering_transform.shape[0], z_flat.numel()),
            device=device,
            dtype=self.z_grid.real.dtype,
        )

        self_h = float(self.dk.item())
        k_rind = k_grid[rind]
        zero_idx = int(torch.argmin(torch.abs(k_rind)).item())
        for batch_idx in range(scattering_transform.shape[0]):
            scatk = scattering_transform[batch_idx].to(device=device, dtype=torch.complex64) * scatk_scale
            scatk[ind0] = 0.0
            init_guess = rhs.clone()
            for z_idx, z in enumerate(z_flat):
                tr = (1.0 / (4.0 * math.pi)) * scatk * torch.exp(
                    -1j * (k_grid * z + torch.conj(k_grid * z))
                )
                operator = DBOperator(
                    fundfft=fundfft,
                    tr=tr,
                    rind=rind,
                    nind=nind,
                    h=self_h,
                )
                solution, info = gmres(
                    operator,
                    rhs,
                    x0=init_guess,
                    restart=self.gmres_restart,
                    rtol=self.gmres_rtol,
                    atol=0.0,
                    maxiter=self.gmres_maxiter,
                )
                if info != 0:
                    raise RuntimeError(
                        f"GMRES failed for batch {batch_idx} at spatial index {z_idx} with info={info}."
                    )
                init_guess = solution
                mu_rind = torch.complex(solution[:nind], solution[nind:])
                recon[batch_idx, z_idx] = mu_rind[zero_idx].abs().square().to(dtype=recon.dtype)

        sigma = recon.to(device=self.z_grid.device, dtype=self.z_grid.real.dtype)
        sigma = sigma.reshape(scattering_transform.shape[0], self.image_size, self.image_size)
        sigma = self._clamp_sigma(sigma)
        return sigma[0] if squeeze else sigma

    def solve_sigma(self, scattering_transform: torch.Tensor) -> torch.Tensor:
        if self.inverse_method == "born":
            return self._solve_sigma_born(scattering_transform)
        return self._solve_sigma_dbar(scattering_transform)

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
        
        device = self.z_grid.device

        lambda_sigma = lambda_sigma.to(device=device, dtype=torch.complex64)

        if lambda_ref is None:
            raise ValueError("Square-domain D-bar reconstruction requires an explicit reference DN map.")
        else:
            lambda_ref_batch, _ = _as_batch_square_matrix(lambda_ref, "lambda_ref")
            lambda_ref_batch = lambda_ref_batch.to(device=device, dtype=torch.complex64)

        if lambda_ref_batch.shape[0] == 1 and lambda_sigma.shape[0] > 1:
            lambda_ref_batch = lambda_ref_batch.expand(lambda_sigma.shape[0], -1, -1)
        elif lambda_ref_batch.shape[0] != lambda_sigma.shape[0]:
            raise ValueError(
                f"lambda_ref batch dimension must be 1 or {lambda_sigma.shape[0]}, got {lambda_ref_batch.shape[0]}."
            )

        t_exp = self.compute_scattering_transform(lambda_sigma, lambda_ref_batch)
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
        if reference_voltages is None:
            raise ValueError("Square-domain D-bar reconstruction requires reference_voltages.")

        if reference_currents is None:
            reference_currents = currents

        lambda_sigma = build_dn_map_from_electrode_data(
            voltages,
            current_patterns=currents,
            domain_size=self.domain_size,
            trig_mode_indices=self.trig_mode_indices,
            n_boundary_samples=self.n_boundary_nodes,
            regularization=regularization,
        )
        lambda_ref = build_dn_map_from_electrode_data(
            reference_voltages,
            current_patterns=reference_currents,
            domain_size=self.domain_size,
            trig_mode_indices=self.trig_mode_indices,
            n_boundary_samples=self.n_boundary_nodes,
            regularization=regularization,
        )
        return self.forward(lambda_sigma=lambda_sigma, lambda_ref=lambda_ref)

