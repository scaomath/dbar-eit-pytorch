import math
from typing import Tuple, Union

import torch
import torch.nn as nn

from .forward_solver import DiffusionEquation2D


def generate_electrodes(
    n_electrodes: int,
    domain_size: Union[float, Tuple[float, float]] = 1.0,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> torch.Tensor:
    r"""Build equally sized electrode arcs on the boundary of a square domain.

    The electrodes are parameterized by arc length along the perimeter,
    starting at 0 and proceeding counterclockwise. The result stores the
    start and end arc coordinate for each electrode.

    Parameters
    ----------
    n_electrodes : int
        Number of electrodes on the boundary.
    domain_size : float or tuple of float, default=1.0
        Side length for a square domain, or (Lx, Ly) for a rectangle.
    dtype : torch.dtype, default=torch.float64
        Tensor dtype for the returned electrode intervals.
    device : torch.device or None, default=None
        Device on which to allocate the result.

    Returns
    -------
    electrodes : torch.Tensor
        Tensor of shape (n_electrodes, 2). Each row contains the start and end
        arc-length coordinate of one electrode.
    """
    if n_electrodes < 1:
        raise ValueError(f"n_electrodes must be positive, got {n_electrodes}.")

    if isinstance(domain_size, (float, int)):
        Lx = Ly = float(domain_size)
    else:
        Lx, Ly = map(float, domain_size)

    perimeter = 2.0 * (Lx + Ly)
    edges = torch.linspace(0.0, perimeter, n_electrodes + 1, dtype=dtype, device=device)
    return torch.stack((edges[:-1], edges[1:]), dim=1)


def generate_adjacent_current_patterns(
    n_electrodes: int,
    *,
    amplitude: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> torch.Tensor:
    r"""Construct adjacent current-injection patterns for boundary electrodes.

    Column: one pattern.
    Electrode j injects the specified
    amplitude and electrode j+1 mod n_electrodes withdraws the same amount,
    so every pattern has zero net current.

    Parameters
    ----------
    n_electrodes : int
        Number of electrodes.
    amplitude : float, default=1.0
        Magnitude of the injected and withdrawn current.
    dtype : torch.dtype, default=torch.float64
        Tensor dtype for the returned pattern matrix.
    device : torch.device or None, default=None
        Device on which to allocate the result.

    Returns
    -------
    patterns : torch.Tensor
        Tensor of shape (n_electrodes, n_electrodes). Column j is the j-th
        adjacent current pattern.
    """
    patterns = torch.zeros((n_electrodes, n_electrodes), dtype=dtype, device=device)
    idx = torch.arange(n_electrodes, device=device)
    patterns[idx, idx] = amplitude
    patterns[(idx + 1) % n_electrodes, idx] = -amplitude
    return patterns


class CompleteElectrodeModel(nn.Module):
    def __init__(self, solver: DiffusionEquation2D, **kwargs):
        super().__init__()
        if not isinstance(solver, DiffusionEquation2D):
            raise TypeError("solver must be a DiffusionEquation2D instance.")
        if solver.Nx_int < 1 or solver.Ny_int < 1:
            raise ValueError("CEM requires at least one interior node in each direction.")
        self.solver = solver

    @property
    def device(self) -> torch.device:
        return self.solver.x_nodes.device

    @property
    def dtype(self) -> torch.dtype:
        return self.solver.dtype

    def _normalize_contact_impedance(self, z_contact, n_electrodes: int) -> torch.Tensor:
        r"""Convert contact impedance data to a validated per-electrode tensor."""
        if isinstance(z_contact, torch.Tensor):
            z = z_contact.to(device=self.device, dtype=self.dtype)
        else:
            z = torch.as_tensor(z_contact, device=self.device, dtype=self.dtype)

        if z.dim() == 0:
            z = z.expand(n_electrodes)
        elif z.shape != (n_electrodes,):
            raise ValueError(
                f"z_contact must be scalar or shape ({n_electrodes},), got {tuple(z.shape)}."
            )

        if torch.any(z <= 0):
            raise ValueError("z_contact must be strictly positive.")
        return z

    def _boundary_face_data(self):
        idx_grid_int = torch.arange(self.solver.M, device=self.device).reshape(
            self.solver.Nx_int, self.solver.Ny_int
        )
        faces = []

        for i in range(self.solver.Nx_int):
            node = int(idx_grid_int[i, 0].item())
            arc = (i + 1) * self.solver.hx
            faces.append((node, arc, self.solver.hx, self.solver.hy))

        for j in range(self.solver.Ny_int):
            node = int(idx_grid_int[self.solver.Nx_int - 1, j].item())
            arc = self.solver.Lx + (j + 1) * self.solver.hy
            faces.append((node, arc, self.solver.hy, self.solver.hx))

        for i in range(self.solver.Nx_int - 1, -1, -1):
            node = int(idx_grid_int[i, self.solver.Ny_int - 1].item())
            arc = 2.0 * self.solver.Lx + self.solver.Ly - (i + 1) * self.solver.hx
            faces.append((node, arc, self.solver.hx, self.solver.hy))

        for j in range(self.solver.Ny_int - 1, -1, -1):
            node = int(idx_grid_int[0, j].item())
            arc = 2.0 * self.solver.Lx + 2.0 * self.solver.Ly - (j + 1) * self.solver.hy
            faces.append((node, arc, self.solver.hy, self.solver.hx))

        return faces

    def _paste_electrode_boundary(self, electrode_potentials: torch.Tensor) -> torch.Tensor:
        batch = electrode_potentials.shape[0]
        boundary = torch.zeros(
            (batch, self.solver.Nx, self.solver.Ny),
            dtype=electrode_potentials.dtype,
            device=electrode_potentials.device,
        )

        n_electrodes = electrode_potentials.shape[1]
        perimeter = 2.0 * (self.solver.Lx + self.solver.Ly)
        electrode_length = perimeter / float(n_electrodes)

        def electrode_idx(arc: float) -> int:
            idx = int(math.floor((arc % perimeter) / electrode_length))
            return min(idx, n_electrodes - 1)

        for i in range(self.solver.Nx):
            arc = i * self.solver.hx
            boundary[:, i, 0] = electrode_potentials[:, electrode_idx(arc)]

        for j in range(1, self.solver.Ny):
            arc = self.solver.Lx + j * self.solver.hy
            boundary[:, self.solver.Nx - 1, j] = electrode_potentials[:, electrode_idx(arc)]

        for i in range(self.solver.Nx - 2, -1, -1):
            arc = 2.0 * self.solver.Lx + self.solver.Ly - i * self.solver.hx
            boundary[:, i, self.solver.Ny - 1] = electrode_potentials[:, electrode_idx(arc)]

        for j in range(self.solver.Ny - 2, 0, -1):
            arc = 2.0 * self.solver.Lx + 2.0 * self.solver.Ly - j * self.solver.hy
            boundary[:, 0, j] = electrode_potentials[:, electrode_idx(arc)]

        return boundary

    def assemble_cem_system(
        self,
        sigma: torch.Tensor,
        *,
        n_electrodes: int,
        z_contact=1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Assemble the dense linear system blocks for the complete electrode model."""
        sigma = sigma.to(device=self.device, dtype=self.dtype)
        z = self._normalize_contact_impedance(z_contact, n_electrodes)

        k_e, k_w, k_n, k_s = self.solver._get_face_coefs(sigma)
        hx2 = self.solver.hx * self.solver.hx
        hy2 = self.solver.hy * self.solver.hy

        diag = (k_e + k_w) / hx2 + (k_n + k_s) / hy2
        diag = diag.clone()
        diag[0, :] -= k_w[0, :] / hx2
        diag[-1, :] -= k_e[-1, :] / hx2
        diag[:, 0] -= k_s[:, 0] / hy2
        diag[:, -1] -= k_n[:, -1] / hy2

        A = torch.zeros((self.solver.M, self.solver.M), dtype=self.dtype, device=self.device)
        idx_grid_int = torch.arange(self.solver.M, device=self.device).reshape(
            self.solver.Nx_int, self.solver.Ny_int
        )
        flat_idx = torch.arange(self.solver.M, device=self.device)
        A[flat_idx, flat_idx] = diag.ravel()

        if self.solver.Nx_int > 1:
            rows = idx_grid_int[:-1, :].ravel()
            cols = idx_grid_int[1:, :].ravel()
            vals = (-k_e[:-1, :] / hx2).ravel()
            A[rows, cols] = vals
            A[cols, rows] = vals

        if self.solver.Ny_int > 1:
            rows = idx_grid_int[:, :-1].ravel()
            cols = idx_grid_int[:, 1:].ravel()
            vals = (-k_n[:, :-1] / hy2).ravel()
            A[rows, cols] = vals
            A[cols, rows] = vals

        C = torch.zeros((self.solver.M, n_electrodes), dtype=self.dtype, device=self.device)
        E = torch.zeros((n_electrodes, self.solver.M), dtype=self.dtype, device=self.device)
        D = torch.zeros((n_electrodes, n_electrodes), dtype=self.dtype, device=self.device)

        perimeter = 2.0 * (self.solver.Lx + self.solver.Ly)
        electrode_length = perimeter / float(n_electrodes)
        counts = torch.zeros(n_electrodes, dtype=torch.int64, device=self.device)

        for node_idx, arc, face_length, normal_spacing in self._boundary_face_data():
            electrode_idx = min(int(math.floor(arc / electrode_length)), n_electrodes - 1)
            diag_coeff = 1.0 / (z[electrode_idx] * normal_spacing)
            current_coeff = face_length / z[electrode_idx]

            A[node_idx, node_idx] += diag_coeff
            C[node_idx, electrode_idx] += diag_coeff
            E[electrode_idx, node_idx] += current_coeff
            D[electrode_idx, electrode_idx] += current_coeff
            counts[electrode_idx] += 1

        if torch.any(counts == 0):
            raise ValueError(
                "Each electrode must cover at least one boundary face. "
                "Use fewer electrodes or a finer grid."
            )

        return A, C, E, D

    def solve_cem(
        self,
        sigma: torch.Tensor,
        currents: torch.Tensor,
        *,
        z_contact=1.0,
        n_electrodes: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Solve the complete electrode model for one or more current patterns."""
        currents = torch.as_tensor(currents, dtype=self.dtype, device=self.device)
        squeeze = currents.dim() == 1
        if squeeze:
            currents = currents.unsqueeze(0)
        elif currents.dim() != 2:
            raise ValueError(
                f"currents must have shape (L,) or (P, L), got {tuple(currents.shape)}."
            )

        n_patterns, inferred_electrodes = currents.shape
        if n_electrodes is None:
            n_electrodes = inferred_electrodes
        elif inferred_electrodes != n_electrodes:
            raise ValueError(
                f"currents have {inferred_electrodes} electrodes, expected {n_electrodes}."
            )
        n_electrodes = int(n_electrodes)

        if not torch.allclose(
            currents.sum(dim=-1),
            torch.zeros(n_patterns, dtype=self.dtype, device=currents.device),
            atol=1e-10,
            rtol=1e-8,
        ):
            raise ValueError("Each current pattern must sum to zero.")

        A, C, E, D = self.assemble_cem_system(
            sigma,
            n_electrodes=n_electrodes,
            z_contact=z_contact,
        )

        n_free = n_electrodes - 1
        system = torch.zeros(
            (self.solver.M + n_free, self.solver.M + n_free),
            dtype=self.dtype,
            device=self.device,
        )
        system[: self.solver.M, : self.solver.M] = A
        system[: self.solver.M, self.solver.M :] = -C[:, :n_free]
        system[self.solver.M :, : self.solver.M] = -E[:n_free, :]
        system[self.solver.M :, self.solver.M :] = D[:n_free, :n_free]

        rhs = torch.zeros(
            (self.solver.M + n_free, n_patterns),
            dtype=self.dtype,
            device=self.device,
        )
        rhs[self.solver.M :, :] = currents[:, :n_free].transpose(0, 1)

        sol = torch.linalg.solve(system, rhs)
        u_int = sol[: self.solver.M, :].transpose(0, 1)
        electrode_potentials = torch.zeros(
            (n_patterns, n_electrodes),
            dtype=self.dtype,
            device=self.device,
        )
        electrode_potentials[:, :n_free] = sol[self.solver.M :, :].transpose(0, 1)

        u = self._paste_electrode_boundary(electrode_potentials)
        u[:, 1:-1, 1:-1] = u_int.reshape(n_patterns, self.solver.Nx_int, self.solver.Ny_int)

        if squeeze:
            return u[0], electrode_potentials[0]
        return u, electrode_potentials

    def neumann_to_dirichlet(
        self,
        sigma: torch.Tensor,
        *,
        n_electrodes: int = 16,
        z_contact=1.0,
        current_patterns: torch.Tensor | None = None,
        rcond: float = 1e-6,
    ) -> torch.Tensor:
        r"""Build an electrode-level Neumann-to-Dirichlet map from CEM solves."""
        if current_patterns is None:
            current_matrix = generate_adjacent_current_patterns(
                n_electrodes,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            current_matrix = torch.as_tensor(
                current_patterns,
                dtype=self.dtype,
                device=self.device,
            )
            if current_matrix.dim() != 2:
                raise ValueError(
                    f"current_patterns must have shape (L, P) or (P, L), got {tuple(current_matrix.shape)}."
                )
            if current_matrix.shape[0] != n_electrodes and current_matrix.shape[1] != n_electrodes:
                raise ValueError(
                    f"current_patterns must include {n_electrodes} electrodes, got {tuple(current_matrix.shape)}."
                )

        if current_matrix.shape[0] == n_electrodes:
            currents_batch = current_matrix.transpose(0, 1)
            currents_for_fit = current_matrix
        else:
            currents_batch = current_matrix
            currents_for_fit = current_matrix.transpose(0, 1)

        _, electrode_voltages = self.solve_cem(
            sigma,
            currents_batch,
            z_contact=z_contact,
            n_electrodes=n_electrodes,
        )
        voltage_matrix = electrode_voltages.transpose(0, 1)
        nd_map = voltage_matrix @ torch.linalg.pinv(currents_for_fit, rtol=rcond)
        return 0.5 * (nd_map + nd_map.transpose(0, 1))

    def dirichlet_to_neumann(
        self,
        sigma: torch.Tensor,
        *,
        n_electrodes: int = 16,
        z_contact=1.0,
        current_patterns: torch.Tensor | None = None,
        rcond: float = 1e-6,
    ) -> torch.Tensor:
        r"""Build an electrode-level Dirichlet-to-Neumann map from the ND map."""
        nd_map = self.neumann_to_dirichlet(
            sigma,
            n_electrodes=n_electrodes,
            z_contact=z_contact,
            current_patterns=current_patterns,
            rcond=rcond,
        )
        dn_map = torch.linalg.pinv(nd_map, rtol=rcond)
        return 0.5 * (dn_map + dn_map.transpose(0, 1))