import math
import warnings
from typing import Callable, Tuple, Union

import torch
import torch.nn as nn
import torch.sparse


def electrode_builder_square(
    n_electrodes: int,
    domain_size: Union[float, Tuple[float, float]] = 1.0,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> torch.Tensor:
    if n_electrodes < 1:
        raise ValueError(f"n_electrodes must be positive, got {n_electrodes}.")

    if isinstance(domain_size, (float, int)):
        Lx = Ly = float(domain_size)
    else:
        Lx, Ly = map(float, domain_size)

    perimeter = 2.0 * (Lx + Ly)
    edges = torch.linspace(0.0, perimeter, n_electrodes + 1, dtype=dtype, device=device)
    return torch.stack((edges[:-1], edges[1:]), dim=1)


def make_adjacent_current_patterns(
    n_electrodes: int,
    *,
    amplitude: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> torch.Tensor:
    patterns = torch.zeros((n_electrodes, n_electrodes), dtype=dtype, device=device)
    idx = torch.arange(n_electrodes, device=device)
    patterns[idx, idx] = amplitude
    patterns[(idx + 1) % n_electrodes, idx] = -amplitude
    return patterns


class DiffusionEquation2D(nn.Module):
    def __init__(
        self,
        grid_size: Union[int, Tuple[int, int]],
        domain_size: Union[float, Tuple[float, float]] = 1.0,
        grid_type: str = "staggered", # "node" or "staggered"
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.grid_type = grid_type
        if isinstance(grid_size, int):
            self.Nx = self.Ny = grid_size
        else:
            self.Nx, self.Ny = grid_size
            
        if isinstance(domain_size, (float, int)):
            Lx = Ly = float(domain_size)
        else:
            Lx, Ly = domain_size

        self.Lx = float(Lx)
        self.Ly = float(Ly)
            
        self.hx = Lx / (self.Nx - 1)
        self.hy = Ly / (self.Ny - 1)
        
        self.dtype = dtype
        self.tensor_device = device
        
        # Grid coordinates
        x = torch.linspace(0, Lx, self.Nx, dtype=dtype, device=device)
        y = torch.linspace(0, Ly, self.Ny, dtype=dtype, device=device)
        self.x_nodes, self.y_nodes = torch.meshgrid(x, y, indexing='ij')

        # Interior mapped to 1D
        self.Nx_int = self.Nx - 2
        self.Ny_int = self.Ny - 2
        self.M = self.Nx_int * self.Ny_int
        
        # Nodal indices
        idx_grid = torch.arange(self.Nx * self.Ny, device=device).reshape(self.Nx, self.Ny)
        self.register_buffer("idx_map", idx_grid)
        self.register_buffer("int_idx", idx_grid[1:-1, 1:-1].ravel())
        
        # Boundary indices
        bnd_mask = torch.ones((self.Nx, self.Ny), dtype=torch.bool, device=device)
        bnd_mask[1:-1, 1:-1] = False
        self.register_buffer("bnd_idx", idx_grid[bnd_mask])

    def _get_face_coefs(self, sigma: torch.Tensor):
        if self.grid_type == "staggered":
            # sigma is (Nc, Nc) = (Nx-1, Ny-1) at cell centers.
            # Interior nodes: (Nx_int, Ny_int) = (Nc-1, Nc-1).
            # Each face conductivity is the arithmetic mean of the two cell-center
            # values that straddle that face.  All four arrays have shape (Nc-1, Nc-1).
            #
            #  k_e[i,j]: east face of interior node (i,j), at x=(i+1.5)h, y=(j+1)h
            #             flanked by cells (i+1,j) and (i+1,j+1)
            k_e = 0.5 * (sigma[1:, :-1] + sigma[1:, 1:])
            #  k_w[i,j]: west face, at x=(i+0.5)h, y=(j+1)h
            #             flanked by cells (i,j) and (i,j+1)
            k_w = 0.5 * (sigma[:-1, :-1] + sigma[:-1, 1:])
            #  k_n[i,j]: north face, at x=(i+1)h, y=(j+1.5)h
            #             flanked by cells (i,j+1) and (i+1,j+1)
            k_n = 0.5 * (sigma[:-1, 1:] + sigma[1:, 1:])
            #  k_s[i,j]: south face, at x=(i+1)h, y=(j+0.5)h
            #             flanked by cells (i,j) and (i+1,j)
            k_s = 0.5 * (sigma[:-1, :-1] + sigma[1:, :-1])
        else:  # node-centered
            # sigma is (Nx, Ny) — same grid as solution nodes.
            # Face value is the arithmetic mean of the two adjacent nodal sigma values.
            # All four arrays have shape (Nx_int, Ny_int) = (Nx-2, Ny-2).
            coef_int = sigma[1:-1, 1:-1]
            k_e = 0.5 * (sigma[2:, 1:-1] + coef_int)
            k_w = 0.5 * (sigma[:-2, 1:-1] + coef_int)
            k_n = 0.5 * (sigma[1:-1, 2:] + coef_int)
            k_s = 0.5 * (sigma[1:-1, :-2] + coef_int)
        return k_e, k_w, k_n, k_s

    def assemble_stiffness(self, sigma: torch.Tensor):
        k_e, k_w, k_n, k_s = self._get_face_coefs(sigma)
        
        diag_data = (k_e + k_w + k_n + k_s).ravel()
        idx_grid_int = torch.arange(self.M, device=self.x_nodes.device).reshape(self.Nx_int, self.Ny_int)

        rows = []
        cols = []
        data = []
        
        # Diagonal
        rows.append(torch.arange(self.M, device=self.x_nodes.device))
        cols.append(torch.arange(self.M, device=self.x_nodes.device))
        data.append(diag_data)
        
        # East/West: coupling along the x-axis (first grid dimension).
        # The face between interior nodes (i,j) and (i+1,j) carries conductivity k_e[i,j].
        data_e = -k_e[:-1, :].ravel()
        rows.append(idx_grid_int[:-1, :].ravel())
        cols.append(idx_grid_int[1:, :].ravel())
        data.append(data_e)
        # Symmetric (west) entry
        rows.append(idx_grid_int[1:, :].ravel())
        cols.append(idx_grid_int[:-1, :].ravel())
        data.append(data_e)

        # North/South: coupling along the y-axis (second grid dimension).
        # The face between interior nodes (i,j) and (i,j+1) carries conductivity k_n[i,j].
        data_n = -k_n[:, :-1].ravel()
        rows.append(idx_grid_int[:, :-1].ravel())
        cols.append(idx_grid_int[:, 1:].ravel())
        data.append(data_n)
        # Symmetric (south) entry
        rows.append(idx_grid_int[:, 1:].ravel())
        cols.append(idx_grid_int[:, :-1].ravel())
        data.append(data_n)

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        indices = torch.stack([rows, cols], dim=0)
        data = torch.cat(data) / (self.hx * self.hy)
        
        A_II = torch.sparse_coo_tensor(indices, data, size=(self.M, self.M)).to_sparse_csr()
        
        # For A_IB, we just construct dense explicitly in a simple way
        # 0: left, -1: right, 0: top, -1: bottom
        A_matrix = torch.zeros((self.Nx * self.Ny, self.Nx * self.Ny), dtype=self.dtype, device=self.x_nodes.device)
        # Populate A_full (slow but ok for demo)
        int_idx = self.int_idx
        # We can construct interior -> boundary couplings.
        
        return A_II, None # simplify for now

    def _parse_bc(self, bc) -> torch.Tensor:
        if callable(bc):
            return torch.as_tensor(
                bc(self.x_nodes, self.y_nodes),
                dtype=self.dtype,
                device=self.x_nodes.device,
            )
        elif isinstance(bc, tuple) and len(bc) == 4:
            b_tensor = torch.zeros((self.Nx, self.Ny), dtype=self.dtype, device=self.x_nodes.device)
            b_tensor[0, :] = bc[0] # left
            b_tensor[-1, :] = bc[1] # right
            b_tensor[:, 0] = bc[2] # bottom
            b_tensor[:, -1] = bc[3] # top
            return b_tensor
        return torch.as_tensor(bc, dtype=self.dtype, device=self.x_nodes.device)

    def _solve_linear(self, A_csr, rhs):
        # Dense solver for now
        A_dense = A_csr.to_dense()
        return torch.linalg.solve(A_dense, rhs)

    def _normalize_contact_impedance(self, z_contact, n_electrodes: int) -> torch.Tensor:
        device = self.x_nodes.device
        if isinstance(z_contact, torch.Tensor):
            z = z_contact.to(device=device, dtype=self.dtype)
        else:
            z = torch.as_tensor(z_contact, device=device, dtype=self.dtype)

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
        idx_grid_int = torch.arange(self.M, device=self.x_nodes.device).reshape(
            self.Nx_int, self.Ny_int
        )
        faces = []

        for i in range(self.Nx_int):
            node = int(idx_grid_int[i, 0].item())
            arc = (i + 1) * self.hx
            faces.append((node, arc, self.hx, self.hy))

        for j in range(self.Ny_int):
            node = int(idx_grid_int[self.Nx_int - 1, j].item())
            arc = self.Lx + (j + 1) * self.hy
            faces.append((node, arc, self.hy, self.hx))

        for i in range(self.Nx_int - 1, -1, -1):
            node = int(idx_grid_int[i, self.Ny_int - 1].item())
            arc = 2.0 * self.Lx + self.Ly - (i + 1) * self.hx
            faces.append((node, arc, self.hx, self.hy))

        for j in range(self.Ny_int - 1, -1, -1):
            node = int(idx_grid_int[0, j].item())
            arc = 2.0 * self.Lx + 2.0 * self.Ly - (j + 1) * self.hy
            faces.append((node, arc, self.hy, self.hx))

        return faces

    def _paste_electrode_boundary(self, electrode_potentials: torch.Tensor) -> torch.Tensor:
        batch = electrode_potentials.shape[0]
        boundary = torch.zeros(
            (batch, self.Nx, self.Ny),
            dtype=electrode_potentials.dtype,
            device=electrode_potentials.device,
        )

        n_electrodes = electrode_potentials.shape[1]
        perimeter = 2.0 * (self.Lx + self.Ly)
        electrode_length = perimeter / float(n_electrodes)

        def electrode_idx(arc: float) -> int:
            idx = int(math.floor((arc % perimeter) / electrode_length))
            return min(idx, n_electrodes - 1)

        for i in range(self.Nx):
            arc = i * self.hx
            boundary[:, i, 0] = electrode_potentials[:, electrode_idx(arc)]

        for j in range(1, self.Ny):
            arc = self.Lx + j * self.hy
            boundary[:, self.Nx - 1, j] = electrode_potentials[:, electrode_idx(arc)]

        for i in range(self.Nx - 2, -1, -1):
            arc = 2.0 * self.Lx + self.Ly - i * self.hx
            boundary[:, i, self.Ny - 1] = electrode_potentials[:, electrode_idx(arc)]

        for j in range(self.Ny - 2, 0, -1):
            arc = 2.0 * self.Lx + 2.0 * self.Ly - j * self.hy
            boundary[:, 0, j] = electrode_potentials[:, electrode_idx(arc)]

        return boundary

    def assemble_cem_system(
        self,
        sigma: torch.Tensor,
        *,
        n_electrodes: int,
        z_contact=1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.Nx_int < 1 or self.Ny_int < 1:
            raise ValueError("CEM requires at least one interior node in each direction.")

        device = self.x_nodes.device
        sigma = sigma.to(device=device, dtype=self.dtype)
        z = self._normalize_contact_impedance(z_contact, n_electrodes)

        k_e, k_w, k_n, k_s = self._get_face_coefs(sigma)
        hx2 = self.hx * self.hx
        hy2 = self.hy * self.hy

        diag = (k_e + k_w) / hx2 + (k_n + k_s) / hy2
        diag = diag.clone()
        diag[0, :] -= k_w[0, :] / hx2
        diag[-1, :] -= k_e[-1, :] / hx2
        diag[:, 0] -= k_s[:, 0] / hy2
        diag[:, -1] -= k_n[:, -1] / hy2

        A = torch.zeros((self.M, self.M), dtype=self.dtype, device=device)
        idx_grid_int = torch.arange(self.M, device=device).reshape(self.Nx_int, self.Ny_int)
        flat_idx = torch.arange(self.M, device=device)
        A[flat_idx, flat_idx] = diag.ravel()

        if self.Nx_int > 1:
            rows = idx_grid_int[:-1, :].ravel()
            cols = idx_grid_int[1:, :].ravel()
            vals = (-k_e[:-1, :] / hx2).ravel()
            A[rows, cols] = vals
            A[cols, rows] = vals

        if self.Ny_int > 1:
            rows = idx_grid_int[:, :-1].ravel()
            cols = idx_grid_int[:, 1:].ravel()
            vals = (-k_n[:, :-1] / hy2).ravel()
            A[rows, cols] = vals
            A[cols, rows] = vals

        C = torch.zeros((self.M, n_electrodes), dtype=self.dtype, device=device)
        E = torch.zeros((n_electrodes, self.M), dtype=self.dtype, device=device)
        D = torch.zeros((n_electrodes, n_electrodes), dtype=self.dtype, device=device)

        perimeter = 2.0 * (self.Lx + self.Ly)
        electrode_length = perimeter / float(n_electrodes)
        counts = torch.zeros(n_electrodes, dtype=torch.int64, device=device)

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

        return A, C, D

    def solve_cem(
        self,
        sigma: torch.Tensor,
        currents: torch.Tensor,
        *,
        z_contact=1.0,
        n_electrodes: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        currents = torch.as_tensor(currents, dtype=self.dtype, device=self.x_nodes.device)
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

        A, C, D = self.assemble_cem_system(
            sigma,
            n_electrodes=n_electrodes,
            z_contact=z_contact,
        )

        E = torch.zeros((n_electrodes, self.M), dtype=self.dtype, device=self.x_nodes.device)
        perimeter = 2.0 * (self.Lx + self.Ly)
        electrode_length = perimeter / float(n_electrodes)
        z = self._normalize_contact_impedance(z_contact, n_electrodes)
        for node_idx, arc, face_length, _ in self._boundary_face_data():
            electrode_idx = min(int(math.floor(arc / electrode_length)), n_electrodes - 1)
            E[electrode_idx, node_idx] += face_length / z[electrode_idx]

        n_free = n_electrodes - 1
        system = torch.zeros(
            (self.M + n_free, self.M + n_free),
            dtype=self.dtype,
            device=self.x_nodes.device,
        )
        system[: self.M, : self.M] = A
        system[: self.M, self.M :] = -C[:, :n_free]
        system[self.M :, : self.M] = -E[:n_free, :]
        system[self.M :, self.M :] = D[:n_free, :n_free]

        rhs = torch.zeros(
            (self.M + n_free, n_patterns),
            dtype=self.dtype,
            device=self.x_nodes.device,
        )
        rhs[self.M :, :] = currents[:, :n_free].transpose(0, 1)

        sol = torch.linalg.solve(system, rhs)
        u_int = sol[: self.M, :].transpose(0, 1)
        electrode_potentials = torch.zeros(
            (n_patterns, n_electrodes),
            dtype=self.dtype,
            device=self.x_nodes.device,
        )
        electrode_potentials[:, :n_free] = sol[self.M :, :].transpose(0, 1)

        u = self._paste_electrode_boundary(electrode_potentials)
        u[:, 1:-1, 1:-1] = u_int.reshape(n_patterns, self.Nx_int, self.Ny_int)

        if squeeze:
            return u[0], electrode_potentials[0]
        return u, electrode_potentials

    def solve(self, sigma, f=None, bc=None, bc_type="dirichlet"):
        g_bnd = self._parse_bc(bc)
        A_II, _ = self.assemble_stiffness(sigma)
        
        if f is None:
            f = torch.zeros((self.Nx, self.Ny), dtype=self.dtype, device=self.x_nodes.device)
            
        rhs = f[1:-1, 1:-1].flatten().clone()
        
        # Add boundary conditions to rhs
        # Simplification: we'll just evaluate the A matrix action explicitly since full construction was skipped
        k_e, k_w, k_n, k_s = self._get_face_coefs(sigma)
        
        # Left boundary
        rhs[:self.Ny_int] -= -k_w[0, :] * g_bnd[0, 1:-1] / (self.hx * self.hy)
        # Right boundary
        rhs[-self.Ny_int:] -= -k_e[-1, :] * g_bnd[-1, 1:-1] / (self.hx * self.hy)
        # Bottom boundary
        v = -k_s[:, 0] * g_bnd[1:-1, 0] / (self.hx * self.hy)
        rhs[0::self.Ny_int] -= v
        # Top boundary
        v2 = -k_n[:, -1] * g_bnd[1:-1, -1] / (self.hx * self.hy)
        rhs[self.Ny_int-1::self.Ny_int] -= v2

        u_int = self._solve_linear(A_II, rhs)
        
        u = g_bnd.clone()
        u[1:-1, 1:-1] = u_int.reshape(self.Nx_int, self.Ny_int)
        return u
        
    def get_flux(
        self, sigma: torch.Tensor, u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Compute the flux field :math:`\mathbf{F} = -\sigma \nabla u` on staggered face grids.

        The two components are sampled on staggered faces between consecutive
        nodes:

        * F_x — at x-faces, shape (Nx_int+1, Ny_int)
        * F_y — at y-faces, shape (Nx_int, Ny_int+1)

        where Nx_int = Nx - 2 and Ny_int = Ny - 2 are the counts of
        interior nodes in each direction.

        Note
            The leading minus sign is included so that
            self.get_div(self.get_flux(sigma, u)) reproduces
            forward(), i.e. it equals -\nabla \cdot (\sigma \nabla u).

        Parameters
        ----------
        sigma : torch.Tensor
            Conductivity field.  Shape (Nx, Ny) for grid_type="node"
            or (Nx-1, Ny-1) for grid_type="staggered".
        u : torch.Tensor
            Full nodal solution, shape (Nx, Ny), including boundary values.

        Returns
        -------
        Fx : torch.Tensor
            x-component of -\sigma \nabla u, shape (Nx_int+1, Ny_int).
        Fy : torch.Tensor
            y-component of -\sigma \nabla u, shape (Nx_int, Ny_int+1).
        """
        k_e, k_w, k_n, k_s = self._get_face_coefs(sigma)
        # All four arrays have shape (Nx_int, Ny_int).

        # --- x-faces --------------------------------------------------------
        # Nx_int+1 faces per y-interior column.
        # Face 0      : west boundary face of the first interior column
        #               → conductivity k_w[0, :]
        # Faces 1..Nx_int : east face of each interior column (ends at the right
        #               boundary face) → conductivity k_e[0:, :]
        sigma_fx = torch.cat([k_w[0:1, :], k_e], dim=0)          # (Nx_int+1, Ny_int)
        # \partial u / \partial x at each x-face (forward difference).
        # u[0:Nx_int+1, 1:-1]  — left  node of each face, shape (Nx_int+1, Ny_int)
        # u[1:Nx_int+2, 1:-1]  — right node of each face, shape (Nx_int+1, Ny_int)
        grad_x = (u[1:self.Nx_int + 2, 1:-1] - u[0:self.Nx_int + 1, 1:-1]) / self.hx
        Fx = -sigma_fx * grad_x

        # --- y-faces --------------------------------------------------------
        # Ny_int+1 faces per x-interior row.
        # Face 0      : south boundary face of the first interior row
        #               → conductivity k_s[:, 0]
        # Faces 1..Ny_int : north face of each interior row (ends at the top
        #               boundary face) → conductivity k_n[:, 0:]
        sigma_fy = torch.cat([k_s[:, 0:1], k_n], dim=1)           # (Nx_int, Ny_int+1)
        grad_y = (u[1:-1, 1:self.Ny_int + 2] - u[1:-1, 0:self.Ny_int + 1]) / self.hy
        Fy = -sigma_fy * grad_y

        return Fx, Fy

    def get_div(self, flux: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        r"""Compute :math:`\nabla \cdot \mathbf{F}` at every interior node.

        The flux tuple is expected on the staggered face grids produced by
        :meth:`get_flux`:

        * Fx — (Nx_int+1, Ny_int)
        * Fy — (Nx_int, Ny_int+1)

        The divergence is approximated by first-order finite differences across
        adjacent faces:

            (\nabla \cdot \mathbf{F})_{i,j}
            \approx \frac{F_x^{i+1,j} - F_x^{i,j}}{h_x}
                  + \frac{F_y^{i,j+1} - F_y^{i,j}}{h_y}

        Parameters
        ----------
        flux : tuple of two torch.Tensor
            (Fx, Fy) as returned by get_flux.

        Returns
        -------
        result : torch.Tensor
            Full (Nx, Ny) tensor; interior entries hold
            \nabla \cdot \mathbf{F}, boundary entries are zero.
        """
        Fx, Fy = flux
        div_x = (Fx[1:, :] - Fx[:-1, :]) / self.hx   # (Nx_int, Ny_int)
        div_y = (Fy[:, 1:] - Fy[:, :-1]) / self.hy   # (Nx_int, Ny_int)
        result = torch.zeros((self.Nx, self.Ny), dtype=Fx.dtype, device=Fx.device)
        result[1:-1, 1:-1] = div_x + div_y
        return result

    def forward(self, sigma: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        r"""Apply the diffusion operator :math:`-\nabla \cdot (\sigma \nabla u)` at every interior node.

        Implemented as :math:`\nabla \cdot \mathbf{F}` where
        :math:`\mathbf{F} = -\sigma \nabla u` is computed by :meth:`get_flux`
        and the divergence is taken by :meth:`get_div`.

        Parameters
        ----------
        sigma : torch.Tensor
            Conductivity field.  Shape ``(Nx, Ny)`` for ``grid_type="node"``
            or ``(Nx-1, Ny-1)`` for ``grid_type="staggered"``.
        u : torch.Tensor
            Full nodal solution, shape ``(Nx, Ny)``, including boundary values.

        Returns
        -------
        result : torch.Tensor
            ``(Nx, Ny)`` tensor; interior entries hold
            :math:`-\nabla \cdot (\sigma \nabla u)`, boundary entries are zero.
        """
        return self.get_div(self.get_flux(sigma, u))

    def neumann_to_dirichlet(
        self,
        sigma: torch.Tensor,
        *,
        n_electrodes: int = 16,
        z_contact=1.0,
        current_patterns: torch.Tensor | None = None,
        rcond: float = 1e-6,
    ) -> torch.Tensor:
        if current_patterns is None:
            current_matrix = make_adjacent_current_patterns(
                n_electrodes,
                dtype=self.dtype,
                device=self.x_nodes.device,
            )
        else:
            current_matrix = torch.as_tensor(
                current_patterns,
                dtype=self.dtype,
                device=self.x_nodes.device,
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
        nd_map = self.neumann_to_dirichlet(
            sigma,
            n_electrodes=n_electrodes,
            z_contact=z_contact,
            current_patterns=current_patterns,
            rcond=rcond,
        )
        dn_map = torch.linalg.pinv(nd_map, rtol=rcond)
        return 0.5 * (dn_map + dn_map.transpose(0, 1))
