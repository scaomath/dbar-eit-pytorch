import math
import warnings
from typing import Callable, Tuple, Union

import torch
import torch.nn as nn
import torch.sparse

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
            
        self.hx = Lx / (self.Nx - 1)
        self.hy = Ly / (self.Ny - 1)
        
        self.dtype = dtype
        self.device = device
        
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
        idx_grid_int = torch.arange(self.M, device=self.device).reshape(self.Nx_int, self.Ny_int)

        rows = []
        cols = []
        data = []
        
        # Diagonal
        rows.append(torch.arange(self.M, device=self.device))
        cols.append(torch.arange(self.M, device=self.device))
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
        A_matrix = torch.zeros((self.Nx * self.Ny, self.Nx * self.Ny), dtype=self.dtype, device=self.device)
        # Populate A_full (slow but ok for demo)
        int_idx = self.int_idx
        # We can construct interior -> boundary couplings.
        
        return A_II, None # simplify for now

    def _parse_bc(self, bc):
        if callable(bc):
            return bc(self.x_nodes, self.y_nodes)
        elif isinstance(bc, tuple) and len(bc) == 4:
            b_tensor = torch.zeros((self.Nx, self.Ny), dtype=self.dtype, device=self.device)
            b_tensor[0, :] = bc[0] # left
            b_tensor[-1, :] = bc[1] # right
            b_tensor[:, 0] = bc[2] # bottom
            b_tensor[:, -1] = bc[3] # top
            return b_tensor
        return bc

    def _solve_linear(self, A_csr, rhs):
        # Dense solver for now
        A_dense = A_csr.to_dense()
        return torch.linalg.solve(A_dense, rhs)

    def solve(self, sigma, f=None, bc=None, bc_type="dirichlet"):
        g_bnd = self._parse_bc(bc)
        A_II, _ = self.assemble_stiffness(sigma)
        
        if f is None:
            f = torch.zeros((self.Nx, self.Ny), dtype=self.dtype, device=self.device)
            
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

        * :math:`F_x` — at x-faces, shape ``(Nx\_int+1, Ny\_int)``
        * :math:`F_y` — at y-faces, shape ``(Nx\_int, Ny\_int+1)``

        where ``Nx_int = Nx - 2`` and ``Ny_int = Ny - 2`` are the counts of
        interior nodes in each direction.

        .. note::
            The leading minus sign is included so that
            ``self.get_div(self.get_flux(sigma, u))`` reproduces
            :meth:`forward`, i.e. it equals :math:`-\nabla \cdot (\sigma \nabla u)`.

        Parameters
        ----------
        sigma : torch.Tensor
            Conductivity field.  Shape ``(Nx, Ny)`` for ``grid_type="node"``
            or ``(Nx-1, Ny-1)`` for ``grid_type="staggered"``.
        u : torch.Tensor
            Full nodal solution, shape ``(Nx, Ny)``, including boundary values.

        Returns
        -------
        Fx : torch.Tensor
            x-component of :math:`-\sigma \nabla u`, shape ``(Nx\_int+1, Ny\_int)``.
        Fy : torch.Tensor
            y-component of :math:`-\sigma \nabla u`, shape ``(Nx\_int, Ny\_int+1)``.
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

        * ``Fx`` — shape ``(Nx\_int+1, Ny\_int)``
        * ``Fy`` — shape ``(Nx\_int, Ny\_int+1)``

        The divergence is approximated by first-order finite differences across
        adjacent faces:

        .. math::

            (\nabla \cdot \mathbf{F})_{i,j}
            \approx \frac{F_x^{i+1,j} - F_x^{i,j}}{h_x}
                  + \frac{F_y^{i,j+1} - F_y^{i,j}}{h_y}

        Parameters
        ----------
        flux : tuple of two torch.Tensor
            ``(Fx, Fy)`` as returned by :meth:`get_flux`.

        Returns
        -------
        result : torch.Tensor
            Full ``(Nx, Ny)`` tensor; interior entries hold
            :math:`\nabla \cdot \mathbf{F}`, boundary entries are zero.
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

    def dirichlet_to_neumann(self, sigma):
        pass

    def neumann_to_dirichlet(self, sigma):
        pass
