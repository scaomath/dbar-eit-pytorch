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
        grid_type: str = "staggered",
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
        
        # East
        data_e = -k_e[:, :-1].ravel()
        rows.append(idx_grid_int[:, :-1].ravel())
        cols.append(idx_grid_int[:, 1:].ravel())
        data.append(data_e)
        
        # West
        data_w = -k_e[:, :-1].ravel()
        rows.append(idx_grid_int[:, 1:].ravel())
        cols.append(idx_grid_int[:, :-1].ravel())
        data.append(data_w)
        
        # North
        data_n = -k_n[:-1, :].ravel()
        rows.append(idx_grid_int[:-1, :].ravel())
        cols.append(idx_grid_int[1:, :].ravel())
        data.append(data_n)
        
        # South
        data_s = -k_n[:-1, :].ravel()
        rows.append(idx_grid_int[1:, :].ravel())
        cols.append(idx_grid_int[:-1, :].ravel())
        data.append(data_s)

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
        
    def forward(self, sigma, u):
        pass

    def dirichlet_to_neumann(self, sigma):
        pass

    def neumann_to_dirichlet(self, sigma):
        pass
