import sys
import os
from forward_solver import DiffusionEquation2D

import math
import torch
from absl.testing import absltest
from absl.testing import parameterized

class TestForwardSolver(parameterized.TestCase):

    @parameterized.named_parameters(
        ('linear_node', 'node', 'linear'),
        ('linear_staggered', 'staggered', 'linear'),
        ('harmonic_quad_node', 'node', 'harmonic_quad'),
        ('harmonic_quad_staggered', 'staggered', 'harmonic_quad'),
    )
    def test_solve_dirichlet(self, grid_type, solution_type):
        n = 16
        eq = DiffusionEquation2D(grid_size=n, domain_size=1.0, grid_type=grid_type)
        
        if grid_type == 'node':
            sigma = torch.ones((n, n), dtype=torch.float64)
        else:
            sigma = torch.ones((n-1, n-1), dtype=torch.float64)

        if solution_type == 'linear':
            # u(x, y) = x
            def bc(x, y): return x
            expected = eq.x_nodes[1:-1, 1:-1]
            
        elif solution_type == 'harmonic_quad':
            # u(x, y) = x^2 - y^2
            def bc(x, y): return x**2 - y**2
            expected = (eq.x_nodes**2 - eq.y_nodes**2)[1:-1, 1:-1]
            
        else:
            raise ValueError()

        u = eq.solve(sigma, f=None, bc=bc)
        torch.testing.assert_close(u[1:-1, 1:-1], expected, atol=1e-12, rtol=1e-5)

    @parameterized.named_parameters(
        ('square_node',         'node',       16,       1.0),
        ('square_staggered',    'staggered',  16,       1.0),
        ('nonsquare_node',      'node',       (20, 10), (2.0, 1.0)),
        ('nonsquare_staggered', 'staggered',  (20, 10), (2.0, 1.0)),
    )
    def test_forward_laplacian(self, grid_type, grid_size, domain_size):
        r"""For :math:`\sigma = 1` and :math:`u = x^2 + y^2`,
        :meth:`forward` should return :math:`-\nabla \cdot (\sigma \nabla u) = -4`
        at every interior node, for both grid types and non-square grids.
        """
        eq = DiffusionEquation2D(
            grid_size=grid_size, domain_size=domain_size, grid_type=grid_type,
        )
        if grid_type == 'node':
            sigma = torch.ones((eq.Nx, eq.Ny), dtype=torch.float64)
        else:
            sigma = torch.ones((eq.Nx - 1, eq.Ny - 1), dtype=torch.float64)

        u = eq.x_nodes ** 2 + eq.y_nodes ** 2
        result = eq.forward(sigma, u)
        expected = torch.full((eq.Nx_int, eq.Ny_int), -4.0, dtype=torch.float64)
        torch.testing.assert_close(result[1:-1, 1:-1], expected, atol=1e-10, rtol=0.0)


if __name__ == '__main__':
    absltest.main()
