from dbar import (
    DbarReconstruction2D,
    arc_length_params_square,
    build_dn_map_from_electrode_data,
    build_nd_map_from_electrode_data,
    make_square_trig_mode_indices,
    square_boundary_points_from_angles,
    compute_psi_BIE_square,
    compute_tBIE_square,
)
from forward_solver import DiffusionEquation2D, make_adjacent_current_patterns

import math
import torch
from absl.testing import absltest
from absl.testing import parameterized


class TestDbarBoundary(parameterized.TestCase):
    """Tests for square boundary cases.
    Project CEM electrode measurements onto the trig basis used in the D-bar formulation
    (port `transfrom_Adj2Trig.m` and `build_ND_from_kit4` from the MATLAB code) and check that the resulting ND and DN maps are finite and Hermitian.
    """
    def test_make_square_trig_mode_indices(self):
        expected = torch.tensor([1, 2, 3, 4, 3, 2, 1], dtype=torch.int64)
        torch.testing.assert_close(make_square_trig_mode_indices(8), expected)

    def test_arc_length_params_square(self):
        fii, Dfii, gamma_mid = arc_length_params_square(8, domain_size=2.0, n_boundary_samples=16)
        expected_gamma = torch.tensor(
            [
                math.pi / 8.0,
                3.0 * math.pi / 8.0,
                5.0 * math.pi / 8.0,
                7.0 * math.pi / 8.0,
                9.0 * math.pi / 8.0,
                11.0 * math.pi / 8.0,
                13.0 * math.pi / 8.0,
                15.0 * math.pi / 8.0,
            ],
            dtype=torch.float64,
        )
        self.assertLen(fii, 16)
        self.assertAlmostEqual(Dfii, 2.0 * math.pi / 16.0)
        torch.testing.assert_close(gamma_mid, expected_gamma)

    @parameterized.named_parameters(
        ("node", "node"),
        ("staggered", "staggered"),
    )
    def test_square_trig_nd_dn_from_cem(self, grid_type):
        eq = DiffusionEquation2D(grid_size=16, domain_size=2.0, grid_type=grid_type)
        if grid_type == "node":
            sigma = torch.ones((eq.Nx, eq.Ny), dtype=torch.float64)
        else:
            sigma = torch.ones((eq.Nx - 1, eq.Ny - 1), dtype=torch.float64)

        currents = make_adjacent_current_patterns(8, dtype=torch.float64).transpose(0, 1)
        _, electrode_voltages = eq.solve_cem(sigma, currents, z_contact=1.0, n_electrodes=8)
        voltage_matrix = electrode_voltages.transpose(0, 1)

        nd_map = build_nd_map_from_electrode_data(
            voltage_matrix,
            domain_size=2.0,
            n_boundary_samples=128,
        )
        dn_map = build_dn_map_from_electrode_data(
            voltage_matrix,
            domain_size=2.0,
            n_boundary_samples=128,
        )

        self.assertEqual(nd_map.shape, (7, 7))
        self.assertEqual(dn_map.shape, (7, 7))
        self.assertTrue(torch.isfinite(nd_map).all().item())
        self.assertTrue(torch.isfinite(dn_map).all().item())
        torch.testing.assert_close(nd_map, nd_map.transpose(0, 1).conj(), atol=1e-10, rtol=0.0)
        torch.testing.assert_close(dn_map, dn_map.transpose(0, 1).conj(), atol=1e-10, rtol=0.0)

class TestDbarCGO(absltest.TestCase):
    r"""Tests for the CGO solution of the BIE on the square boundary.
    For each k in the k-grid (|k| < R_freq), compute the boundary trace of the
    Faddeev CGO solution \psi(x, k) on the square perimeter. This replaces `comp04_psi_BIE.m` in the original D-bar MATLAB code
    (which uses a BIE on the unit circle; here we use the square perimeter).
    """
    def test_square_boundary_points_from_angles(self):
        theta = torch.tensor(
            [0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0],
            dtype=torch.float64,
        )
        points = square_boundary_points_from_angles(theta, domain_size=2.0)
        expected = torch.tensor([0.0 + 0.0j, 2.0 + 0.0j, 2.0 + 2.0j, 0.0 + 2.0j], dtype=torch.complex128)
        torch.testing.assert_close(points, expected)

    def test_compute_psi_BIE_square_zero_k(self):
        theta_arc, Dtheta, _ = arc_length_params_square(8, domain_size=2.0, n_boundary_samples=128)
        ntrig = make_square_trig_mode_indices(8)
        fpsi = compute_psi_BIE_square(
            torch.tensor([0.0 + 0.0j], dtype=torch.complex128),
            theta_arc,
            ntrig,
            domain_size=2.0,
            Dtheta=Dtheta,
        )
        torch.testing.assert_close(fpsi, torch.zeros_like(fpsi), atol=1e-8, rtol=0.0)

    def test_compute_tBIE_square_zero_reference(self):
        eq = DiffusionEquation2D(grid_size=16, domain_size=2.0, grid_type="node")
        sigma = torch.ones((eq.Nx, eq.Ny), dtype=torch.float64)
        currents = make_adjacent_current_patterns(8, dtype=torch.float64).transpose(0, 1)
        _, electrode_voltages = eq.solve_cem(sigma, currents, z_contact=1.0, n_electrodes=8)
        voltage_matrix = electrode_voltages.transpose(0, 1)
        dn_map = build_dn_map_from_electrode_data(voltage_matrix, domain_size=2.0, n_boundary_samples=128)

        theta_arc, Dtheta, _ = arc_length_params_square(8, domain_size=2.0, n_boundary_samples=128)
        ntrig = make_square_trig_mode_indices(8)
        kvec = torch.tensor([0.0 + 0.0j, 0.25 + 0.5j, -0.4 + 0.1j], dtype=torch.complex128)
        fpsi = compute_psi_BIE_square(kvec, theta_arc, ntrig, domain_size=2.0, Dtheta=Dtheta)
        tbie = compute_tBIE_square(
            Kvec=kvec,
            DN=dn_map,
            DN1=dn_map,
            Fpsi_BIE=fpsi,
            theta_arc=theta_arc,
            domain_size=2.0,
            trig_mode_indices=ntrig,
            Dtheta=Dtheta,
        )
        torch.testing.assert_close(tbie, torch.zeros_like(tbie), atol=1e-10, rtol=0.0)


class TestDbarReconstruction(absltest.TestCase):
    def test_forward_square_zero_scattering_returns_ones(self):
        eq = DiffusionEquation2D(grid_size=16, domain_size=2.0, grid_type="node")
        sigma = torch.ones((eq.Nx, eq.Ny), dtype=torch.float64)
        currents = make_adjacent_current_patterns(8, dtype=torch.float64).transpose(0, 1)
        _, electrode_voltages = eq.solve_cem(sigma, currents, z_contact=1.0, n_electrodes=8)
        voltage_matrix = electrode_voltages.transpose(0, 1)
        dn_map = build_dn_map_from_electrode_data(voltage_matrix, domain_size=2.0, n_boundary_samples=128)

        recon = DbarReconstruction2D(
            image_size=16,
            n_boundary_nodes=128,
            n_modes=7,
            n_electrodes=8,
            k_grid_size=16,
            k_radius=2.0,
            domain_shape="square",
            domain_size=2.0,
        )
        sigma_rec = recon.forward(lambda_sigma=dn_map, lambda_ref=dn_map)
        self.assertEqual(sigma_rec.shape, (1, 16, 16))
        torch.testing.assert_close(sigma_rec, torch.ones_like(sigma_rec), atol=1e-10, rtol=0.0)

    def test_forward_from_measurements_square_zero_scattering_returns_ones(self):
        eq = DiffusionEquation2D(grid_size=16, domain_size=2.0, grid_type="node")
        sigma = torch.ones((eq.Nx, eq.Ny), dtype=torch.float64)
        currents = make_adjacent_current_patterns(8, dtype=torch.float64).transpose(0, 1)
        _, electrode_voltages = eq.solve_cem(sigma, currents, z_contact=1.0, n_electrodes=8)
        voltage_matrix = electrode_voltages.transpose(0, 1)

        recon = DbarReconstruction2D(
            image_size=16,
            n_boundary_nodes=128,
            n_modes=7,
            n_electrodes=8,
            k_grid_size=16,
            k_radius=2.0,
            domain_shape="square",
            domain_size=2.0,
        )
        sigma_rec = recon.forward_from_measurements(
            currents=currents,
            voltages=voltage_matrix,
            reference_voltages=voltage_matrix,
        )
        self.assertEqual(sigma_rec.shape, (1, 16, 16))
        torch.testing.assert_close(sigma_rec, torch.ones_like(sigma_rec), atol=1e-10, rtol=0.0)

if __name__ == "__main__":
    absltest.main()