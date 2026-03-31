from dbar import (
    DBOperator,
    DbarReconstruction2D,
    _interp2_bicubic_rect_grid,
    _interp_cubic_scattered,
    arc_length_params_square,
    build_dn_map_from_electrode_data,
    build_nd_map_from_electrode_data,
    make_trig_mode_indices,
    boundary_points_from_angles,
    compute_psi_BIE_square,
    compute_tBIE_square,
)
from cem_solver import CompleteElectrodeModel, generate_adjacent_current_patterns
from forward_solver import DiffusionEquation2D

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
        torch.testing.assert_close(make_trig_mode_indices(8), expected)

    def test_build_nd_map_uses_default_electrode_scale(self):
        eq = DiffusionEquation2D(grid_size=16, domain_size=2.0, grid_type="node")
        cem = CompleteElectrodeModel(eq)
        sigma = torch.ones((eq.Nx, eq.Ny), dtype=torch.float64)
        currents = generate_adjacent_current_patterns(8, dtype=torch.float64).transpose(0, 1)
        _, electrode_voltages = cem.solve_cem(sigma, currents, z_contact=1.0, n_electrodes=8)
        voltage_matrix = electrode_voltages.transpose(0, 1)

        nd_default = build_nd_map_from_electrode_data(
            voltage_matrix,
            domain_size=2.0,
            n_boundary_samples=128,
        )
        nd_manual = build_nd_map_from_electrode_data(
            voltage_matrix,
            domain_size=2.0,
            n_boundary_samples=128,
            electrode_data_scale=math.pi / 8.0,
        )
        nd_unscaled = build_nd_map_from_electrode_data(
            voltage_matrix,
            domain_size=2.0,
            n_boundary_samples=128,
            electrode_data_scale=1.0,
        )

        torch.testing.assert_close(nd_default, nd_manual, atol=1e-10, rtol=0.0)
        torch.testing.assert_close(nd_default, (math.pi / 8.0) * nd_unscaled, atol=1e-10, rtol=1e-10)

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
        cem = CompleteElectrodeModel(eq)
        if grid_type == "node":
            sigma = torch.ones((eq.Nx, eq.Ny), dtype=torch.float64)
        else:
            sigma = torch.ones((eq.Nx - 1, eq.Ny - 1), dtype=torch.float64)

        currents = generate_adjacent_current_patterns(8, dtype=torch.float64).transpose(0, 1)
        _, electrode_voltages = cem.solve_cem(sigma, currents, z_contact=1.0, n_electrodes=8)
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
        points = boundary_points_from_angles(theta, domain_size=2.0)
        expected = torch.tensor([0.0 + 0.0j, 2.0 + 0.0j, 2.0 + 2.0j, 0.0 + 2.0j], dtype=torch.complex128)
        torch.testing.assert_close(points, expected)

    def test_compute_psi_BIE_square_zero_k(self):
        theta_arc, Dtheta, _ = arc_length_params_square(8, domain_size=2.0, n_boundary_samples=128)
        ntrig = make_trig_mode_indices(8)
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
        cem = CompleteElectrodeModel(eq)
        sigma = torch.ones((eq.Nx, eq.Ny), dtype=torch.float64)
        currents = generate_adjacent_current_patterns(8, dtype=torch.float64).transpose(0, 1)
        _, electrode_voltages = cem.solve_cem(sigma, currents, z_contact=1.0, n_electrodes=8)
        voltage_matrix = electrode_voltages.transpose(0, 1)
        dn_map = build_dn_map_from_electrode_data(voltage_matrix, domain_size=2.0, n_boundary_samples=128)

        theta_arc, Dtheta, _ = arc_length_params_square(8, domain_size=2.0, n_boundary_samples=128)
        ntrig = make_trig_mode_indices(8)
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
    def test_db_operator_matches_reference_formula(self):
        """
        Reference: DB_oper.m
        """
        fund = torch.tensor(
            [
                [0.0 + 0.0j, 0.10 - 0.20j, 0.05 + 0.03j],
                [0.07 + 0.01j, -0.02 + 0.04j, 0.11 - 0.06j],
                [0.03 - 0.08j, 0.09 + 0.02j, -0.05 + 0.07j],
            ],
            dtype=torch.complex128,
        )
        fundfft = torch.fft.fft2(torch.fft.fftshift(fund, dim=(-2, -1)), dim=(-2, -1))
        tr = torch.tensor(
            [
                [0.0 + 0.0j, 0.20 + 0.10j, -0.15 + 0.05j],
                [0.04 - 0.07j, 0.00 + 0.00j, 0.12 + 0.03j],
                [-0.05 + 0.02j, 0.08 - 0.04j, 0.00 + 0.00j],
            ],
            dtype=torch.complex128,
        )
        rind = torch.tensor(
            [
                [False, True, True],
                [True, False, True],
                [True, True, False],
            ],
            dtype=torch.bool,
        )
        w_vec = torch.tensor(
            [0.4, -0.2, 0.1, 0.3, -0.5, 0.2, -0.1, 0.6, -0.4, 0.15, -0.35, 0.05],
            dtype=torch.float64,
        )
        nind = int(rind.sum().item())

        wtmp = torch.zeros_like(tr)
        wtmp[rind] = torch.complex(w_vec[:nind], w_vec[nind:])
        transformed = torch.fft.fft2(torch.fft.fftshift(tr * torch.conj(wtmp), dim=(-2, -1)), dim=(-2, -1))
        conv = (0.25 ** 2) * torch.fft.ifftshift(torch.fft.ifft2(fundfft * transformed, dim=(-2, -1)), dim=(-2, -1))
        expected = torch.cat(((wtmp - conv).real[rind], (wtmp - conv).imag[rind]), dim=0)

        operator = DBOperator(
            fundfft=fundfft,
            tr=tr,
            rind=rind,
            nind=nind,
            h=0.25,
        )
        actual = operator.matvec(w_vec)

        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0.0)

    def test_db_operator_supports_autograd(self):
        fund = torch.tensor(
            [
                [0.0 + 0.0j, 0.10 - 0.20j, 0.05 + 0.03j],
                [0.07 + 0.01j, -0.02 + 0.04j, 0.11 - 0.06j],
                [0.03 - 0.08j, 0.09 + 0.02j, -0.05 + 0.07j],
            ],
            dtype=torch.complex128,
        )
        fundfft = torch.fft.fft2(torch.fft.fftshift(fund, dim=(-2, -1)), dim=(-2, -1))
        tr = torch.tensor(
            [
                [0.0 + 0.0j, 0.20 + 0.10j, -0.15 + 0.05j],
                [0.04 - 0.07j, 0.00 + 0.00j, 0.12 + 0.03j],
                [-0.05 + 0.02j, 0.08 - 0.04j, 0.00 + 0.00j],
            ],
            dtype=torch.complex128,
            requires_grad=True,
        )
        rind = torch.tensor(
            [
                [False, True, True],
                [True, False, True],
                [True, True, False],
            ],
            dtype=torch.bool,
        )
        w_vec = torch.tensor(
            [0.4, -0.2, 0.1, 0.3, -0.5, 0.2, -0.1, 0.6, -0.4, 0.15, -0.35, 0.05],
            dtype=torch.float64,
        )

        operator = DBOperator(
            fundfft=fundfft,
            tr=tr,
            rind=rind,
            nind=int(rind.sum().item()),
            h=0.25,
        )
        loss = operator.matvec(w_vec).square().sum()
        loss.backward()

        self.assertIsNotNone(tr.grad)
        grad = tr.grad
        assert grad is not None
        self.assertTrue(torch.isfinite(grad).all().item())

    def test_db_operator_cuda_smoke(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available.")

        device = torch.device("cuda")
        fund = torch.tensor(
            [
                [0.0 + 0.0j, 0.10 - 0.20j, 0.05 + 0.03j],
                [0.07 + 0.01j, -0.02 + 0.04j, 0.11 - 0.06j],
                [0.03 - 0.08j, 0.09 + 0.02j, -0.05 + 0.07j],
            ],
            dtype=torch.complex64,
            device=device,
        )
        fundfft = torch.fft.fft2(torch.fft.fftshift(fund, dim=(-2, -1)), dim=(-2, -1))
        tr = torch.tensor(
            [
                [0.0 + 0.0j, 0.20 + 0.10j, -0.15 + 0.05j],
                [0.04 - 0.07j, 0.00 + 0.00j, 0.12 + 0.03j],
                [-0.05 + 0.02j, 0.08 - 0.04j, 0.00 + 0.00j],
            ],
            dtype=torch.complex64,
            device=device,
        )
        rind = torch.tensor(
            [
                [False, True, True],
                [True, False, True],
                [True, True, False],
            ],
            dtype=torch.bool,
            device=device,
        )
        w_vec = torch.tensor(
            [0.4, -0.2, 0.1, 0.3, -0.5, 0.2, -0.1, 0.6, -0.4, 0.15, -0.35, 0.05],
            dtype=torch.float32,
            device=device,
        )

        operator = DBOperator(
            fundfft=fundfft,
            tr=tr,
            rind=rind,
            nind=int(rind.sum().item()),
            h=0.25,
        )
        actual = operator.matvec(w_vec)

        self.assertEqual(actual.device.type, "cuda")
        self.assertTrue(torch.isfinite(actual).all().item())

    def test_dbar_inverse_requires_square_domain(self):
        with self.assertRaisesRegex(ValueError, "supported only for domain_shape='square'"):
            DbarReconstruction2D(
                image_size=8,
                n_boundary_nodes=16,
                n_modes=3,
                k_grid_size=9,
                k_radius=2.0,
                inverse_method="dbar",
            )

    def test_interp2_bicubic_rect_grid_matches_grid_nodes(self):
        x = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64)
        y = torch.linspace(-1.5, 1.5, 7, dtype=torch.float64)
        K2, K1 = torch.meshgrid(y, x, indexing="ij")
        values = (K1.square() - 0.5 * K2) + 1j * (K1 + K2.square())

        query_x = torch.tensor([x[0], x[3], x[-1], x[5]], dtype=torch.float64)
        query_y = torch.tensor([y[0], y[2], y[-1], y[4]], dtype=torch.float64)
        expected = values[torch.tensor([0, 2, 6, 4]), torch.tensor([0, 3, 8, 5])]

        interp_vals = _interp2_bicubic_rect_grid(K1.flip(0).flip(1), K2.flip(0).flip(1), values.flip(0).flip(1), query_x, query_y)
        torch.testing.assert_close(interp_vals, expected, atol=1e-8, rtol=0.0)

    def test_interp_cubic_scattered_preserves_constant_inside_bbox(self):
        sample_xy = torch.tensor(
            [
                [-1.0, -0.75],
                [-0.5, 0.5],
                [0.0, -0.25],
                [0.25, 0.75],
                [0.8, -0.6],
                [1.0, 0.9],
            ],
            dtype=torch.float64,
        )
        Kvec = sample_xy[:, 1] + 1j * sample_xy[:, 0]
        values = torch.full((sample_xy.shape[0],), 2.5 - 0.75j, dtype=torch.complex128)
        query_x = torch.tensor([-0.5, 0.0, 0.6, 1.5], dtype=torch.float64)
        query_y = torch.tensor([0.0, -0.2, 0.5, 0.0], dtype=torch.float64)

        interp_vals = _interp_cubic_scattered(Kvec, values, query_x, query_y)
        torch.testing.assert_close(interp_vals[:3], values[:3], atol=2e-2, rtol=0.0)
        torch.testing.assert_close(interp_vals[3:], torch.zeros(1, dtype=torch.complex128), atol=1e-12, rtol=0.0)

    def test_interpolate_precomputed_scattering_cutoff_zeros_large_values(self):
        recon = DbarReconstruction2D(
            image_size=8,
            n_boundary_nodes=32,
            n_modes=15,
            k_grid_size=8,
            k_radius=2.0,
        )
        x = torch.linspace(-2.0, 2.0, 9, dtype=torch.float32)
        y = torch.linspace(-2.0, 2.0, 9, dtype=torch.float32)
        K2, K1 = torch.meshgrid(y, x, indexing="ij")
        tBIE = torch.full(K1.shape, 100.0 + 100.0j, dtype=torch.complex64)

        out = recon.interpolate_precomputed_scattering(tBIE, K1=K1, K2=K2, cutoff=25.0)
        torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-6, rtol=0.0)

    def test_forward_square_zero_scattering_returns_ones(self):
        eq = DiffusionEquation2D(grid_size=16, domain_size=2.0, grid_type="node")
        cem = CompleteElectrodeModel(eq)
        sigma = torch.ones((eq.Nx, eq.Ny), dtype=torch.float64)
        currents = generate_adjacent_current_patterns(8, dtype=torch.float64).transpose(0, 1)
        _, electrode_voltages = cem.solve_cem(sigma, currents, z_contact=1.0, n_electrodes=8)
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

    def test_forward_square_zero_scattering_returns_ones_with_dbar_inverse(self):
        eq = DiffusionEquation2D(grid_size=12, domain_size=2.0, grid_type="node")
        cem = CompleteElectrodeModel(eq)
        sigma = torch.ones((eq.Nx, eq.Ny), dtype=torch.float64)
        currents = generate_adjacent_current_patterns(8, dtype=torch.float64).transpose(0, 1)
        _, electrode_voltages = cem.solve_cem(sigma, currents, z_contact=1.0, n_electrodes=8)
        voltage_matrix = electrode_voltages.transpose(0, 1)
        dn_map = build_dn_map_from_electrode_data(voltage_matrix, domain_size=2.0, n_boundary_samples=64)

        recon = DbarReconstruction2D(
            image_size=8,
            n_boundary_nodes=64,
            n_modes=7,
            n_electrodes=8,
            k_grid_size=9,
            k_radius=2.0,
            domain_shape="square",
            domain_size=2.0,
            inverse_method="dbar",
            gmres_restart=10,
            gmres_rtol=1e-10,
            gmres_maxiter=10,
        )
        sigma_rec = recon.forward(lambda_sigma=dn_map, lambda_ref=dn_map)
        self.assertEqual(sigma_rec.shape, (1, 8, 8))
        torch.testing.assert_close(sigma_rec, torch.ones_like(sigma_rec), atol=1e-8, rtol=0.0)

    def test_square_reconstruction_grid_matches_domain_size(self):
        recon = DbarReconstruction2D(
            image_size=16,
            n_boundary_nodes=128,
            n_modes=7,
            n_electrodes=8,
            k_grid_size=16,
            k_radius=2.0,
            domain_shape="square",
            domain_size=(1.0, math.pi),
        )
        z_grid = torch.as_tensor(recon.z_grid)

        torch.testing.assert_close(z_grid.real[0, 0], torch.tensor(0.0, dtype=z_grid.real.dtype), atol=0.0, rtol=0.0)
        torch.testing.assert_close(z_grid.real[-1, 0], torch.tensor(1.0, dtype=z_grid.real.dtype), atol=1e-12, rtol=0.0)
        torch.testing.assert_close(z_grid.imag[0, 0], torch.tensor(0.0, dtype=z_grid.imag.dtype), atol=0.0, rtol=0.0)
        torch.testing.assert_close(z_grid.imag[0, -1], torch.tensor(math.pi, dtype=z_grid.imag.dtype), atol=1e-12, rtol=0.0)

    def test_forward_from_measurements_square_zero_scattering_returns_ones(self):
        eq = DiffusionEquation2D(grid_size=32, domain_size=2.0, grid_type="node")
        cem = CompleteElectrodeModel(eq)
        sigma = torch.ones((eq.Nx, eq.Ny), dtype=torch.float64)
        currents = generate_adjacent_current_patterns(8, dtype=torch.float64).transpose(0, 1)
        _, electrode_voltages = cem.solve_cem(sigma, currents, z_contact=1.0, n_electrodes=8)
        voltage_matrix = electrode_voltages.transpose(0, 1)

        recon = DbarReconstruction2D(
            image_size=32,
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
        self.assertEqual(sigma_rec.shape, (1, 32, 32))
        torch.testing.assert_close(sigma_rec, torch.ones_like(sigma_rec), atol=1e-10, rtol=0.0)

if __name__ == "__main__":
    absltest.main()