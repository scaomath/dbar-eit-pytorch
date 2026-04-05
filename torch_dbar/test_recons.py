import torch
from absl.testing import absltest

from dbar import (
    DbarReconstruction2D,
    arc_length_params_square,
    build_dn_map_from_electrode_data,
    compute_psi_BIE,
    compute_tBIE,
    make_trig_mode_indices,
)
from cem_solver import CompleteElectrodeModel, generate_adjacent_current_patterns
from forward_solver import DiffusionEquation2D


def _build_reference_and_currents(grid_size: int = 24):
    eq = DiffusionEquation2D(grid_size=grid_size, domain_size=2.0, grid_type="node")
    cem = CompleteElectrodeModel(eq)
    currents = generate_adjacent_current_patterns(8, dtype=torch.float64).transpose(0, 1)
    sigma_ref = torch.ones((eq.Nx, eq.Ny), dtype=torch.float64)
    _, electrode_voltages_ref = cem.solve_cem(sigma_ref, currents, z_contact=1.0, n_electrodes=8)
    dn_ref = build_dn_map_from_electrode_data(
        electrode_voltages_ref.transpose(0, 1),
        domain_size=2.0,
        n_boundary_samples=128,
    )
    return eq, currents, sigma_ref, dn_ref


class TestReconstructionVerification(absltest.TestCase):
    def test_reconstruction_grid_matches_forward_solver_order(self):
        eq = DiffusionEquation2D(grid_size=8, domain_size=2.0, grid_type="node")
        recon = DbarReconstruction2D(
            image_size=8,
            n_boundary_nodes=32,
            n_modes=7,
            n_electrodes=8,
            k_grid_size=9,
            k_radius=2.0,
            domain_size=2.0,
        )

        self.assertGreater(float(eq.x_nodes[1, 0] - eq.x_nodes[0, 0]), 0.0)
        self.assertEqual(float(eq.x_nodes[0, 1] - eq.x_nodes[0, 0]), 0.0)
        self.assertGreater(float(recon.z_grid.real[1, 0] - recon.z_grid.real[0, 0]), 0.0)
        self.assertEqual(float(recon.z_grid.real[0, 1] - recon.z_grid.real[0, 0]), 0.0)

    def test_sigma_one_trig_dn_spectrum_example(self):
        _, _, _, dn_ref = _build_reference_and_currents(grid_size=16)

        eigvals = torch.linalg.eigvalsh(dn_ref).real
        self.assertTrue(torch.all(eigvals[:-1] <= eigvals[1:]).item())
        self.assertGreater(float(eigvals[-1]), 1.0)
        self.assertGreater(float(eigvals[-1] / eigvals[1]), 5.0)

    def test_scattering_reference_and_sampled_high_k_example(self):
        eq, currents, _, dn_ref = _build_reference_and_currents(grid_size=16)
        cem = CompleteElectrodeModel(eq)

        sigma_smooth = torch.ones((eq.Nx, eq.Ny), dtype=torch.float64)
        mask = (eq.x_nodes - 1.0) ** 2 + (eq.y_nodes - 1.0) ** 2 <= 0.45**2
        sigma_smooth[mask] = 1.1
        _, electrode_voltages = cem.solve_cem(sigma_smooth, currents, z_contact=1.0, n_electrodes=8)
        dn_smooth = build_dn_map_from_electrode_data(
            electrode_voltages.transpose(0, 1),
            domain_size=2.0,
            n_boundary_samples=128,
        )

        theta_arc, Dtheta, _ = arc_length_params_square(8, domain_size=2.0, n_boundary_samples=128)
        trig_mode_indices = make_trig_mode_indices(8)
        kvec = torch.tensor([0.25 + 0.0j, 0.5 + 0.0j, 1.0 + 0.0j, 2.0 + 0.0j, 4.0 + 0.0j], dtype=torch.complex128)
        fpsi = compute_psi_BIE(
            kvec,
            theta_arc,
            trig_mode_indices,
            domain_size=2.0,
            Dtheta=Dtheta,
        )

        t_ref = compute_tBIE(
            Kvec=kvec,
            DN=dn_ref,
            DN1=dn_ref,
            Fpsi_BIE=fpsi,
            theta_arc=theta_arc,
            domain_size=2.0,
            trig_mode_indices=trig_mode_indices,
            Dtheta=Dtheta,
        )
        torch.testing.assert_close(t_ref, torch.zeros_like(t_ref), atol=1e-10, rtol=0.0)

        t_smooth = compute_tBIE(
            Kvec=kvec,
            DN=dn_smooth,
            DN1=dn_ref,
            Fpsi_BIE=fpsi,
            theta_arc=theta_arc,
            domain_size=2.0,
            trig_mode_indices=trig_mode_indices,
            Dtheta=Dtheta,
        )
        self.assertTrue(torch.isfinite(t_smooth).all().item())
        self.assertGreater(float(t_smooth.abs().max()), 0.0)

    def test_end_to_end_off_center_inclusion_example(self):
        eq, currents, _, dn_ref = _build_reference_and_currents(grid_size=24)
        cem = CompleteElectrodeModel(eq)

        sigma_target = torch.ones((eq.Nx, eq.Ny), dtype=torch.float64)
        mask = (eq.x_nodes - 1.5) ** 2 + (eq.y_nodes - 0.6) ** 2 <= 0.35**2
        sigma_target[mask] = 0.5
        _, electrode_voltages = cem.solve_cem(sigma_target, currents, z_contact=1.0, n_electrodes=8)
        dn_target = build_dn_map_from_electrode_data(
            electrode_voltages.transpose(0, 1),
            domain_size=2.0,
            n_boundary_samples=128,
        )

        recon = DbarReconstruction2D(
            image_size=24,
            n_boundary_nodes=128,
            n_modes=7,
            n_electrodes=8,
            k_grid_size=32,
            k_radius=4.0,
            domain_size=2.0,
        )
        sigma_rec = recon.forward(lambda_sigma=dn_target, lambda_ref=dn_ref)[0]

        xx, yy = torch.meshgrid(
            torch.linspace(0.0, 2.0, 24),
            torch.linspace(0.0, 2.0, 24),
            indexing="ij",
        )
        correct_mask = (xx - 1.5) ** 2 + (yy - 0.6) ** 2 <= 0.35**2
        center_mean = sigma_rec[correct_mask].mean()
        outer_mean = sigma_rec[~correct_mask].mean()

        self.assertGreater(float(torch.abs(center_mean - outer_mean)), 1e-3)
        self.assertGreater(float(torch.linalg.norm(sigma_rec - 1.0)), 0.5)


if __name__ == "__main__":
    absltest.main()