# Plan: Square-domain D-bar port from MATLAB circular domain codes

## Decisions from user
- Boundary: Complete Electrode Model (CEM) - discrete electrodes on 4 sides
- D-bar solver: Keep Born approximation, not GMRES for now
- Scattering transform: Faddeev CGO / BIE on square perimeter, option 1

## TL;DR
Port the KIT4 circular-domain D-bar pipeline to a square domain. The forward
solver `DiffusionEquation2D` is done. The next work is:
(A) CEM forward problem on the square,
(B) square-domain trigonometric basis projection,
(C) Faddeev CGO solutions on the square perimeter,
(D) scattering transform using those CGO traces.

---

## Phase 1 - CEM Forward Solver

**Goal**: Given `$\sigma$` on a square grid, solve the CEM forward problem for `L` electrodes placed
uniformly on the 4 sides. Output electrode voltages `U` and interior potential `u`.

### Steps
1. Add `electrode_builder_square(L, domain_size)` helper - returns an `(L, 2)` array of arc-length
   ranges of each electrode on the square perimeter, mirroring `electrodeBuilder.m`.
2. Add `assemble_cem_stiffness(sigma, z_contact, electrode_ranges)` to `DiffusionEquation2D`
   or introduce a dedicated `CEM2D` helper.
   - Augment `A_II` with boundary electrode terms.
- **Later**: replace the Born approximation solve with GMRES for the dense D-bar integral equation,
  not for the sparse CEM forward system. Concretely, this future phase would solve a discretized
  equation of the form `$(I - \mathcal{K}_z[t])\mu_z = 1$` in the `k` variable, ideally with a
  matrix-free Krylov method rather than by assembling a full dense matrix.
   - Assemble the block system
     `[(A_II + A_bnd), -B; -B^T, diag((1 / z_l) * |E_l|)] [u_int; U] = [0; I]`.
3. Implement `solve_cem(sigma, I, z_contact)` - assemble and solve the augmented system,
   returning `(u_full, U)`.
4. Implement `dirichlet_to_neumann(sigma, z_contact)` - for each adjacent-current pattern `I`,
   call `solve_cem`, collect `U`, assemble the `(L, L)` ND map, then convert it to DN in a form that
   can later feed `nd_to_dn_map` or `build_dn_map_from_electrode_data` style downstream code.

### Implementation notes
1. Do not reuse the current Dirichlet-oriented assembly unchanged. The existing `assemble_stiffness`
   keeps boundary-face conductances in the diagonal and moves only Dirichlet data into the RHS;
   that is appropriate for lifted Dirichlet boundaries but not for pure Neumann gaps or CEM boundaries.
2. Build a dedicated interior matrix for the CEM case where non-electrode boundary faces contribute
   zero flux. In practice, that means removing the outer-face terms from the diagonal for the left,
   right, top, and bottom boundary-adjacent interior nodes unless that face belongs to an electrode.
3. On an electrode face `E_l`, impose the Robin-type CEM law
   `$\sigma \partial_n u = (U_l - u) / z_l$`, equivalently
   `$u + z_l \sigma \partial_n u = U_l$`.
   This adds a diagonal contribution for each boundary-adjacent interior node and a coupling term to
   the corresponding electrode voltage DOF.
4. The clean implementation is to construct the full augmented `(M + L) x (M + L)` system from scratch:
   interior unknowns remain on the structured grid, electrode voltages add `L` new DOFs, and the
   interior-electrode coupling is assembled face by face.
5. Parameterize the square boundary by arc length, counterclockwise:
   - bottom edge: `s in [0, Lx]`
   - right edge: `s in [Lx, Lx + Ly]`
   - top edge: `s in [Lx + Ly, 2Lx + Ly]`
   - left edge: `s in [2Lx + Ly, 2Lx + 2Ly]`
   Use boundary-face centers to assign each face to an electrode interval.
6. For uniform electrode placement, divide the total perimeter `2Lx + 2Ly` into `L` equal arc-length
   intervals and assign each boundary face to the electrode whose interval contains that face center.
7. To remove the additive constant in electrode potentials, ground one electrode, for example by setting
   the last electrode voltage to zero and eliminating its row and column. Solve the reduced
   `(M + L - 1) x (M + L - 1)` system and reconstruct the full `U` afterward.
8. Use adjacent current patterns that sum to zero. The raw ND map can remain `(L, L)` even though its
   natural action is on the mean-zero subspace.
9. A dense solve is acceptable for the initial implementation and moderate grids, but note that a sparse
   solver is the natural follow-up if the mesh size grows.

**Relevant existing code**:
- `DiffusionEquation2D` in `forward_solver.py` - reuse `_get_face_coefs`, existing indexing logic,
  and `_solve_linear` where appropriate
- `make_adjacent_current_patterns(n_electrodes)` in `dbar.py` - reuse for current patterns
- `build_nd_map_from_electrode_data(...)`, `build_dn_map_from_electrode_data(...)`, and `nd_to_dn_map(...)`
   in `dbar.py` - match the Phase 1 output conventions to these downstream interfaces
- MATLAB `electrodeBuilder.m` and `comp02_ND_buildFromKIT4.m` as references

**Files to modify**: `torch-dbar/forward_solver.py`

---

## Phase 2 - Square-boundary trig basis projection

**Goal**: Project CEM electrode measurements onto the trig basis used in the D-bar formulation,
matching `transform_adjacent_to_square_trig(...)`, `build_nd_map_from_electrode_data(...)`,
and `build_dn_map_from_electrode_data(...)` in `dbar.py`.

### Steps
1. Add `arc_length_params_square(L, domain_size)` - returns `(fii, Dfii, gammaMid)` for the square
   perimeter, analogous to `angPars_adj.mat`. Arc-length coordinate runs from `0` to `8` for a `2 x 2` square.
2. Port the `transfrom_Adj2Trig.m` logic into the existing `transform_adjacent_to_square_trig(...)`
   flow, but use square-boundary arc-length angles rather than circular angles.
   Key basis functions are `cos(n * theta_arc)` and `sin(n * theta_arc)` with
   `$\theta_{arc} = ({arc length} / {total perimeter}) 2\pi$`.
3. After projection, obtain `(nBasis, nBasis)` ND and DN matrices in the trig basis using
   `build_nd_map_from_electrode_data(...)` and `build_dn_map_from_electrode_data(...)`.

**Relevant existing code**:
- `transform_adjacent_to_square_trig(...)` in `dbar.py` - direct Python target for the square trig projection
- `build_nd_map_from_electrode_data(...)` and `build_dn_map_from_electrode_data(...)` in `dbar.py` -
   downstream map construction helpers
- `make_square_trig_mode_indices(...)` in `dbar.py` - mode indexing helper for the square basis
- `transfrom_Adj2Trig` in `KIT4_recons_standalone.py` - MATLAB-side logic template

**Files to modify**: `torch-dbar/dbar.py`, likely via new helper functions or an extension of `DbarReconstruction2D`

---

## Phase 3 - Faddeev CGO solutions on square boundary

**Goal**: For each `k` in the `k`-grid with `|k| < R_freq`, compute the boundary trace of the
Faddeev CGO solution `$\psi(x, k)$` on the square perimeter. This replaces `comp04_psi_BIE.m`,
which uses a BIE on the unit circle.

### Steps
1. For a given `k`, construct the Lippmann-Schwinger integral equation on the square boundary:
   `$\psi(x, k) = e^{ikx} - \int_{\partial \Omega} G(x-y, k) (\Delta \sigma)(...) \psi(y, k)\,dy$`.
   In the homogeneous reference case `$\sigma = 1$`, `$\psi_0 = e^{ikx}$` is exact, so the CGO traces
   `Fpsi_BIE` are the trig Fourier coefficients of `$e^{ik e^{i\theta_{arc}}}$`.
2. Implement or extend `compute_psi_BIE_square(Kvec, theta_arc, trig_mode_indices)` - computes an `(nBasis, num_k)` matrix
   `Fpsi_BIE` using the square arc-length parameterization. For `$\sigma_{ref} = 1$`:
   `Fpsi_BIE[j, ki] = Dtheta * sum_n cos/sin(trig_mode_indices[j] * theta_arc_n) * e^{i Kvec[ki] * x_n}`
   where `x_n` are the complex boundary coordinates.
3. Port the MATLAB `compute_tBIE` logic into `compute_tBIE_square(...)`, using square-parameterized
   boundary points.

**Relevant existing code**:
- `compute_psi_BIE_square(...)` in `dbar.py` - existing square CGO trace helper
- `compute_tBIE_square(...)` in `dbar.py` - existing square scattering helper
- `compute_tBIE` in `KIT4_recons_standalone.py` - direct template, with only boundary parameterization changed
- `psi_BIE.mat` - precomputed on the unit disk and therefore not reusable for the square domain

**Files to modify**: `torch-dbar/dbar.py`

---

## Phase 4 - Scattering grid assembly + Born reconstruction

**Goal**: Assemble the full scattering grid and run the Born-approximation D-bar reconstruction.
The Born solver is already in `dbar.py::solve_sigma()`.

### Steps
1. Update `DbarReconstruction2D.__init__` to accept a `domain_shape='square'` flag and replace
   the circular `boundary_points` with square arc-length-parameterized points.
2. Use `compute_scattering_transform_square()` for the square-domain path so it calls the Phase 3 logic.
3. Wire `forward()` and `forward_from_measurements()` to use the Phase 1 DN map and the Phase 3 scattering machinery end to end.

**Files to modify**: `torch-dbar/dbar.py`

---

## Verification
1. Unit test `dirichlet_to_neumann` on `$\sigma = 1$`: DN-map eigenvalues should scale like `|n|` in the Fourier basis-like solution $\sin(n_1 x)\sinh(n_2 y)$.
2. Sanity-check that scattering transform `T(k)` tends to `0` as `k -> infinity` for smooth `$\sigma$`.
3. Run an end-to-end synthetic case, for example a disk inclusion in a square, and confirm the reconstruction qualitatively matches the target `$\sigma$`.
4. Run the regression suite in `test_forward_solver.py` after the Phase 1 changes.

---

## Scope boundaries
- **Excluded**: GMRES D-bar solve, per the current decision to keep Born approximation for now
- **Excluded**: full port of `comp04_psi_BIE.m` for non-homogeneous `$\sigma_{ref}$`; use `$\sigma_{ref} = 1$` for now
- **Excluded**: GPU-specific optimizations and `k`-batching for the CGO solve
- **Later**: replace the Born approximation solve with GMRES in a later phase
