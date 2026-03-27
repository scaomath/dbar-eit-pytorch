## 

## Porting `KIT4_recons_numpy_standalone.py`

| MATLAB | Python function | Porting notes |
| --- | --- | --- |
| ex2Kvec_comp.m | build_kvec_grid() | Uses `.flatten(order='F')` to replicate MATLAB's column-major logical-index extraction, preserving Kvec ordering consistent with Fpsi_BIE columns |
| electrodeBuilder.m | electrode_builder() | `reshape(2,L,order='F').T` replicates MATLAB column-major fill of `zeros(2,L)` then transpose |
| transfrom_Adj2Trig.m | transfrom_Adj2Trig() | `coeff = Atrig.T @ Aad` replaces the nested loop; `lstsq(coeff.T, U.T)[0].T` is the Python equivalent of MATLAB's `U_in / coeff` right-division |
| comp02_ND_buildFromKIT4.m | build_ND_from_kit4() | Inner loop piecewise-constant trace assignment is ported verbatim; NtoD computed as `Dfii * B @ u_trace` |
| comp03_DN_build.m | build_DN_from_ND() | The MATLAB `[DN(:,1:8),DN(:,9:end)]` / row rearrangement on a 15×15 matrix with parVal=8 is a no-op - only inv() remains |
| comp05_tBIE_psi.m | compute_tBIE() | Fully vectorised over all k; `einsum('ij,ij->j',...)` replaces the scalar dot-product loop |