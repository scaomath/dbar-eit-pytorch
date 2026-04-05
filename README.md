# D-bar Method for Electrical Impedance Tomography in PyTorch 

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/) ![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?&logo=pytorch)

This repository has a WIP implementation of D-bar method (both solve and approximation) natively in PyTorch using `nn.Module`, compatible with `torch.autograd`.


## Instructions
You need to download the `KIT4_Dbar_recon.zip` data file from the official MATLAB implementation of D-bar method at https://fips.fi/blog/the-d-bar-method-for-electrical-impedance-tomography-experimental-data/ and https://arxiv.org/pdf/1704.01178, then the data should be extracted to `data` and `KIT4_measdata` folders.


## Notes

For the classical circular-domain D-bar method on the unit disk, the boundary is parameterized by an angle
$$
z(\theta) = e^{i\theta}, \qquad \theta \in [0,2\pi),  ds = d\theta.
$$
and since the radius is usually normalized to $1$.
In the circular-domain setting, the usual trigonometric boundary basis is
$$
\phi_n^{\cos}(\theta) = \frac{1}{\sqrt{\pi}}\cos(n\theta),
\qquad
\phi_n^{\sin}(\theta) = \frac{1}{\sqrt{\pi}}\sin(n\theta),
$$
so boundary projections and scattering quantities are written as angle integrals, for example
$$
F_\psi(k) = \int_0^{2\pi} \phi(\theta) e^{ik z(\theta)} \, d\theta,
$$
and
$$
t^{exp}(k) = \int_0^{2\pi} e^{i\bar{k}\overline{z(\theta)}} (\Lambda_\sigma - \Lambda_1)\psi^{exp}(z(\theta),k) \, d\theta,
$$
where $\Lambda_\sigma$ is the usual NtD map and $\Lambda_1$ is the NtD map with no inclusion.
For the square-domain port, the boundary is not naturally parameterized by the polar angle, the implementation uses arclength.
$$
s \in [0, |\partial\Omega|), \qquad |\partial\Omega| = 2(L_x + L_y),
$$
with piecewise square-boundary parameterization
$$
z(s) =
\begin{cases}
x + 0i, & s=x, \quad 0 \le x < L_x,\\
L_x + iy, & s=L_x+y, \quad 0 \le y < L_y,\\
x + iL_y, & s=2L_x+L_y-x, \quad 0 \le x < L_x,\\
0 + iy, & s=|\partial\Omega|-y, \quad 0 \le y < L_y.
\end{cases}
$$
On each side of the square, $ds = \pm dx$, $ds = \pm dy$, $d\theta$.

## References
Our implementation is based on the official implementation based on KIT4 dataset:  https://fips.fi/blog/the-d-bar-method-for-electrical-impedance-tomography-experimental-data/ and their [wiki page](https://wiki.helsinki.fi/xwiki/bin/view/mathstatHenkilokunta/Henkil%C3%B6t/Siltanen%2C%20Samuli/Inverse%20Problems%20Book%20Page/EIT%20with%20the%20D-bar%20method%3A%20discontinuous%20heart-and-lungs%20phantom/).
The initial porting draft of the MATLAB code based on KIT4 on circular domain is done by GPT 5.4 using VSCode Copilot. The `numpy`-based iterative method in https://github.com/eitcom/pyEIT is also used as a reference code.

