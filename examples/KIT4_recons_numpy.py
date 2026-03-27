#%%
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.interpolate import RectBivariateSpline
from scipy.io import loadmat
from scipy.sparse.linalg import LinearOperator, gmres
from tqdm.auto import tqdm

#%%
ROOT = Path(__file__).resolve().parents[1]

# This is a numpy port of the comp01_KIT4_recons.m example
# from the D-bar code repository
# https://fips.fi/blog/the-d-bar-method-for-electrical-impedance-tomography-experimental-data/
EX = 3
VER = 5

# comp06_Dbarsolve.m
CUTOFF = 25.0
R = 4.0
M = 6
RESTART = 50
RTOL = 1e-5
MAXITER = 500
#%%
def grid_like_gv(M: int, radius: float) -> tuple[np.ndarray, np.ndarray, float]:
    n = 2**M
    axis = np.linspace(-radius, radius, n, endpoint=False, dtype=np.float64)
    h = float(axis[1] - axis[0]) if n > 1 else 1.0
    k1, k2 = np.meshgrid(axis, axis, indexing="xy")
    return k1, k2, h


def interp2_bicubic(K1: np.ndarray, K2: np.ndarray, values: np.ndarray, xq: np.ndarray, yq: np.ndarray) -> np.ndarray:
    x = np.asarray(K1[0, :], dtype=np.float64)
    y = np.asarray(K2[:, 0], dtype=np.float64)
    v = np.asarray(values)

    if x[0] > x[-1]:
        x = x[::-1]
        v = v[:, ::-1]
    if y[0] > y[-1]:
        y = y[::-1]
        v = v[::-1, :]

    spl_re = RectBivariateSpline(y, x, np.real(v), kx=3, ky=3)
    spl_im = RectBivariateSpline(y, x, np.imag(v), kx=3, ky=3)
    return spl_re.ev(yq, xq) + 1j * spl_im.ev(yq, xq)


def centered_convolution(fundfft: np.ndarray, values: np.ndarray, h: float) -> np.ndarray:
    return (h**2) * ifftshift(ifft2(fundfft * fft2(fftshift(values))))


def db_oper_real(
    w_vec: np.ndarray,
    *,
    fundfft: np.ndarray,
    TR: np.ndarray,
    Rind: np.ndarray,
    Nind: int,
    h: float,
) -> np.ndarray:
    """
    DB_oper in the matlab routine
    """
    w = np.zeros(TR.shape, dtype=np.complex128)
    w[Rind] = w_vec[:Nind] + 1j * w_vec[Nind:]
    conv = centered_convolution(fundfft, TR * np.conj(w), h)
    out = w - conv
    return np.concatenate((np.real(out[Rind]), np.imag(out[Rind])))


def build_scat_from_matlab_outputs(
    data_dir: Path,
    *,
    cutoff: float,
    R: float,
    M: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, int]:
    mesh = loadmat(data_dir / "mesh.mat")
    kdat = loadmat(data_dir / "ex2Kvec.mat")
    tdat = loadmat(data_dir / "ex2tBIE.mat")

    # p: spatial mesh nodes where the conductivity is reconstructed/evaluated
    p = np.asarray(mesh["p"], dtype=np.float64)

    # zvec: same mesh nodes written as complex points z = x + i y
    xvec = p[0, :]
    yvec = p[1, :]
    zvec = xvec.astype(np.complex128) + 1j * yvec.astype(np.complex128)

    # K1, K2: Cartesian coordinates of the precomputed scattering grid in k-space
    K1 = np.asarray(kdat["K1"], dtype=np.float64)
    K2 = np.asarray(kdat["K2"], dtype=np.float64)

    # tMAX: radius of the k-space disk on which the precomputed scattering data is available
    tMAX = float(np.asarray(kdat["tMAX"]).squeeze())

    # tBIE: precomputed scattering transform values from the BIE / CGO
    # (complex geometrical optics) stage, i.e. after solving for the special
    # D-bar boundary solutions psi and converting them into scattering data
    tBIE = np.asarray(tdat["tBIE"]).reshape(-1).astype(np.complex128)

    # scatBIE_34: full 2D k-grid version of the scattering transform;
    # outside |k| < tMAX it is zero, and large outliers are truncated
    scatBIE_34 = np.zeros_like(K1, dtype=np.complex128)
    inside_tmax = np.abs(K1 + 1j * K2) < tMAX
    if tBIE.size != int(inside_tmax.sum()):
        raise ValueError(
            f"tBIE length {tBIE.size} does not match number of |K|<tMAX points {int(inside_tmax.sum())}."
        )
    scatBIE_34[inside_tmax] = tBIE
    scatBIE_34[np.abs(np.real(scatBIE_34)) > cutoff] = 0
    scatBIE_34[np.abs(np.imag(scatBIE_34)) > cutoff] = 0

    # k1, k2: computational k-grid used by the D-bar solve
    # Rind/Nind: the truncated disk |k| < R where the unknown is solved for
    k1, k2, h = grid_like_gv(M=M, radius=2.3 * R)
    k = k1 + 1j * k2
    Rind = np.abs(k) < R
    Nind = int(Rind.sum())

    # scat: scattering transform interpolated from the precomputed grid
    # onto the computational D-bar grid, restricted to |k| < R
    scatvec = interp2_bicubic(K1, K2, scatBIE_34, k1[Rind], k2[Rind])
    scat = np.zeros_like(k, dtype=np.complex128)
    scat[Rind] = scatvec

    return p, zvec, k1, k2, h, scat, Nind


def solve_comp06_like(
    *,
    zvec: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
    h: float,
    scat: np.ndarray,
    Nind: int,
    R: float,
) -> np.ndarray:
    """
    gmres solve of the integral eq
    """
    k = k1 + 1j * k2

    ktmp = k.copy()
    ind0 = np.abs(k) < 1e-14
    if not np.any(ind0):
        raise ValueError("Origin not found on computational k-grid.")
    ktmp[ind0] = 1.0

    scatk = scat / np.conj(ktmp)
    scatk[ind0] = 0.0

    fund = 1.0 / (np.pi * ktmp)
    fund[ind0] = 0.0

    s = abs(np.min(k1))
    ep = s / 10.0
    RR = (s - ep) / 2.0
    fund[np.abs(k) >= s] = 0.0
    medind = (np.abs(k) < s) & (np.abs(k) > 2.0 * RR)
    fund[medind] = fund[medind] * (1.0 - (np.abs(k[medind]) - 2.0 * RR) / ep)

    fundfft = fft2(fftshift(fund))

    Rind = np.abs(k) < R
    rhs = np.concatenate((np.ones(Nind, dtype=np.float64), np.zeros(Nind, dtype=np.float64)))
    iniguess = rhs.copy()

    recon = np.ones(zvec.shape[0], dtype=np.complex128)

    for iii, z in enumerate(tqdm(zvec, total=zvec.size, desc="D-bar solve"), start=1):
        # comp06_Dbarsolve.m
        # zvec = xvec(:) + 1i*yvec(:)
        TR = (1.0 / (4.0 * np.pi)) * scatk * np.exp(-1j * (k * z + np.conj(k * z)))

        A = LinearOperator(
            shape=(2 * Nind, 2 * Nind),
            matvec=lambda v: db_oper_real(v, fundfft=fundfft, TR=TR, Rind=Rind, Nind=Nind, h=h),
            dtype=np.float64,
        )

        w, info = gmres(
            A,
            rhs,
            x0=iniguess,
            restart=RESTART,
            rtol=RTOL,
            atol=0.0,
            maxiter=MAXITER,
        )
        if info != 0:
            raise RuntimeError(f"GMRES failed at node {iii}/{zvec.size} with info={info}.")

        mu = np.zeros_like(k, dtype=np.complex128)
        mu[Rind] = w[:Nind] + 1j * w[Nind:]
        recon[iii - 1] = mu[ind0][0] ** 2

    return np.real(recon)


def plot_reconstruction(p: np.ndarray, recon: np.ndarray, data_dir: Path):
    mesh = loadmat(data_dir / "mesh.mat")
    t = np.asarray(mesh["t"])
    triangles = np.asarray(t[:3, :].T - 1, dtype=np.int32)

    fig = plt.figure(2, clear=True)
    ax = fig.add_subplot(111)
    triang = mtri.Triangulation(p[0, :], p[1, :], triangles)
    ax.tripcolor(triang, recon, shading="gouraud", cmap="jet")
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax

#%%

if __name__ == "__main__":
    p, zvec, k1, k2, h, scat, Nind = build_scat_from_matlab_outputs(
        ROOT / "data",
        cutoff=CUTOFF,
        R=R,
        M=M,
    )
    recon = solve_comp06_like(
        zvec=zvec,
        k1=k1,
        k2=k2,
        h=h,
        scat=scat,
        Nind=Nind,
        R=R,
    )
    fig, ax = plot_reconstruction(p, recon, ROOT / "data")
    plt.show()

