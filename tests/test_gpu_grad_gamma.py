"""Gamma-point GPU pp-RPA gradient: the low-rank pairing force against the AFT
kernel it replaces, and the full gradient against the CPU reference, with the
CPHF warm start."""

import numpy as np
import pytest


def _gpu4pyscf_available():
    try:
        import cupy as cp
        import gpu4pyscf  # noqa: F401
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _gpu4pyscf_available(), reason="GPU4PySCF CUDA device is unavailable")

XC = "pbe"
CHANNEL = "pp"
KPTS = np.zeros((1, 3))


@pytest.fixture(scope="module")
def system():
    """Distorted C2 diamond cell, charge +2, GPU SCF, MO ERIs through the GPU ao2mo."""
    import cupy as cp
    from pyscf.pbc import dft as cdft, gto
    from gpu4pyscf.pbc import dft as gdft
    from lib_pprpa.gpu_ao2mo import gpu_ao2mo_blocks
    a0 = 3.5668
    cell = gto.Cell()
    cell.atom = [["C", [0.0, 0.0, 0.0]], ["C", [a0 / 4 + 0.05, a0 / 4, a0 / 4]]]
    cell.a = np.array([[0, a0 / 2, a0 / 2], [a0 / 2, 0, a0 / 2], [a0 / 2, a0 / 2, 0]])
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.charge = 2
    cell.ke_cutoff = 100.0
    cell.verbose = 0
    cell.build()
    kg = gdft.KRKS(cell, kpts=KPTS, xc=XC)
    kg.exxdiv = None
    kg.conv_tol = 1e-10
    kg.kernel()
    mf = cdft.RKS(cell, xc=XC)
    mf.exxdiv = None
    mf.mo_coeff = cp.asnumpy(kg.mo_coeff[0])
    mf.mo_energy = cp.asnumpy(kg.mo_energy[0])
    mf.mo_occ = cp.asnumpy(kg.mo_occ[0])
    mf.e_tot = float(kg.e_tot)
    mf.converged = True
    nocc = cell.nelectron // 2
    eri = gpu_ao2mo_blocks(cell, mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:], cell.mesh)
    return cell, mf, nocc, eri


def _solve(system, mult):
    from lib_pprpa.pprpa_davidson import ppRPA_Davidson
    from lib_pprpa.pprpa_eri_gpu import attach_gpu_eri_contraction
    cell, mf, nocc, eri = system
    mp = ppRPA_Davidson(nocc, mf.mo_energy, Lpq=None, channel=CHANNEL, nroot=2,
                        residue_thresh=1e-10, trial="identity")
    mp.mu = 0.0
    attach_gpu_eri_contraction(mp, *eri, mode="resident")
    mp.kernel(mult)
    xy = np.array((mp.xy_s if mult == "s" else mp.xy_t)[0], copy=True)
    return mp, xy


def _dense_get_k(cell):
    import cupy as cp
    from gpu4pyscf.pbc.df import fft_jk
    from gpu4pyscf.pbc.df.fft import FFTDF
    fdf = FFTDF(cell, KPTS)

    def get_k(dm=None, hermi=1, **kw):
        dmg = cp.asarray(dm)
        single = dmg.ndim == 2
        k = cp.asnumpy(fft_jk.get_k(fdf, dmg[None] if single else dmg, hermi=0,
                                    kpt=np.zeros(3), exxdiv=None))
        return k[0] if single else k
    return get_k


def _aft_pairing_force(cell, mf, X):
    """The AFT kernel the low-rank force replaces: +Q_K(antisym X) - Q_K(sym X), sr=lr=2."""
    import cupy as cp
    from gpu4pyscf.pbc import dft as gdft
    from gpu4pyscf.pbc.grad import krhf as krhf_g
    from lib_pprpa.grad.pprpa_gamma_gpu import _aftdf
    kf = gdft.KRKS(cell, kpts=KPTS, xc=mf.xc)
    kf.exxdiv = None
    kf.mo_coeff = [cp.asarray(mf.mo_coeff)]
    kf.mo_energy = [cp.asarray(mf.mo_energy)]
    kf.mo_occ = [cp.asarray(mf.mo_occ)]
    kf.with_df = _aftdf(cell, KPTS)
    kf.rsjk = None
    Xg = cp.asarray(X)
    Xa, Xs = (Xg - Xg.T) * 0.5, (Xg + Xg.T) * 0.5

    def pair(dm):
        return cp.asarray(krhf_g.jk_energy_per_atom(
            kf, dm[None], KPTS, j_factor=0.0, sr_factor=2.0, lr_factor=2.0,
            omega=0.0, exxdiv=None))
    de = cp.zeros((cell.natm, 3))
    if float(cp.abs(Xa).max()) > 1e-10:
        de += pair(Xa)
    if float(cp.abs(Xs).max()) > 1e-10:
        de -= pair(Xs)
    return de.get()


@pytest.mark.parametrize("mult", ("s", "t"))
def test_pairing_force_matches_aft(system, mult):
    """AFT and FFT exchange derivatives agree to ~1e-8; the low-rank form is exact in X."""
    from lib_pprpa.grad.grad_utils import get_xy_full
    from lib_pprpa.gpu_pairing_force import pairing_k_force_lowrank
    cell, mf, nocc, _ = system
    mp, xy = _solve(system, mult)
    nvir = cell.nao - nocc
    occ_y, vir_x = get_xy_full(xy, mp.oo_dim, mult, nocc=nocc, nvir=nvir)
    cocc, cvir = mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:]
    X = cvir @ vir_x @ cvir.T + cocc @ occ_y @ cocc.T
    ref = _aft_pairing_force(cell, mf, X)
    got = pairing_k_force_lowrank(cell, cell.mesh, np.hstack([cvir @ vir_x, cocc @ occ_y]),
                                  np.hstack([cvir, cocc]))
    assert np.abs(got - ref).max() < 1e-6 * np.abs(ref).max()


@pytest.mark.parametrize("mult", ("s", "t"))
def test_grad_elec_matches_cpu_and_warm_start(system, mult):
    from lib_pprpa.grad import pprpa_gamma as cpu
    from lib_pprpa.grad import pprpa_gamma_gpu as gpu
    cell, mf, nocc, _ = system
    mp, xy = _solve(system, mult)
    mf.get_k = _dense_get_k(cell)          # the CPU driver's 2-RDM exchange
    g_cpu = cpu.Gradients(mp, mf, mult, 0)
    g_gpu = gpu.Gradients(mp, mf, mult, 0)
    for g in (g_cpu, g_gpu):
        g.cphf_conv_tol = 1e-10
        g.cphf_max_cycle = 100
    de_cpu = g_cpu.grad_elec(xy, mult, range(cell.natm))
    assert g_gpu.cphf_x0 is None
    de_gpu = g_gpu.grad_elec(xy, mult, range(cell.natm))
    scale = np.abs(de_cpu).max()
    assert np.abs(de_gpu - de_cpu).max() < 1e-7 * scale
    # the warm start seeds the second solve and leaves the result unchanged
    assert g_gpu.cphf_x0 is not None and g_gpu.cphf_x0.shape == (cell.nao, cell.nao)
    de_warm = g_gpu.grad_elec(xy, mult, range(cell.natm))
    assert np.abs(de_warm - de_gpu).max() < 1e-9 * scale
