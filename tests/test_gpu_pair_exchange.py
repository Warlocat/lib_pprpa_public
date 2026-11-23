"""The exchange of the pp-RPA pair density X + Y from the pair potentials against
gpu4pyscf's dense fft_jk.get_k, and through make_rdm1_relaxed_rhf_pprpa's pair_k."""

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


@pytest.fixture(scope="module")
def diamond():
    from pyscf.pbc import dft, gto
    a0 = 3.370137329
    cell = gto.M(atom=[["C", [0., 0., 0.]], ["C", [a0 / 2, a0 / 2, a0 / 2]]],
                 a=np.array([[0, a0, a0], [a0, 0, a0], [a0, a0, 0]]),
                 unit="bohr", basis="gth-dzv", pseudo="gth-pade", mesh=[20] * 3, verbose=0)
    mf = dft.RKS(cell, xc="pbe")
    mf.exxdiv = None
    mf.conv_tol = 1e-10
    mf.kernel()
    return cell, mf


def _dense_k(cell, dms):
    import cupy as cp
    from gpu4pyscf.pbc.df import fft_jk
    from gpu4pyscf.pbc.df.fft import FFTDF
    fdf = FFTDF(cell, np.zeros((1, 3)))
    k = fft_jk.get_k(fdf, cp.asarray(dms), hermi=0, kpt=np.zeros(3), exxdiv=None)
    return cp.asnumpy(k)


@pytest.mark.parametrize("mult", ("s", "t"))
def test_relaxed_density_from_pair_potential(diamond, mult):
    """make_rdm1_relaxed_rhf_pprpa with pair_k from the pair potentials equals the one
    with the dense exchange callback."""
    from scipy.linalg import block_diag
    from lib_pprpa.grad.grad_utils import get_xy_full
    from lib_pprpa.grad.pprpa import make_rdm1_relaxed_rhf_pprpa
    from lib_pprpa.gpu_pairing_force import pair_exchange, pair_potential_lowrank
    from lib_pprpa.pprpa_davidson import ppRPA_Davidson
    cell, mf = diamond
    nocc = cell.nelectron // 2
    nmo = cell.nao
    eri = mf.with_df.get_mo_eri(mf.mo_coeff, compact=False)
    eri = eri.reshape(nmo, nmo, nmo, nmo).transpose(0, 2, 1, 3)
    solver = ppRPA_Davidson(nocc, mf.mo_energy, Lpq=None, channel="hh", nroot=2,
                            residue_thresh=1e-10, trial="identity")
    solver.mu = 0.0
    solver.use_eri(np.ascontiguousarray(eri[nocc:, nocc:, nocc:, nocc:]),
                   np.ascontiguousarray(eri[:nocc, :nocc, nocc:, nocc:]),
                   np.ascontiguousarray(eri[:nocc, :nocc, :nocc, :nocc]))
    solver.kernel(mult)
    xy = (solver.xy_s if mult == "s" else solver.xy_t)[0]
    occ_y, vir_x = get_xy_full(xy, solver.oo_dim, mult, nocc=nocc, nvir=nmo - nocc)
    W = pair_potential_lowrank(cell, cell.mesh, mf.mo_coeff, block_diag(occ_y, vir_x))

    def dense(dms, hermi):
        return _dense_k(cell, dms)

    ref = make_rdm1_relaxed_rhf_pprpa(solver, mf, xy=xy, mult=mult, cphf_conv_tol=1e-10,
                                      cphf_max_cycle=50, pair_get_k=dense)
    got = make_rdm1_relaxed_rhf_pprpa(solver, mf, xy=xy, mult=mult, cphf_conv_tol=1e-10,
                                      cphf_max_cycle=50,
                                      pair_k=pair_exchange(cell, cell.mesh, W))
    np.testing.assert_allclose(got[0], ref[0], rtol=0, atol=1e-9)
    np.testing.assert_allclose(got[1], ref[1], rtol=0, atol=1e-9)


@pytest.mark.parametrize("mult", ("s", "t"))
def test_shared_pair_potential(diamond, mult):
    """One pass of pair potentials gives K[X + Y] @ orbp (against the dense kernel) and
    the pairing force (against the force that builds its own potentials)."""
    from scipy.linalg import block_diag
    from lib_pprpa.gpu_pairing_force import (pair_exchange, pair_potential_lowrank,
                                             pairing_k_force_lowrank)
    cell, mf = diamond
    rng = np.random.default_rng(2)
    nocc = cell.nelectron // 2
    nvir = cell.nao - nocc
    orbp = mf.mo_coeff
    x = rng.standard_normal((nvir, nvir))
    y = rng.standard_normal((nocc, nocc))
    x, y = (x + x.T, y + y.T) if mult == "s" else (x - x.T, y - y.T)
    M = block_diag(y, x)
    dm = orbp @ M @ orbp.T
    W = pair_potential_lowrank(cell, cell.mesh, orbp, M)
    got = pair_exchange(cell, cell.mesh, W)
    ref = (_dense_k(cell, dm[None]) @ orbp)[0]
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 1e-9 * np.abs(ref).max()
    f_shared = pairing_k_force_lowrank(cell, cell.mesh, orbp, M, W=W)
    f_own = pairing_k_force_lowrank(cell, cell.mesh, orbp, M)
    assert np.abs(f_shared - f_own).max() < 1e-10 * np.abs(f_own).max()
