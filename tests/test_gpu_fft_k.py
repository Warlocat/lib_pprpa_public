"""Low-rank FFT exchange against gpu4pyscf's dense fft_jk.get_k, and through the
pair_get_k hook of the relaxed density."""

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


def test_get_k_lowrank_matches_dense(diamond):
    """Symmetric, antisymmetric and general low-rank densities, full K and K @ ket.

    The Coulomb operator acts on the ket-side codensity here and on the bra side in
    the dense kernel; the sums over the grid then round differently at 1e-11.
    """
    from lib_pprpa.gpu_fft_k import get_k_lowrank
    cell, mf = diamond
    rng = np.random.default_rng(0)
    nao = cell.nao
    nocc = cell.nelectron // 2
    C = mf.mo_coeff
    x = rng.standard_normal((3, 3))
    y = rng.standard_normal((nocc, nocc))
    factors = [(C[:, nocc:nocc + 3] @ (x + x.T), C[:, nocc:nocc + 3]),      # symmetric
               (C[:, :nocc] @ (y - y.T), C[:, :nocc]),                      # antisymmetric
               (rng.standard_normal((nao, 4)), rng.standard_normal((nao, 4)))]   # general
    dms = np.stack([L @ R.T for L, R in factors])
    ref = _dense_k(cell, dms)
    scale = np.abs(ref).max()
    got = get_k_lowrank(cell, cell.mesh, factors)
    assert np.abs(got - ref).max() < 1e-9 * scale
    ket = C[:, 1:6]
    got_ket = get_k_lowrank(cell, cell.mesh, factors, ket=ket)
    assert np.abs(got_ket - ref @ ket).max() < 1e-9 * scale


@pytest.mark.parametrize("mult", ("s", "t"))
def test_pair_get_k_lowrank_callback(diamond, mult):
    """The callback recovers the factors from the dense pair densities and returns K @ orbp."""
    from lib_pprpa.gpu_fft_k import pair_get_k_lowrank
    cell, mf = diamond
    rng = np.random.default_rng(1)
    nocc = cell.nelectron // 2
    nvir = cell.nao - nocc
    C = mf.mo_coeff
    orbi, orba, orbp = C[:, :nocc], C[:, nocc:], C
    x = rng.standard_normal((nvir, nvir))
    y = rng.standard_normal((nocc, nocc))
    x, y = (x + x.T, y + y.T) if mult == "s" else (x - x.T, y - y.T)
    dms = np.stack((orba @ x @ orba.T, orbi @ y @ orbi.T))
    ref = _dense_k(cell, dms) @ orbp
    cb = pair_get_k_lowrank(cell, mf, orbp)
    assert cb.accepts_ket
    got = cb(dms, hermi=1 if mult == "s" else 2, ket=orbp)
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 1e-9 * np.abs(ref).max()


@pytest.mark.parametrize("mult", ("s", "t"))
def test_relaxed_density_through_hook(diamond, mult):
    """make_rdm1_relaxed_rhf_pprpa with the low-rank callback equals the dense one."""
    from lib_pprpa.grad.pprpa import make_rdm1_relaxed_rhf_pprpa
    from lib_pprpa.gpu_fft_k import pair_get_k_lowrank
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

    def dense(dms, hermi):
        return _dense_k(cell, dms)

    ref = make_rdm1_relaxed_rhf_pprpa(solver, mf, mult=mult, cphf_conv_tol=1e-10,
                                      cphf_max_cycle=50, pair_get_k=dense)
    got = make_rdm1_relaxed_rhf_pprpa(solver, mf, mult=mult, cphf_conv_tol=1e-10,
                                      cphf_max_cycle=50,
                                      pair_get_k=pair_get_k_lowrank(cell, mf, mf.mo_coeff))
    np.testing.assert_allclose(got[0], ref[0], rtol=0, atol=1e-9)
    np.testing.assert_allclose(got[1], ref[1], rtol=0, atol=1e-9)
