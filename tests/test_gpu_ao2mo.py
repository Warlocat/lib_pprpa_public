"""Tiled GPU FFT ao2mo against pyscf's FFT MO integrals; pair layout and tile scatter."""

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


def _diamond(basis="gth-szv"):
    from pyscf.pbc import dft, gto
    a0 = 3.370137329
    cell = gto.M(atom=[["C", [0., 0., 0.]], ["C", [a0 / 2, a0 / 2, a0 / 2]]],
                 a=np.array([[0, a0, a0], [a0, 0, a0], [a0, a0, 0]]),
                 unit="bohr", basis=basis, pseudo="gth-pade", mesh=[20] * 3, verbose=0)
    mf = dft.RKS(cell, xc="pbe")
    mf.exxdiv = None
    mf.conv_tol = 1e-10
    mf.kernel()
    nocc = cell.nelectron // 2
    nmo = cell.nao
    eri = mf.with_df.get_mo_eri(mf.mo_coeff, compact=False)
    eri = eri.reshape(nmo, nmo, nmo, nmo).transpose(0, 2, 1, 3)      # <pq|rs>
    ref = (eri[nocc:, nocc:, nocc:, nocc:], eri[:nocc, :nocc, nocc:, nocc:],
           eri[:nocc, :nocc, :nocc, :nocc])
    return cell, mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:], ref


def _reference_block(phiA, phiB, W):
    # out[a, b, c, d] = <ab|cd> = (ac|bd) = sum_gh phiA_a phiB_c (g) W[g, h] phiA_b phiB_d (h)
    rho = np.einsum("ag,cg->acg", phiA, phiB)
    return np.einsum("acg,gh,bdh->acbd", rho, W, rho).transpose(0, 2, 1, 3)


@pytest.mark.parametrize("compact", (False, True))
@pytest.mark.parametrize("blk", (1, 3, 7, 10**6))
def test_pair_layout_and_tile_scatter(compact, blk):
    """Every strip width reproduces the brute-force tensor and writes each element."""
    from lib_pprpa.gpu_ao2mo import _PairLayout, _write_tile
    rng = np.random.default_rng(0)
    nA, nB, ng = 4, (4 if compact else 3), 9
    phiA = rng.standard_normal((nA, ng))
    phiB = phiA if compact else rng.standard_normal((nB, ng))
    W = rng.standard_normal((ng, ng))
    W = W + W.T
    layout = _PairLayout(nA, nB, compact)
    pidx, qidx = (x.get() for x in layout.index_arrays())
    rho = phiA[pidx] * phiB[qidx]
    E = rho @ W @ rho.T
    out = np.full((nA, nA, nB, nB), np.nan)
    for pa, pb in layout.split(0, nA, blk):
        P0, P1 = layout.pairs(pa, pb)
        for ra, rb in layout.split(0, pb, blk):
            Q0, Q1 = layout.pairs(ra, rb)
            _write_tile(out, E[P0:P1, Q0:Q1], layout, pa, pb, ra, rb)
    assert not np.isnan(out).any()
    np.testing.assert_allclose(out, _reference_block(phiA, phiB, W), rtol=0, atol=1e-12)


def test_max_fft_batch():
    from lib_pprpa.gpu_mem import max_fft_batch, needs_bluestein
    assert needs_bluestein((151, 151, 151)) and not needs_bluestein((107, 128, 159))
    assert max_fft_batch(151**3, (151, 151, 151)) == (2**31 - 1) // 151**3
    assert max_fft_batch(159**3, (159, 159, 159)) == int(1.5 * 2**31) // 159**3


def test_coulomb_potential_matches_complex_chain():
    """R2C with the symmetrised kernel equals pyscf's ifft(fft(rho) w).real on an even mesh."""
    import cupy as cp
    from gpu4pyscf.pbc import tools as gtools
    from lib_pprpa.gpu_coulomb import coulomb_potential, half_kernel
    cell, cocc, cvir, _ = _diamond()
    mesh = cell.mesh
    ng = int(np.prod(mesh))
    w = gtools.get_coulG(cell, mesh=mesh) * (cell.vol / ng)
    rho = cp.asarray(np.random.default_rng(1).standard_normal((5, ng)))
    ref = gtools.ifft(gtools.fft(rho, mesh) * w, mesh).real
    got = coulomb_potential(rho, mesh, half_kernel(w, mesh))
    assert float(cp.abs(got - ref).max()) < 1e-12 * float(cp.abs(ref).max())


@pytest.mark.parametrize("pair_blk", (None, 3))
@pytest.mark.parametrize("stage_host", (False, True))
def test_ao2mo_blocks_match_pyscf(pair_blk, stage_host):
    import cupy as cp
    from lib_pprpa.gpu_ao2mo import gpu_ao2mo_blocks
    from lib_pprpa.gpu_mem import is_pinned
    cell, cocc, cvir, ref = _diamond("gth-dzv")
    got = gpu_ao2mo_blocks(cell, cocc, cvir, cell.mesh, pair_blk=pair_blk,
                           stage_host=stage_host, return_gpu=True)
    for g, r in zip(got, ref):
        # a small cell always fits: staged finals are uploaded after the build
        assert isinstance(g, cp.ndarray)
        # 8000-point grid sums in a different order than pyscf's: ~5e-11 relative
        np.testing.assert_allclose(g.get(), r, rtol=0, atol=1e-11)


def test_ao2mo_host_staging_is_pinned(monkeypatch):
    """With the upload suppressed, host-staged finals come back as pinned NumPy arrays."""
    import cupy as cp
    from lib_pprpa import gpu_ao2mo
    from lib_pprpa.gpu_mem import is_pinned
    cell, cocc, cvir, ref = _diamond()
    monkeypatch.setattr(gpu_ao2mo, "fits_resident", lambda *a, **k: False)
    got = gpu_ao2mo.gpu_ao2mo_blocks(cell, cocc, cvir, cell.mesh, stage_host=True)
    for g, r in zip(got, ref):
        assert isinstance(g, np.ndarray) and is_pinned(g)
        np.testing.assert_allclose(g, r, rtol=0, atol=1e-12)
    cp.get_default_pinned_memory_pool().free_all_blocks()
