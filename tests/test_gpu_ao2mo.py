"""Tiled GPU FFT ao2mo against pyscf's FFT MO integrals; pair layout and packed tile scatter."""

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


def test_pack_offset_and_eri_bytes():
    from lib_pprpa.gpu_mem import eri_bytes, pack_offset
    assert [pack_offset(n) for n in range(5)] == [0, 1, 3, 6, 10]
    # NV216 at AS=300: 49 GB packed against 194 GB of physicist tensors
    assert eri_bytes(300, 300) == 8 * (2 * pack_offset(pack_offset(300)) + pack_offset(300 * 300))
    assert abs(eri_bytes(300, 300) / 1e9 - 48.71) < 0.01


@pytest.mark.parametrize("compact", (False, True))
@pytest.mark.parametrize("blk", (1, 3, 7, 10**6))
def test_pair_layout_and_pack_tile(compact, blk):
    """Every strip width packs the lower triangle of the Gram matrix exactly once."""
    import cupy as cp
    from lib_pprpa.gpu_ao2mo import _PairLayout, _pack_tile
    from lib_pprpa.gpu_mem import pack_offset
    rng = np.random.default_rng(0)
    nA, nB = 4, (4 if compact else 3)
    layout = _PairLayout(nA, nB, compact)
    npair = layout.npair
    assert npair == (pack_offset(nA) if compact else nA * nB)
    E = rng.standard_normal((npair, npair))
    E = E + E.T
    out = np.full(pack_offset(npair), np.nan)
    for pa, pb in layout.split(0, nA, blk):
        P0, P1 = layout.pairs(pa, pb)
        base = pack_offset(P0)
        buf = cp.full(pack_offset(P1) - base, np.nan)
        for ra, rb in layout.split(0, pb, blk):
            Q0, Q1 = layout.pairs(ra, rb)
            _pack_tile(buf, cp.asarray(E[P0:P1, Q0:Q1]), P0, Q0, base)
        buf.get(out=out[base:base + buf.size])
    assert not np.isnan(out).any()
    np.testing.assert_array_equal(out, E[np.tril_indices(npair)])


def test_free_bytes_releases_cached_memory():
    """Freed pool blocks and cached cuFFT plans count as free."""
    import cupy as cp
    import cupyx.scipy.fft as cufft
    from gpu4pyscf.lib import cupy_helper  # noqa: F401  (the >100 MB cudaMalloc allocator)
    from lib_pprpa.gpu_mem import free_bytes
    n = 96

    def work():
        rows = [cp.ones(4 * 1024**2) for _ in range(64)]      # 64 pooled 32 MiB blocks
        cufft.irfftn(cufft.rfftn(cp.ones((8, n, n, n)), axes=(1, 2, 3)), s=(n, n, n),
                     axes=(1, 2, 3))
        return len(rows)

    work()                       # cuFFT loads its kernels for this shape once per process
    before = free_bytes()
    work()
    cache = cp.fft.config.get_plan_cache()
    assert cache.get_curr_size() > 0
    after = free_bytes()
    assert cache.get_curr_size() == 0
    assert after > before - 16 * 1024**2


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
def test_ao2mo_blocks_match_pyscf(pair_blk):
    """The packed blocks equal the packed pyscf integrals and come back pinned."""
    import cupy as cp
    from lib_pprpa.gpu_ao2mo import gpu_ao2mo_blocks
    from lib_pprpa.gpu_mem import is_pinned
    from lib_pprpa.pprpa_eri_gpu import pack_eri
    cell, cocc, cvir, ref = _diamond("gth-dzv")
    got = gpu_ao2mo_blocks(cell, cocc, cvir, cell.mesh, pair_blk=pair_blk)
    for g, r in zip(got, pack_eri(*ref)):
        assert isinstance(g, np.ndarray) and g.ndim == 1 and is_pinned(g)
        # 8000-point grid sums in a different order than pyscf's: ~5e-11 relative
        np.testing.assert_allclose(g, r, rtol=0, atol=1e-11)
    cp.get_default_pinned_memory_pool().free_all_blocks()
