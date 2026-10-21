"""Scoped cuBLAS fp64 emulation: the width rule, the per-handle switch and the emulated ao2mo."""

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


def _needs_emulation():
    from lib_pprpa.gpu_emulation import emulation_available
    if not emulation_available():
        pytest.skip("cuBLAS fp64 emulation is unavailable on this device")


def test_emulation_pays_on_harmonic_width(monkeypatch):
    from lib_pprpa import gpu_emulation
    monkeypatch.setattr(gpu_emulation, '_arch', lambda device_id: 10)
    w = gpu_emulation.EMULATION_MIN_WIDTH[10]
    assert gpu_emulation.emulation_pays(2 * w, 2 * w)
    assert not gpu_emulation.emulation_pays(2 * w - 1, 2 * w)
    assert gpu_emulation.emulation_pays(6 * w, int(1.2 * w))       # 1/m + 1/n = 1/w
    assert not gpu_emulation.emulation_pays(10**6, w - 1)
    assert not gpu_emulation.emulation_pays(0, 10**6)
    assert gpu_emulation.emulation_available('performant')
    monkeypatch.setattr(gpu_emulation, '_arch', lambda device_id: 9)
    assert not gpu_emulation.emulation_pays(2 * w, 2 * w)
    assert not gpu_emulation.emulation_available('performant')
    monkeypatch.setattr(gpu_emulation, '_arch', lambda device_id: None)
    assert not gpu_emulation.emulation_pays(10**4, 10**4)
    assert not gpu_emulation.emulation_available()


def test_switch_is_scoped_to_the_block():
    import cupy as cp
    from cupy.cuda import cublas

    from lib_pprpa.gpu_emulation import fp64_emulation
    _needs_emulation()
    handle = cp.cuda.device.get_cublas_handle()
    mode = cublas.getMathMode(handle)
    rng = cp.random.default_rng(1)
    A = rng.standard_normal((1024, 20000))
    B = rng.standard_normal((1024, 20000))
    ref = A.get() @ B.get().T
    with fp64_emulation('eager'):
        assert cublas.getMathMode(handle) != mode
        C = A.dot(B.T)
    assert cublas.getMathMode(handle) == mode
    np.testing.assert_allclose(C.get(), ref, rtol=0, atol=1e-11 * np.abs(ref).max())
    with fp64_emulation(None):
        assert cublas.getMathMode(handle) == mode


def test_emulated_ao2mo_matches_native(monkeypatch):
    """Every tile emulated (width rule off) gives the native packed blocks."""
    import cupy as cp
    from pyscf.pbc import dft, gto

    from lib_pprpa import gpu_emulation
    from lib_pprpa.gpu_ao2mo import gpu_ao2mo_blocks
    _needs_emulation()
    a0 = 3.370137329
    cell = gto.M(atom=[["C", [0., 0., 0.]], ["C", [a0 / 2, a0 / 2, a0 / 2]]],
                 a=np.array([[0, a0, a0], [a0, 0, a0], [a0, a0, 0]]),
                 unit="bohr", basis="gth-dzvp", pseudo="gth-pade", mesh=[20] * 3, verbose=0)
    mf = dft.RKS(cell, xc="pbe")
    mf.kernel()
    nocc = cell.nelectron // 2
    cocc, cvir = mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:]
    native = gpu_ao2mo_blocks(cell, cocc, cvir, cell.mesh, pair_blk=40, emulate=False)
    major = int(cp.cuda.Device().compute_capability[:-1])
    monkeypatch.setitem(gpu_emulation.EMULATION_MIN_WIDTH, major, 0)
    emulated = gpu_ao2mo_blocks(cell, cocc, cvir, cell.mesh, pair_blk=40)
    for e, n in zip(emulated, native):
        np.testing.assert_allclose(e, n, rtol=0, atol=1e-12 * np.abs(n).max())
