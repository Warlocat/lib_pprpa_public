"""GPU Davidson ERI contraction against the CPU routine, resident and streamed."""

import numpy as np
import pytest

from lib_pprpa.gpu_mem import eri_bytes, fits_resident


def _gpu4pyscf_available():
    try:
        import cupy as cp
        import gpu4pyscf  # noqa: F401
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


needs_gpu = pytest.mark.skipif(
    not _gpu4pyscf_available(), reason="GPU4PySCF CUDA device is unavailable")


def test_eri_bytes_and_fits_resident():
    assert eri_bytes(300, 300) == 3 * 300**4 * 8
    assert eri_bytes(128, 124) == (124**4 + 128**4 + 128**2 * 124**2) * 8
    b200 = 183359 * 1024**2
    assert not fits_resident(300, 300, total_bytes=b200)
    assert fits_resident(128, 124, total_bytes=b200)
    tiny = eri_bytes(8, 8)
    assert fits_resident(8, 8, total_bytes=int(tiny / 0.75) + 8)
    assert not fits_resident(8, 8, total_bytes=int(tiny / 0.75) - 8)


@needs_gpu
def test_is_pinned():
    import cupyx
    from gpu4pyscf.lib.cupy_helper import pin_memory
    from lib_pprpa.gpu_mem import is_pinned
    a = np.ones((4, 6))
    assert not is_pinned(a)
    p = cupyx.empty_pinned((4, 6))
    assert is_pinned(p) and is_pinned(p.reshape(24)) and is_pinned(p[1:, :3])
    assert is_pinned(pin_memory(a))


def _random_eri(rng, nocc, nvir):
    # <ab|cd> = <cd|ab> for real orbitals: the flattened vvvv / oooo are symmetric
    nv2, no2 = nvir * nvir, nocc * nocc
    V = rng.standard_normal((nv2, nv2))
    O = rng.standard_normal((no2, no2))
    vvvv = ((V + V.T) * 0.5).reshape(nvir, nvir, nvir, nvir)
    oooo = ((O + O.T) * 0.5).reshape(nocc, nocc, nocc, nocc)
    oovv = rng.standard_normal((nocc, nocc, nvir, nvir))
    return vvvv, oovv, oooo


def _solver(nocc, mo_energy, multi, eri=None):
    from lib_pprpa.pprpa_davidson import ppRPA_Davidson
    mp = ppRPA_Davidson(nocc, mo_energy, Lpq=None, channel="hh", nroot=2,
                        residue_thresh=1e-10, trial="identity")
    mp.mu = 0.0
    mp.multi = multi
    if eri is not None:
        mp.use_eri(*eri)
    mp.check_parameter()
    return mp


@needs_gpu
@pytest.mark.parametrize("multi", ("s", "t"))
@pytest.mark.parametrize("ntri", (1, 7))
def test_mvp_matches_cpu(multi, ntri):
    from lib_pprpa.pprpa_davidson import _pprpa_contraction
    from lib_pprpa.pprpa_eri_gpu import attach_gpu_eri_contraction, release_gpu_eri

    rng = np.random.default_rng(4)
    nocc, nvir = 6, 8
    moe = rng.standard_normal(nocc + nvir)
    vvvv, oovv, oooo = _random_eri(rng, nocc, nvir)
    cpu = _solver(nocc, moe, multi, eri=(vvvv, oovv, oooo))
    tv = rng.standard_normal((ntri, cpu.full_dim))
    ref = _pprpa_contraction(cpu, tv)

    gpu = _solver(nocc, moe, multi)
    attach_gpu_eri_contraction(gpu, vvvv, oovv, oooo, mode="resident")
    np.testing.assert_allclose(gpu.contraction(tv), ref, rtol=1e-11, atol=1e-11)
    for strip in (1, 5, 64):
        attach_gpu_eri_contraction(gpu, vvvv, oovv, oooo, mode="streamed", strip=strip)
        np.testing.assert_allclose(gpu.contraction(tv), ref, rtol=1e-11, atol=1e-11)
    release_gpu_eri(gpu)
    assert not hasattr(gpu, "_gpu_vvvv")
    assert "contraction" not in gpu.__dict__


@needs_gpu
def test_auto_mode_and_device_input():
    import cupy as cp
    import cupyx
    from lib_pprpa.gpu_mem import is_pinned
    from lib_pprpa.pprpa_eri_gpu import attach_gpu_eri_contraction, release_gpu_eri

    rng = np.random.default_rng(2)
    nocc, nvir = 4, 5
    moe = rng.standard_normal(nocc + nvir)
    vvvv, oovv, oooo = _random_eri(rng, nocc, nvir)
    gpu = _solver(nocc, moe, "t")
    attach_gpu_eri_contraction(gpu, vvvv, oovv, oooo)
    assert gpu._gpu_eri_strip is None and isinstance(gpu._gpu_vvvv, cp.ndarray)
    attach_gpu_eri_contraction(gpu, cp.asarray(vvvv), cp.asarray(oovv), cp.asarray(oooo))
    assert gpu._gpu_eri_strip is None and isinstance(gpu._gpu_vvvv, cp.ndarray)
    attach_gpu_eri_contraction(gpu, vvvv, oovv, oooo, mode="streamed")
    assert 1 <= gpu._gpu_eri_strip <= nvir * nvir
    assert isinstance(gpu._gpu_vvvv, np.ndarray)
    assert all(is_pinned(b) for b in (gpu._gpu_vvvv, gpu._gpu_oooo, gpu._gpu_oovv))
    # blocks already in pinned memory are streamed in place, not copied
    pinned_vvvv = cupyx.empty_pinned(vvvv.shape)
    pinned_vvvv[...] = vvvv
    attach_gpu_eri_contraction(gpu, pinned_vvvv, oovv, oooo, mode="streamed", strip=5)
    assert np.shares_memory(gpu._gpu_vvvv, pinned_vvvv)
    attach_gpu_eri_contraction(gpu, vvvv, oovv, oooo, mode="streamed", pin=False)
    assert not is_pinned(gpu._gpu_vvvv)
    release_gpu_eri(gpu)


@needs_gpu
@pytest.mark.parametrize("multi", ("s", "t"))
def test_davidson_energies_match_cpu(multi):
    """Full Davidson solve on a diamond cell: streamed GPU MVP vs the CPU use_eri path."""
    from pyscf.pbc import dft, gto
    from lib_pprpa.pprpa_eri_gpu import attach_gpu_eri_contraction, release_gpu_eri

    a0 = 3.370137329
    cell = gto.M(atom=[["C", [0., 0., 0.]], ["C", [a0 / 2, a0 / 2, a0 / 2]]],
                 a=np.array([[0, a0, a0], [a0, 0, a0], [a0, a0, 0]]),
                 unit="bohr", basis="gth-dzv", pseudo="gth-pade", mesh=[20] * 3, verbose=0)
    mf = dft.RKS(cell, xc="pbe")
    mf.exxdiv = None
    mf.conv_tol = 1e-10
    mf.kernel()
    nocc = cell.nelectron // 2
    nmo = cell.nao
    eri = mf.with_df.get_mo_eri(mf.mo_coeff, compact=False)
    eri = eri.reshape(nmo, nmo, nmo, nmo).transpose(0, 2, 1, 3)
    vvvv = np.ascontiguousarray(eri[nocc:, nocc:, nocc:, nocc:])
    oovv = np.ascontiguousarray(eri[:nocc, :nocc, nocc:, nocc:])
    oooo = np.ascontiguousarray(eri[:nocc, :nocc, :nocc, :nocc])

    cpu = _solver(nocc, mf.mo_energy, multi, eri=(vvvv, oovv, oooo))
    cpu.kernel(multi)
    gpu = _solver(nocc, mf.mo_energy, multi)
    attach_gpu_eri_contraction(gpu, vvvv, oovv, oooo, mode="streamed", strip=3)
    gpu.kernel(multi)
    exci_cpu = cpu.exci_s if multi == "s" else cpu.exci_t
    exci_gpu = gpu.exci_s if multi == "s" else gpu.exci_t
    np.testing.assert_allclose(exci_gpu, exci_cpu, rtol=0, atol=1e-8)
    release_gpu_eri(gpu)
