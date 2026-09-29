"""GPU Davidson ERI contraction from packed blocks against the CPU routine, resident and streamed."""

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
    assert eri_bytes(300, 300) == 8 * (2 * 45150 * 45151 // 2 + 90000 * 90001 // 2)
    assert eri_bytes(8, 8) == 8 * (2 * 36 * 37 // 2 + 64 * 65 // 2)
    b200 = 183359 * 1024**2
    assert fits_resident(300, 300, total_bytes=b200)
    assert not fits_resident(400, 400, total_bytes=b200)
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


def _random_eri(rng, nocc, nvir, ng=40):
    """Physicist blocks of random real orbitals on a grid: every ERI symmetry holds."""
    phi = rng.standard_normal((nocc + nvir, ng))
    W = rng.standard_normal((ng, ng)) / ng
    W = W + W.T
    rho = np.einsum("pg,qg->pqg", phi, phi)
    phys = np.einsum("pqg,gh,rsh->pqrs", rho, W, rho).transpose(0, 2, 1, 3)   # <pr|qs>
    return (np.ascontiguousarray(phys[nocc:, nocc:, nocc:, nocc:]),
            np.ascontiguousarray(phys[:nocc, :nocc, nocc:, nocc:]),
            np.ascontiguousarray(phys[:nocc, :nocc, :nocc, :nocc]))


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
def test_pack_eri():
    """pack_eri stores (pq|rs) for Q <= P at P(P+1)/2 + Q, on NumPy and CuPy input."""
    import cupy as cp
    from lib_pprpa.gpu_mem import pack_offset
    from lib_pprpa.pprpa_eri_gpu import pack_eri
    rng = np.random.default_rng(1)
    nocc, nvir = 3, 4
    vvvv, oovv, oooo = _random_eri(rng, nocc, nvir)
    pv, pov, po = pack_eri(vvvv, oovv, oooo)
    assert pv.shape == (pack_offset(pack_offset(nvir)),)
    assert po.shape == (pack_offset(pack_offset(nocc)),)
    assert pov.shape == (pack_offset(nocc * nvir),)
    a, b, c, d = 3, 1, 2, 0                         # <ab|cd> = (ac|bd): P = (a,c), Q = (b,d)
    P, Q = pack_offset(a) + c, pack_offset(b) + d
    assert pv[pack_offset(P) + Q] == vvvv[a, b, c, d]
    i, j, a, b = 2, 1, 3, 0                         # <ij|ab> = (ia|jb): P = (i,a), Q = (j,b)
    P, Q = i * nvir + a, j * nvir + b
    assert pov[pack_offset(P) + Q] == oovv[i, j, a, b]
    gv, gov, go = pack_eri(cp.asarray(vvvv), cp.asarray(oovv), cp.asarray(oooo))
    for g, h in zip((gv, gov, go), (pv, pov, po)):
        assert isinstance(g, cp.ndarray)
        np.testing.assert_array_equal(g.get(), h)


@needs_gpu
@pytest.mark.parametrize("multi", ("s", "t"))
@pytest.mark.parametrize("ntri", (1, 7))
def test_mvp_matches_cpu(multi, ntri):
    from lib_pprpa.pprpa_davidson import _pprpa_contraction
    from lib_pprpa.pprpa_eri_gpu import attach_gpu_eri_contraction, pack_eri, release_gpu_eri

    rng = np.random.default_rng(4)
    nocc, nvir = 6, 8
    moe = rng.standard_normal(nocc + nvir)
    vvvv, oovv, oooo = _random_eri(rng, nocc, nvir)
    cpu = _solver(nocc, moe, multi, eri=(vvvv, oovv, oooo))
    tv = rng.standard_normal((ntri, cpu.full_dim))
    ref = _pprpa_contraction(cpu, tv)
    packed = pack_eri(vvvv, oovv, oooo)

    gpu = _solver(nocc, moe, multi)
    attach_gpu_eri_contraction(gpu, *packed, mode="resident")
    np.testing.assert_allclose(gpu.contraction(tv), ref, rtol=1e-11, atol=1e-11)
    for rows in (1, 2, 5, 64):
        attach_gpu_eri_contraction(gpu, *packed, mode="resident", rows=rows)
        np.testing.assert_allclose(gpu.contraction(tv), ref, rtol=1e-11, atol=1e-11)
        attach_gpu_eri_contraction(gpu, *packed, mode="streamed", rows=rows)
        np.testing.assert_allclose(gpu.contraction(tv), ref, rtol=1e-11, atol=1e-11)
    release_gpu_eri(gpu)
    assert not hasattr(gpu, "_gpu_eri_pieces")
    assert "contraction" not in gpu.__dict__


@needs_gpu
def test_auto_mode_and_device_input():
    import cupy as cp
    import cupyx
    from lib_pprpa.gpu_mem import is_pinned
    from lib_pprpa.pprpa_eri_gpu import attach_gpu_eri_contraction, pack_eri, release_gpu_eri

    rng = np.random.default_rng(2)
    nocc, nvir = 4, 5
    moe = rng.standard_normal(nocc + nvir)
    vvvv, oovv, oooo = _random_eri(rng, nocc, nvir)
    with pytest.raises(ValueError):
        attach_gpu_eri_contraction(_solver(nocc, moe, "t"), vvvv, oovv, oooo)
    pv, pov, po = pack_eri(vvvv, oovv, oooo)
    gpu = _solver(nocc, moe, "t")
    attach_gpu_eri_contraction(gpu, pv, pov, po)
    assert gpu._gpu_eri_mode == "resident"
    assert all(isinstance(piece[3], cp.ndarray) for piece in gpu._gpu_eri_pieces)
    attach_gpu_eri_contraction(gpu, cp.asarray(pv), cp.asarray(pov), cp.asarray(po))
    assert gpu._gpu_eri_mode == "resident"
    attach_gpu_eri_contraction(gpu, pv, pov, po, mode="streamed")
    assert all(isinstance(piece[3], np.ndarray) and is_pinned(piece[3])
               for piece in gpu._gpu_eri_pieces)
    # blocks already in pinned memory are streamed in place, not copied
    pinned = cupyx.empty_pinned(pv.shape)
    pinned[...] = pv
    attach_gpu_eri_contraction(gpu, pinned, pov, po, mode="streamed", rows=2)
    strips = [piece for piece in gpu._gpu_eri_pieces if piece[0] == "vvvv"]
    assert len(strips) == 3 and all(np.shares_memory(s[3], pinned) for s in strips)
    release_gpu_eri(gpu)


@needs_gpu
@pytest.mark.parametrize("multi", ("s", "t"))
def test_davidson_energies_match_cpu(multi):
    """Full Davidson solve on a diamond cell: streamed GPU MVP vs the CPU use_eri path."""
    from pyscf.pbc import dft, gto
    from lib_pprpa.pprpa_eri_gpu import attach_gpu_eri_contraction, pack_eri, release_gpu_eri

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
    attach_gpu_eri_contraction(gpu, *pack_eri(vvvv, oovv, oooo), mode="streamed", rows=3)
    gpu.kernel(multi)
    exci_cpu = cpu.exci_s if multi == "s" else cpu.exci_t
    exci_gpu = gpu.exci_s if multi == "s" else gpu.exci_t
    np.testing.assert_allclose(exci_gpu, exci_cpu, rtol=0, atol=1e-8)
    release_gpu_eri(gpu)
