"""GPU Davidson products from the pair-space operator against the CPU use_eri routine,
resident, streamed and split; Gram-block packing and pinned host memory."""

import numpy as np
import pytest

from lib_pprpa.gpu_mem import eri_bytes


def _gpu4pyscf_available():
    try:
        import cupy as cp
        import gpu4pyscf  # noqa: F401
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


needs_gpu = pytest.mark.skipif(
    not _gpu4pyscf_available(), reason="GPU4PySCF CUDA device is unavailable")


def test_eri_bytes():
    assert eri_bytes(300, 300) == 8 * (2 * 45150 * 45151 // 2 + 90000 * 90001 // 2)
    assert eri_bytes(8, 8) == 8 * (2 * 36 * 37 // 2 + 64 * 65 // 2)


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
    from lib_pprpa.gpu_mem import empty_pinned
    q = empty_pinned((3, 5))
    assert q.shape == (3, 5) and q.dtype == np.float64
    assert is_pinned(q) and is_pinned(q[1:, 2:])
    assert empty_pinned(0).shape == (0,)


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
def test_strips_cover_the_operator():
    """Strips tile every block; their storage is M plus the full diagonal blocks."""
    from lib_pprpa.pprpa_eri_gpu import _pair_offset, _strips, pair_operator_bytes
    nocc, nvir = 30, 70
    for multi in ("s", "t"):
        for rows in (None, 1, 4):
            plan = _strips(nocc, nvir, multi, rows)
            extra = 0
            for kind, n in (("oo", nocc), ("vv", nvir), ("ov", nocc)):
                spans = [(R0, R1) for k, _, _, R0, R1, _ in plan if k == kind]
                assert spans[0][0] == 0 and spans[-1][1] == _pair_offset(n, multi)
                assert [s[0] for s in spans[1:]] == [s[1] for s in spans[:-1]]
                if kind != "ov":
                    extra += sum(8 * (R1 - R0) * (R1 - R0 - 1) // 2 for R0, R1 in spans)
            assert sum(s[-1] for s in plan) == pair_operator_bytes(nocc, nvir, multi) + extra


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
    assert cpu.eri_nbytes() == 8 * (nocc**4 + nvir**4 + nocc**2 * nvir**2)
    tv = rng.standard_normal((ntri, cpu.full_dim))
    ref = _pprpa_contraction(cpu, tv)
    packed = pack_eri(vvvv, oovv, oooo)

    for mode, devices in (("resident", None), ("streamed", None), ("split", [0, 0])):
        for rows in (None, 1, 2, 64):
            gpu = _solver(nocc, moe, multi)
            attach_gpu_eri_contraction(gpu, *packed, mode=mode, rows=rows, devices=devices)
            np.testing.assert_allclose(gpu.contraction(tv), ref, rtol=1e-11, atol=1e-11)
            release_gpu_eri(gpu)
            assert not hasattr(gpu, "_gpu_eri") and "contraction" not in gpu.__dict__
            assert "eri_nbytes" not in gpu.__dict__


@needs_gpu
def test_operator_is_built_for_one_multiplicity():
    import cupy as cp
    from lib_pprpa.pprpa_eri_gpu import attach_gpu_eri_contraction, pack_eri, release_gpu_eri
    rng = np.random.default_rng(2)
    nocc, nvir = 4, 5
    moe = rng.standard_normal(nocc + nvir)
    gpu = _solver(nocc, moe, "s")
    packed = pack_eri(*_random_eri(rng, nocc, nvir))
    attach_gpu_eri_contraction(gpu, *packed)
    assert gpu._gpu_eri["mode"] == "resident"
    before = gpu.eri_nbytes()
    gpu.contraction(rng.standard_normal((2, gpu.full_dim)))
    op = gpu._gpu_eri
    assert op["multi"] == "s" and op["blocks"] is None
    assert all(isinstance(s[3], cp.ndarray) for s in op["strips"])
    # the Gram blocks are counted until the first product converts them
    assert gpu.eri_nbytes() == sum(s[3].nbytes for s in op["strips"])
    assert before == gpu.eri_nbytes() + sum(b.nbytes for b in packed)
    gpu.multi = "t"
    gpu.check_parameter()
    with pytest.raises(ValueError):
        gpu.contraction(rng.standard_normal((2, gpu.full_dim)))
    release_gpu_eri(gpu)


@needs_gpu
def test_auto_mode_follows_free_memory(monkeypatch):
    """'auto' streams when the operator and the product arrays do not fit the free
    memory left after the workspace reserve, and keeps it resident when they do."""
    from lib_pprpa import pprpa_eri_gpu as mod
    from lib_pprpa.pprpa_davidson import _pprpa_contraction
    rng = np.random.default_rng(8)
    nocc, nvir = 4, 5
    moe = rng.standard_normal(nocc + nvir)
    eri = _random_eri(rng, nocc, nvir)
    cpu = _solver(nocc, moe, "s", eri=eri)
    tv = rng.standard_normal((3, cpu.full_dim))
    for free, mode in ((512 * 1024**2, "streamed"), (2 * 1024**3, "resident")):
        monkeypatch.setattr(mod, "free_bytes", lambda f=free: f)
        gpu = _solver(nocc, moe, "s")
        mod.attach_gpu_eri_contraction(gpu, *mod.pack_eri(*eri))
        assert gpu._gpu_eri["mode"] == mode
        np.testing.assert_allclose(gpu.contraction(tv), _pprpa_contraction(cpu, tv),
                                   rtol=1e-11, atol=1e-11)
        mod.release_gpu_eri(gpu)


@needs_gpu
@pytest.mark.parametrize("multi", ("s", "t"))
@pytest.mark.parametrize("pair_blk, devices", ((None, None), (3, None), (30, [0, 0])))
def test_fused_ao2mo_operator_matches_cpu(multi, pair_blk, devices):
    """The operator built inside the FFT ao2mo gives the CPU products on pyscf's
    integrals, placed in every mode, and only for its own multiplicity."""
    from test_gpu_ao2mo import _diamond
    from lib_pprpa.pprpa_davidson import _pprpa_contraction
    from lib_pprpa.pprpa_eri_gpu import (attach_gpu_eri_contraction, gpu_ao2mo_pair_operator,
                                            release_gpu_eri)
    cell, cocc, cvir, ref = _diamond("gth-dzv")
    nocc = cocc.shape[1]
    rng = np.random.default_rng(6)
    moe = np.sort(rng.standard_normal(nocc + cvir.shape[1]))
    cpu = _solver(nocc, moe, multi, eri=ref)
    tv = rng.standard_normal((5, cpu.full_dim))
    expect = _pprpa_contraction(cpu, tv)
    op = gpu_ao2mo_pair_operator(cell, cocc, cvir, cell.mesh, multi, pair_blk=pair_blk,
                                 devices=devices)
    assert op.multi == multi and all(isinstance(s[3], np.ndarray) for s in op.strips)
    for mode, devs in (("resident", None), ("streamed", None), ("split", [0, 0])):
        gpu = _solver(nocc, moe, multi)
        attach_gpu_eri_contraction(gpu, op, mode=mode, devices=devs)
        assert gpu.eri_nbytes() == op.nbytes
        # ao2mo sums the 8000-point grid in a different order than pyscf (~5e-11 per element)
        np.testing.assert_allclose(gpu.contraction(tv), expect, rtol=1e-9, atol=1e-9)
        release_gpu_eri(gpu)
    gpu = _solver(nocc, moe, "t" if multi == "s" else "s")
    attach_gpu_eri_contraction(gpu, op)
    with pytest.raises(ValueError):
        gpu.contraction(np.zeros((1, gpu.full_dim)))
    release_gpu_eri(gpu)


@needs_gpu
@pytest.mark.parametrize("multi", ("s", "t"))
def test_davidson_energies_match_cpu(multi):
    """Full Davidson solve on a diamond cell: streamed pair operator vs the CPU use_eri path."""
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
