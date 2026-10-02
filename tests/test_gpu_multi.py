"""Multi-device dispatch: two ranks on one GPU (run one after another) must
reproduce the single-device ao2mo blocks, the split-resident Davidson product,
the low-rank exchange and the full gradient."""

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

TWO = [0, 0]     # two ranks on the first device: the split bookkeeping without a second GPU


def test_run_and_each():
    import cupy as cp
    from lib_pprpa.gpu_multi import default_devices, each, run
    assert default_devices(None) == [cp.cuda.Device().id]
    ranks = each(TWO, lambda rank: rank)
    assert ranks == [0, 1]
    results, states = run(TWO, lambda rank: {"n": 0},
                          lambda rank, st, task: st.__setitem__("n", st["n"] + 1) or task * 2,
                          range(7))
    assert results == [2 * t for t in range(7)]
    assert sum(st["n"] for st in states) == 7
    # ranks sharing a device run in one thread, one after the other
    threads = each(TWO, lambda rank: __import__("threading").get_ident())
    assert threads[0] == threads[1]

    def boom(rank, st, task):
        raise ValueError("task %d" % task)
    with pytest.raises(ValueError):
        run(TWO, lambda rank: None, boom, range(3))


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


def test_ao2mo_two_workers_match_one(diamond):
    from lib_pprpa.gpu_ao2mo import gpu_ao2mo_blocks
    from lib_pprpa.gpu_mem import is_pinned
    cell, mf = diamond
    nocc = cell.nelectron // 2
    cocc, cvir = mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:]
    one = gpu_ao2mo_blocks(cell, cocc, cvir, cell.mesh, pair_blk=3)
    two = gpu_ao2mo_blocks(cell, cocc, cvir, cell.mesh, pair_blk=3, devices=TWO)
    for a, b in zip(one, two):
        np.testing.assert_allclose(b, a, rtol=0, atol=1e-13)


@pytest.mark.parametrize("multi", ("s", "t"))
def test_split_mvp_matches_resident(diamond, multi):
    from lib_pprpa.pprpa_davidson import ppRPA_Davidson, _pprpa_contraction
    from lib_pprpa.pprpa_eri_gpu import attach_gpu_eri_contraction, pack_eri, release_gpu_eri
    rng = np.random.default_rng(3)
    nocc, nvir = 5, 7
    # real orbitals on a grid: the packed form assumes every ERI symmetry
    phi = rng.standard_normal((nocc + nvir, 40))
    W = rng.standard_normal((40, 40)) / 40
    rho = np.einsum("pg,qg->pqg", phi, phi)
    phys = np.einsum("pqg,gh,rsh->pqrs", rho, W + W.T, rho).transpose(0, 2, 1, 3)
    vvvv = np.ascontiguousarray(phys[nocc:, nocc:, nocc:, nocc:])
    oovv = np.ascontiguousarray(phys[:nocc, :nocc, nocc:, nocc:])
    oooo = np.ascontiguousarray(phys[:nocc, :nocc, :nocc, :nocc])
    moe = rng.standard_normal(nocc + nvir)

    def solver():
        mp = ppRPA_Davidson(nocc, moe, Lpq=None, channel="hh", nroot=2,
                            residue_thresh=1e-10, trial="identity")
        mp.mu = 0.0
        mp.multi = multi
        mp.check_parameter()
        return mp
    cpu = solver()
    cpu.use_eri(vvvv, oovv, oooo)
    tv = rng.standard_normal((6, cpu.full_dim))
    ref = _pprpa_contraction(cpu, tv)
    gpu = solver()
    packed = pack_eri(vvvv, oovv, oooo)
    for rows in (None, 2):
        attach_gpu_eri_contraction(gpu, *packed, mode="split", devices=TWO, rows=rows)
        assert gpu._gpu_eri_split is not None and gpu._gpu_eri_pieces is None
        assert all(len(part) > 0 for part in gpu._gpu_eri_split[1])
        np.testing.assert_allclose(gpu.contraction(tv), ref, rtol=1e-11, atol=1e-11)
    release_gpu_eri(gpu)
    assert not hasattr(gpu, "_gpu_eri_split")


def test_lowrank_k_two_workers_match_one(diamond):
    from lib_pprpa.gpu_fft_k import get_k_lowrank
    cell, mf = diamond
    rng = np.random.default_rng(5)
    nao = cell.nao
    factors = [(rng.standard_normal((nao, 3)), rng.standard_normal((nao, 3))),
               (rng.standard_normal((nao, 5)), rng.standard_normal((nao, 5)))]
    ket = mf.mo_coeff[:, :7]
    one = get_k_lowrank(cell, cell.mesh, factors, ket=ket)
    two = get_k_lowrank(cell, cell.mesh, factors, ket=ket, devices=TWO)
    assert np.abs(two - one).max() < 1e-12 * np.abs(one).max()


def test_gradient_two_workers_match_one(diamond):
    from lib_pprpa.gpu_ao2mo import gpu_ao2mo_blocks
    from lib_pprpa.grad import pprpa_gamma_gpu as gpu
    from lib_pprpa.pprpa_davidson import ppRPA_Davidson
    from lib_pprpa.pprpa_eri_gpu import attach_gpu_eri_contraction
    cell, mf = diamond
    nocc = cell.nelectron // 2
    eri = gpu_ao2mo_blocks(cell, mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:], cell.mesh)
    mp = ppRPA_Davidson(nocc, mf.mo_energy, Lpq=None, channel="hh", nroot=2,
                        residue_thresh=1e-10, trial="identity")
    mp.mu = 0.0
    attach_gpu_eri_contraction(mp, *eri, mode="resident")
    mp.kernel("t")
    xy = np.array(mp.xy_t[0], copy=True)
    de = []
    for devices in (None, TWO):
        g = gpu.Gradients(mp, mf, "t", 0)
        g.cphf_conv_tol = 1e-10
        g.cphf_max_cycle = 100
        g.devices = devices
        de.append(g.grad_elec(xy, "t", range(cell.natm)))
    assert np.abs(de[1] - de[0]).max() < 1e-9 * np.abs(de[0]).max()
