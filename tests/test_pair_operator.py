"""The CPU pair-space operator: Gram packing, the strip layout, and the use_eri products
and Davidson roots against the Lpq path on the same integrals."""

import numpy as np
import pytest

import lib_pprpa.pair_operator as pair_operator_module
from lib_pprpa.pair_operator import (operator_strips, pack_eri, pack_offset, pair_offset,
                                     pair_operator, pair_operator_bytes)
from lib_pprpa.pprpa_davidson import _pprpa_contraction, ppRPA_Davidson


def _system(seed, nocc=4, nvir=6, naux=30):
    """Orbital energies and random real three-index integrals, symmetric in the pair."""
    rng = np.random.default_rng(seed)
    L = rng.standard_normal((naux, nocc + nvir, nocc + nvir)) / naux
    moe = np.concatenate((np.sort(rng.uniform(-1.5, -0.5, nocc)),
                          np.sort(rng.uniform(0.5, 1.5, nvir))))
    return nocc, moe, L + L.transpose(0, 2, 1)


def _physicist(Lpq, nocc):
    """vvvv, oovv, oooo with <pq|rs> = (pr|qs) = sum_L L_pr L_qs."""
    eri = np.einsum("Lpr,Lqs->pqrs", Lpq, Lpq)
    o, v = slice(0, nocc), slice(nocc, None)
    return (np.ascontiguousarray(eri[v, v, v, v]), np.ascontiguousarray(eri[o, o, v, v]),
            np.ascontiguousarray(eri[o, o, o, o]))


def _solver(nocc, moe, multi, Lpq=None):
    mp = ppRPA_Davidson(nocc, moe, Lpq=Lpq, channel="hh", nroot=2, residue_thresh=1e-10,
                        trial="identity")
    mp.mu = 0.0
    mp.multi = multi
    mp.check_parameter()
    return mp


def _lpq_solver(nocc, moe, multi, Lpq):
    """The Lpq path with the blocks kernel() would slice."""
    mp = _solver(nocc, moe, multi, Lpq)
    o, v = slice(0, nocc), slice(nocc, None)
    mp.Loo, mp.Lov, mp.Lvo, mp.Lvv = (np.ascontiguousarray(Lpq[:, a, b])
                                      for a, b in ((o, o), (o, v), (v, o), (v, v)))
    mp.oLo, mp.oLv, mp.vLo, mp.vLv = (np.ascontiguousarray(x.transpose(1, 0, 2))
                                      for x in (mp.Loo, mp.Lov, mp.Lvo, mp.Lvv))
    return mp


def _eri_solver(nocc, moe, multi, Lpq):
    mp = _solver(nocc, moe, multi)
    mp.use_eri(*_physicist(Lpq, nocc))
    return mp


def test_pack_eri():
    """pack_eri stores (pq|rs) for Q <= P at P(P+1)/2 + Q: the lower triangle of the
    whole Gram matrix, built one first orbital at a time."""
    nocc, _, Lpq = _system(1, nocc=3, nvir=4)
    nvir = Lpq.shape[1] - nocc
    vvvv, oovv, oooo = _physicist(Lpq, nocc)
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

    def whole(block, compact):
        if compact:
            n = block.shape[0]
            E = block.transpose(0, 2, 1, 3).reshape(n * n, n * n)
            p, q = np.tril_indices(n)
            E = E[p * n + q][:, p * n + q]
        else:
            no, nv = block.shape[0], block.shape[2]
            E = block.transpose(0, 2, 1, 3).reshape(no * nv, no * nv)
        return E[np.tril_indices(E.shape[0])]
    for got, block, compact in ((pv, vvvv, True), (pov, oovv, False), (po, oooo, True)):
        np.testing.assert_array_equal(got, whole(block, compact))


def test_strips_cover_the_operator():
    """Strips tile every block; their storage is M plus the full diagonal blocks."""
    nocc, nvir = 30, 70
    for multi in ("s", "t"):
        for rows in (None, 1, 4):
            plan = operator_strips(nocc, nvir, multi, rows)
            extra = 0
            for kind, n in (("oo", nocc), ("vv", nvir), ("ov", nocc)):
                spans = [(R0, R1) for k, _, _, R0, R1, _ in plan if k == kind]
                assert spans[0][0] == 0 and spans[-1][1] == pair_offset(n, multi)
                assert [s[0] for s in spans[1:]] == [s[1] for s in spans[:-1]]
                if kind != "ov":
                    extra += sum(8 * (R1 - R0) * (R1 - R0 - 1) // 2 for R0, R1 in spans)
            assert sum(s[-1] for s in plan) == pair_operator_bytes(nocc, nvir, multi) + extra


@pytest.mark.parametrize("multi", ("s", "t"))
def test_build_tasks_do_not_change_the_operator(multi, monkeypatch):
    """The threaded build fills the strips one block of rows per task: any block size
    gives the same operator, bit for bit."""
    nocc, _, Lpq = _system(7)
    nvir = Lpq.shape[1] - nocc
    gram = pack_eri(*_physicist(Lpq, nocc))
    ref = pair_operator(*gram, nocc, nvir, multi)
    for build_rows in (1, 2, 5):
        monkeypatch.setattr(pair_operator_module, "_BUILD_ROWS", build_rows)
        got = pair_operator(*gram, nocc, nvir, multi)
        for (k1, a1, b1, S1), (k2, a2, b2, S2) in zip(ref.strips, got.strips):
            assert (k1, a1, b1) == (k2, a2, b2)
            np.testing.assert_array_equal(S1, S2)


@pytest.mark.parametrize("multi", ("s", "t"))
@pytest.mark.parametrize("rows", (None, 1, 3))
def test_products_match_lpq(multi, rows):
    nocc, moe, Lpq = _system(2)
    ref = _lpq_solver(nocc, moe, multi, Lpq)
    tv = np.random.default_rng(3).standard_normal((5, ref.full_dim))
    eri = _eri_solver(nocc, moe, multi, Lpq)
    if rows is not None:
        eri._pair_op = pair_operator(*eri.eri_gram, nocc, eri.nvir, multi, rows)
    np.testing.assert_allclose(_pprpa_contraction(eri, tv), _pprpa_contraction(ref, tv),
                               rtol=1e-12, atol=1e-12)


def test_operator_follows_the_multiplicity():
    """One operator at a time, rebuilt from the Gram blocks when the multiplicity
    changes; eri_nbytes counts the Gram blocks and the operator of the multiplicity."""
    nocc, moe, Lpq = _system(4)
    eri = _eri_solver(nocc, moe, "s", Lpq)
    gram = sum(g.nbytes for g in eri.eri_gram)
    for multi in ("s", "t"):
        eri.multi = multi
        eri.check_parameter()
        plan = sum(s[-1] for s in operator_strips(nocc, eri.nvir, multi))
        assert eri.eri_nbytes() == gram + plan
        ref = _lpq_solver(nocc, moe, multi, Lpq)
        tv = np.random.default_rng(5).standard_normal((3, ref.full_dim))
        np.testing.assert_allclose(_pprpa_contraction(eri, tv), _pprpa_contraction(ref, tv),
                                   rtol=1e-12, atol=1e-12)
        assert eri._pair_op.multi == multi and eri._pair_op.nbytes == plan


@pytest.mark.parametrize("multi", ("s", "t"))
def test_davidson_roots_match_lpq(multi):
    nocc, moe, Lpq = _system(6)
    lpq = ppRPA_Davidson(nocc, moe, Lpq=Lpq.copy(), channel="hh", nroot=2,
                         residue_thresh=1e-10, trial="identity")
    eri = _eri_solver(nocc, moe, multi, Lpq)
    eri.mu = None
    lpq.kernel(multi)
    eri.kernel(multi)
    exci = (lambda mp: mp.exci_s) if multi == "s" else (lambda mp: mp.exci_t)
    np.testing.assert_allclose(exci(eri), exci(lpq), rtol=0, atol=1e-9)
