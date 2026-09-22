"""The Davidson space expansion must not index past the end of the channel slice.

Each subspace vector is assigned to the particle-particle or hole-hole channel
by the sign of its metric norm, and ``first_state`` is the index where the
requested channel begins.  When the subspace happens to hold no state of that
channel, that index equals the subspace size and the slice of eigenvectors is
empty.  The routine used to index element zero of it and die with an
IndexError many hours into a run; it must report the condition instead.
"""

import types

import numpy as np
import pytest

from lib_pprpa.pprpa_davidson import _pprpa_expand_space


def _mock(nroot=3, nocc=4, nvir=4, channel="pp"):
    oo_dim = nocc * (nocc + 1) // 2
    vv_dim = nvir * (nvir + 1) // 2
    return types.SimpleNamespace(
        nocc=nocc, nvir=nvir, nroot=nroot, channel=channel, multi="s",
        mo_energy=np.linspace(-1.0, 1.0, nocc + nvir), mu=0.0,
        oo_dim=oo_dim, exci=np.full(nroot, 0.5), max_vec=100,
        residue_thresh=1e-9, _compact_subspace=False, xy=None,
    ), oo_dim + vv_dim


def test_absent_channel_reports_instead_of_indexerror():
    pprpa, dim = _mock()
    ntri = 4
    rng = np.random.default_rng(0)
    tri_vec = rng.standard_normal((pprpa.max_vec, dim))
    mv_prod = rng.standard_normal((pprpa.max_vec, dim))
    v_tri = rng.standard_normal((ntri, ntri))
    tri_vec_sig = np.ones(pprpa.max_vec, dtype=int)

    # first_state == ntri: the requested channel is absent from the subspace
    with pytest.raises(RuntimeError, match="no state in the current subspace"):
        _pprpa_expand_space(
            pprpa=pprpa, first_state=ntri, tri_vec=tri_vec,
            tri_vec_sig=tri_vec_sig, mv_prod=mv_prod, v_tri=v_tri)


def test_partial_channel_uses_what_is_there_and_keeps_expanding():
    pprpa, dim = _mock(nroot=3)
    ntri = 4
    rng = np.random.default_rng(1)
    tri_vec = rng.standard_normal((pprpa.max_vec, dim))
    mv_prod = rng.standard_normal((pprpa.max_vec, dim))
    v_tri = rng.standard_normal((ntri, ntri))
    tri_vec_sig = np.ones(pprpa.max_vec, dtype=int)

    # only two states of the channel are present, fewer than the three roots
    conv, ntri_new = _pprpa_expand_space(
        pprpa=pprpa, first_state=ntri - 2, tri_vec=tri_vec,
        tri_vec_sig=tri_vec_sig, mv_prod=mv_prod, v_tri=v_tri)
    assert pprpa.xy.shape[0] == 2
    assert conv is False, "must not claim convergence while roots are missing"
    assert ntri_new >= ntri
