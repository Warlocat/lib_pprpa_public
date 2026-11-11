"""GppRPA_Davidson roots against the GppRPA_direct diagonalisation (GHF, CPU).

The complex case rotates every MO by a phase, L_pq -> exp(-i t_p) L_pq exp(i t_q),
which leaves the spectrum unchanged and makes Lpq genuinely complex.
"""

import numpy as np
import pytest
from pyscf import gto, scf

from lib_pprpa.gpprpa_davidson import GppRPA_Davidson
from lib_pprpa.gpprpa_direct import GppRPA_direct
from lib_pprpa.pyscf_util import get_pyscf_input_mol_g

NROOT = 5


@pytest.fixture(scope="module")
def ghf_input():
    mol = gto.M(
        atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
        basis="6-31g", verbose=0)
    mf = scf.GHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    return get_pyscf_input_mol_g(mf)


def _lpq(ghf_input, dtype):
    nocc, mo_energy, Lpq = ghf_input
    if dtype == np.complex128:
        phase = np.exp(1j * np.random.default_rng(7).uniform(0, 2 * np.pi, len(mo_energy)))
        Lpq = phase.conj()[None, :, None] * Lpq * phase[None, None, :]
    return nocc, mo_energy, np.asarray(Lpq, dtype=dtype)


@pytest.mark.parametrize("dtype", [np.double, np.complex128])
@pytest.mark.parametrize("channel", ["pp", "hh"])
@pytest.mark.parametrize("trial", ["identity", "subspace"])
def test_gpprpa_davidson_matches_direct(ghf_input, dtype, channel, trial):
    nocc, mo_energy, Lpq = _lpq(ghf_input, dtype)
    oo_dim = nocc * (nocc - 1) // 2

    ref = GppRPA_direct(nocc, mo_energy, Lpq)
    ref.kernel()
    if channel == "pp":
        e_ref = np.sort(ref.exci[oo_dim:])[:NROOT]
    else:
        e_ref = np.sort(ref.exci[:oo_dim])[::-1][:NROOT]

    mp = GppRPA_Davidson(nocc, mo_energy, Lpq, channel=channel, nroot=NROOT, trial=trial)
    mp.kernel()
    assert mp.xy_t.dtype == dtype
    assert mp.xy_t.shape == (NROOT, oo_dim + mp.vv_dim)
    e = np.sort(mp.exci_t)
    if channel == "hh":
        e = e[::-1]
    assert np.abs(e - e_ref).max() < 1e-9
