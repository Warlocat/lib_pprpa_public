import numpy as np
import pytest

from pyscf import dft, gto

from lib_pprpa.grad.pprpa import make_rdm1_relaxed_rhf_pprpa
from lib_pprpa.pprpa_davidson import ppRPA_Davidson
from lib_pprpa.pyscf_util import get_pyscf_input_mol


@pytest.mark.parametrize("mult", ["s", "t"])
def test_pair_get_k_matches_full_lpq_with_frozen_orbitals(mult):
    mol = gto.M(
        atom="O 0 0 0; H 0 -.757 .587; H 0 .757 .587",
        basis="6-31g", verbose=0)
    mf = dft.RKS(mol, xc="pbe").density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()

    nocc, mo_energy, lpq = get_pyscf_input_mol(
        mf, nocc_act=2, nvir_act=2)
    solver = ppRPA_Davidson(
        nocc, mo_energy, lpq, channel="pp", nroot=1,
        residue_thresh=1e-9, max_iter=100, max_vec=100,
        trial="identity")
    solver.mu = 0.0
    solver.kernel(mult)

    full_lpq = make_rdm1_relaxed_rhf_pprpa(
        solver, mf, mult=mult, cphf_conv_tol=1e-9,
        cphf_max_cycle=50)

    def pair_get_k(dms, hermi):
        return mf.get_k(dm=dms, hermi=hermi)

    direct_pair = make_rdm1_relaxed_rhf_pprpa(
        solver, mf, mult=mult, cphf_conv_tol=1e-9,
        cphf_max_cycle=50, pair_get_k=pair_get_k)
    np.testing.assert_allclose(direct_pair[0], full_lpq[0], atol=3e-12, rtol=0)
    np.testing.assert_allclose(direct_pair[1], full_lpq[1], atol=3e-12, rtol=0)
