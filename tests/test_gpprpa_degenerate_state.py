"""Generalized pp-RPA relaxed density of a degenerate state: independent of the degenerate orbitals' basis."""

import numpy as np
import pytest

pytest.importorskip("socutils")
from pyscf import gto, scf

from lib_pprpa.gpprpa_direct import GppRPA_direct
from lib_pprpa.grad.gpprpa import make_rdm1_relaxed_ghf_pprpa
from lib_pprpa.grad.grad_utils import get_xy_full
from lib_pprpa.pyscf_util import get_pyscf_input_mol


def test_relaxed_density_independent_of_degenerate_orbital_basis():
    # C3v NH3 in a real GHF: every degenerate set is alpha/beta x (a or e), up to 4-fold
    ang = 2 * np.pi * np.arange(3) / 3
    h = np.stack([1.77 * np.cos(ang), 1.77 * np.sin(ang), np.full(3, -0.7)], axis=1)
    xyz = np.vstack([np.zeros(3), h])
    mol = gto.M(atom=[("N", xyz[0])] + [("H", x) for x in xyz[1:]], basis="6-31g", unit="Bohr",
                verbose=0)
    mf = scf.GHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    nocc, mo_energy, lpq = get_pyscf_input_mol(mf)
    mp = GppRPA_direct(nocc, mo_energy, lpq, hh_state=10, pp_state=0)
    mp.kernel()

    # a two-electron removal root with a degenerate partner
    oo_dim = nocc * (nocc - 1) // 2
    xy_all, exci = np.asarray(mp.xy), np.asarray(mp.exci)
    hh = [k for k in range(len(exci)) if np.linalg.norm(xy_all[k, :oo_dim]) > 0.9]
    k = next(k for k in hh if any(j != k and abs(exci[j] - exci[k]) < 1e-6 for j in hh))
    xy = xy_all[k]

    e = mf.mo_energy
    rot = np.eye(len(e))
    groups, cur = [], [0]
    for p in range(1, len(e)):
        if abs(e[p] - e[cur[-1]]) < 1e-6:
            cur.append(p)
        else:
            groups.append(cur)
            cur = [p]
    groups.append(cur)
    rng = np.random.default_rng(7)
    for g in groups:
        if len(g) > 1:
            rot[np.ix_(g, g)] = np.linalg.qr(rng.standard_normal((len(g), len(g))))[0]
    nvir = len(e) - nocc
    occ_y, vir_x = get_xy_full(xy, oo_dim)
    occ_y = rot[:nocc, :nocc].T @ occ_y @ rot[:nocc, :nocc]
    vir_x = rot[nocc:, nocc:].T @ vir_x @ rot[nocc:, nocc:]
    xy_rot = np.concatenate([occ_y[np.tril_indices(nocc, -1)], vir_x[np.tril_indices(nvir, -1)]])
    mf_rot = mf.copy()
    mf_rot.mo_coeff = mf.mo_coeff @ rot

    mp.Lpq = None           # rebuilt from each mean field's orbitals
    ao = []
    for m, v in ((mf, xy), (mf_rot, xy_rot)):
        den, i_int = make_rdm1_relaxed_ghf_pprpa(mp, m, v, cphf_max_cycle=100, cphf_conv_tol=1e-11)
        c = m.mo_coeff
        ao.append((c @ den @ c.conj().T, c @ i_int @ c.conj().T))
    np.testing.assert_allclose(ao[1][0], ao[0][0], atol=1e-7, rtol=0)
    np.testing.assert_allclose(ao[1][1], ao[0][1], atol=1e-7, rtol=0)
