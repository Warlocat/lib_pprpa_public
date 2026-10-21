"""Gradient of a degenerate (E) pp-RPA state: independent of the mixture and the orbital basis."""

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import gto, scf

from lib_pprpa.grad.grad_utils import get_xy_full
from lib_pprpa.grad.pprpa import Gradients
from lib_pprpa.pprpa_davidson import ppRPA_Davidson
from lib_pprpa.pyscf_util import get_pyscf_input_mol


def _nh3(dx=0.0):
    """C3v NH3 (C3 along z, Bohr), RHF/DF; ``dx`` (4, 3) displaces the atoms."""
    ang = 2 * np.pi * np.arange(3) / 3
    h = np.stack([1.77 * np.cos(ang), 1.77 * np.sin(ang), np.full(3, -0.7)], axis=1)
    xyz = np.vstack([np.zeros(3), h]) + dx
    mol = gto.M(atom=[("N", xyz[0])] + [("H", x) for x in xyz[1:]], basis="6-31g", unit="Bohr",
                verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-13
    mf.kernel()
    assert mf.converged
    return mf


def _pprpa(mf):
    nocc, mo_energy, lpq = get_pyscf_input_mol(mf)
    mp = ppRPA_Davidson(nocc, mo_energy, lpq, channel="hh", nroot=4, residue_thresh=1e-12,
                        trial="identity")
    mp.mu = 0.0
    mp.kernel("t")
    return mp


def _force(mf, mp, xy):
    grad = Gradients(mp, mf, "t", 0)
    grad.cphf_max_cycle = 100
    grad.cphf_conv_tol = 1e-11
    return -(grad.grad_elec(xy, "t", range(mf.mol.natm)) + grad.grad_nuc())


def _rotate(mf, mp, xy, rng):
    """The same state with the orbitals of every degenerate set rotated at random."""
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
    for g in groups:
        if len(g) > 1:
            rot[np.ix_(g, g)] = np.linalg.qr(rng.standard_normal((len(g), len(g))))[0]
    assert any(len(g) > 1 for g in groups)
    nocc = mp.nocc
    occ_y, vir_x = get_xy_full(xy, mp.oo_dim, "t", nocc=nocc, nvir=mp.nvir)
    ro, rv = rot[:nocc, :nocc], rot[nocc:, nocc:]
    occ_y, vir_x = ro.T @ occ_y @ ro, rv.T @ vir_x @ rv
    xy_rot = np.concatenate([occ_y[np.tril_indices(nocc, -1)], vir_x[np.tril_indices(mp.nvir, -1)]])
    mf_rot = mf.copy()
    mf_rot.mo_coeff = mf.mo_coeff @ rot
    return mf_rot, xy_rot


@pytest.fixture(scope="module")
def e_pair():
    mf = _nh3()
    mp = _pprpa(mf)
    assert abs(mp.exci_t[0] - mp.exci_t[1]) < 1e-10      # roots 0, 1: the 3E pair
    return mf, mp, np.array(mp.xy_t[0]), np.array(mp.xy_t[1])


def _mix(a, b, theta):
    return np.cos(theta) * a + np.sin(theta) * b


def _e_part(mf, mp, a, b, theta):
    # F(theta) - F(theta + 90 deg) removes the totally symmetric part; for an E pair what is
    # left is twice the e part of F(theta)
    return 0.5 * (_force(mf, mp, _mix(a, b, theta)) - _force(mf, mp, _mix(a, b, theta + np.pi / 2)))


def test_e_part_independent_of_mixture_and_orbital_basis(e_pair):
    mf, mp, a, b = e_pair
    fe = [np.linalg.norm(_e_part(mf, mp, a, b, theta)) for theta in (0.0, 0.7)]
    np.testing.assert_allclose(fe[1], fe[0], rtol=1e-8)

    xy = _mix(a, b, 0.7)
    mf_rot, xy_rot = _rotate(mf, mp, xy, np.random.default_rng(3))
    np.testing.assert_allclose(_force(mf_rot, mp, xy_rot), _force(mf, mp, xy), atol=1e-9, rtol=0)


def test_e_part_matches_branch_splitting(e_pair):
    mf, mp, a, b = e_pair
    fe = _e_part(mf, mp, a, b, 0.0)
    u = fe / np.linalg.norm(fe)
    delta = 2e-4
    half = []
    for sign in (1, -1):
        exci = _pprpa(_nh3(sign * delta * u)).exci_t
        half.append(0.5 * abs(exci[1] - exci[0]))
    # first-order splitting of the pair along u is +-|F_e|; the +- average removes the second order
    np.testing.assert_allclose(np.mean(half) / delta, np.linalg.norm(fe), rtol=1e-6)
