"""Validate an equilibrium-PCM molecular ppRPA gradient on water.

This example uses density fitting throughout and compares the analytical
gradient with a central finite difference of the complete solvated-reference
state energy. The ppRPA eigenvalue equation itself is unchanged.
"""

import numpy as np
from pyscf import dft, gto

from lib_pprpa.grad.pprpa_pcm import Gradients
from lib_pprpa.pprpa_davidson import ppRPA_Davidson
from lib_pprpa.pyscf_util import get_pyscf_input_mol
from lib_pprpa.solvent.pcm import attach_pcm, state_energy


MULT = "s"
STATE = 0
DISPLACEMENT = 1.0e-3
PCM_METHOD = "IEF-PCM"


def make_mf(displacement=0.0):
    mol = gto.M(
        atom=[
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, 1.0, 1.0 + displacement)),
            ("H", (0.0, -1.0, 1.0)),
        ],
        basis="sto-3g",
        unit="Bohr",
        verbose=0,
    )
    mf = dft.RKS(mol, xc="pbe").density_fit()
    mf = attach_pcm(mf, eps=78.3553, method=PCM_METHOD)
    mf.conv_tol = 1.0e-12
    mf.grids.level = 5
    mf.kernel()
    assert mf.converged
    return mf


def make_pprpa(mf):
    nocc, mo_energy, lpq = get_pyscf_input_mol(mf)
    pprpa = ppRPA_Davidson(
        nocc,
        mo_energy,
        lpq,
        channel="pp",
        nroot=1,
        trial="identity",
        residue_thresh=1.0e-10,
    )
    pprpa.mu = 0.0
    pprpa.kernel(MULT)
    return pprpa


def calculate_state_energy(displacement):
    mf = make_mf(displacement)
    pprpa = make_pprpa(mf)
    return state_energy(mf, pprpa, mult=MULT, state=STATE)


if __name__ == "__main__":
    mf = make_mf()
    pprpa = make_pprpa(mf)
    analytical = Gradients(pprpa, mf, mult=MULT, state=STATE).kernel()

    e_plus = calculate_state_energy(DISPLACEMENT)
    e_minus = calculate_state_energy(-DISPLACEMENT)
    numerical = (e_plus - e_minus) / (2.0 * DISPLACEMENT)

    print("PCM method:             ", PCM_METHOD)
    print("Analytical dE/dz(H1):", analytical[1, 2])
    print("Numerical  dE/dz(H1):", numerical)
    print("Difference:           ", numerical - analytical[1, 2])
    print("Translational sum:    ", np.sum(analytical, axis=0))
