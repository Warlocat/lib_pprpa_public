"""
For ppRPA gradients, a fake pyscf method need to be created before passing to
berny_solver.
"""

import numpy as np
from pyscf import gto, scf, dft
from pyscf.geomopt import berny_solver, as_pyscf_method
from lib_pprpa.pyscf_util import get_pyscf_input_mol
from lib_pprpa.pprpa_davidson import ppRPA_Davidson
from lib_pprpa.grad import pprpa as ppRPA_grad


def pprpaobj(mf, nfrozen_occ, vir_cut):
    mo_ene = mf.mo_energy
    mol = mf.mol
    import numpy

    nvircut = numpy.sum(mo_ene > vir_cut)
    nvir_act = mo_ene.shape[0] - nvircut - mol.nelectron // 2
    nocc = mol.nelectron // 2 - nfrozen_occ
    nocc, mo_energy, Lpq = get_pyscf_input_mol(mf, nocc_act=nocc, nvir_act=nvir_act)
    # pprpa = GppRPA_direct(nocc, mo_energy, Lpq, nelec="n-2", hh_state=0, pp_state=5)
    pprpa = ppRPA_Davidson(nocc, mo_energy, Lpq, channel='pp', nroot=3, residue_thresh=1e-10, trial='subspace')
    pprpa.mu = 0.0
    return pprpa


mol = gto.Mole()
mol.verbose = 0
mol.atom = [
    ['O', (0.0, 0.0, -4.09461424e-02)],
    ['H', (0.0, 1.37496115e00, 9.95189266e-01)],
    ['H', (0.0, -1.37496115e00, 9.95189266e-01)],
]
mol.basis = '6-311++g**'
mol.charge = 2
mol.unit = 'Bohr'
mol.cart = True  # Use Cartesian coordinates to compare with the PCCP paper
mol.build()


def f(mol):
    istate = 0
    mult = 's'

    # mf = scf.RHF(mol).density_fit()

    mf = dft.RKS(mol, xc='LDA,VWN').density_fit()
    # One has to use high grid level for benchmarking since PySCF does not include the fxc grid response
    mf.grids.level = 9

    e_hf = mf.kernel()
    nfrozen_occ = 0
    mp = pprpaobj(mf, nfrozen_occ, 1e5)
    mp.kernel(mult)
    e_pprpa = mp.exci_s[istate] if mult == 's' else mp.exci_t[istate]
    e = e_hf + e_pprpa
    mpg = mp.Gradients(mf, mult, istate)
    mpg.kernel()
    return e, mpg.de


#
# Function as_pyscf_method is a wrapper that convert the "energy-gradients"
# function to berny_solver.  The "energy-gradients" function takes the Mole
# object as geometry input, and returns the energy and gradients of that
# geometry.
#
fake_method = as_pyscf_method(mol, f)
conv_params = {
    'gradientrms': 1e-6,  # Eh/Bohr
}
new_mol = berny_solver.optimize(fake_method, **conv_params)


print('Old geometry (Bohr)')
print(mol.atom_coords())

print('New geometry (Bohr)')
print(new_mol.atom_coords())

oh1 = new_mol.atom_coords()[1] - new_mol.atom_coords()[0]
oh2 = new_mol.atom_coords()[2] - new_mol.atom_coords()[0]
print('OH bond lengths (Angstrom)')
print(np.linalg.norm(oh1) / 1.8897259886)
print(np.linalg.norm(oh2) / 1.8897259886)
print('Bond angle (degree)')
print(np.arccos(np.dot(oh1, oh2) / np.linalg.norm(oh1) / np.linalg.norm(oh2)) * 180 / np.pi)
