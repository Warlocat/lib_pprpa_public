
from pyscf import gto, scf
from lib_pprpa.gpprpa_davidson import GppRPA_Davidson
from lib_pprpa.pyscf_util import get_pyscf_input_mol_g

import numpy as np
import scipy
from functools import reduce


mol = gto.Mole()
mol.verbose = 5
mol.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.0, -0.7571, 0.5861)], [1, (0.0, 0.7571, 0.5861)]]
mol.basis = 'def2-svp'
mol.build()

dip_int = -mol.intor('int1e_r', comp=3)

mf = scf.RHF(mol)
mf.kernel()
print(mf.mo_coeff.shape)
mo_coeff = mf.mo_coeff[:,mol.nelectron//2:]

nao = mol.nao_nr()
nmo = mo_coeff.shape[1]

mo_coeff_ghf = np.zeros((mo_coeff.shape[0]*2, mo_coeff.shape[1]*2))
for i in range(nmo):
    mo_coeff_ghf[:nao, 2*i] = mo_coeff[:, i]
    mo_coeff_ghf[nao:, 2*i+1] = mo_coeff[:, i]

dip_int_ghf = np.zeros((3, nao*2, nao*2))
for i in range(3):
    dip_int_ghf[i][nao:,nao:] = dip_int[i]
    dip_int_ghf[i][:nao,:nao] = dip_int[i]
dip_int_ghf = np.einsum('xij,ip,jq->xpq', dip_int_ghf, mo_coeff_ghf, mo_coeff_ghf)

nocc_act, mo_energy_act, Lpq = get_pyscf_input_mol_g(mf, nocc_act=0)
nroot = 15
mp = GppRPA_Davidson(nocc_act, mo_energy_act, Lpq, channel="pp", nroot=nroot, residue_thresh=1e-10)
mp.kernel()
mp.analyze()

from lib_pprpa.grad.grad_utils import make_tdm1
for i in range(1,nroot):
    tdm, diag = make_tdm1(mp.xy[0], mp.xy[i], 0)
    # np.fill_diagonal(tdm, tdm.diagonal() + diag)
    dipole = np.einsum('xij,ji->x', dip_int_ghf, tdm)
    # print("Transition dipole moment between state 0 and state",i)
    # print(dipole)
    print("Oscillator strength between state 0 and state",i+1)
    print(mp.exci[i],np.linalg.norm(dipole)**2)


