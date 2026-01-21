r"""
    This is an example to calculate the molecular properties using the pp-RPA method
    using the analytical pp-RPA energy derivatives (relaxed 1e-rdm). 
    This example calculates the electronic dipole moment (without picture change correction).
    It is important to note that the density fitting is required in the mean-field
    step to ensure the numerical and analytical gradients are the same.
"""
from pyscf import gto, scf
from lib_pprpa.gpprpa_direct import GppRPA_direct
from lib_pprpa.gpprpa_davidson import GppRPA_Davidson
from lib_pprpa.pyscf_util import get_pyscf_input_mol_g
from pyscf.data.nist import LIGHT_SPEED

import numpy as np
import scipy
from functools import reduce

def kernel(method, dm=None):
    ints_nr = -method.mol.intor('int1e_r', comp=3).reshape(3, method.mol.nao, method.mol.nao)
    ints_2c = np.zeros((3, method.mol.nao_2c(), method.mol.nao_2c()))
    ints_2c[:, :method.mol.nao, :method.mol.nao] = ints_nr
    ints_2c[:, method.mol.nao:, method.mol.nao:] = ints_nr
    dipole = np.einsum('xij,ji->', ints_2c, dm)
    return dipole

mol = gto.Mole()
mol.verbose = 4
mol.atom = """
O 0.          0. -0.12390941
H 0. -1.42993701  0.98326612
H 0.  1.42993701  0.98326612
"""
mol.basis = "unc-sto-3g"
mol.charge = 0
mol.unit = 'Bohr'
mol.build()

def mfobj(dx):
    # density fitting is required to ensure the numerical and analytical gradients are the same
    mf = scf.GHF(mol).sfx2c1e().density_fit() 
    
    def get_hcore(mol, dx):
        ints_nr = -mol.intor('int1e_r', comp=3).reshape(3, mol.nao, mol.nao)
        ints_nr = np.einsum('xij,x->ij', ints_nr, dx) + mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        ints_2c = np.zeros((mol.nao_2c(), mol.nao_2c()))
        ints_2c[:mol.nao, :mol.nao] = ints_nr
        ints_2c[mol.nao:, mol.nao:] = ints_nr

        return ints_2c
    mf.get_hcore = lambda mol = None: get_hcore(mol=mol, dx=dx)
    mf.conv_tol = 1e-12
    return mf
def pprpaobj(mf, nfrozen_occ, vir_cut):
    mo_ene = mf.mo_energy
    import numpy
    nvircut = numpy.sum(mo_ene > vir_cut)
    nvir_act = mo_ene.shape[0] - nvircut - mol.nelectron
    nocc = mol.nelectron - nfrozen_occ
    nocc, mo_energy, Lpq = get_pyscf_input_mol_g(mf, nocc_act=nocc, nvir_act=nvir_act)
    # pprpa = GppRPA_direct(nocc, mo_energy, Lpq, nelec="n-2", hh_state=0, pp_state=5)
    pprpa = GppRPA_Davidson(nocc, mo_energy, Lpq, channel="pp", nroot=5, residue_thresh=1e-13)
    pprpa.mu = 0.0
    return pprpa


mf = mfobj([0.0,0.0,0.0])
mf.kernel()
mo_ene_full = mf.mo_energy
nfrozen_occ = 0
vir_cut = 1e5
nfrozen_vir = np.sum(mo_ene_full > vir_cut)
nvir = mo_ene_full.shape[0] - nfrozen_vir - mol.nelectron
nocc = mol.nelectron - nfrozen_occ
oo_dim = round(nocc * (nocc - 1) / 2)

mp = pprpaobj(mf, nfrozen_occ, vir_cut)
mp.kernel()
mp.analyze()

istate = 1
from lib_pprpa.grad import grad_utils
den, i_int = grad_utils.make_rdm1_relaxed_pprpa(mp, mf, istate=istate)
mo_coeff = mf.mo_coeff
den = np.einsum('pi,ij,qj->pq', mo_coeff, den, mo_coeff.conj())
eefg_scf = kernel(mf, dm=mf.make_rdm1())
eefg_rpa = kernel(mf, dm=den)

dx = 0.00001
mf1 = mfobj([0.0,0.0,dx])
mf2 = mfobj([0.0,0.0,-dx])
e1 = mf1.kernel()
e2 = mf2.kernel()
mp1 = pprpaobj(mf1, nfrozen_occ, vir_cut)
mp1.kernel()
e1 = mp1.exci[istate]
mp2 = pprpaobj(mf2, nfrozen_occ, vir_cut)
mp2.kernel()
e2 = mp2.exci[istate]

eefg2 = (e1 - e2)/dx/2.0

print("analytical: ", eefg_rpa.real)
print("numerical:  ", eefg2)

den = grad_utils.make_rdm1_unrelaxed(mp.xy[istate], oo_dim)
den = np.einsum('pi,i,qi->pq', mo_coeff, den, mo_coeff.conj())
eefg_rpa = kernel(mf, dm=den)
print("unrelaxed: ", eefg_rpa.real)