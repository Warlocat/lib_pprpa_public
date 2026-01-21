r"""
    This is an example to calculate the molecular properties using the pp-RPA method
    using the analytical pp-RPA energy derivatives. 
    The zz-component of electronic field gradient (EFG) at oxygen atom is calculated 
    in this example with both analytical and numerical methods.
    It is important to note that the density fitting is required in the mean-field
    step to ensure the numerical and analytical gradients are the same.
"""
from pyscf import gto, scf
from lib_pprpa.pprpa_davidson import ppRPA_Davidson
from lib_pprpa.pyscf_util import get_pyscf_input_mol
from pyscf.data.nist import LIGHT_SPEED

import numpy as np
import scipy
from functools import reduce

def kernel(method, dm=None):
    ints = -method.mol.intor('int1e_r', comp=3).reshape(3, method.mol.nao, method.mol.nao)
    dipole = np.einsum('xij,ji->', ints, dm)
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

mult = "s"

def mfobj(dx):
    # density fitting is required to ensure the numerical and analytical gradients are the same
    mf = scf.RHF(mol).density_fit() 
    
    def get_hcore(mol, dx):
        h1e = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        ints = -mol.intor('int1e_r', comp=3).reshape(3, mol.nao, mol.nao)
        return h1e + np.einsum('xij,x->ij', ints, dx)
        
    mf.get_hcore = lambda mol = None: get_hcore(mol=mol, dx=dx)
    mf.conv_tol = 1e-12
    return mf
def pprpaobj(mf, nfrozen_occ, vir_cut):
    mo_ene = mf.mo_energy
    import numpy
    nvircut = numpy.sum(mo_ene > vir_cut)
    nvir_act = mo_ene.shape[0] - nvircut - mol.nelectron//2
    nocc = mol.nelectron//2 - nfrozen_occ
    nocc, mo_energy, Lpq = get_pyscf_input_mol(mf, nocc_act=nocc, nvir_act=nvir_act)
    # pprpa = GppRPA_direct(nocc, mo_energy, Lpq, nelec="n-2", hh_state=0, pp_state=5)
    pprpa = ppRPA_Davidson(nocc, mo_energy, Lpq, channel="pp", nroot=5, residue_thresh=1e-13)
    pprpa.mu = 0.0
    return pprpa


mf = mfobj([0.0,0.0,0.0])
mf.kernel()
mo_ene_full = mf.mo_energy
nfrozen_occ = 1
vir_cut = 10
nfrozen_vir = np.sum(mo_ene_full > vir_cut)
nvir = mo_ene_full.shape[0] - nfrozen_vir - mol.nelectron//2
nocc = mol.nelectron//2 - nfrozen_occ
if mult == "s":
    oo_dim = nocc * (nocc + 1) // 2
else:
    oo_dim = nocc * (nocc - 1) // 2

mp = pprpaobj(mf, nfrozen_occ, vir_cut)
mp.kernel(mult)
mp.analyze()
istate = 0
from lib_pprpa.grad.grad_utils import make_rdm1_relaxed_pprpa, make_rdm1_unrelaxed
den, i_int = make_rdm1_relaxed_pprpa(mp, mf, mult=mult, istate=istate)
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
mp1.kernel(mult)
if mult == "s":
    e1 = mp1.exci_s[istate]
else:
    e1 = mp1.exci_t[istate]
mp2 = pprpaobj(mf2, nfrozen_occ, vir_cut)
mp2.kernel(mult)
if mult == "s":
    e2 = mp2.exci_s[istate]
else:
    e2 = mp2.exci_t[istate]

eefg2 = (e1 - e2)/dx/2.0

print("analytical: ", eefg_rpa.real)
print("numerical:  ", eefg2)

if mult == "s":
    den = make_rdm1_unrelaxed(mp.xy_s[istate], oo_dim, mult=mult, diag=False)
else:
    den = make_rdm1_unrelaxed(mp.xy_t[istate], oo_dim, mult=mult, diag=False)
mo_coeff = mo_coeff[:,nfrozen_occ:nfrozen_occ+nocc+nvir]
den = np.einsum('pi,ij,qj->pq', mo_coeff, den, mo_coeff.conj())
eefg_rpa = kernel(mf, dm=den)
print("unrelaxed: ", eefg_rpa.real)