r"""
    This is an example to calculate the molecular geometrical gradient of the ppRPA energy.
"""
from pyscf import gto, scf, dft
from lib_pprpa.pprpa_davidson import ppRPA_Davidson
from lib_pprpa.pyscf_util import get_pyscf_input_mol_eri_r
from pyscf.data.nist import LIGHT_SPEED

import numpy as np

mult = "s"
nfrozen_occ = 1
vir_cut = 1e5
istate = 0

def mfobj(dx):
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [["O", (0.0, 0.0, 0.0)],
                ["H", (0.0, 1.0, 1.0+dx)],
                ["H", (0.0, -1.0, 1.0)]]
    mol.basis = "cc-pvdz"
    mol.charge = 0
    mol.unit = "Bohr"
    mol.build()
    
    # mf = scf.RHF(mol)
    mf = dft.RKS(mol, xc="b3lyp")
    mf.grids.level = 9
    mf.conv_tol = 1e-12
    return mf
def pprpaobj(mf, nfrozen_occ, vir_cut):
    mo_ene = mf.mo_energy
    mol = mf.mol
    import numpy
    nvircut = numpy.sum(mo_ene > vir_cut)
    nvir_act = mo_ene.shape[0] - nvircut - mol.nelectron//2
    nocc = mol.nelectron//2 - nfrozen_occ
    nocc, mo_energy, vvvv, oooo, oovv = get_pyscf_input_mol_eri_r(mf, nocc_act=nocc, nvir_act=nvir_act)
    pprpa = ppRPA_Davidson(nocc, mo_energy, None, channel="pp", nroot=3, residue_thresh=1e-12)
    pprpa.use_eri(vvvv,oovv,oooo)
    pprpa.mu = 0.0
    return pprpa




dx = 0.0001
mf1 = mfobj(dx)
mf2 = mfobj(-dx)

e1 = mf1.kernel()
# f1 = mf1.get_fock(dm=dm0_hf)
h1 = mf1.get_hcore()

e2 = mf2.kernel()
# f2 = mf2.get_fock(dm=dm0_hf)
h2 = mf2.get_hcore()

# dfock = (f1 - f2)/(2.0*dx)
dh = (h1 - h2)/(2.0*dx)
e_hf = (e1 - e2)/dx/2.0

mp1 = pprpaobj(mf1, nfrozen_occ, vir_cut)
mp1.kernel(mult)
mp1.analyze()
if mult == "s":
    e1 = mp1.exci_s[istate]
else:
    e1 = mp1.exci_t[istate]
mp2 = pprpaobj(mf2, nfrozen_occ, vir_cut)
mp2.kernel(mult)
mp2.analyze()
if mult == "s":
    e2 = mp2.exci_s[istate]
else:
    e2 = mp2.exci_t[istate]

e_rpa = (e1 - e2)/dx/2.0

mf = mfobj(0.0)
mf.kernel()
dm0_hf = mf.make_rdm1()
mo_ene_full = mf.mo_energy
nfrozen_vir = np.sum(mo_ene_full > vir_cut)
nvir = mo_ene_full.shape[0] - nfrozen_vir - mf.mol.nelectron//2
nocc = mf.mol.nelectron//2 - nfrozen_occ
if mult == "s":
    oo_dim = nocc * (nocc + 1) // 2
else:
    oo_dim = nocc * (nocc - 1) // 2

mp = pprpaobj(mf, nfrozen_occ, vir_cut)
mp.kernel(mult)
mp.analyze()

xy = mp.xy_s[istate] if mult == "s" else mp.xy_t[istate]
from lib_pprpa.grad import pprpa
mpg = mp.Gradients(mf,mult,istate)
mpg.kernel()

print("analytical (Total):  ", mpg.de)
print("numerical (Total):  ", e_rpa + e_hf)
print("difference:  ", mpg.de[1,2] - (e_rpa + e_hf))
print("Symmetry check: (should be zero)")
print(mpg.de[0]+mpg.de[1]+mpg.de[2])