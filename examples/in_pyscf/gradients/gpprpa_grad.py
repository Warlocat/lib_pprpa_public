r"""
    This is an example to calculate the molecular geometrical gradient of the ppRPA energy.
"""
from pyscf import gto, scf, dft
from lib_pprpa.gpprpa_davidson import GppRPA_Davidson
from lib_pprpa.gpprpa_direct import GppRPA_direct
from lib_pprpa.pyscf_util import get_pyscf_input_mol
from socutils.somf import eamf

import numpy as np

nfrozen_occ = 18
vir_cut = 1e50
istate = 0

def mfobj(dx):
    mol = gto.Mole()
    mol.verbose = 4
    mol.atom = [["Se", (0.0, 0.0, 0.0)],
                ["H", (0.0, 1.0, 2.5+dx)],
                ["H", (0.0, 0.0, -3.5)]]
    mol.basis = {"H":"cc-pvdz-dk", "Se":"cc-pvdz-dk"}
    # mol.symmetry = True
    mol.charge = 2
    mol.unit = "Bohr"
    mol.build()
    
    # density fitting is required to ensure the numerical and analytical gradients are the same
    mf = scf.GHF(mol).x2c1e().density_fit()
    # mf.with_x2c = eamf.SpinOrbitalEAMFX2CHelper(mol, eamf = "x2camf", with_gaunt=True, with_breit=True)
    mf.conv_tol = 1e-12
    return mf
def pprpaobj(mf, nfrozen_occ, vir_cut):
    assert nfrozen_occ < mf.mol.nelectron
    mo_ene = mf.mo_energy
    mol = mf.mol
    import numpy
    nvircut = numpy.sum(mo_ene > vir_cut)
    nvir_act = mo_ene.shape[0] - nvircut - mol.nelectron
    nocc = mol.nelectron - nfrozen_occ
    nocc, mo_energy, Lpq = get_pyscf_input_mol(mf, nocc_act=nocc, nvir_act=nvir_act)
    # pprpa = GppRPA_direct(nocc, mo_energy, Lpq, nelec="n-2", hh_state=0, pp_state=5)
    pprpa = GppRPA_Davidson(nocc, mo_energy, Lpq, channel="hh", nroot=5, residue_thresh=1e-10, trial="subspace")
    pprpa._use_Lov = True
    pprpa.mu = 0.0
    return pprpa




dx = 0.02
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
e_hf_elec = (mf1.energy_elec()[0] - mf2.energy_elec()[0])/dx/2.0

mp1 = pprpaobj(mf1, nfrozen_occ, vir_cut)
mp1.kernel()
e1 += mp1.exci[istate] if mp1.channel == "pp" else -mp1.exci[istate]
mp2 = pprpaobj(mf2, nfrozen_occ, vir_cut)
mp2.kernel()
mp2.analyze()
e2 += mp2.exci[istate] if mp2.channel == "pp" else -mp2.exci[istate]

e_rpa = (e1 - e2)/dx/2.0

mf = mfobj(0.0)
mf.kernel()
dm0_hf = mf.make_rdm1()
mo_ene_full = mf.mo_energy
nfrozen_vir = np.sum(mo_ene_full > vir_cut)
nvir = mo_ene_full.shape[0] - nfrozen_vir - mf.mol.nelectron
nocc = mf.mol.nelectron - nfrozen_occ
oo_dim = nocc * (nocc + 1) // 2

mp = pprpaobj(mf, nfrozen_occ, vir_cut)
mp.kernel()
mp.analyze()
xy = mp.xy[istate]
from lib_pprpa.grad import gpprpa
mpg = mp.Gradients(mf,istate)
mpg.kernel()

print("analytical (Total):  ", mpg.de)
print("numerical (Total):  ", e_rpa)
print(mpg.de[0]+mpg.de[1]+mpg.de[2]) # Should be zero
print(e1,e2)
# print("analytical (HF elec):  ", gpprpa.grad_elec_mf(mf))
# print("analytical (HF elec):  ", mf.Gradients().grad_elec())
# print("numerical (HF elec):  ", e_hf_elec)