r"""
    This is an example to calculate the molecular geometrical gradient of the ppRPA energy.
"""
import pyscf
import numpy as np
from pyscf.pbc import gto, scf, dft
from lib_pprpa.pprpa_davidson import ppRPA_Davidson

mult = "t"
nfrozen_occ = 0
vir_cut = 1e5
istate = 0

def mfobj(dx):
    cell = pyscf.M(
    atom = [['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.695068664391+dx]]],
    a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000
    ''',
    unit = 'bohr',
    basis = 'gth-szv',
    pseudo = 'gth-pade',
    precision = 1e-12,
    verbose = 5
)
    
    # For Gamma-point gradient calculations, one must use FFTDF with pseudopotential.
    # mf = scf.RHF(cell)
    mf = dft.RKS(cell,xc="pbe")
    mf.exxdiv = None
    mf.conv_tol = 1e-12
    return mf
def pprpaobj(mf, nfrozen_occ, vir_cut, mo_eri=True):
    mo_ene = mf.mo_energy
    mol = mf.mol
    import numpy
    nvircut = numpy.sum(mo_ene > vir_cut)
    nvir_act = mo_ene.shape[0] - nvircut - mol.nelectron//2
    nocc = mol.nelectron//2 - nfrozen_occ
    nmo = nocc + nvir_act
    mo_energy = mo_ene[nfrozen_occ:mol.nelectron//2+nvir_act]
    mo_coeff = mf.mo_coeff[:,nfrozen_occ:mol.nelectron//2+nvir_act]
    
    pprpa = ppRPA_Davidson(nocc, mo_energy, Lpq=None, channel="pp", nroot=3, residue_thresh=1e-12)

    # One can use either the MO eri or the ao direct approach.
    # For small active spaces, MO eri should be faster.
    if mo_eri:
        eri = mf.with_df.get_mo_eri(mo_coeff, compact=False)
        eri = eri.reshape(nmo, nmo, nmo, nmo).transpose(0, 2, 1, 3)
        vvvv = np.ascontiguousarray(eri[nocc:, nocc:, nocc:, nocc:])
        oovv = np.ascontiguousarray(eri[:nocc, :nocc, nocc:, nocc:])
        oooo = np.ascontiguousarray(eri[:nocc, :nocc, :nocc, :nocc])
        pprpa.use_eri(vvvv, oovv, oooo)
    else:
        pprpa._ao_direct = True
        pprpa._scf = mf
    
    pprpa.mu = 0.0
    return pprpa




dx = 0.0001
mf1 = mfobj(dx)
mf2 = mfobj(-dx)

e1 = mf1.kernel()
# f1 = mf1.get_fock(dm=dm0_hf)
h1 = mf1.get_hcore()[0]

e2 = mf2.kernel()
# f2 = mf2.get_fock(dm=dm0_hf)
h2 = mf2.get_hcore()[0]

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
from lib_pprpa.grad import pprpa_gamma
mpg = mp.Gradients(mf,mult,istate)
mpg.kernel()

print("analytical (Total):  ", mpg.de)
print("numerical (Total):  ", e_rpa + e_hf)
print("difference:  ", mpg.de[1,2] - (e_rpa + e_hf))
print("Symmetry check: (should be zero)")
print(mpg.de[0]+mpg.de[1])