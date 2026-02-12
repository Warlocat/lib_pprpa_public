r"""
    This is an example to calculate the molecular geometrical gradient of the ppRPA energy.
"""
import pyscf
import numpy as np
from pyscf.pbc import dft
from lib_pprpa.grad.ase_utils import pprpaobj

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

mult = "t"
nfrozen_occ = 0
vir_cut = 1e5
istate = 0
channel = "pp"
params = {"nroot": 3, "residue_thresh": 1e-12, "mo_eri": True, "nfrozen_occ": nfrozen_occ, "vir_cut": vir_cut}

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

mp1 = pprpaobj(mf1, channel, **params)
mp1.kernel(mult)
mp1.analyze()
if mult == "s":
    e1 = mp1.exci_s[istate]
else:
    e1 = mp1.exci_t[istate]
mp2 = pprpaobj(mf2, channel, **params)
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

mp = pprpaobj(mf, channel, **params)
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