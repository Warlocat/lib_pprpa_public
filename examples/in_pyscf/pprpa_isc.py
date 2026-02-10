from pyscf import scf, gto, dft
import numpy

mol = gto.Mole()
mol.atom = '''
O 0 0 0
O 0 0 1.20752
'''
mol.basis = 'cc-pvdz'
mol.charge = 2
mol.verbose = 4
mol.build()

# set nocc and nvir for ppRPA
# TDM is in principle not well-defined for full ppRPA, similar to TDDFT.
# In practice, TDA and full RPA tend to give similar results.
nocc = mol.nelectron // 2 
nvir = mol.nao - mol.nelectron//2

mf = dft.RKS(mol, xc="b3lyp") #.sfx2c1e() # one can add scalar effects using sfx2c1e
mf.kernel()

from lib_pprpa import pyscf_util, pprpa_davidson
nocc_act, mo_energy_act, Lpq = pyscf_util.get_pyscf_input_mol_r(mf, nocc_act=nocc, nvir_act=nvir)
nroots = 5
pprpa = pprpa_davidson.ppRPA_Davidson(nocc_act, mo_energy_act, Lpq, channel="pp", nroot=nroots, trial="subspace")
pprpa.max_vec = 1000
pprpa.kernel("s")
pprpa.kernel("t")
pprpa.analyze()

from lib_pprpa.grad.grad_utils import get_isc_S_T
# available soc_variants: 
#     bp1e: Breit-Pauli one-electron SOC, default option for ISC calculations
#     socecp: spin-orbit effective core potentials
# if https://github.com/xubwa/socutils is installed, the following variants are also available:
#     mfbp: mean-field Breit-Pauli SOC = bp1e + mean-field two-electron spin-same and spin-other SOC
#     x2c1e: exact two-component one-electron SOC, similar to bp1e but more accurate for heavy elements
#     x2cmmf: exact two-component molecular mean-field SOC, similar to mfbp but more accurate for heavy elements
#     x2camf: exact two-component atomic mean-field SOC, efficient approximation to x2cmmf
# 1e approximation tends to overestimate ISC rates, especially for light elements
soc_matrix = get_isc_S_T(mf, pprpa, soc_variants="mfbp")

