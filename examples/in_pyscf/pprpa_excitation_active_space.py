import numpy

from pyscf import df, gto, scf
from pyscf.ao2mo import _ao2mo

from lib_pprpa.pprpa_direct import ppRPA_direct

mol = gto.Mole()
mol.verbose = 0
mol.atom = [
    ["O",  (0.00000000,  -0.00000000,  -0.00614048)],
    ["H",  (0.76443318,  -0.00000000,  0.58917024)],
    ["H",  (-0.76443318,  0.00000000,  0.58917024)],
]
mol.basis = "def2svp"
mol.charge = 2  # start from the N-2 electron system
mol.build()

mf = scf.RHF(mol)
mf.kernel()

# get density-fitting matrix in AO
if getattr(mf, 'with_df', None):
    pass
else:
    mf.with_df = df.DF(mf.mol)
    try:
        mf.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
    except:
        mf.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=False)
    mf._keys.update(['with_df'])

nmo = len(mf.mo_energy)
nocc = mf.mol.nelectron // 2
nvir = nmo - nocc
naux = mf.with_df.get_naoaux()
nocc_act = 0  # number of active occupied orbitals
nvir_act = 10  # number of active virtual orbitals
nmo_act = nocc_act + nvir_act  # number of active orbitals
nocc_fro = nocc - nocc_act  # number of frozen occupied orbitals
nvir_fro = nvir - nvir_act  # number of frozen virtual orbitals

# 1. get density-fitting matrix in full MO space and assign active space
mo = numpy.asarray(mf.mo_coeff, order='F')
ijslice = (0, nmo, 0, nmo)
Lpq = None
Lpq = _ao2mo.nr_e2(mf.with_df._cderi, mo, ijslice, aosym='s2', out=Lpq)
Lpq = Lpq.reshape(naux, nmo, nmo)

pprpa = ppRPA_direct(nocc, mf.mo_energy, Lpq, nocc_act=nocc_act, nvir_act=nvir_act)
pprpa.kernel("s")
pprpa.kernel("t")
pprpa.analyze()

# 2. get density-fitting matrix in active MO space
mo = numpy.asarray(mf.mo_coeff, order='F')
ijslice = (nocc_fro, nmo-nvir_fro, nocc_fro, nmo-nvir_fro)
Lpq = None
Lpq = _ao2mo.nr_e2(mf.with_df._cderi, mo, ijslice, aosym='s2', out=Lpq)
Lpq = Lpq.reshape(naux, nmo_act, nmo_act)

pprpa = ppRPA_direct(nocc_act, mf.mo_energy[nocc_fro:(nmo-nvir_fro)], Lpq)
pprpa.kernel("s")
pprpa.kernel("t")
pprpa.analyze(nocc_fro=nocc_fro)  # manually assign the index of the first active occupied orbital
