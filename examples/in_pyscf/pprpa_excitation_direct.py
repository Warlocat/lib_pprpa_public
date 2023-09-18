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
mol.charge = 2 # start from the N-2 electron system
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

# get density-fitting matrix in MO space
nmo = len(mf.mo_energy)
nocc = mf.mol.nelectron // 2
nvir = nmo - nocc
naux = mf.with_df.get_naoaux()
mo = numpy.asarray(mf.mo_coeff, order='F')
ijslice = (0, nmo, 0, nmo)
Lpq = None
Lpq = _ao2mo.nr_e2(mf.with_df._cderi, mo, ijslice, aosym='s2', out=Lpq)
Lpq = Lpq.reshape(naux, nmo, nmo)

# 1. simple ppRPA calculation
pprpa = ppRPA_direct(nocc, mf.mo_energy, Lpq)
pprpa.kernel("s")
pprpa.kernel("t")
pprpa.analyze()

# 2. ppTDA calculation
pprpa = ppRPA_direct(nocc, mf.mo_energy, Lpq, TDA=True)
pprpa.kernel("s")
pprpa.kernel("t")
pprpa.analyze()
