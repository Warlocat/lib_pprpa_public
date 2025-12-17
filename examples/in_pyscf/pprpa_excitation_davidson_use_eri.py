import numpy

from pyscf import df, gto, scf
from pyscf.ao2mo import _ao2mo

from lib_pprpa.pprpa_davidson import ppRPA_Davidson

mol = gto.Mole()
mol.verbose = 0
mol.atom = [
    ["O",  (0.00000000,  -0.00000000,  -0.00614048)],
    ["H",  (0.76443318,  -0.00000000,  0.58917024)],
    ["H",  (-0.76443318,  0.00000000,  0.58917024)],
]
mol.basis = "def2svp"
mol.build()

mf = scf.RHF(mol)
mf.kernel()

# 1. simple ppRPA calculation use eri
nmo = len(mf.mo_energy)
nocc = mf.mol.nelectron // 2
nvir = nmo - nocc
from lib_pprpa.pyscf_util import get_pyscf_input_mol_eri_r
nocc_act, mo_energy_act, vvvv, oooo, oovv = get_pyscf_input_mol_eri_r(mf, nocc_act=nocc, nvir_act=nvir)
Lpq = None
pprpa = ppRPA_Davidson(nocc, mf.mo_energy, Lpq)
pprpa.use_eri(vvvv, oovv, oooo)
pprpa.kernel("s")
pprpa.kernel("t")
pprpa.analyze()

# 2. simple ppRPA calculation use eri generated from Lpq
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
naux = mf.with_df.get_naoaux()
mo = numpy.asarray(mf.mo_coeff, order='F')
ijslice = (0, nmo, 0, nmo)
Lpq = None
Lpq = _ao2mo.nr_e2(mf.with_df._cderi, mo, ijslice, aosym='s2', out=Lpq)
Lpq = Lpq.reshape(naux, nmo*nmo)
eri_raw = numpy.matmul(Lpq.T, Lpq).reshape(nmo,nmo,nmo,nmo)
eri_raw = eri_raw.transpose(0, 2, 1, 3)
vvvv = eri_raw[nocc:, nocc:, nocc:, nocc:]
oooo = eri_raw[:nocc, :nocc, :nocc, :nocc]
oovv = eri_raw[:nocc, :nocc, nocc:, nocc:]
pprpa = ppRPA_Davidson(nocc, mf.mo_energy, Lpq)
pprpa.use_eri(vvvv, oovv, oooo)
pprpa.kernel("s")
pprpa.kernel("t")
pprpa.analyze()
               
