import numpy

from pyscf import df, gto, scf
from pyscf.ao2mo import _ao2mo

from lib_pprpa.pair_operator import gram_from_lpq
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
# packed=True: the Gram triangles use_eri keeps, without the four-index blocks
nocc_act, mo_energy_act, vvvv, oooo, oovv = get_pyscf_input_mol_eri_r(
    mf, nocc_act=nocc, nvir_act=nvir, packed=True)
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
Lpq = Lpq.reshape(naux, nmo, nmo)
# the packed Gram triangles straight from Lpq, without the four-index blocks
vvvv, oovv, oooo = gram_from_lpq(Lpq, nocc)
pprpa = ppRPA_Davidson(nocc, mf.mo_energy, Lpq)
pprpa.use_eri(vvvv, oovv, oooo)
pprpa.kernel("s")
pprpa.kernel("t")
pprpa.analyze()
               
