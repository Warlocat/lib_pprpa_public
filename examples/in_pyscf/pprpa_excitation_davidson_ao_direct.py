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

# 1. simple ppRPA calculation use eri_ao
nmo = len(mf.mo_energy)
nocc = mf.mol.nelectron // 2
nvir = nmo - nocc

Lpq = None
pprpa = ppRPA_Davidson(nocc, mf.mo_energy, Lpq)
pprpa._ao_direct = True
pprpa._scf = mf  # assign SCF object for eri_ao_direct use
pprpa.kernel("s")
pprpa.kernel("t")
pprpa.analyze()

# 2. simple ppRPA calculation use eri_ao for with_df
if getattr(mf, 'with_df', None):
    pass
else:
    mf = mf.density_fit() # only with this line can change the get_jk behavior
    mf.with_df = df.DF(mf.mol)
    try:
        mf.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
    except:
        mf.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=False)
    mf._keys.update(['with_df'])
    
Lpq = None
pprpa = ppRPA_Davidson(nocc, mf.mo_energy, Lpq)
pprpa._ao_direct = True
pprpa._scf = mf  # assign SCF object for eri_ao_direct use
pprpa.kernel("s")
pprpa.kernel("t")
pprpa.analyze()
               
