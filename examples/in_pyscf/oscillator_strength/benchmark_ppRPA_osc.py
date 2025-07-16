from pyscf import gto, scf

from lib_pprpa.pprpa_davidson import ppRPA_Davidson
from lib_pprpa.pyscf_util import get_pyscf_input_mol

molecule = [
    ["H", (0,0,0)],
    ["H", (.741,0,0)],
]
basis = "631++g**"

mol = gto.Mole()
mol.verbose = 0
mol.atom = molecule
mol.basis = basis
mol.charge = +2  # start from the N-2 electron system
mol.build()

mf = scf.RHF(mol)
mf.kernel()

nocc, mo_energy, Lpq, mo_dip = get_pyscf_input_mol(mf, with_dip=True)
pprpa = ppRPA_Davidson(
    nocc, mo_energy, Lpq, mo_dip=mo_dip)
pprpa.kernel("s")
pprpa.kernel("t")
pprpa.analyze()

mol2 = gto.Mole()
mol2.verbose = 0
mol2.atom = molecule
mol2.basis = basis
mol2.build()

mf2 = scf.RHF(mol2)
mf2.kernel()

mytd = mf2.TDA()
mytd.singlet = True
mytd.nstates = 8
mytd.kernel()
mytd.analyze(verbose=3)