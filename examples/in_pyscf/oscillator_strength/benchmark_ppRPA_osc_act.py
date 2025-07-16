from pyscf import gto, scf

from lib_pprpa.pprpa_davidson import ppRPA_Davidson
from lib_pprpa.pyscf_util import get_pyscf_input_mol
molecule = [
    ["C", (0.0000, 0.0000, 0.7680)],
    ["C", (0.0000, 0.0000, -0.7680)],
    ["H", (-1.0192, 0.0000, 1.1573)],
    ["H", (0.5096, 0.8826, 1.1573)],
    ["H", (0.5096, -0.8826, 1.1573)],
    ["H", (1.0192, 0.0000, -1.1573)],
    ["H", (-0.5096, -0.8826, -1.1573)],
    ["H", (-0.5096, 0.8826, -1.1573)],
]
# molecule = [
#     ["O", (0.00000000, -0.00000000, -0.00614048)],
#     ["H", (0.76443318, -0.00000000, 0.58917024)],
#     ["H", (-0.76443318, 0.00000000, 0.58917024)],
# ]
basis = "631++g**"

mol = gto.Mole()
mol.verbose = 0
mol.atom = molecule
mol.basis = basis
mol.charge = +2  # start from the N-2 electron system
mol.build()

mf = scf.RHF(mol)
mf.kernel()

nocc, mo_energy, Lpq, mo_dip = get_pyscf_input_mol(mf, with_dip=True, nocc_act=20, nvir_act=20)
pprpa = ppRPA_Davidson(
    nocc, mo_energy, Lpq, mo_dip=mo_dip, nroot=5)
pprpa.kernel("s")
pprpa.kernel("t")
pprpa.analyze()
# quit()
nocc, mo_energy, Lpq, mo_dip = get_pyscf_input_mol(mf, with_dip=True)
pprpa = ppRPA_Davidson(
    nocc, mo_energy, Lpq, mo_dip=mo_dip, nroot=5)
pprpa.kernel("s")
pprpa.kernel("t")
pprpa.analyze()
# quit()

mol2 = gto.Mole()
mol2.verbose = 0
mol2.atom = molecule
mol2.basis = basis
mol2.build()

mf2 = scf.UHF(mol2)
mf2.kernel()

mytd = mf2.TDA()
mytd.singlet = False
mytd.nstates = 10
mytd.kernel()
mytd.analyze(verbose=4)