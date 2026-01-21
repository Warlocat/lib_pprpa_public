from pyscf import gto, scf
from lib_pprpa.upprpa_direct import UppRPA_direct
from lib_pprpa.pyscf_util import get_pyscf_input_mol_u

mol = gto.Mole()
mol.verbose = 0
mol.atom = [
    ["O",  (0.00000000,  -0.00000000,  -0.00614048)],
    ["H",  (0.76443318,  -0.00000000,  0.58917024)],
    ["H",  (-0.76443318,  0.00000000,  0.58917024)],
]
mol.basis = "def2svp"
mol.charge = 1
mol.spin = 1
mol.build()

mf = scf.UHF(mol)
mf.kernel()

nocc, mo_energy, Lpq = get_pyscf_input_mol_u(mf, nocc_act=None, nvir_act=None)
pprpa = UppRPA_direct(nocc, mo_energy, Lpq)
pprpa.kernel()
pprpa.analyze()

