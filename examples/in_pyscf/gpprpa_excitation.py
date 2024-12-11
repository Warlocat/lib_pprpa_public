from pyscf import gto, scf
from lib_pprpa.gpprpa_direct import GppRPA_direct
from lib_pprpa.gpprpa_davidson import GppRPA_Davidson
from lib_pprpa.pyscf_util import get_pyscf_input_mol

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

# Non-relativistic GHF (real-valued)
mf = scf.GHF(mol)
mf.kernel()
nocc, mo_energy, Lpq = get_pyscf_input_mol(mf, nocc_act=None, nvir_act=None)
pprpa = GppRPA_direct(nocc, mo_energy, Lpq)
pprpa.kernel()
pprpa.analyze()

# Relativistic X2C-1e GHF (complex-valued)
mf = scf.GHF(mol).x2c1e()
mf.kernel()
nocc, mo_energy, Lpq = get_pyscf_input_mol(mf, nocc_act=None, nvir_act=None)
pprpa = GppRPA_direct(nocc, mo_energy, Lpq)
pprpa.kernel()
pprpa.analyze()

# Davidson
pprpa = GppRPA_Davidson(nocc, mo_energy, Lpq, channel="pp", nroot=5, trial="subspace")
pprpa.kernel()
pprpa.analyze()

