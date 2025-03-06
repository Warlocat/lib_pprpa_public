'''
    This is an example using j-adapted spinor HF (JHF)
    and GppRPA_direct to calculate the excitation energies.
    GppRPA is designed to support both GHF and JHF.

    It requires the socutils package.
'''

from pyscf import gto, scf
from socutils.scf import spinor_hf
from socutils.somf import amf
from lib_pprpa.gpprpa_direct import GppRPA_direct
from lib_pprpa.pyscf_util import get_pyscf_input_mol_g

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

mf = spinor_hf.JHF(mol)
mf.with_x2c = amf.SpinorX2CAMFHelper(mol)
mf.kernel()

nocc, mo_energy, Lpq = get_pyscf_input_mol_g(mf, nocc_act=None, nvir_act=None)
pprpa = GppRPA_direct(nocc, mo_energy, Lpq)
pprpa.kernel()
pprpa.analyze()

