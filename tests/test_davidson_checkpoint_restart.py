import h5py
import numpy
from pyscf import gto, scf

from lib_pprpa.pprpa_davidson import get_identity_trial_vector, ppRPA_Davidson
from lib_pprpa.pyscf_util import get_pyscf_input_mol


def _build_water_rhf():
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        ["O", (0.00000000, -0.00000000, -0.00614048)],
        ["H", (0.76443318, -0.00000000, 0.58917024)],
        ["H", (-0.76443318, 0.00000000, 0.58917024)],
    ]
    mol.basis = "def2svp"
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    return mf


def test_davidson_checkpoint_restart_expands_subspace(tmp_path):
    mf = _build_water_rhf()
    nocc, mo_energy, Lpq = get_pyscf_input_mol(mf)

    nroot = 3
    checkpoint_file = tmp_path / "pprpa_davidson_restart.h5"

    pp = ppRPA_Davidson(
        nocc,
        mo_energy,
        Lpq,
        nroot=nroot,
        checkpoint_file=str(checkpoint_file),
    )
    pp.max_vec = 500
    pp.multi = "s"
    pp.check_parameter()

    tri_vec, tri_sig = get_identity_trial_vector(pp, ntri=nroot)
    pp._save_pprpa_checkpoint(
        conv=False,
        ntri=nroot,
        tri_vec=tri_vec,
        tri_vec_sig=tri_sig,
    )

    with h5py.File(checkpoint_file, "r") as f:
        g = f["singlet"]
        assert int(numpy.asarray(g["ntri"])) == nroot
        assert g["tri_vec"].shape == (nroot, pp.full_dim)
        assert g["tri_vec_sig"].shape == (nroot,)

    pp_restart = ppRPA_Davidson(
        nocc,
        mo_energy,
        Lpq,
        nroot=nroot,
        checkpoint_file=str(checkpoint_file),
    )
    pp_restart.max_vec = 500
    pp_restart.kernel("s")

    with h5py.File(checkpoint_file, "r") as f:
        ntri_final = int(numpy.asarray(f["singlet"]["ntri"]))
        assert ntri_final > nroot
