"""The CPHF solution checkpoint: read back only for the state it was written for, and kept
apart from an SCF record in the same chkfile."""

import numpy as np
from pyscf import lib

from lib_pprpa.grad.pprpa import load_cphf_x0, save_cphf_x0


def test_a_cphf_checkpoint_is_read_only_by_its_own_state(tmp_path):
    chk = str(tmp_path / "run.chk")
    assert load_cphf_x0(chk, "t", 0) is None
    lib.chkfile.save(chk, "scf", {"e_tot": -1.5})
    assert load_cphf_x0(chk, "t", 0) is None
    x0 = np.arange(9.0).reshape(3, 3)
    save_cphf_x0(chk, x0, "t", 0)
    assert np.array_equal(load_cphf_x0(chk, "t", 0), x0)
    assert load_cphf_x0(chk, "t", 1) is None
    assert load_cphf_x0(chk, "s", 0) is None
    assert lib.chkfile.load(chk, "scf/e_tot") == -1.5
    save_cphf_x0(chk, 2 * x0, "s", 1)
    assert np.array_equal(load_cphf_x0(chk, "s", 1), 2 * x0)
    assert load_cphf_x0(chk, "t", 0) is None
