"""An interrupted ASE optimisation resumes from its restart file and trajectory."""

import numpy as np
from ase.io import read
from pyscf.pbc import gto

from lib_pprpa.grad import ase_utils


def _cell():
    cell = gto.Cell()
    cell.atom = [["H", [0.0, 0.0, 0.0]], ["H", [1.5, 0.0, 0.0]]]
    cell.a = np.eye(3) * 8.0
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.verbose = 0
    cell.build()
    return cell


def _anharmonic(target):
    # E = 1/2 |d|^2 + 1/4 sum d^4 (Ha, Bohr), d = R - R0: the minimum is at R0 and
    # BFGS needs several steps to find it
    def grad_func(cell, **kw):
        d = cell.atom_coords() - target
        return 0.5 * float((d * d).sum()) + 0.25 * float((d**4).sum()), d + d**3

    def ene_func(cell, **kw):
        return grad_func(cell)[0]
    return grad_func, ene_func


def test_bfgs_resumes_from_restart_and_trajectory(tmp_path):
    cell = _cell()
    target = cell.atom_coords() + np.array([[0.8, -0.5, 0.3], [-0.6, 0.7, -0.4]])
    grad_func, ene_func = _anharmonic(target)
    restart, traj = str(tmp_path / "bfgs.pkl"), str(tmp_path / "opt.traj")

    # interrupted after two steps
    converged, _ = ase_utils.kernel(cell.copy(), grad_func, ene_func, logfile=None,
                                    fmax=1e-4, max_steps=2, restart=restart, trajectory=traj)
    assert not converged
    frames = read(traj, index=":")
    assert len(frames) == 3

    # resume from the last frame with the Hessian from the restart file
    last = frames[-1]
    cell2 = cell.set_geom_(last.get_positions(), unit="Ang", a=last.cell, inplace=False)
    converged, cell_opt = ase_utils.kernel(cell2, grad_func, ene_func, logfile=None,
                                           fmax=1e-4, max_steps=100, restart=restart,
                                           trajectory=traj, append_trajectory=True)
    assert converged
    assert len(read(traj, index=":")) > 3
    np.testing.assert_allclose(cell_opt.atom_coords(), target, atol=1e-3)
