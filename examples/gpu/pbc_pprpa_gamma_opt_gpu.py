"""GPU-accelerated Gamma-point pp-RPA geometry optimization (periodic).

Full GPU pipeline driving the existing ASE interface
(``lib_pprpa.grad.ase_utils.kernel`` -> ASE BFGS):

  GPU KRKS SCF (gpu4pyscf)
    -> AO-direct GPU pp-RPA Davidson  (lib_pprpa.pprpa_davidson_gpu)
    -> GPU Gamma-point pp-RPA nuclear gradient (lib_pprpa.grad.pprpa_gamma_gpu)
    -> ASE BFGS optimizer

This example uses only ``lib_pprpa`` + ``gpu4pyscf`` + ``ase`` (no extra files).
It optimizes one pp-RPA state (lowest singlet or triplet) of a small periodic
diamond cell.  The cell is intentionally tiny so the whole loop runs in a few
minutes on one GPU; the same script scales to large cells / defects by swapping
in a bigger geometry and a frozen-core active space (see notes at the bottom and
examples/gpu/README.md for what is required at scale).

Run:
    # in the GPU env (see README.md), with lib_pprpa + gpu4pyscf on PYTHONPATH
    python pbc_pprpa_gamma_opt_gpu.py            # default: lowest triplet, pp channel
    python pbc_pprpa_gamma_opt_gpu.py s          # lowest singlet

A rerun in the same directory resumes from the last frame of opt.traj with the
BFGS Hessian in bfgs.pkl and the SCF orbitals in scf.chk (see README.md).
"""
import os
import sys
import numpy as np
import cupy as cp
from pyscf.lib import chkfile
from pyscf.pbc import gto, dft as cdft
from pyscf.pbc.tools.pyscf_ase import pyscf_to_ase_atoms
from gpu4pyscf.pbc import dft as gdft

from ase.io import read
from ase.io.extxyz import write_extxyz
from lib_pprpa.grad.ase_utils import pprpaobj, kernel as ase_opt
from lib_pprpa.pprpa_davidson_gpu import attach_gpu_contraction
from lib_pprpa.grad import pprpa_gamma          # noqa: F401  (attaches .Gradients)
from lib_pprpa.grad import pprpa_gamma_gpu as gpugrad

# ---------------- settings ----------------
XC       = "pbe"          # functional for the KS reference (use "hf" for HF)
CHANNEL  = "pp"           # "pp" (charge +2 ref) or "hh" (charge -2 ref)
CHARGE   = +2 if CHANNEL == "pp" else -2
MULT     = sys.argv[1] if len(sys.argv) > 1 else "t"   # 's' or 't'
ISTATE   = 0
NROOT    = 2
FMAX     = 0.05          # eV/Ang convergence
MAXSTEPS = 15
KPTS     = np.zeros((1, 3))     # Gamma point
CHK, RESTART, TRAJ = "scf.chk", "bfgs.pkl", "opt.traj"
_state   = {"cphf_x0": None}    # CPHF warm start between geometry steps


def build_cell(coords=None):
    """Small FCC-diamond primitive cell (2 C). One atom is displaced off the
    symmetric site so the optimizer has a real force to relax."""
    a0 = 3.5668          # Angstrom (conventional diamond lattice constant)
    cell = gto.Cell()
    if coords is None:
        coords = [[0.0, 0.0, 0.0], [a0/4 + 0.05, a0/4, a0/4]]   # 0.05 A distortion
    cell.atom = [["C", coords[0]], ["C", coords[1]]]
    cell.a = np.array([[0, a0/2, a0/2], [a0/2, 0, a0/2], [a0/2, a0/2, 0]])
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.charge = CHARGE
    cell.ke_cutoff = 100.0
    cell.verbose = 0
    cell.build()
    return cell


def scf_guess(mf):
    """Density from the orbitals in mf.chkfile (the previous geometry), or None.

    Built from the stored mo_coeff / mo_occ directly: pyscf's from_chk expects
    a kpts record that the final gpu4pyscf dump does not keep.
    """
    if not os.path.exists(mf.chkfile):
        return None
    mo = chkfile.load(mf.chkfile, "scf/mo_coeff")
    occ = chkfile.load(mf.chkfile, "scf/mo_occ")
    return mf.make_rdm1(cp.asarray(mo), cp.asarray(occ))


def _pipeline(cell, want_grad):
    """GPU SCF + GPU AO-direct pp-RPA (+ GPU gradient).  Returns e_state, or
    (e_state, dE/dR) when want_grad."""
    cell.build()

    # 1) GPU KRKS SCF for orbitals
    kg = gdft.KRKS(cell, kpts=KPTS, xc=XC)
    kg.exxdiv = None
    kg.conv_tol = 1e-9
    kg.chkfile = CHK
    kg.kernel(scf_guess(kg))

    # 2) lightweight CPU RKS shell carrying the GPU orbitals (pprpaobj reads numpy)
    mf = cdft.RKS(cell, xc=XC)
    mf.exxdiv = None
    mf.mo_coeff = cp.asnumpy(kg.mo_coeff[0])
    mf.mo_energy = cp.asnumpy(kg.mo_energy[0])
    mf.mo_occ = cp.asnumpy(kg.mo_occ[0])
    mf.e_tot = float(kg.e_tot)
    mf.converged = True

    # 3) AO-direct pp-RPA, contraction routed to the GPU KRKS get_k
    mp = pprpaobj(mf, CHANNEL, nroot=NROOT, mo_eri=False,
                  nfrozen_occ=0, vir_cut=1e5)
    mp.residue_thresh = 1e-9
    attach_gpu_contraction(mp, kg)
    mp.kernel(MULT)

    exci = (mp.exci_s if MULT == "s" else mp.exci_t)[ISTATE]
    sign = 1.0 if CHANNEL == "pp" else -1.0     # E_state = e_scf +/- exci
    e_state = mf.e_tot + sign * exci

    if not want_grad:
        cp.get_default_memory_pool().free_all_blocks()
        return e_state

    # 4) GPU Gamma-point gradient (low-rank exchange of the pair densities built in)
    xy = (mp.xy_s if MULT == "s" else mp.xy_t)[ISTATE]
    g = gpugrad.Gradients(mp, mf, MULT, ISTATE)
    g.cphf_x0 = _state["cphf_x0"]
    de = g.grad_elec(xy, MULT, range(cell.natm)) + g.grad_nuc()
    _state["cphf_x0"] = g.cphf_x0
    cp.get_default_memory_pool().free_all_blocks()
    return e_state, de


# ASE callback signatures expected by ase_utils.kernel:
#   grad_func(cell, **kw) -> (E, dE/dR);   ene_func(cell, **kw) -> E
_step = [0]
def grad_func(cell, **kw):
    e, de = _pipeline(cell, want_grad=True)
    _step[0] += 1
    print(f"[opt step {_step[0]:2d}]  E_state = {e:.8f} Ha   "
          f"|F| = {np.linalg.norm(de):.4e}  max|F| = {np.abs(de).max():.4e} a.u.",
          flush=True)
    return e, de

def ene_func(cell, **kw):
    return _pipeline(cell, want_grad=False)


if __name__ == "__main__":
    resume = os.path.exists(TRAJ)
    cell = build_cell(read(TRAJ, index=-1).get_positions() if resume else None)
    print(f"### GPU Gamma pp-RPA opt: {cell.natm} atoms, nao={cell.nao}, "
          f"mesh={list(cell.mesh)}, channel={CHANNEL}, mult={MULT}, xc={XC}"
          f"{', resumed from ' + TRAJ if resume else ''} ###", flush=True)
    converged, cell_opt = ase_opt(cell, grad_func=grad_func, ene_func=ene_func,
                                  logfile="-", fmax=FMAX, max_steps=MAXSTEPS,
                                  restart=RESTART, trajectory=TRAJ, append_trajectory=resume)
    opt_ase = pyscf_to_ase_atoms(cell_opt)
    info={"converged":converged, "charge":CHARGE, "mult":MULT, "channel":CHANNEL, "istate":ISTATE, "xc":XC, "nao":cell.nao}
    opt_ase.info.update(info)
    write_extxyz("optimized.xyz", opt_ase)

# ---------------------------------------------------------------------------
# Scaling to large cells / defects (e.g. NV center in diamond):
#   * read the geometry from a POSCAR/xyz instead of build_cell()
#   * use a frozen-core active space: pprpaobj(mf, CHANNEL, AS_size=N, mo_eri=False)
#   * for the energy solve, the MO-eri path (GPU FFT ao2mo + batched use_eri
#     contraction) is much faster than AO-direct at large active spaces
#   * hybrid functionals at large nao / fine mesh need the aft_jk blksize patch — see README.md
# ---------------------------------------------------------------------------
