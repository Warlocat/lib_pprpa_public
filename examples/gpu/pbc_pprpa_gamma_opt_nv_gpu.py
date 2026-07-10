"""GPU Gamma-point pp-RPA geometry optimization of a defect (e.g. NV center),
using the fast MO-eri energy path — suitable for large cells with an active space.

Pipeline (all on GPU except the small relaxed-density block algebra):
  GPU KRKS SCF (gpu4pyscf)
    -> GPU FFT ao2mo  (lib_pprpa.gpu_ao2mo) : active-space vvvv/oovv/oooo
    -> GPU batched use_eri Davidson (lib_pprpa.pprpa_eri_gpu)
    -> GPU Gamma-point pp-RPA gradient (lib_pprpa.grad.pprpa_gamma_gpu)
    -> ASE BFGS (lib_pprpa.grad.ase_utils.kernel)

This is the path used for the 63-atom NV-center production runs. The MO-eri energy
solve is orders of magnitude faster than AO-direct at large active spaces
(AS=100: ~0.3 s vs ~90 min per Davidson solve).

Usage:
    python pbc_pprpa_gamma_opt_nv_gpu.py <geometry> <ke_cutoff> [mult] [AS] [istate] [fmax] [maxsteps]

      geometry  : POSCAR/.vasp or .xyz file (read via ASE); lattice is taken from it
      ke_cutoff : plane-wave kinetic-energy cutoff in Hartree  (TUNE THIS)
      mult      : 's' or 't'            (default 't')
      AS        : active-space size     (default 100; occ/vir each capped at this)
      istate    : which root to optimize, 0-based (default 0 = lowest)
      fmax      : force convergence eV/A (default 0.05)
      maxsteps  : max BFGS steps        (default 80)

Example:
    python pbc_pprpa_gamma_opt_nv_gpu.py 3A2-geo.POSCAR.vasp 90 t 100 0

REQUIREMENTS: see examples/gpu/README.md — in particular, at this scale the two
gpu4pyscf blksize patches (fft_jk.py, aft_jk.py) ARE required to avoid OOM.
"""
import sys, os
import numpy as np
import cupy as cp
from ase.io import read
from ase.io.extxyz import write_extxyz
from pyscf.data.nist import BOHR
from pyscf.pbc import gto, dft as cdft
from pyscf.pbc.tools.pyscf_ase import pyscf_to_ase_atoms
from gpu4pyscf.pbc import dft as gdft
from gpu4pyscf.pbc.df.fft import FFTDF
from gpu4pyscf.pbc.df import fft_jk

from lib_pprpa.pprpa_davidson import ppRPA_Davidson
from lib_pprpa.gpu_ao2mo import gpu_ao2mo_blocks
from lib_pprpa.pprpa_eri_gpu import attach_gpu_eri_contraction
from lib_pprpa.grad import pprpa_gamma          # noqa: F401 (attaches .Gradients)
from lib_pprpa.grad import pprpa_gamma_gpu as gpugrad
from lib_pprpa.grad import ase_utils

# ---------------- arguments ----------------
GEOM     = sys.argv[1]
KE       = float(sys.argv[2])
MULT     = sys.argv[3] if len(sys.argv) > 3 else "t"
AS       = int(sys.argv[4]) if len(sys.argv) > 4 else 100
ISTATE   = int(sys.argv[5]) if len(sys.argv) > 5 else 0
FMAX     = float(sys.argv[6]) if len(sys.argv) > 6 else 0.05
MAXSTEPS = int(sys.argv[7]) if len(sys.argv) > 7 else 80

# ---------------- defect / method settings (NV defaults) ----------------
XC      = "pbe"
CHANNEL = "hh"          # NV (-3) ground state from the dianion-like reference
CHARGE  = -3
BASIS   = "gth-dzvp"
PSEUDO  = "gth-pbe"
NROOT   = max(5, ISTATE + 1)
GAMMA   = np.zeros((1, 3))      # pp-RPA is Gamma-point only

# read geometry + lattice from the input file (POSCAR or xyz)
_at = read(GEOM)
_SYM = _at.get_chemical_symbols()
_A_BOHR = np.asarray(_at.cell) / BOHR          # lattice fixed during the opt
_NATM = len(_SYM)
_step = [0]


def build_cell(coords_bohr):
    cell = gto.Cell()
    cell.atom = [(_SYM[i], coords_bohr[i]) for i in range(_NATM)]
    cell.a = _A_BOHR
    cell.unit = "Bohr"
    cell.basis = BASIS
    cell.pseudo = PSEUDO
    cell.charge = CHARGE
    cell.spin = 0
    cell.ke_cutoff = KE
    cell.verbose = 0
    cell.build()
    return cell


def _patch_gpu_getk(mf, cell):
    fdf = FFTDF(cell, GAMMA)
    def gk(dm=None, hermi=1, **kw):
        dmg = cp.asarray(dm); single = dmg.ndim == 2
        if single:
            dmg = dmg[None]
        K = cp.asnumpy(fft_jk.get_k(fdf, dmg, hermi=0, kpt=np.zeros(3), exxdiv=None))
        return K[0] if single else K
    mf.get_k = gk


def _pipeline(cell, want_grad):
    cell.build()
    nocc_all = cell.nelectron // 2
    nvir_all = cell.nao - nocc_all
    nocc = min(AS, nocc_all)
    nvir = min(AS, nvir_all)
    nfo = nocc_all - nocc
    nact = nocc + nvir

    # GPU SCF
    kg = gdft.KRKS(cell, kpts=GAMMA, xc=XC)
    kg.exxdiv = None
    kg.conv_tol = 1e-9
    kg.kernel()
    mo = cp.asnumpy(kg.mo_coeff[0])
    moe = cp.asnumpy(kg.mo_energy[0])
    mo_occ = cp.asnumpy(kg.mo_occ[0])
    e_tot = float(kg.e_tot)
    # release the SCF GPU arrays before the (memory-heavy) ao2mo
    kg = None
    cp.get_default_memory_pool().free_all_blocks()

    # CPU RKS shell carrying GPU orbitals (for the gradient relaxed density)
    mf = cdft.RKS(cell, xc=XC)
    mf.exxdiv = None
    mf.mo_coeff = mo
    mf.mo_energy = moe
    mf.mo_occ = mo_occ
    mf.e_tot = e_tot
    mf.converged = True

    # GPU FFT ao2mo (active-space MO ERI) + GPU batched use_eri Davidson
    cocc = mo[:, nfo:nfo + nocc]
    cvir = mo[:, nfo + nocc:nfo + nocc + nvir]
    vvvv, oovv, oooo = gpu_ao2mo_blocks(cell, cocc, cvir, cell.mesh, return_gpu=True)
    mp = ppRPA_Davidson(nocc, moe[nfo:nfo + nact], Lpq=None, channel=CHANNEL,
                        nroot=NROOT, residue_thresh=1e-9, trial="identity")
    mp.mu = 0.0
    mp.max_vec = 1000
    attach_gpu_eri_contraction(mp, vvvv, oovv, oooo)
    mp.kernel(MULT)

    exci = (mp.exci_s if MULT == "s" else mp.exci_t)[ISTATE]
    sign = 1.0 if CHANNEL == "pp" else -1.0
    e_state = float(mf.e_tot) + sign * float(exci)

    if not want_grad:
        cp.get_default_memory_pool().free_all_blocks()
        return e_state

    _patch_gpu_getk(mf, cell)
    xy = (mp.xy_s if MULT == "s" else mp.xy_t)[ISTATE]
    g = gpugrad.Gradients(mp, mf, MULT, ISTATE)
    g.cphf_max_cycle = 100
    g.cphf_conv_tol = 1e-7
    de = g.grad_elec(xy, MULT, range(_NATM)) + g.grad_nuc()
    cp.get_default_memory_pool().free_all_blocks()
    return e_state, de


def grad_func(cell, **kw):
    e, de = _pipeline(cell, want_grad=True)
    _step[0] += 1
    print(f"[opt step {_step[0]:3d}]  E_state = {e:.8f} Ha   "
          f"|F| = {np.linalg.norm(de):.4e}  max|F| = {np.abs(de).max():.4e} a.u.",
          flush=True)
    return e, de


def ene_func(cell, **kw):
    return _pipeline(cell, want_grad=False)


if __name__ == "__main__":
    cell = build_cell(_at.get_positions() / BOHR)
    print(f"### NV-style GPU Gamma pp-RPA opt: {cell.natm} atoms, nao={cell.nao}, "
          f"mesh={list(cell.mesh)}, ke={KE} Ha, channel={CHANNEL}, mult={MULT}, "
          f"AS={AS} ###", flush=True)
    converged, cell_opt = ase_utils.kernel(
        cell, grad_func=grad_func, ene_func=ene_func,
        logfile="opt_ase.log", fmax=FMAX, max_steps=MAXSTEPS)
    opt_ase = pyscf_to_ase_atoms(cell_opt)
    info={"converged":converged, "charge":CHARGE, "mult":MULT, "channel":CHANNEL, "istate":ISTATE, "ke_Ha":KE, "xc":XC, "nao":cell.nao}
    opt_ase.info.update(info)
    write_extxyz("opt_final.xyz", opt_ase)
    print(f"\nconverged = {converged}   (final geometry -> opt_final.xyz)", flush=True)
