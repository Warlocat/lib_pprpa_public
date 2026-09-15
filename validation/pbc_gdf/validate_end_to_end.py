#!/usr/bin/env python
"""End-to-end finite-difference validation of the periodic GDF backend.

The reference energy, ppRPA operator, CPHF response, and analytical two-electron
terms all use Gaussian density fitting.  This script is intentionally small
enough for an allocated development GPU while exercising both pair symmetries.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import cupy as cp
import numpy as np
from pyscf.pbc import df, dft, gto

from lib_pprpa.grad.pprpa_gamma_gdf_gpu import Gradients
from lib_pprpa.grad.pprpa_gamma_gdf_gpu import make_gpu_gdf_mf
from lib_pprpa.pbc_gdf import make_pprpa_gdf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--system", default="c2", choices=("c2",))
    parser.add_argument("--channel", default="pp", choices=("pp", "hh"))
    parser.add_argument("--xc", default="pbe", choices=("pbe", "b3lyp", "hf"))
    parser.add_argument("--auxbasis", default="compact",
                        help="PySCF auxiliary basis name, 'auto', or 'compact'")
    parser.add_argument("--mult", nargs="+", default=("s", "t"), choices=("s", "t"))
    parser.add_argument("--steps", nargs="+", type=float, default=(0.002, 0.004),
                        help="central finite-difference half steps in Bohr")
    parser.add_argument("--analytic-only", action="store_true")
    parser.add_argument("--cell-precision", type=float, default=1e-8)
    parser.add_argument("--ke-cutoff", type=float, default=100.0)
    return parser.parse_args()


ARGS = parse_args()
OUTPUT = Path(ARGS.output)
GAMMA = np.zeros((1, 3))
SYSTEMS = {
    "c2": {
        # Four occupied orbitals make every S/T pair and response block
        # nontrivial.  The nonsymmetric origin prevents accidental cancellation
        # while the internal coordinate removes common translational grid error.
        "symbols": ("C", "C"),
        "coords": np.array([[0.031, -0.019, 0.013], [1.72, 1.63, 1.81]]),
        "lattice": np.array([
            [0.0, 3.4, 3.4],
            [3.4, 0.0, 3.4],
            [3.4, 3.4, 0.0],
        ]),
        "basis": "gth-szv",
    },
}
SYSTEM = SYSTEMS[ARGS.system]
BASE_COORDS = SYSTEM["coords"]


def build_cell(coords):
    cell = gto.Cell()
    cell.atom = list(zip(SYSTEM["symbols"], coords))
    cell.a = SYSTEM["lattice"]
    cell.unit = "Bohr"
    cell.basis = SYSTEM["basis"]
    cell.pseudo = "gth-pbe"
    cell.ke_cutoff = ARGS.ke_cutoff
    cell.precision = ARGS.cell_precision
    cell.max_memory = 120000
    cell.verbose = 3
    cell.build()
    return cell


def run_reference(coords):
    cell = build_cell(coords)
    if ARGS.xc == "hf":
        from pyscf.pbc import scf
        mf = scf.RHF(cell)
    else:
        mf = dft.RKS(cell, xc=ARGS.xc)
        mf.grids.mesh = cell.mesh
    mf.exxdiv = None
    mf.with_df = df.GDF(cell, kpts=GAMMA)
    if ARGS.auxbasis.lower() == "auto":
        mf.with_df.auxbasis = None
    elif ARGS.auxbasis.lower() == "compact":
        # A deliberately compact, well-conditioned fit for a fast algebraic
        # FD gate.  Physical auxiliary-basis sensitivity is tested separately.
        mf.with_df.auxbasis = {
            symbol: [[0, [3.0, 1.0]], [0, [0.8, 1.0]], [1, [1.0, 1.0]]]
            for symbol in set(SYSTEM["symbols"])
        }
    else:
        mf.with_df.auxbasis = ARGS.auxbasis
    mf.with_df.linear_dep_threshold = 1e-10
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        # Continue a difficult fixed-point SCF from its current orbitals with
        # PySCF's second-order solver.  This retains the same Cell, GDF object,
        # auxiliary basis, and XC grid.
        mo_coeff, mo_occ = mf.mo_coeff, mf.mo_occ
        mf = mf.newton()
        mf.conv_tol = 1e-11
        mf.conv_tol_grad = 1e-9
        mf.max_cycle = 100
        mf.kernel(mo_coeff, mo_occ)
    if not mf.converged:
        raise RuntimeError("CPU GDF reference did not converge, including Newton fallback")
    return mf


def solve_state(coords, mult, want_gradient=False):
    mf = run_reference(coords)
    solver = make_pprpa_gdf(
        mf, channel=ARGS.channel, nroot=3, residue_thresh=1e-10,
        max_iter=200, max_vec=500, trial="identity")
    solver.mu = 0.0
    solver.kernel(mult)
    spectrum = solver.exci_s if mult == "s" else solver.exci_t
    # pp eigenvalues add two particles; hh eigenvalues are positive removal
    # energies and therefore enter the physical N-2 state with the opposite
    # sign.  The analytical gradient driver already follows this convention.
    state_sign = 1.0 if ARGS.channel == "pp" else -1.0
    state_energy = float(mf.e_tot + state_sign * spectrum[0])
    result = {
        "reference_energy_hartree": float(mf.e_tot),
        "spectrum_hartree": np.asarray(spectrum).tolist(),
        "state_energy_hartree": state_energy,
        "nao": int(mf.cell.nao),
        "naux": int(mf.with_df.get_naoaux()),
        "mesh": [int(x) for x in mf.cell.mesh],
    }
    if want_gradient:
        reference_gradient = make_gpu_gdf_mf(mf).nuc_grad_method().kernel()
        result["reference_gradient_hartree_per_bohr"] = np.asarray(
            reference_gradient).tolist()
        gradient = Gradients(solver, mf, mult=mult, state=0)
        gradient.cphf_max_cycle = 100
        gradient.cphf_conv_tol = 1e-10
        result["gradient_hartree_per_bohr"] = np.asarray(
            gradient.kernel()).tolist()
    del solver, mf
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    return result


def write(payload):
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main():
    payload = {
        "config": {
            "xc": ARGS.xc,
            "channel": ARGS.channel,
            "system": ARGS.system,
            "auxbasis": ARGS.auxbasis,
            "cell_precision": ARGS.cell_precision,
            "ke_cutoff_hartree": ARGS.ke_cutoff,
            "multiplicities": list(ARGS.mult),
            "steps_bohr": list(ARGS.steps),
        },
        "results": {},
    }
    # Use an internal, translation-free coordinate.  Periodic pseudopotential
    # calculations on a finite uniform FFT grid have a small common-force
    # ("egg-box") component if only one atom is displaced; projecting that
    # component out makes this a test of the GDF ppRPA derivative rather than
    # of the deliberately inexpensive validation grid.
    direction = np.zeros_like(BASE_COORDS)
    direction[0, 2] = -0.5
    direction[1, 2] = 0.5
    for mult in ARGS.mult:
        center = solve_state(BASE_COORDS, mult, want_gradient=True)
        analytic = float(np.einsum(
            "ax,ax->", center["gradient_hartree_per_bohr"], direction))
        reference_analytic = float(np.einsum(
            "ax,ax->", center["reference_gradient_hartree_per_bohr"], direction))
        excitation_analytic = analytic - reference_analytic
        entry = {
            "center": center,
            "analytic_directional": analytic,
            "reference_analytic_directional": reference_analytic,
            "excitation_analytic_directional": excitation_analytic,
            "fd": {},
        }
        payload["results"][mult] = entry
        write(payload)
        if not ARGS.analytic_only:
            for step in ARGS.steps:
                plus = solve_state(BASE_COORDS + step * direction, mult)
                minus = solve_state(BASE_COORDS - step * direction, mult)
                ep = plus["state_energy_hartree"]
                em = minus["state_energy_hartree"]
                finite_difference = (ep - em) / (2.0 * step)
                reference_fd = (
                    plus["reference_energy_hartree"]
                    - minus["reference_energy_hartree"]
                ) / (2.0 * step)
                excitation_fd = (
                    (1.0 if ARGS.channel == "pp" else -1.0)
                    * (plus["spectrum_hartree"][0]
                       - minus["spectrum_hartree"][0])
                ) / (2.0 * step)
                entry["fd"][str(step)] = {
                    "plus_energy_hartree": ep,
                    "minus_energy_hartree": em,
                    "finite_difference": finite_difference,
                    "reference_finite_difference": reference_fd,
                    "excitation_finite_difference": excitation_fd,
                    "error": finite_difference - analytic,
                    "reference_error": reference_fd - reference_analytic,
                    "excitation_error": excitation_fd - excitation_analytic,
                }
                write(payload)

    errors = [
        abs(fd["error"])
        for entry in payload["results"].values()
        for fd in entry["fd"].values()
    ]
    payload["max_abs_fd_error_hartree_per_bohr"] = max(errors) if errors else None
    write(payload)


if __name__ == "__main__":
    main()
