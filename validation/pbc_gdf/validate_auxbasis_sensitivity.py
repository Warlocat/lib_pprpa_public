#!/usr/bin/env python
"""Measure periodic GDF reference and ppRPA sensitivity to the fit basis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from pyscf.pbc import df, dft, gto

from lib_pprpa.pbc_gdf import make_pprpa_gdf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--betas", nargs="+", type=float,
                        default=(2.4, 2.0, 1.8, 1.6))
    parser.add_argument("--include-weigend", action="store_true")
    return parser.parse_args()


def build_cell():
    cell = gto.Cell()
    cell.atom = [
        ("H", (0.031, -0.019, 0.013)),
        ("H", (1.57, 0.21, -0.13)),
    ]
    cell.a = np.diag([7.2, 7.4, 7.6])
    cell.unit = "Bohr"
    cell.basis = "gth-dzvp"
    cell.pseudo = "gth-pbe"
    cell.ke_cutoff = 35.0
    cell.precision = 1e-8
    cell.max_memory = 120000
    cell.verbose = 0
    cell.build()
    return cell


def evaluate(label, auxbasis):
    start = time.perf_counter()
    cell = build_cell()
    mf = dft.RKS(cell, xc="pbe")
    mf.exxdiv = None
    mf.grids.mesh = cell.mesh
    mf.with_df = df.GDF(cell, kpts=np.zeros((1, 3)))
    mf.with_df.auxbasis = auxbasis
    mf.with_df.linear_dep_threshold = 1e-10
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"GDF reference {label} did not converge")

    result = {
        "reference_energy_hartree": float(mf.e_tot),
        "naux": int(mf.with_df.get_naoaux()),
    }
    for mult in ("s", "t"):
        solver = make_pprpa_gdf(
            mf, channel="pp", nroot=1, residue_thresh=1e-10,
            max_iter=200, max_vec=500, trial="identity")
        solver.mu = 0.0
        solver.kernel(mult)
        spectrum = solver.exci_s if mult == "s" else solver.exci_t
        result[f"{mult}_excitation_hartree"] = float(spectrum[0])
        result[f"{mult}_state_energy_hartree"] = float(mf.e_tot + spectrum[0])
    result["wall_seconds"] = time.perf_counter() - start
    return result


def main():
    args = parse_args()
    output = Path(args.output)
    cell = build_cell()
    cases = []
    if args.include_weigend:
        cases.append(("weigend", "weigend"))
    for beta in args.betas:
        cases.append((f"etb_beta_{beta:g}", df.aug_etb(cell, beta=beta)))

    payload = {"cases": {}}
    for label, auxbasis in cases:
        payload["cases"][label] = evaluate(label, auxbasis)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    labels = [label for label, _ in cases]
    reference = payload["cases"][labels[-1]]
    for label in labels:
        case = payload["cases"][label]
        case["delta_vs_last"] = {
            key: case[key] - reference[key]
            for key in (
                "reference_energy_hartree",
                "s_excitation_hartree", "t_excitation_hartree",
                "s_state_energy_hartree", "t_state_energy_hartree",
            )
        }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
