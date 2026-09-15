"""One-step, fully GDF-consistent Gamma-point ppRPA energy and gradient.

The PySCF reference and ppRPA Davidson operator use the same periodic Gaussian
density-fitting (GDF) object.  The analytical gradient mirrors that object with
GPU4PySCF GDF and uses GDF for the reference J/K derivative, native CPHF
response, and singlet/triplet pair-density derivative.  It never substitutes
FFTDF or AFTDF for an exchange term.

Run on a GPU node with the compatible PySCF/GPU4PySCF environment documented
in ``examples/gpu/README.md``::

    python pbc_pprpa_gamma_gdf_gpu.py t
    python pbc_pprpa_gamma_gdf_gpu.py s
"""

import sys

import numpy as np
from pyscf.pbc import df, dft, gto

from lib_pprpa.grad.pprpa_gamma_gdf_gpu import Gradients
from lib_pprpa.pbc_gdf import make_pprpa_gdf


MULT = sys.argv[1] if len(sys.argv) > 1 else "t"
if MULT not in ("s", "t"):
    raise SystemExit("multiplicity must be 's' or 't'")

cell = gto.Cell()
cell.atom = [
    ("C", (0.031, -0.019, 0.013)),
    ("C", (1.72, 1.63, 1.81)),
]
cell.a = np.array([
    [0.0, 3.4, 3.4],
    [3.4, 0.0, 3.4],
    [3.4, 3.4, 0.0],
])
cell.unit = "Bohr"
cell.basis = "gth-szv"
cell.pseudo = "gth-pbe"
cell.ke_cutoff = 100.0
cell.precision = 1e-8
cell.verbose = 4
cell.build()

# The CPU object is the authoritative energy/operator reference.  Do not call
# density_fit() again after make_pprpa_gdf: its identity and settings are part
# of the provenance checked by the gradient driver.
mf = dft.RKS(cell, xc="pbe")
mf.exxdiv = None
mf.grids.mesh = cell.mesh
mf.with_df = df.GDF(cell, kpts=np.zeros((1, 3)))
mf.with_df.auxbasis = "weigend"
mf.with_df.linear_dep_threshold = 1e-10
mf.conv_tol = 1e-11
mf.kernel()
if not mf.converged:
    mo_coeff, mo_occ = mf.mo_coeff, mf.mo_occ
    mf = mf.newton()
    mf.conv_tol = 1e-11
    mf.max_cycle = 100
    mf.kernel(mo_coeff, mo_occ)
if not mf.converged:
    raise RuntimeError("GDF reference did not converge, including Newton fallback")

solver = make_pprpa_gdf(
    mf, channel="pp", nroot=2, residue_thresh=1e-10,
    max_iter=200, max_vec=500, trial="identity")
solver.mu = 0.0
solver.kernel(MULT)

spectrum = solver.exci_s if MULT == "s" else solver.exci_t
state_energy = mf.e_tot + spectrum[0]
gradient = Gradients(solver, mf, mult=MULT, state=0)
gradient.cphf_max_cycle = 100
gradient.cphf_conv_tol = 1e-10
de = gradient.kernel()

print(f"multiplicity = {MULT}")
print(f"GDF auxiliary functions = {mf.with_df.get_naoaux()}")
print(f"state energy = {state_energy:.12f} Ha")
print("gradient (Ha/Bohr):")
print(np.asarray(de))
