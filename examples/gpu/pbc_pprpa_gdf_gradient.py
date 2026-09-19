"""Gamma-point pp-RPA nuclear gradient of a periodic defect with Gaussian
density fitting, on the GPU.

Builds on the energy example, pbc_pprpa_gdf_energy.py: same reference, same
GDF factors, then the analytical gradient of the selected root.

The gradient reuses the GPU density fitting and grids that the reference
already built, so pass them to Gradients rather than letting it rebuild them
for every evaluation.

Translational invariance is the cheapest check that the forces are right: the
raw gradient of a periodic system should sum to zero over atoms.  It is
printed below.  Along an optimization it is also worth projecting the residual
net force out, since a small uniform component otherwise translates the whole
cell.

Successive evaluations can reuse the previous CPHF solution.  Gradients
carries it in the AO basis as .cphf_x0; assign it onto the next Gradients
object (or keep the same one) and the solver iterates only on the correction,
which on a geometry optimization saves roughly a third of the CPHF time.  The
second solve below demonstrates it.

Usage:
    python pbc_pprpa_gdf_gradient.py <geometry> [xc] [charge] [channel] [mult] [AS] [state]

Example:
    python pbc_pprpa_gdf_gradient.py nv_63.vasp pbe -3 hh t 100 0
"""

import copy
import sys

import numpy as np
from ase.io import read
from pyscf.data.nist import BOHR, HARTREE2EV
from pyscf.pbc import df, dft, gto

from gpu4pyscf.pbc import dft as gdft
from gpu4pyscf.pbc.df.df import GDF as GPU_GDF

from lib_pprpa.grad.pprpa_gamma_gdf_gpu import Gradients
from lib_pprpa.pbc_gdf import make_pprpa_gdf

geometry = sys.argv[1]
xc = sys.argv[2] if len(sys.argv) > 2 else 'pbe'
charge = int(sys.argv[3]) if len(sys.argv) > 3 else -3
channel = sys.argv[4] if len(sys.argv) > 4 else 'hh'
mult = sys.argv[5] if len(sys.argv) > 5 else 't'
as_size = int(sys.argv[6]) if len(sys.argv) > 6 else 100
state = int(sys.argv[7]) if len(sys.argv) > 7 else 0

atoms = read(geometry)
cell = gto.Cell()
cell.atom = list(zip(atoms.get_chemical_symbols(), atoms.positions / BOHR))
cell.a = np.asarray(atoms.cell) / BOHR
cell.unit = 'Bohr'
cell.basis = 'gth-cc-pvdz-lc'
cell.pseudo = 'gth-pbe'
cell.charge = charge
cell.spin = 0
cell.verbose = 4
cell.build()
assert cell.nelectron % 2 == 0, 'the pp-RPA reference must be closed shell'

kpts = np.zeros((1, 3))

cpu_df = df.GDF(cell, kpts=kpts)
cpu_df.auxbasis = None
cpu_df.linear_dep_threshold = 1e-10
cpu_df.build(j_only=False, with_j3c=False)

gpu_df = GPU_GDF(cell, kpts)
gpu_df.auxbasis = copy.deepcopy(cpu_df.auxcell.basis)
gpu_df.exp_to_discard = None
gpu_df.linear_dep_threshold = cpu_df.linear_dep_threshold
gpu_df.mesh = np.asarray(cpu_df.mesh if cpu_df.mesh is not None else cell.mesh)
gpu_df.is_gamma_point = True
gpu_df.build(j_only=False)

gpu_mf = gdft.KRKS(cell, kpts=kpts, xc=xc)
gpu_mf.with_df = gpu_df
gpu_mf.rsjk = None
gpu_mf.exxdiv = None
gpu_mf.grids = gdft.BeckeGrids(cell)
gpu_mf.grids.level = 3
gpu_mf.small_rho_cutoff = 0
gpu_mf.conv_tol = 1e-9
gpu_mf.kernel()
assert gpu_mf.converged, 'the GPU reference did not converge'

mf = dft.RKS(cell, xc=xc)
mf.exxdiv = None
mf.grids = dft.BeckeGrids(cell)
mf.grids.level = 3
mf.small_rho_cutoff = 0
mf.with_df = cpu_df
mf.mo_coeff = np.asarray(gpu_mf.mo_coeff[0].get()).real
mf.mo_energy = np.asarray(gpu_mf.mo_energy[0].get()).real
mf.mo_occ = np.asarray(gpu_mf.mo_occ[0].get()).real
mf.e_tot = float(gpu_mf.e_tot)
mf.converged = True

pp = make_pprpa_gdf(mf, channel=channel, nocc_act=as_size, nvir_act=as_size,
                    mu=0.0, nroot=max(8, state + 1), residue_thresh=1e-9,
                    max_iter=400, max_vec=1000, trial='identity')
pp.kernel(mult)

omega = np.asarray(pp.exci_s if mult == 's' else pp.exci_t)
sign = -1.0 if channel == 'hh' else 1.0
print('state %d energy %20.10f Ha' % (state, mf.e_tot + sign * omega[state]))

# Hand the GPU density fitting and grids to the gradient so it does not build
# a second copy.  cphf_conv_tol 1e-4 is enough for forces at this scale; the
# error it leaves is far below any sensible convergence criterion.  Raise
# cphf_max_cycle as well: the class default of 20 is aimed at molecules and a
# defect supercell needs more, and lib.krylov raises rather than warning when
# it runs out of cycles.
grad = Gradients(pp, mf, mult=mult, state=state,
                 gpu_df=gpu_df, gpu_grids=gpu_mf.grids)
grad.cphf_conv_tol = 1e-4
grad.cphf_max_cycle = 100
de = grad.kernel()

force = -de * HARTREE2EV / BOHR
print('max |force| %12.6f eV/A' % np.abs(force).max())
print('net force   %12.3e eV/A   (translational invariance)'
      % np.abs(force.sum(axis=0)).max())
for ia in range(min(len(force), 6)):
    print('  atom %3d %-2s %12.6f %12.6f %12.6f'
          % (ia, cell.atom_symbol(ia), *force[ia]))
if len(force) > 6:
    print('  ... %d more atoms' % (len(force) - 6))

# Reuse the CPHF solution.  At an unchanged geometry the guess is already the
# answer, so the second solve returns immediately; along an optimization it
# leaves only the correction to iterate on.  Run with verbose >= 4 on the cell
# to see the response-evaluation count drop.
grad2 = Gradients(pp, mf, mult=mult, state=state,
                  gpu_df=gpu_df, gpu_grids=gpu_mf.grids)
grad2.cphf_conv_tol = 1e-4
grad2.cphf_max_cycle = 100
grad2.cphf_x0 = grad.cphf_x0
de2 = grad2.kernel()
print('warm-started gradient differs by %.3e Ha/Bohr'
      % np.abs(de2 - de).max())
