"""Gamma-point pp-RPA excitation energies of a periodic defect with Gaussian
density fitting, mean field on the GPU.  Energies only; no gradient.

Pipeline:
  CPU GDF auxiliary basis (pyscf.pbc.df.GDF, with_j3c=False)
    -> GPU mirror of the same fitting + GPU KRKS SCF (gpu4pyscf)
    -> CPU reference object carrying the converged GPU orbitals
    -> Davidson pp-RPA on the GDF factors (lib_pprpa.pbc_gdf.make_pprpa_gdf)

Two details are easy to get wrong.

The density fitting is built twice on purpose.  The CPU object owns the
auxiliary basis and is what the pp-RPA side reads; the GPU mirror does the
three-center integrals and the SCF.  Mirroring the auxiliary basis that
actually generated the CPU factors, rather than a basis name, keeps the two
consistent after any exponent pruning.

The channel fixes the sign of the excitation energies.  The hole-hole channel
starts from a closed-shell N+2 reference and removes two electrons, so the
N-electron energy is E_reference - omega; that is the NV center, reached from
charge -3.  The particle-particle channel starts from N-2 and adds two, giving
E_reference + omega; that is the neutral carbon vacancy, reached from charge
+2.  Reading the spectrum with the wrong sign silently inverts the state
ordering.

Usage:
    python pbc_pprpa_gdf_energy.py <geometry> [xc] [charge] [channel] [mult] [AS] [nroot]

      geometry : POSCAR/.vasp or .xyz read through ASE; the lattice comes from it
      xc       : functional                       (default 'pbe')
      charge   : reference charge                 (default -3, the NV center)
      channel  : 'hh' or 'pp'                     (default 'hh')
      mult     : 's' or 't'                       (default 't')
      AS       : active occupied = active virtual (default 100)
      nroot    : number of roots                  (default 8)

Examples:
    # NV- center: 3A2 ground state and the 3E pair
    python pbc_pprpa_gdf_energy.py nv_63.vasp pbe -3 hh t 100 8

    # neutral carbon vacancy: the 1E and 1T2 manifolds
    python pbc_pprpa_gdf_energy.py vc_215.vasp pbe 2 pp s 100 10

Choose the active space so that neither boundary falls inside a
near-degenerate pair of orbitals: the relaxed-density terms divide by the
orbital energy difference across the cut, and a cut through a degenerate shell
makes that denominator meaningless.  Print mo_energy around the boundary and
check the gaps before trusting a large calculation.
"""

import copy
import sys

import numpy as np
from ase.io import read
from pyscf.data.nist import BOHR, HARTREE2EV
from pyscf.pbc import df, dft, gto

from gpu4pyscf.pbc import dft as gdft
from gpu4pyscf.pbc.df.df import GDF as GPU_GDF

from lib_pprpa.pbc_gdf import make_pprpa_gdf

geometry = sys.argv[1]
xc = sys.argv[2] if len(sys.argv) > 2 else 'pbe'
charge = int(sys.argv[3]) if len(sys.argv) > 3 else -3
channel = sys.argv[4] if len(sys.argv) > 4 else 'hh'
mult = sys.argv[5] if len(sys.argv) > 5 else 't'
as_size = int(sys.argv[6]) if len(sys.argv) > 6 else 100
nroot = int(sys.argv[7]) if len(sys.argv) > 7 else 8
assert channel in ('hh', 'pp')

# The supercell.  exxdiv is switched off below, so no Madelung correction is
# applied to the orbitals of the charged reference.
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

# The CPU GDF object owns the auxiliary basis.  with_j3c=False builds the
# auxiliary cell only; the three-center integrals are made on the GPU below.
cpu_df = df.GDF(cell, kpts=kpts)
cpu_df.auxbasis = None
cpu_df.linear_dep_threshold = 1e-10
cpu_df.build(j_only=False, with_j3c=False)

# Mirror that fitting on the GPU and converge the reference there.
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
gpu_mf.grids = gdft.BeckeGrids(cell)   # atom-centered, as for molecules
gpu_mf.grids.level = 3
gpu_mf.small_rho_cutoff = 0
gpu_mf.conv_tol = 1e-9
gpu_mf.kernel()
assert gpu_mf.converged, 'the GPU reference did not converge'

# Carry the converged orbitals over to a CPU reference object.  pp-RPA reads
# mo_coeff/mo_energy/mo_occ and the GDF factors from it; no SCF is repeated.
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

# mu = 0 keeps the eigenvalues as plain two-electron addition/removal energies.
pp = make_pprpa_gdf(mf, channel=channel, nocc_act=as_size, nvir_act=as_size,
                    mu=0.0, nroot=nroot, residue_thresh=1e-9,
                    max_iter=400, max_vec=1000, trial='identity')
pp.kernel(mult)

omega = np.asarray(pp.exci_s if mult == 's' else pp.exci_t)
sign = -1.0 if channel == 'hh' else 1.0
e_state = mf.e_tot + sign * omega
print('reference energy %22.10f Ha' % mf.e_tot)
print('root      omega (Ha)     E_state (Ha)   excitation (eV)')
for i, (w, e) in enumerate(zip(omega, e_state)):
    print('%4d %15.8f %17.8f %14.4f'
          % (i, w, e, (e - e_state[0]) * HARTREE2EV))
