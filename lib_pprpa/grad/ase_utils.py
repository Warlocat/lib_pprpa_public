#!/usr/bin/env python
# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Garnet Chan <gkc1000@gmail.com>
#         Xing Zhang <zhangxing.nju@gmail.com>
#

'''
ASE package interface
'''

try:
    from ase.calculators.calculator import Calculator, all_properties
    from ase.optimize import BFGS
except ImportError:
    print("""ASE is not found. Please install ASE via
pip3 install ase
          """)
    raise RuntimeError("ASE is not found")

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.data.nist import BOHR, HARTREE2EV
from pyscf.gto.mole import charge
from pyscf.pbc.tools.pyscf_ase import pyscf_to_ase_atoms
from lib_pprpa.pprpa_davidson import ppRPA_Davidson

def pprpaobj(mf, channel, nocc=None, nvir=None, mo_eri=False):
    mo_ene = mf.mo_energy
    mol = mf.mol
    if nocc is None:
        nocc = mol.nelectron // 2
    if nvir is None:
        nvir = mo_ene.shape[0] - nocc
    nmo = nocc + nvir
    mo_energy = mo_ene[mol.nelectron//2 - nocc:mol.nelectron//2 + nvir]
    mo_coeff = mf.mo_coeff[:,mol.nelectron//2 - nocc:mol.nelectron//2 + nvir]
    
    pprpa = ppRPA_Davidson(nocc, mo_energy, Lpq=None, channel=channel, nroot=1, residue_thresh=1e-12)
    pprpa.cell = mol

    # One can use either the MO eri or the ao direct approach.
    # For small active spaces, MO eri should be faster.
    if mo_eri:
        eri = mf.with_df.get_mo_eri(mo_coeff, compact=False)
        eri = eri.reshape(nmo, nmo, nmo, nmo).transpose(0, 2, 1, 3)
        vvvv = np.ascontiguousarray(eri[nocc:, nocc:, nocc:, nocc:])
        oovv = np.ascontiguousarray(eri[:nocc, :nocc, nocc:, nocc:])
        oooo = np.ascontiguousarray(eri[:nocc, :nocc, :nocc, :nocc])
        pprpa.use_eri(vvvv, oovv, oooo)
    else:
        pprpa._ao_direct = True
        pprpa._scf = mf
    
    pprpa.mu = 0.0
    return pprpa

def pprpa_energy(cell, xc, channel, mult, istate):
    nroot = istate + 1
    mf = cell.RKS(xc=xc)
    mf.exxdiv = None
    mf.conv_tol = 1e-12
    mf.kernel()
    e = mf.e_tot

    mp = pprpaobj(mf, channel)
    mp.nroot = nroot
    mp.kernel(mult)
    mp.analyze()
    e_pprpa = mp.exci_s[istate] if mult == 's' else mp.exci_t[istate]
    e = e + e_pprpa if channel == "pp" else e - e_pprpa
    return e

def pprpa_grad(cell, xc, channel, mult, istate):
    from lib_pprpa.grad import pprpa_gamma
    nroot = max(istate, 3)
    mf = cell.RKS(xc=xc)
    mf.exxdiv = None
    mf.conv_tol = 1e-12
    mf.kernel()
    e = mf.e_tot
    mp = pprpaobj(mf, channel)
    mp.nroot = nroot
    mp.kernel(mult)
    mp.analyze()
    e_pprpa = mp.exci_s[istate] if mult == 's' else mp.exci_t[istate]
    e = e + e_pprpa if channel == "pp" else e - e_pprpa
    mpg = mp.Gradients(mf, mult, istate)
    mpg.kernel()
    return e, mpg.de


class ppRPA(Calculator):
    implemented_properties = ['energy', 'forces']

    default_parameters = {}

    def __init__(self, cell, xc, channel, mult, state, restart=None, label='ppRPA', atoms=None, directory='.',
                 **kwargs):
        """Construct ppRPA-calculator object.
        """
        Calculator.__init__(self, restart, label=label, atoms=atoms,
                            directory=directory, **kwargs)
        self.cell = cell
        self.mult = mult
        self.state = state
        self.xc = xc
        self.channel = channel

    def set(self, **kwargs):
        changed_parameters = Calculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_properties):
        Calculator.calculate(self, atoms)

        positions = atoms.get_positions()
        atomic_numbers = atoms.get_atomic_numbers()
        Z = np.array([charge(x) for x in self.cell.elements])
        if all(Z == atomic_numbers):
            _atoms = positions
        else:
            _atoms = list(zip(atomic_numbers, positions))

        self.cell.set_geom_(_atoms, a=np.asarray(atoms.cell), unit='Angstrom')

        with_grad = 'forces' in properties
        with_energy = with_grad or 'energy' in properties

        if with_energy and with_grad:
            e_tot, grad = pprpa_grad(self.cell, self.xc, self.channel, self.mult, self.state)
            self.results['energy'] = e_tot * HARTREE2EV
            self.results['forces'] = -grad * (HARTREE2EV / BOHR)
        elif with_energy:
            e_tot = pprpa_energy(self.cell, self.xc, self.channel, self.mult, self.state)
            self.results['energy'] = e_tot * HARTREE2EV
        else:
            raise NotImplementedError("Only energy and forces are implemented for ppRPA calculator.")
        
def kernel(cell, xc, channel, mult, state, logfile=None, fmax=0.05, max_steps=100):
    '''Optimize the geometry using ASE.
    '''
    atoms = pyscf_to_ase_atoms(cell)
    atoms.calc = ppRPA(cell, xc, channel, mult, state)
    if logfile is None:
        logfile = '-' # stdout

    opt = BFGS(atoms, logfile=logfile)
    converged = opt.run(fmax=fmax, steps=max_steps)

    cell = cell.set_geom_(atoms.get_positions(), unit='Ang', a=atoms.cell, inplace=False)

    if converged:
        logger.note(cell, 'Geometry optimization converged')
    else:
        logger.note(cell, 'Geometry optimization not converged')
    if cell.verbose >= logger.NOTE:
        coords = cell.atom_coords() * lib.param.BOHR
        for ia in range(cell.natm):
            logger.note(cell, ' %3d %-4s %16.9f %16.9f %16.9f AA',
                        ia+1, cell.atom_symbol(ia), *coords[ia])
        a = cell.lattice_vectors() * lib.param.BOHR
        logger.note(cell, 'lattice vectors  a1 [%.9f, %.9f, %.9f]', *a[0])
        logger.note(cell, '                 a2 [%.9f, %.9f, %.9f]', *a[1])
        logger.note(cell, '                 a3 [%.9f, %.9f, %.9f]', *a[2])
    return converged, cell
