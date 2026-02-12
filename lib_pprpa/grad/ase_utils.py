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

def pprpaobj(mf, channel, **kwargs):
    mo_ene = mf.mo_energy
    mol = mf.mol

    nfrozen_occ = kwargs.get("nfrozen_occ", 0)
    vir_cut = kwargs.get("vir_cut", np.inf)
    AS_size = kwargs.get("AS_size", None)
    nocc = kwargs.get("nocc", mol.nelectron // 2)
    nvir = kwargs.get("nvir", mo_ene.shape[0] - nocc)
    mo_eri = kwargs.get("mo_eri", False)
    nroot = kwargs.get("nroot", 1)
    checkpoint = kwargs.get("checkpoint", None)
    max_mem = kwargs.get("max_mem", None)
    trial = kwargs.get("trial", "identity")
    full_nocc = nocc
    nmo = nocc + nvir

    # create active space by explicitly excluding number of occupied orbitals or by vir energy cutoff
    if nfrozen_occ > 0  or vir_cut < np.inf:
        if AS_size is not None:
            raise ValueError("Cannot specify both nfrozen_occ/vir_cut and AS_size")
        nvir -= np.sum(mo_ene > vir_cut)
        nocc -= nfrozen_occ
        assert nocc > 1, "Too many frozen occupied orbitals!"
        assert nvir > 1, "Too many excluded virtual orbitals!"
    # create active space by explicitly setting the size of the active space
    elif AS_size is not None and (AS_size < full_nocc or AS_size < nvir):
        nvir = min(nvir, AS_size)
        nfrozen_occ = full_nocc - min(full_nocc, AS_size)
        nocc = full_nocc - nfrozen_occ

    vir_act_idx = full_nocc + nvir
    mo_energy = mo_ene[nfrozen_occ:vir_act_idx]

    pprpa = ppRPA_Davidson(nocc, mo_energy, Lpq=None, channel=channel, nroot=nroot, residue_thresh=1e-12, checkpoint_file=checkpoint, trial=trial)
    pprpa.cell = mol

    # One can use either the MO eri or the ao direct approach.
    # For small active spaces, MO eri should be faster.
    if mo_eri:
        if max_mem is not None:
            mf.with_df.max_memory = max_mem
        eri = mf.with_df.get_mo_eri(mf.mo_coeff, compact=False)
        eri = eri.reshape(nmo, nmo, nmo, nmo).transpose(0, 2, 1, 3)
        vvvv = eri[full_nocc:vir_act_idx, full_nocc:vir_act_idx, full_nocc:vir_act_idx, full_nocc:vir_act_idx]
        oovv = eri[nfrozen_occ:full_nocc, nfrozen_occ:full_nocc, full_nocc:vir_act_idx, full_nocc:vir_act_idx]
        oooo = eri[nfrozen_occ:full_nocc, nfrozen_occ:full_nocc, nfrozen_occ:full_nocc, nfrozen_occ:full_nocc]
        pprpa.use_eri(vvvv, oovv, oooo)
    else:
        pprpa._ao_direct = True
        pprpa._scf = mf
    
    pprpa.mu = 0.0
    return pprpa

def pprpa_energy(cell, with_extras=False, **kwargs):
    mf = cell.RKS(xc=kwargs.get("xc", "pbe"))
    mf.exxdiv = None
    mf.conv_tol = kwargs.get("conv_tol", 1e-8)
    mf.chkfile = kwargs.get("chkfile", None)
    mf.init_guess = "chk" if mf.chkfile is not None else "minao"
    mf.kernel()
    e = mf.e_tot

    istate = kwargs.get("istate", 0)
    mult = kwargs.get("mult", "t")
    channel = kwargs.pop("channel", "pp")
    mp = pprpaobj(mf, channel, **kwargs)
    mp.kernel(mult)
    mp.analyze()
    e_pprpa = mp.exci_s[istate] if mult == 's' else mp.exci_t[istate]
    e = e + e_pprpa if mp.channel == "pp" else e - e_pprpa
    if with_extras:
        return e, mp, mf, mult, istate
    return e

def pprpa_grad(cell, **kwargs):
    e, mp, mf, mult, istate = pprpa_energy(cell, with_extras=True, **kwargs)
    from lib_pprpa.grad import pprpa_gamma
    mpg = mp.Gradients(mf, mult, istate)
    mpg.kernel()
    return e, mpg.de


class ASE_calculator(Calculator):
    implemented_properties = ['energy', 'forces']

    default_parameters = {}

    def __init__(self, cell, grad_func, ene_func=None, restart=None, label='ase_tils', atoms=None, directory='.',
                 **kwargs):
        """Construct calculator object.
        """
        Calculator.__init__(self, restart, label=label, atoms=atoms,
                            directory=directory, **kwargs)
        self.cell = cell
        self.grad_func = grad_func
        self.ene_func = ene_func
        self.kwargs = kwargs

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
            e_tot, grad = self.grad_func(self.cell, **self.kwargs)
            self.results['energy'] = e_tot * HARTREE2EV
            self.results['forces'] = -grad * (HARTREE2EV / BOHR)
        elif with_energy:
            e_tot = self.ene_func(self.cell, **self.kwargs)
            self.results['energy'] = e_tot * HARTREE2EV
        else:
            raise NotImplementedError("Only energy and forces are implemented for ppRPA calculator.")
        
def kernel(cell, grad_func, ene_func=None, logfile=None, fmax=0.05, max_steps=100, **kwargs):
    '''Optimize the geometry using ASE.
    '''
    atoms = pyscf_to_ase_atoms(cell)
    atoms.calc = ASE_calculator(cell, grad_func=grad_func, ene_func=ene_func, **kwargs)
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
