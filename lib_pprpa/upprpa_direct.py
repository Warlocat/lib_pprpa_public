"""Direct solver for unrestricted particle-particle random phase approximation.
Author: Jincheng Yu <pimetamon@gmail.com>
"""
import h5py
import numpy
import scipy
from lib_pprpa.pprpa_davidson import pprpa_orthonormalize_eigenvector, pprpa_print_a_pair
from lib_pprpa.pprpa_util import get_chemical_potential, start_clock, stop_clock, print_citation


class UppRPA_direct():
    """Direct solver class for unrestricted ppRPA.
    
    Args:
        nocc (tuple): number of occupied orbitals, (nalpha, nbeta)
        mo_energy (list of double arrays): orbital energies, [alpha, beta]
        Lpq (list of double ndarrays): 
            three-center RI matrices in MO space, [alpha, beta]

    Kwargs:
        hh_state (int): number of hole-hole states to print
        pp_state (int): number of particle-particle states to print
        nelec (str): 'n-2' for ppRPA and 'n+2' for hhRPA
        print_thresh (float): threshold for printing component
    """
    def __init__(
            self, nocc, mo_energy, Lpq, hh_state=5, pp_state=5, nelec='n-2',
            print_thresh=0.1):
        self.nocc = nocc
        self.mo_energy = mo_energy
        self.Lpq = Lpq
        self.hh_state = hh_state
        self.pp_state = pp_state
        self.print_thresh = print_thresh

        #======================> internal flags <=======================
        # number of orbitals
        self.nmo = (len(self.mo_energy[0]), len(self.mo_energy[1]))
        # number of virtual orbitals
        self.nvir = (self.nmo[0] - self.nocc[0], self.nmo[1] - self.nocc[1])
        # number of auxiliary basis functions
        self.naux = (self.Lpq[0].shape[0], self.Lpq[1].shape[0])
        # chemical potential
        self.mu = None

        #=========================> results <===========================
        self.ec = None  # correlation energy
        self.ec_s = None  # singlet correlation energy
        self.ec_t = None  # triplet correlation energy
        self.exci = None  # two-electron addition energy
        self.xy = None  # ppRPA eigenvector
        self.exci_s = None  # singlet two-electron addition energy
        self.xy_s = None  # singlet two-electron addition eigenvector
        self.exci_t = None  # triplet two-electron addition energy
        self.xy_t = None  # triplet two-electron addition eigenvector

        print_citation()

        return
    
    def check_parameter(self):
        assert 0.0 < self.print_thresh < 1.0
        assert self.nelec in ["n-2", "n+2"]
        if self.mu is None:
            self.mu = get_chemical_potential(nocc=self.nocc,
                                             mo_energy=self.mo_energy)
        return
