"""Direct solver for unrestricted particle-particle random phase approximation.
Author: Jincheng Yu <pimetamon@gmail.com>
"""
import h5py
import numpy
import scipy
from lib_pprpa.pprpa_davidson import pprpa_orthonormalize_eigenvector, pprpa_print_a_pair
from lib_pprpa.pprpa_util import get_chemical_potential, start_clock, stop_clock, print_citation

def upprpa_orthonormalize_eigenvector(nocc, exci, xy):
    pass


def diagonalize_pprpa_subspace_same_spin(nocc, mo_energy, Lpq, mu=None):
    """Diagonalize ppRPA matrix in subspace (alpha alpha, alpha, alpha)
    or (beta beta, beta beta).

    Reference:
    [1] https://doi.org/10.1063/1.4828728 (equation 14)

    Args:
        nocc(int): number of occupied orbitals.
        mo_energy (double array): orbital energies.
        Lpq (double ndarray): three-center RI matrices in MO space.
    
    Kwarg:
        mu (double): chemical potential.
    
    Returns:
        exci (double array): ppRPA eigenvalue.
        xy (double ndarray): ppRPA eigenvector.
        ec (double): correlation energy from one subspace.
    """

def diagonalize_pprpa_subspace_diff_spin(nocc, mo_energy, Lpq, mu=None):
    """Diagonalize ppRPA matrix in subspace (alpha beta, alpha, beta).

    Reference:
    [1] https://doi.org/10.1063/1.4828728 (equation 14)

    Args:
        nocc(tuple of int): number of occupied orbitals, (nalpha, nbeta).
        mo_energy (list of double array): orbital energies.
        Lpq (list of double ndarray): three-center RI matrices in MO space.
    
    Kwarg:
        mu (double): chemical potential.
    
    Returns:
        exci (double array): ppRPA eigenvalue.
        xy (double ndarray): ppRPA eigenvector.
        ec (double): correlation energy from one subspace.
    """
    nmo = (len(mo_energy[0]), len(mo_energy[1]))
    nvir = (nmo[0]-nocc[0], nmo[1]-nocc[1])
    if mu is None:
        mu = get_chemical_potential(nocc, mo_energy)

    #===========================> A matrix <============================
    # <ab|cd>
    A = numpy.einsum(
        'Pac,Pbd->abcd', Lpq[0][:, nocc[0]:, nocc[0]:], 
        Lpq[1][:, nocc[1]:, nocc[1]:], optimize=True)
    # delta_ac delta_bd (e_a + e_b - 2 * mu)
    A = A.reshape(nvir[0]*nvir[1], nvir[0]*nvir[1])
    orb_sum = numpy.asarray(
        mo_energy[0][nocc[0]:, None] + mo_energy[1][None, nocc[1]:]
    ).reshape(-1)
    orb_sum -= 2.0 * mu
    numpy.fill_diagonal(A, A.diagonal() + orb_sum)
    trace_A = numpy.trace(A)

    #===========================> B matrix <============================
    # <ab|ij>
    B = numpy.einsum(
        'Pai,Pbj->abij', Lpq[0][:, nocc[0]:, :nocc[0]], 
        Lpq[1][:, nocc[1]:, :nocc[1]], optimize=True)
    B = B.reshape(nvir[0]*nvir[1], nocc[0]*nocc[1])

    #===========================> C matrix <============================
    # <ij|kl>
    C = numpy.einsum(
        'Pik,Pjl->ijkl', Lpq[0][:, :nocc[0], :nocc[0]], 
        Lpq[1][:, :nocc[1], :nocc[1]], optimize=True)
    # - delta_ik delta_jl (e_i + e_j - 2 * mu)
    C = C.reshape(nocc[0]*nocc[1], nocc[0]*nocc[1])
    orb_sum = numpy.asarray(
        mo_energy[0][:nocc[0], None] + mo_energy[1][None, :nocc[1]]
    ).reshape(-1)
    orb_sum -= 2.0 * mu
    numpy.fill_diagonal(C, C.diagonal() - orb_sum)

    #==================> whole matrix in the subspace<==================
    # C    B^T
    # B     A
    M_upper = numpy.concatenate((C, B.T), axis=1)
    M_lower = numpy.concatenate((B, A), axis=1)
    M = numpy.concatenate((M_upper, M_lower), axis=0)
    del A, B, C
    # M to WM, where W is the metric matrix [[-I, 0], [0, I]]
    M[:nocc[0]*nocc[1]][:] *= -1.0

    #=====================> solve for eigenpairs <======================
    exci, xy = scipy.linalg.eig(M)
    exci = exci.real
    xy = xy.T  # Fortran to Python order

    # sort eigenpairs
    idx = exci.argsort()
    exci = exci[idx]
    xy = xy[idx, :]
    upprpa_orthonormalize_eigenvector(nocc, exci, xy)

    sum_exci = numpy.sum(exci[nocc[0]*nocc[1]:])
    ec = sum_exci - trace_A

    return exci, xy, ec


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
        self.naux = self.Lpq[0].shape[0]
        # chemical potential
        self.mu = None
        # 'n-2' for ppRPA, 'n+2' for hhRPA
        self.nelec = nelec

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

    def dump_flags(self):
        #====================> calculate dimensions <===================
        # (alpha, alpha) subspace
        aavv_dim = int(self.nvir[0] * (self.nvir[0] + 1) / 2)
        aaoo_dim = int(self.nocc[0] * (self.nocc[0] + 1) / 2)
        # (alpha, beta) subspace
        abvv_dim = int(self.nvir[0] * self.nvir[1])
        aboo_dim = int(self.nocc[0] * self.nocc[1])
        # (beta, beta) subspace
        bbvv_dim = int(self.nvir[1] * (self.nvir[1] + 1) / 2)
        bboo_dim = int(self.nocc[1] * (self.nocc[1] + 1) / 2)

        print('\n******** %s ********' % self.__class__)
        print('naux = %d' % self.naux)
        print('nmo = %d (%d alpha, %d beta)' 
              % (self.nmo[0]+self.nmo[1], self.nmo[0], self.nmo[1]))
        print('nocc = %d (%d alpha, %d beta), nvir = %d (%d alpha, %d beta)' 
              % (
                  self.nocc[0] + self.nocc[1], self.nocc[0], self.nocc[1],
                  self.nvir[0] + self.nvir[1], self.nvir[0], self.nvir[1]))
        print('for (alpha alpha, alpha alpha) subspace:')
        print('  occ-occ dimension = %d vir-vir dimension = %d' %
              (aaoo_dim, aavv_dim))
        print('for (alpha beta, alpha beta) subspace:')
        print('  occ-occ dimension = %d vir-vir dimension = %d' %
              (aboo_dim, abvv_dim))
        print('for (beta beta, beta beta) subspace:')
        print('  occ-occ dimension = %d vir-vir dimension = %d' %
              (bboo_dim, bbvv_dim))
        print('interested hh state = %d' % self.hh_state)
        print('interested pp state = %d' % self.pp_state)
        print('ground state = %s' % self.nelec)
        print('print threshold = %.2f%%' % (self.print_thresh*100))
        print('')
        return

    def check_memory(self):
        #====================> calculate dimensions <===================
        # (alpha, alpha) subspace
        aavv_dim = int(self.nvir[0] * (self.nvir[0] + 1) / 2)
        aaoo_dim = int(self.nocc[0] * (self.nocc[0] + 1) / 2)
        aafull_dim = aavv_dim + aaoo_dim
        # (alpha, beta) subspace
        abvv_dim = int(self.nvir[0] * self.nvir[1])
        aboo_dim = int(self.nocc[0] * self.nocc[1])
        abfull_dim = abvv_dim + aboo_dim
        # (beta, beta) subspace
        bbvv_dim = int(self.nvir[1] * (self.nvir[1] + 1) / 2)
        bboo_dim = int(self.nocc[1] * (self.nocc[1] + 1) / 2)
        bbfull_dim = bbvv_dim + bboo_dim

        full_dim = max(aafull_dim, abfull_dim, bbfull_dim)

        mem = (3 * full_dim * full_dim) * 8 / 1.0e6
        if mem < 1000:
            print("ppRPA needs at least %d MB memory." % mem)
        else:
            print("ppRPA needs at least %.1f GB memory." % (mem / 1.0e3))
        return

    def kernel(self):
        self.check_parameter()
        self.dump_flags()
        self.check_memory()
        #start_clock("ppRPA direct")
        #============> (alpha alpha, alpha alpha) subspace <============
        # (alpha beta, alpha beta) subspace
        ab_exci, ab_xy, ab_ec = diagonalize_pprpa_subspace_diff_spin(
            self.nocc, self.mo_energy, self.Lpq, mu=self.mu
        )
        #end("ppRPA direct")
