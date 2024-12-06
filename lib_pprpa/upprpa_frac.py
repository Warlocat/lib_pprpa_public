"""Direct solver for unrestricted particle-particle random phase approximtion
for systems with fractional occupation numbers.
Author: Jincheng Yu <pimetamon@gmail.com>
"""
import h5py
import numpy
import scipy
from lib_pprpa.pprpa_davidson import pprpa_orthonormalize_eigenvector, pprpa_print_a_pair
from lib_pprpa.pprpa_util import get_chemical_potential, start_clock, stop_clock, print_citation, inner_product, get_nocc_nvir_frac
from lib_pprpa.pprpa_direct import diagonalize_pprpa_triplet
from lib_pprpa.upprpa_direct import upprpa_orthonormalize_eigenvector, _analyze_pprpa_direct

def diagonalize_pprpa_subspace_same_spin(
        nocc, nvir, mo_occ, mo_energy, Lpq, mu=None):
    """Diagonalize fractional ppRPA matrix in subspace 
    (alpha alpha, alpha, alpha) or (beta beta, beta beta).

    Args:
        int_nocc (int): number of fully occupied orbitals.
        frac_nocc (int): number of fractionally occupied orbitals.
        frac_occ (array of float): fractional occupation number.
        mo_energy (double array): orbital energies.
        Lpq (double ndarray): three-center RI matrices in MO space.

    Kwarg:
        mu (double): chemical potential.

    Returns:
        exci (double array): ppRPA eigenvalue.
        xy (double ndarray): ppRPA eigenvector.
        ec (double): correlation energy from one subspace.
    """
    nmo = len(mo_energy)
    frac_nocc = nocc + nvir - nmo
    int_nocc = nocc - frac_nocc
    if mu is None:
        mu = get_chemical_potential(nocc, mo_energy)

    oo_dim = int((nocc - 1) * nocc / 2)  # number of hole-hole pairs

    # low triangular index (not including diagonal)
    tri_row_o, tri_col_o = numpy.tril_indices(nocc, -1)
    tri_row_v, tri_col_v = numpy.tril_indices(nvir, -1)

    # 1 - na
    one_m_occ = 1 - mo_occ

    # ===========================> A matrix <===========================
    # (1-na) * (1-nb) * <ab||cd>
    A = numpy.einsum(
        "a,b,Pac,Pbd->abcd", one_m_occ[int_nocc:], one_m_occ[int_nocc:],
        Lpq[:, int_nocc:, int_nocc:], Lpq[:, int_nocc:, int_nocc:], 
        optimize=True)
    A -= numpy.einsum(
        "a,b,Pad,Pbc->abcd", one_m_occ[int_nocc:], one_m_occ[int_nocc:],
        Lpq[:, int_nocc:, int_nocc:], Lpq[:, int_nocc:, int_nocc:], 
        optimize=True)
    # delta_ac delta_bd (e_a + e_b - 2 * mu)
    A = A.reshape(nvir*nvir, nvir*nvir)
    orb_sum = numpy.asarray(mo_energy[int_nocc:, None] \
                            + mo_energy[None, int_nocc:]).reshape(-1)
    orb_sum -= 2.0 * mu
    numpy.fill_diagonal(A, A.diagonal() + orb_sum)
    A = A.reshape(nvir, nvir, nvir, nvir)

    A = A[tri_row_v, tri_col_v, ...]
    A = A[..., tri_row_v, tri_col_v]

    trace_A = numpy.trace(A)

    # ===========================> B matrix <===========================
    # (1-na) * (1-nb) * <ab||ij>
    B = numpy.einsum(
        "a,b,Pai,Pbj->abij", one_m_occ[int_nocc:], one_m_occ[int_nocc:],
        Lpq[:, int_nocc:, :nocc], Lpq[:, int_nocc:, :nocc], optimize=True)
    B -= numpy.einsum(
        "a,b,Paj,Pbi->abij", one_m_occ[int_nocc:], one_m_occ[int_nocc:],
        Lpq[:, int_nocc:, :nocc], Lpq[:, int_nocc:, :nocc], optimize=True)
    B = B[tri_row_v, tri_col_v, ...]
    B = B[..., tri_row_o, tri_col_o]

    # ==========================> B' matrix <===========================
    # ni * nj * <ij||ab>
    Bp = numpy.einsum(
        "i,j,Pia,Pjb->ijab", mo_occ[:nocc], mo_occ[:nocc],
        Lpq[:, :nocc, int_nocc:], Lpq[:, :nocc, int_nocc:], optimize=True)
    Bp -= numpy.einsum(
        "a,b,Paj,Pbi->abij", mo_occ[:nocc], mo_occ[:nocc],
        Lpq[:, :nocc, int_nocc:], Lpq[:, :nocc, int_nocc:], optimize=True)
    Bp = Bp[tri_row_o, tri_col_o, ...]
    Bp = Bp[..., tri_row_v, tri_col_v]

    # ===========================> C matrix <===========================
    # ni * nj * <ij||kl>
    C = numpy.einsum(
        "i,j,Pik,Pjl->ijkl", mo_occ[:nocc], mo_occ[:nocc],
        Lpq[:, :nocc, :nocc], Lpq[:, :nocc, :nocc], optimize=True)
    C -= numpy.einsum(
        "i,j,Pil,Pjk->ijkl", mo_occ[:nocc], mo_occ[:nocc],
        Lpq[:, :nocc, :nocc], Lpq[:, :nocc, :nocc], optimize=True)
    # delta_ik delta_jl (e_i + e_j - 2 * mu)
    C = C.reshape(nocc*nocc, nocc*nocc)
    orb_sum = numpy.asarray(mo_energy[:nocc, None] \
                            + mo_energy[None, :nocc]).reshape(-1)
    orb_sum -= 2.0 * mu
    numpy.fill_diagonal(C, C.diagonal() - orb_sum)
    C = C.reshape(nocc, nocc, nocc, nocc)
    # take only low-triangular part
    C = C[tri_row_o, tri_col_o, ...]
    C = C[..., tri_row_o, tri_col_o]

    # =================> whole matrix in the subspace<==================
    # C     B'
    # B     A
    M_upper = numpy.concatenate((C, Bp), axis=1)
    M_lower = numpy.concatenate((B, A), axis=1)
    M = numpy.concatenate((M_upper, M_lower), axis=0)
    del A, B, Bp, C
    # M to WM, where W is the metric matrix [[-I, 0], [0, I]]
    M[:oo_dim][:] *= -1.0

    # ====================> solve for eigenpairs <======================
    exci, xy = scipy.linalg.eig(M)
    exci = exci.real
    xy = xy.T  # Fortran to Python order

    # sort eigenpairs
    idx = exci.argsort()
    exci = exci[idx]
    xy = xy[idx, :]
    upprpa_orthonormalize_eigenvector('t', nocc, exci, xy)

    sum_exci = numpy.sum(exci[oo_dim:])
    ec = sum_exci - trace_A

    return exci, xy, ec

def diagonalize_pprpa_subspace_diff_spin(nocc, nvir, mo_occ, mo_energy, 
                                         Lpq, mu=None):
    """Diagonalize fractional ppRPA matrix in subspace (alpha beta, alpha beta).

    Args:
        int_nocc (tuple of int): number of fully occupied orbitals (alpha, beta)
        frac_nocc (tuple of int): number of fractionally occupied orbitals
            (alpha, beta).
        frac_occ (double ndarray): fractional occupation numbers.
        mo_energy(double ndarray): orbital energies.
        Lpq(double ndarray): three-center RI matrices in MO space.

    Kwarg:
        mu (double): chemical potential.

    Returns:
        exci (double array): ppRPA eigenvalue.
        xy (double ndarray): ppRPA eigenvector.
        ec (double): correlation energy from one subspace.
    """
    nmo = (len(mo_energy[0]), len(mo_energy[1]))
    frac_nocc = (nocc[0] + nvir[0] - nmo[0],
                 nocc[1] + nvir[1] - nmo[1])
    int_nocc = (nocc[0] - frac_nocc[0], nocc[1] - frac_nocc[1])
    if mu is None:
        mu = get_chemical_potential(nocc, mo_energy)

    # 1 - occ
    one_m_occ = [1 - mo_occ[0], 1 - mo_occ[1]]

    # ===========================> A matrix <===========================
    # (1-na) * (1-nb) * <ab|cd>
    A = numpy.einsum(
        "a,b,Pac,Pbd->abcd", 
        one_m_occ[0][int_nocc[0]:], one_m_occ[1][int_nocc[1]:],
        Lpq[0][:, int_nocc[0]:, int_nocc[0]:], 
        Lpq[1][:, int_nocc[1]:, int_nocc[1]:], optimize=True)
    # delta_ac delta_bd (e_a + e_b - 2 * mu)
    A = A.reshape(nvir[0]*nvir[1], nvir[0]*nvir[1])
    orb_sum = numpy.asarray(mo_energy[0][int_nocc[0]:, None] \
        + mo_energy[1][None, int_nocc[1]:]).reshape(-1)
    orb_sum -= 2.0 * mu
    numpy.fill_diagonal(A, A.diagonal() + orb_sum)

    trace_A = numpy.trace(A)

    # ===========================> B matrix <===========================
    # (1-na) * (1-nb) * <ab|ij>
    B = numpy.einsum(
        "a,b,Pai,Pbj->abij", 
        one_m_occ[0][int_nocc[0]:], one_m_occ[1][int_nocc[1]:],
        Lpq[0][:, int_nocc[0]:, :nocc[0]], 
        Lpq[1][:, int_nocc[1]:, :nocc[1]], optimize=True)
    B = B.reshape(nvir[0]*nvir[1], nocc[0]*nocc[1])

    # ==========================> B' matrix <===========================
    # ni * nj * <ij|ab>
    Bp = numpy.einsum(
        "i,j,Pia,Pjb->ijab", mo_occ[0][:nocc[0]], mo_occ[1][:nocc[1]],
        Lpq[0][:, :nocc[0], int_nocc[0]:], 
        Lpq[1][:, :nocc[1], int_nocc[1]:], optimize=True)
    Bp = Bp.reshape(nocc[0]*nocc[1], nvir[0]*nvir[1])

    # ===========================> C matrix <===========================
    # ni * nj * <ij|kl>
    C = numpy.einsum(
        'i,j,Pik,Pjl->ijkl', mo_occ[0][:nocc[0]], mo_occ[1][:nocc[1]],
        Lpq[0][:, :nocc[0], :nocc[0]],
        Lpq[1][:, :nocc[1], :nocc[1]], optimize=True)
    # delta_ik delta_jl (e_i + e_j - 2 * mu)
    C = C.reshape(nocc[0]*nocc[1], nocc[0]*nocc[1])
    orb_sum = numpy.asarray(mo_energy[0][:nocc[0], None] \
        + mo_energy[1][None, :nocc[1]]).reshape(-1)
    orb_sum -= 2.0 * mu
    numpy.fill_diagonal(C, C.diagonal() - orb_sum)

    # =================> whole matrix in the subspace<==================
    # C     B'
    # B     A
    M_upper = numpy.concatenate((C, Bp), axis=1)
    M_lower = numpy.concatenate((B, A), axis=1)
    M = numpy.concatenate((M_upper, M_lower), axis=0)
    del A, B, Bp, C
    # M to WM, where W is the metric matrix [[-I, 0], [0, I]]
    M[:nocc[0]*nocc[1]][:] *= -1.0

    # ====================> solve for eigenpairs <======================
    exci, xy = scipy.linalg.eig(M)
    exci = exci.real
    xy = xy.T  # Fortran to Python order

    # sort eigenpairs
    idx = exci.argsort()
    exci = exci[idx]
    xy = xy[idx, :]
    upprpa_orthonormalize_eigenvector('abab', nocc, exci, xy)

    sum_exci = numpy.sum(exci[nocc[0]*nocc[1]:])
    ec = sum_exci - trace_A

    return exci, xy, ec

class FppRPADirect():
    """Direct solver class for unrestricted ppPRA for systems with 
    fractional occupation numbers.
    """
    def __init__(
            self, mo_occ, mo_energy, Lpq, hh_state=5, pp_state=5, 
            nelec='n-2', print_thresh=0.1):
        #self.mo_occ = numpy.asarray(mo_occ)
        self.mo_occ = mo_occ
        #self.mo_energy = numpy.asarray(mo_energy)
        self.mo_energy = mo_energy
        #self.Lpq = numpy.asarray(Lpq)
        self.Lpq = Lpq
        self.hh_state = hh_state
        self.pp_state = pp_state
        self.print_thresh = print_thresh

        # ======================> internal flags <=======================
        # number of orbitals
        self.nmo = (len(self.mo_energy[0]), len(self.mo_energy[1]))
        # number of occupied (fully occ + frac occ), 
        #   virtual (frac occ + truly unocc), and fractionally occupied orbitals
        self.nocc, self.nvir, self.frac_nocc = get_nocc_nvir_frac(self.mo_occ)
        # number of auxiliary basis functions
        self.naux = self.Lpq[0].shape[0]
        # chemical potential
        self.mu = None
        # 'n-2' for ppRPA, 'n+2' for hhRPA
        self.nelec = nelec

        # =========================> results <==========================
        self.ec = None  # correlation energy [aaaa, bbbb, abab]
        self.exci = None  # two-electron addition energy [aaaa, bbbb, abab]
        self.xy = None  # ppRPA eigenvector [aaaa, bbbb, abab]

        # ===================> methods from UppRPA <====================

        print_citation()

        return

    def check_parameter(self):
        assert 0.0 < self.print_thresh < 1.0
        assert self.nelec in ['n-2', 'n+2']
        if self.mu is None:
            self.mu = get_chemical_potential(nocc=self.nocc,
                                             mo_energy=self.mo_energy)
        return

    def dump_flags(self):
        # ====================> calculate dimensions <===================
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
        print('N(fully occupied orbitals) = %d (%d alpha, %d beta)' % (
            self.nocc[0] + self.nocc[1] - self.frac_nocc[0] - self.frac_nocc[1], 
            self.nocc[0] - self.frac_nocc[0], self.nocc[1] - self.frac_nocc[1]
        ))
        print('N(fractionally occupied orbitals) = %d (%d alpha, %d beta)' % (
            self.frac_nocc[0] + self.frac_nocc[1], 
            self.frac_nocc[0], self.frac_nocc[1]
        ))
        print('N(zero-occupied orbitals) = %d (%d alpha, %d beta)' % (
            self.nvir[0] + self.nvir[1] - self.frac_nocc[0] - self.frac_nocc[1], 
            self.nvir[0] - self.frac_nocc[0], self.nvir[1] - self.frac_nocc[1]
        ))
        print('for (alpha alpha, alpha alpha) subspace:')
        print('  occ-occ dimension = %d vir-vir dimension = %d' %
              (aaoo_dim, aavv_dim))
        print('for (beta beta, beta beta) subspace:')
        print('  occ-occ dimension = %d vir-vir dimension = %d' %
              (bboo_dim, bbvv_dim))
        print('for (alpha beta, alpha beta) subspace:')
        print('  occ-occ dimension = %d vir-vir dimension = %d' %
              (aboo_dim, abvv_dim))
        print('interested hh state = %d' % self.hh_state)
        print('interested pp state = %d' % self.pp_state)
        print('ground state = %s' % self.nelec)
        print('print threshold = %.2f%%' % (self.print_thresh*100))
        print('')
        return

    def check_memory(self):
        # ====================> calculate dimensions <===================
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
            print("Frac ppRPA needs at least %d MB memory." % mem)
        else:
            print("Frac ppRPA needs at least %.1f GB memory." % (mem / 1.0e3))
        return

    def kernel(self, subspace=['aa', 'bb', 'ab']):
        self.check_parameter()
        self.dump_flags()
        self.check_memory()

        if 'aa' in subspace:
            start_clock("Frac ppRPA direct: (alpha alpha, alpha alpha)")
            aa_exci, aa_xy, aa_ec = diagonalize_pprpa_subspace_same_spin(
                self.nocc[0], self.nvir[0], self.mo_occ[0],
                self.mo_energy[0], self.Lpq[0], mu=self.mu
            )
            stop_clock("Frac ppRPA direct: (alpha alpha, alpha alpha)")
        else:
            aa_exci = aa_xy = aa_ec = None

        if 'bb' in subspace:
            start_clock("Frac ppRPA direct: (beta beta, beta beta)")
            bb_exci, bb_xy, bb_ec = diagonalize_pprpa_subspace_same_spin(
                self.nocc[1], self.nvir[1], self.mo_occ[1],
                self.mo_energy[1], self.Lpq[1], mu=self.mu
            )
            stop_clock("Frac ppRPA direct: (beta beta, beta beta)")
        else:
            bb_exci = bb_xy = bb_ec = None

        if 'ab' in subspace:
            start_clock("Frac ppRPA direct: (alpha beta, alpha beta)")
            ab_exci, ab_xy, ab_ec = diagonalize_pprpa_subspace_diff_spin(
                self.nocc, self.nvir, self.mo_occ, self.mo_energy, 
                self.Lpq, mu=self.mu
            )
            stop_clock("Frac ppRPA direct: (alpha beta, alpha beta)")
        else:
            ab_exci = ab_xy = ab_ec = None

        self.ec = [aa_ec, bb_ec, ab_ec]
        self.exci = [aa_exci, bb_exci, ab_exci]
        self.xy = [aa_xy, bb_xy, ab_xy]

        return

    def analyze(self, nocc_fro=(0, 0)):
        _analyze_pprpa_direct(
            self.exci, self.xy, self.nocc, self.nvir, nelec=self.nelec,
            print_thresh=self.print_thresh, hh_state=self.hh_state,
            pp_state=self.pp_state, nocc_fro=nocc_fro)
        return
