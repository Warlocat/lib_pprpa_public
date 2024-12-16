"""Direct solver for unrestricted particle-particle random phase approximation.
Author: Jincheng Yu <pimetamon@gmail.com>
"""
import h5py
import numpy
import scipy

from lib_pprpa.analyze import pprpa_print_a_pair
from lib_pprpa.pprpa_davidson import pprpa_orthonormalize_eigenvector
from lib_pprpa.pprpa_util import get_chemical_potential, start_clock, stop_clock, print_citation, inner_product
from lib_pprpa.pprpa_direct import diagonalize_pprpa_triplet


# TODO: move this function to orthonormalize.py
def upprpa_orthonormalize_eigenvector(subspace, nocc, exci, xy):
    """Orthonormalize U-ppRPA eigenvectors.
    The eigenvector is normalized as Y^2 - X^2 = 1.
    This function will rewrite input exci and xy, after calling this function,
    exci and xy will be re-ordered as [hole-hole, particle-particle].

    Args:
        subspace (str): subspace, 'aaaa', 'bbbb', or 'abab'.
        nocc (int/tuple of int): number of occupied orbitals.
        exci (double array): ppRPA eigenvalue.
        xy (double ndarray): ppRPA eigenvector.
    """
    nroot = xy.shape[0]

    if subspace == 'abab':
        oo_dim = int(nocc[0] * nocc[1])

        # determine the vector is pp or hh
        sig = numpy.zeros(shape=[nroot], dtype=numpy.double)
        for i in range(nroot):
            sig[i] = 1 if inner_product(xy[i].conj(), xy[i], oo_dim) > 0 else -1

        # eliminate parallel component
        for i in range(nroot):
            for j in range(i):
                if abs(exci[i] - exci[j]) < 1.0e-7:
                    norm_j = inner_product(xy[j].conj(), xy[j], oo_dim)
                    inp = inner_product(xy[i].conj(), xy[j], oo_dim)/ norm_j
                    xy[i] -= xy[j] * inp

        # normalize
        for i in range(nroot):
            inp = inner_product(xy[i].conj(), xy[i], oo_dim)
            inp = numpy.sqrt(abs(inp))
            xy[i] /= inp

        # re-order all states by signs, first hh then pp
        hh_index = numpy.where(sig < 0)[0]
        pp_index = numpy.where(sig > 0)[0]
        exci_hh = exci[hh_index]
        exci_pp = exci[pp_index]
        exci[:len(hh_index)] = exci_hh
        exci[len(hh_index):] = exci_pp
        xy_hh = xy[hh_index]
        xy_pp = xy[pp_index]
        xy[:len(hh_index)] = xy_hh
        xy[len(hh_index):] = xy_pp

        # change |X -Y> to |X Y>
        xy[:][:oo_dim] *= -1

    else:
        pprpa_orthonormalize_eigenvector('t', nocc, exci, xy)

    return


def diagonalize_pprpa_subspace_same_spin(nocc, mo_energy, Lpq, mu=None):
    """Diagonalize ppRPA matrix in subspace (alpha alpha, alpha, alpha)
    or (beta beta, beta beta).

    See function `lib_pprpa.pprpa_direct.diagonalize_pprpa_triplet`.

    """
    exci, xy, ec = diagonalize_pprpa_triplet(nocc, mo_energy, Lpq, mu=mu)

    return exci, xy, ec/3.0


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

    # ===========================> A matrix <============================
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

    # ===========================> B matrix <============================
    # <ab|ij>
    B = numpy.einsum(
        'Pai,Pbj->abij', Lpq[0][:, nocc[0]:, :nocc[0]],
        Lpq[1][:, nocc[1]:, :nocc[1]], optimize=True)
    B = B.reshape(nvir[0]*nvir[1], nocc[0]*nocc[1])

    # ===========================> C matrix <============================
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

    # ==================> whole matrix in the subspace<==================
    # C    B^T
    # B     A
    M_upper = numpy.concatenate((C, B.T), axis=1)
    M_lower = numpy.concatenate((B, A), axis=1)
    M = numpy.concatenate((M_upper, M_lower), axis=0)
    del A, B, C
    # M to WM, where W is the metric matrix [[-I, 0], [0, I]]
    M[:nocc[0]*nocc[1]][:] *= -1.0

    # =====================> solve for eigenpairs <======================
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


def _pprpa_print_eigenvector(subspace, nocc, nvir, nocc_fro, thresh, hh_state,
                             pp_state, exci0, exci, xy):
    """Print components of an eigenvector.

    Args:
        subspace (str): subspace, 'aaaa', 'bbbb', or 'abab'.
        nocc (int/tuple of int): number of occupied orbitals.
        nvir (int/tuple of int): number of virtual orbitals.
        nocc_fro (int/tuple of int): number of frozen occupied orbitals.
        thresh (double): threshold to print a pair.
        hh_state (int): number of interested hole-hole states.
        pp_state (int): number of interested particle-particle states.
        exci0 (double): lowest eigenvalue.
        exci (double array): ppRPA eigenvalue.
        xy (double ndarray): ppRPA eigenvector.
    """
    if subspace == 'aaaa':
        oo_dim = int((nocc - 1) * nocc / 2)
        vv_dim = int((nvir - 1) * nvir / 2)
        print("\n     print U-ppRPA excitations: (alpha alpha, alpha alpha)\n")
    elif subspace == 'bbbb':
        oo_dim = int((nocc - 1) * nocc / 2)
        vv_dim = int((nvir - 1) * nvir / 2)
        print("\n     print U-ppRPA excitations: (beta beta, beta beta)\n")
    elif subspace == 'abab':
        oo_dim = int(nocc[0] * nocc[1])
        vv_dim = int(nvir[0] * nvir[1])
        print("\n     print U-ppRPA excitations: (alpha beta, alpha beta)\n")
    else:
        raise ValueError("Not recognized subspace: %s." % subspace)

    if subspace == 'aaaa' or subspace == 'bbbb':
        tri_row_o, tri_col_o = numpy.tril_indices(nocc, -1)
        tri_row_v, tri_col_v = numpy.tril_indices(nvir, -1)

    au2ev = 27.211386

    # =====================> two-electron removal <======================
    for istate in range(min(hh_state, oo_dim)):
        print("#%-d %s de-excitation:  exci= %-12.6f  eV   2e=  %-12.6f  eV" %
              (istate + 1, subspace[:2], (exci[oo_dim-istate-1] - exci0) * au2ev,
               exci[oo_dim-istate-1] * au2ev))
        if subspace == 'aaaa' or subspace == 'bbbb':
            full = numpy.zeros(shape=[nocc, nocc], dtype=numpy.double)
            full[tri_row_o, tri_col_o] = xy[oo_dim-istate-1][:oo_dim]
            full = numpy.power(full, 2)
            pairs = numpy.argwhere(full > thresh)
            for i, j in pairs:
                pprpa_print_a_pair(is_pp=False, p=i+nocc_fro, q=j+nocc_fro,
                                   percentage=full[i, j])

            full = numpy.zeros(shape=[nvir, nvir], dtype=numpy.double)
            full[tri_row_v, tri_col_v] = xy[oo_dim-istate-1][oo_dim:]
            full = numpy.power(full, 2)
            pairs = numpy.argwhere(full > thresh)
            for a, b in pairs:
                pprpa_print_a_pair(is_pp=True, p=a+nocc_fro+nocc,
                                   q=b+nocc_fro+nocc, percentage=full[a, b])

        else:
            full = xy[oo_dim-istate-1][:oo_dim].reshape(nocc[0], nocc[1])
            full = numpy.power(full, 2)
            pairs = numpy.argwhere(full > thresh)
            for i, j in pairs:
                pprpa_print_a_pair(is_pp=False, p=i+nocc_fro[0], q=j+nocc_fro[1],
                                   percentage=full[i, j])

            full = xy[oo_dim-istate-1][oo_dim:].reshape(nvir[0], nvir[1])
            full = numpy.power(full, 2)
            pairs = numpy.argwhere(full > thresh)
            for a, b in pairs:
                pprpa_print_a_pair(is_pp=True, p=a+nocc_fro[0]+nocc[0],
                                   q=b+nocc_fro[1]+nocc[1], percentage=full[a, b])
        print("")

    # =====================> two-electron addition <=====================
    for istate in range(min(pp_state, vv_dim)):
        print("#%-d %s excitation:  exci= %-12.6f  eV   2e=  %-12.6f  eV" %
              (istate + 1, subspace[:2], (exci[oo_dim+istate] - exci0) * au2ev,
               exci[oo_dim+istate] * au2ev))
        if subspace == 'aaaa' or subspace == 'bbbb':
            full = numpy.zeros(shape=[nocc, nocc], dtype=numpy.double)
            full[tri_row_o, tri_col_o] = xy[oo_dim+istate][:oo_dim]
            full = numpy.power(full, 2)
            pairs = numpy.argwhere(full > thresh)
            for i, j in pairs:
                pprpa_print_a_pair(is_pp=False, p=i+nocc_fro, q=j+nocc_fro,
                                   percentage=full[i, j])

            full = numpy.zeros(shape=[nvir, nvir], dtype=numpy.double)
            full[tri_row_v, tri_col_v] = xy[oo_dim+istate][oo_dim:]
            full = numpy.power(full, 2)
            pairs = numpy.argwhere(full > thresh)
            for a, b in pairs:
                pprpa_print_a_pair(is_pp=True, p=a+nocc_fro+nocc,
                                   q=b+nocc_fro+nocc, percentage=full[a, b])

        else:
            full = xy[oo_dim+istate][:oo_dim].reshape(nocc[0], nocc[1])
            full = numpy.power(full, 2)
            pairs = numpy.argwhere(full > thresh)
            for i, j in pairs:
                pprpa_print_a_pair(
                    is_pp=False, p=i+nocc_fro[0], q=j+nocc_fro[1],
                    percentage=full[i, j])

            full = xy[oo_dim+istate][oo_dim:].reshape(nvir[0], nvir[1])
            full = numpy.power(full, 2)
            pairs = numpy.argwhere(full > thresh)
            for a, b in pairs:
                pprpa_print_a_pair(
                    is_pp=True, p=a+nocc_fro[0]+nocc[0],
                    q=b+nocc_fro[1]+nocc[1], percentage=full[a, b])
        print("")

    return


def _analyze_pprpa_direct(
        exci, xy, nocc, nvir, nelec='n-2', print_thresh=0.1, hh_state=5,
        pp_state=5, nocc_fro=0):
    print('\nanalyze U-ppRPA results.')
    oo_dim_aa = int((nocc[0] - 1) * nocc[0] / 2)
    oo_dim_bb = int((nocc[1] - 1) * nocc[1] / 2)
    oo_dim_ab = int(nocc[0] * nocc[1])

    exci_aa = exci[0]
    exci_bb = exci[1]
    exci_ab = exci[2]

    exci0_list = []
    if exci_aa is not None:
        print('(alpha alpha, alpha alpha) results found.')
        if nelec == 'n-2':
            exci0_list.append(exci_aa[oo_dim_aa])
        else:
            exci0_list.append(exci_aa[oo_dim_aa - 1])
    if exci_bb is not None:
        print('(beta beta, beta beta) results found.')
        if nelec == 'n-2':
            exci0_list.append(exci_bb[oo_dim_bb])
        else:
            exci0_list.append(exci_bb[oo_dim_bb - 1])
    if exci_ab is not None:
        print('(alpha beta, alpha beta) results found.')
        if nelec == 'n-2':
            exci0_list.append(exci_ab[oo_dim_ab])
        else:
            exci0_list.append(exci_ab[oo_dim_ab])

    if nelec == 'n-2':
        exci0 = min(exci0_list)
    else:
        exci0 = max(exci0_list)

    if exci_aa is not None:
        _pprpa_print_eigenvector(
            'aaaa', nocc[0], nvir[0], nocc_fro[0], print_thresh, hh_state,
            pp_state, exci0, exci_aa, xy[0])
    if exci_bb is not None:
        _pprpa_print_eigenvector(
            'bbbb', nocc[1], nvir[1], nocc_fro[1], print_thresh, hh_state,
            pp_state, exci0, exci_bb, xy[1])
    if exci_ab is not None:
        _pprpa_print_eigenvector(
            'abab', nocc, nvir, nocc_fro, print_thresh, hh_state,
            pp_state, exci0, exci_ab, xy[2])

    pass


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

        # ======================> internal flags <=======================
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

        # =========================> results <===========================
        self.ec = None  # correlation energy [aaaa, bbbb, abab]
        self.exci = None  # two-electron addition energy [aaaa, bbbb, abab]
        self.xy = None  # ppRPA eigenvector [aaaa, bbbb, abab]

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
        print('nocc = %d (%d alpha, %d beta), nvir = %d (%d alpha, %d beta)'
              % (
                  self.nocc[0] + self.nocc[1], self.nocc[0], self.nocc[1],
                  self.nvir[0] + self.nvir[1], self.nvir[0], self.nvir[1]))
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
            print("U-ppRPA needs at least %d MB memory." % mem)
        else:
            print("U-ppRPA needs at least %.1f GB memory." % (mem / 1.0e3))
        return

    def kernel(self, subspace=['aa', 'bb', 'ab']):
        self.check_parameter()
        self.dump_flags()
        self.check_memory()

        if 'aa' in subspace:
            start_clock("U-ppRPA direct: (alpha alpha, alpha alpha)")
            aa_exci, aa_xy, aa_ec = diagonalize_pprpa_subspace_same_spin(
                self.nocc[0], self.mo_energy[0], self.Lpq[0], mu=self.mu
            )
            stop_clock("U-ppRPA direct: (alpha alpha, alpha alpha)")
        else:
            aa_exci = aa_xy = aa_ec = None

        if 'bb' in subspace:
            start_clock("U-ppRPA direct: (beta beta, beta beta)")
            bb_exci, bb_xy, bb_ec = diagonalize_pprpa_subspace_same_spin(
                self.nocc[1], self.mo_energy[1], self.Lpq[1], mu=self.mu
            )
            stop_clock("U-ppRPA direct: (beta beta, beta beta)")
        else:
            bb_exci = bb_xy = bb_ec = None

        if 'ab' in subspace:
            start_clock("U-ppRPA direct: (alpha beta, alpha beta)")
            ab_exci, ab_xy, ab_ec = diagonalize_pprpa_subspace_diff_spin(
                self.nocc, self.mo_energy, self.Lpq, mu=self.mu
            )
            stop_clock("U-ppRPA direct: (alpha beta, alpha beta)")
        else:
            ab_exci = ab_xy = ab_ec = None

        self.ec = [aa_ec, bb_ec, ab_ec]
        self.exci = [aa_exci, bb_exci, ab_exci]
        self.xy = [aa_xy, bb_xy, ab_xy]

        return

    def save_pprpa(self, fn):
        assert self.exci is not None
        print('\nsave U-ppRPA results to %s.\n' % fn)
        f = h5py.File(fn, 'w')
        f['nocc_alpha'] = numpy.asarray(self.nocc[0])
        f['nocc_beta'] = numpy.asarray(self.nocc[1])
        f['nvir_alpha'] = numpy.asarray(self.nvir[0])
        f['nvir_beta'] = numpy.asarray(self.nvir[1])

        f['exci_aaaa'] = numpy.asarray(self.exci[0])
        f['xy_aaaa'] = numpy.asarray(self.xy[0])
        f['exci_bbbb'] = numpy.asarray(self.exci[1])
        f['xy_bbbb'] = numpy.asarray(self.xy[1])
        f['exci_abab'] = numpy.asarray(self.exci[2])
        f['xy_abab'] = numpy.asarray(self.xy[2])

        f.close()
        return

    def read_pprpa(self, fn):
        if self.exci is None:
            self.exci = [None, None, None]
        if self.xy is None:
            self.xy = [None, None, None]
        print('\nread U-ppRPA results from %s.\n' % fn)
        f = h5py.File(fn, 'r')
        if "exci_aaaa" in f.keys():
            self.exci[0] = numpy.asarray(f['exci_aaaa'])
            self.xy[0] = numpy.asarray(f['xy_aaaa'])
        if "exci_bbbb" in f.keys():
            self.exci[1] = numpy.asarray(f['exci_bbbb'])
            self.xy[1] = numpy.asarray(f['xy_bbbb'])
        if "exci_abab" in f.keys():
            self.exci[2] = numpy.asarray(f['exci_abab'])
            self.xy[2] = numpy.asarray(f['xy_abab'])
        f.close()
        return

    def analyze(self, nocc_fro=(0, 0)):
        _analyze_pprpa_direct(
            self.exci, self.xy, self.nocc, self.nvir, nelec=self.nelec,
            print_thresh=self.print_thresh, hh_state=self.hh_state,
            pp_state=self.pp_state, nocc_fro=nocc_fro)
        return
