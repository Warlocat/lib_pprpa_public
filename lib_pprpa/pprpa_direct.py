import h5py
import numpy
import scipy

from numpy import einsum

from lib_pprpa.pprpa_davidson import pprpa_orthonormalize_eigenvector, pprpa_print_a_pair
from lib_pprpa.pprpa_util import get_chemical_potential, start_clock, stop_clock, print_citation


def diagonalize_pprpa_singlet(nocc, mo_energy, Lpq, mu=None):
    """Diagonalize singlet ppRPA matrix.

    Reference:
    [1] https://doi.org/10.1063/1.4828728

    Args:
        nocc (int): number of occupied orbitals.
        mo_energy (double array): orbital energy.
        Lpq (double ndarray): three-center density-fitting matrix in MO space.
        mu (double, optional): chemical potential. Defaults to None.

    Returns:
        exci (double array): ppRPA eigenvalue.
        xy (double ndarray): ppRPA eigenvector.
        ec (double): singlet correlation energy.
    """
    nmo = len(mo_energy)
    nvir = nmo - nocc
    if mu is None:
        mu = get_chemical_potential(nocc=nocc, mo_energy=mo_energy)

    oo_dim = int((nocc + 1) * nocc / 2)  # number of hole-hole pairs

    # low triangular index (including diagonal)
    tri_row_o, tri_col_o = numpy.tril_indices(nocc)
    tri_row_v, tri_col_v = numpy.tril_indices(nvir)

    # A matrix: particle-particle block
    # two-electron integral part, <ab|cd>+<ab|dc>
    A = einsum("Pac,Pbd->abcd", Lpq[:, nocc:, nocc:], Lpq[:, nocc:, nocc:], optimize=True)
    A += einsum("Pad,Pbc->abcd", Lpq[:, nocc:, nocc:], Lpq[:, nocc:, nocc:], optimize=True)
    # scale the diagonal elements
    A[numpy.diag_indices(nvir)] *= 1.0 / numpy.sqrt(2)  # a=b
    A = A.transpose(2, 3, 0, 1)  # A_{ab,cd} to A_{cd,ab}
    A[numpy.diag_indices(nvir)] *= 1.0 / numpy.sqrt(2)  # c=d
    A = A.transpose(2, 3, 0, 1)  # A_{cd,ab} to A_{ab,cd}
    # orbital energy part
    A = A.reshape(nvir*nvir, nvir*nvir)
    orb_sum = numpy.asarray(mo_energy[nocc:, None] + mo_energy[None, nocc:]).reshape(-1)
    orb_sum -= 2.0 * mu
    numpy.fill_diagonal(A, A.diagonal() + orb_sum)
    A = A.reshape(nvir, nvir, nvir, nvir)
    # take only low-triangular part
    A = A[tri_row_v, tri_col_v, ...]
    A = A[..., tri_row_v, tri_col_v]
    trace_A = numpy.trace(A)

    # B matrix: particle-hole block
    # two-electron integral part, <ab|ij>+<ab|ji>
    B = einsum("Pai,Pbj->abij", Lpq[:, nocc:, :nocc], Lpq[:, nocc:, :nocc], optimize=True)
    B += einsum("Paj,Pbi->abij", Lpq[:, nocc:, :nocc], Lpq[:, nocc:, :nocc], optimize=True)
    # scale the diagonal elements
    B[numpy.diag_indices(nvir)] *= 1.0 / numpy.sqrt(2)  # a=b
    B = B.transpose(2, 3, 0, 1)  # B_{ab,ij} to B_{ij,ab}
    B[numpy.diag_indices(nocc)] *= 1.0 / numpy.sqrt(2)  # i=j
    B = B.transpose(2, 3, 0, 1)  # B_{ij,ab} to B_{ab,ij}
    # take only low-triangular part
    B = B[tri_row_v, tri_col_v, ...]
    B = B[..., tri_row_o, tri_col_o]

    # C matrix: hole-hole block
    # two-electron integral part, <ij|kl>+<ij|lk>
    C = einsum("Pik,Pjl->ijkl", Lpq[:, :nocc, :nocc], Lpq[:, :nocc, :nocc], optimize=True)
    C += einsum("Pil,Pjk->ijkl", Lpq[:, :nocc, :nocc], Lpq[:, :nocc, :nocc], optimize=True)
    # scale the diagonal elements
    C[numpy.diag_indices(nocc)] *= 1.0 / numpy.sqrt(2)  # i=j
    C = C.transpose(2, 3, 0, 1)  # C_{ij,kl} to C_{kl,ij}
    C[numpy.diag_indices(nocc)] *= 1.0 / numpy.sqrt(2)  # k=l
    C = C.transpose(2, 3, 0, 1)  # C_{kl,ij} to C_{ij,kl}
    # orbital energy part
    C = C.reshape(nocc*nocc, nocc*nocc)
    orb_sum = numpy.asarray(mo_energy[:nocc, None] + mo_energy[None, :nocc]).reshape(-1)
    orb_sum -= 2.0 * mu
    numpy.fill_diagonal(C, C.diagonal() - orb_sum)
    C = C.reshape(nocc, nocc, nocc, nocc)
    # take only low-triangular part
    C = C[tri_row_o, tri_col_o, ...]
    C = C[..., tri_row_o, tri_col_o]

    # combine A, B and C matrix as
    # | C B^T |
    # |B A|
    M_upper = numpy.concatenate((C, B.T), axis=1)
    M_lower = numpy.concatenate((B, A), axis=1)
    M = numpy.concatenate((M_upper, M_lower), axis=0)
    del A, B, C
    # M to MW, W is the metric matrix [[-I,0],[0,I]]
    M[:oo_dim][:] *= -1.0

    # diagonalize ppRPA matrix
    exci, xy = scipy.linalg.eig(M)
    exci = exci.real
    xy = xy.T  # Fortran to Python order
    xy = numpy.asarray(list(x for _, x in sorted(zip(exci, xy), reverse=False)))
    exci = numpy.sort(exci)

    pprpa_orthonormalize_eigenvector(multi="s", nocc=nocc, TDA=None, exci=exci, xy=xy)

    sum_exci = numpy.sum(exci[oo_dim:])
    ec = sum_exci - trace_A

    return exci, xy, ec


def diagonalize_pprpa_triplet(nocc, mo_energy, Lpq, mu=None):
    """Diagonalize triplet ppRPA matrix.

    Reference:
    [1] https://doi.org/10.1063/1.4828728

    Args:
        nocc (int): number of occupied orbitals.
        mo_energy (double array): orbital energy.
        Lpq (double ndarray): three-center density-fitting matrix in MO space.
        mu (double, optional): chemical potential. Defaults to None.

    Returns:
        exci (double array): ppRPA eigenvalue.
        xy (double ndarray): ppRPA eigenvector.
        ec (double): triplet correlation energy, with a factor of 3.
    """
    nmo = len(mo_energy)
    nvir = nmo - nocc
    if mu is None:
        mu = get_chemical_potential(nocc=nocc, mo_energy=mo_energy)

    oo_dim = int((nocc - 1) * nocc / 2)  # number of hole-hole pairs

    # low triangular index (not including diagonal)
    tri_row_o, tri_col_o = numpy.tril_indices(nocc, -1)
    tri_row_v, tri_col_v = numpy.tril_indices(nvir, -1)

    # A matrix: particle-particle block
    # two-electron integral part, <ab|cd>-<ab|dc>
    A = einsum("Pac,Pbd->abcd", Lpq[:, nocc:, nocc:], Lpq[:, nocc:, nocc:], optimize=True)
    A -= einsum("Pad,Pbc->abcd", Lpq[:, nocc:, nocc:], Lpq[:, nocc:, nocc:], optimize=True)
    # orbital energy part
    A = A.reshape(nvir*nvir, nvir*nvir)
    orb_sum = numpy.asarray(mo_energy[nocc:, None] + mo_energy[None, nocc:]).reshape(-1)
    orb_sum -= 2.0 * mu
    numpy.fill_diagonal(A, A.diagonal() + orb_sum)
    A = A.reshape(nvir, nvir, nvir, nvir)
    # take only low-triangular part
    A = A[tri_row_v, tri_col_v, ...]
    A = A[..., tri_row_v, tri_col_v]
    trace_A = numpy.trace(A)

    # B matrix: particle-hole block
    # two-electron integral part, <ab|ij>-<ab|ji>
    B = einsum("Pai,Pbj->abij", Lpq[:, nocc:, :nocc], Lpq[:, nocc:, :nocc], optimize=True)
    B -= einsum("Paj,Pbi->abij", Lpq[:, nocc:, :nocc], Lpq[:, nocc:, :nocc], optimize=True)
    # take only low-triangular part
    B = B[tri_row_v, tri_col_v, ...]
    B = B[..., tri_row_o, tri_col_o]

    # C matrix: hole-hole block
    # two-electron integral part, <ij|kl>-<ij|lk>
    C = einsum("Pik,Pjl->ijkl", Lpq[:, :nocc, :nocc], Lpq[:, :nocc, :nocc], optimize=True)
    C -= einsum("Pil,Pjk->ijkl", Lpq[:, :nocc, :nocc], Lpq[:, :nocc, :nocc], optimize=True)
    # orbital energy part
    C = C.reshape(nocc*nocc, nocc*nocc)
    orb_sum = numpy.asarray(mo_energy[:nocc, None] + mo_energy[None, :nocc]).reshape(-1)
    orb_sum -= 2.0 * mu
    numpy.fill_diagonal(C, C.diagonal() - orb_sum)
    C = C.reshape(nocc, nocc, nocc, nocc)
    # take only low-triangular part
    C = C[tri_row_o, tri_col_o, ...]
    C = C[..., tri_row_o, tri_col_o]

    # combine A, B and C matrix as
    # | C B^T |
    # |B A|
    M_upper = numpy.concatenate((C, B.T), axis=1)
    M_lower = numpy.concatenate((B, A), axis=1)
    M = numpy.concatenate((M_upper, M_lower), axis=0)
    del A, B, C
    # M to MW, W is the metric matrix [[-I,0],[0,I]]
    M[:oo_dim][:] *= -1.0

    # diagonalize ppRPA matrix
    exci, xy = scipy.linalg.eig(M)
    exci = exci.real
    xy = xy.T  # Fortran to Python order
    xy = numpy.asarray(list(x for _, x in sorted(zip(exci, xy), reverse=False)))
    exci = numpy.sort(exci)

    pprpa_orthonormalize_eigenvector(multi="t", nocc=nocc, TDA="None", exci=exci, xy=xy)

    sum_exci = numpy.sum(exci[oo_dim:])
    ec = (sum_exci - trace_A) * 3.0

    return exci, xy, ec


# analysis functions
def _pprpa_print_eigenvector(multi, nocc, nvir, nocc_fro, thresh, hh_state, pp_state, exci0, exci, xy):
    """Print dominant components of an eigenvector.

    Args:
        multi (string): multiplicity.
        nocc (int): number of occupied orbitals.
        nvir (int): number of virtual orbitals.
        nocc_fro (int): number of frozen occupied orbitals.
        thresh (double): threshold to print a pair.
        hh_state (int): number of interested hole-hole states.
        pp_state (int): number of interested particle-particle states.
        exci0 (double): lowest eigenvalue.
        exci (double array): ppRPA eigenvalue.
        xy (double ndarray): ppRPA eigenvector.
    """
    if multi == "s":
        oo_dim = int((nocc + 1) * nocc / 2)
        vv_dim = int((nvir + 1) * nvir / 2)
        is_singlet = 1
        print("\n     print ppRPA excitations: singlet\n")
    elif multi == "t":
        oo_dim = int((nocc - 1) * nocc / 2)
        vv_dim = int((nvir - 1) * nvir / 2)
        is_singlet = 0
        print("\n     print ppRPA excitations: triplet\n")

    tri_row_o, tri_col_o = numpy.tril_indices(nocc, is_singlet-1)
    tri_row_v, tri_col_v = numpy.tril_indices(nvir, is_singlet-1)

    au2ev = 27.211386

    for istate in range(min(hh_state, oo_dim)):
        print("#%-d %s de-excitation:  exci= %-12.4f  eV   2e=  %-12.4f  eV" %
              (istate + 1, multi, (exci[oo_dim-istate-1] - exci0) * au2ev, exci[oo_dim-istate-1] * au2ev))
        full = numpy.zeros(shape=[nocc, nocc], dtype=numpy.double)
        full[tri_row_o, tri_col_o] = xy[oo_dim-istate-1][:oo_dim]
        full = numpy.power(full, 2)
        pairs = numpy.argwhere(full > thresh)
        for i, j in pairs:
            pprpa_print_a_pair(is_pp=False, p=i+nocc_fro, q=j+nocc_fro, percentage=full[i, j])

        full = numpy.zeros(shape=[nvir, nvir], dtype=numpy.double)
        full[tri_row_v, tri_col_v] = xy[oo_dim-istate-1][oo_dim:]
        full = numpy.power(full, 2)
        pairs = numpy.argwhere(full > thresh)
        for a, b in pairs:
            pprpa_print_a_pair(is_pp=True, p=a+nocc_fro+nocc, q=b+nocc_fro+nocc, percentage=full[a, b])

        print("")

    for istate in range(min(pp_state, vv_dim)):
        print("#%-d %s excitation:  exci= %-12.4f  eV   2e=  %-12.4f  eV" %
              (istate + 1, multi, (exci[oo_dim+istate] - exci0) * au2ev, exci[oo_dim+istate] * au2ev))
        full = numpy.zeros(shape=[nocc, nocc], dtype=numpy.double)
        full[tri_row_o, tri_col_o] = xy[oo_dim+istate][:oo_dim]
        full = numpy.power(full, 2)
        pairs = numpy.argwhere(full > thresh)
        for i, j in pairs:
            pprpa_print_a_pair(is_pp=False, p=i+nocc_fro, q=j+nocc_fro, percentage=full[i, j])

        full = numpy.zeros(shape=[nvir, nvir], dtype=numpy.double)
        full[tri_row_v, tri_col_v] = xy[oo_dim+istate][oo_dim:]
        full = numpy.power(full, 2)
        pairs = numpy.argwhere(full > thresh)
        for a, b in pairs:
            pprpa_print_a_pair(is_pp=True, p=a+nocc_fro+nocc, q=b+nocc_fro+nocc, percentage=full[a, b])

        print("")

    return


def _analyze_pprpa_direct(
        exci_s, xy_s, exci_t, xy_t, nocc, nvir, nelec="n-2", print_thresh=0.1, hh_state=5, pp_state=5, nocc_fro=0):
    print("\nanalyze ppRPA results.")
    oo_dim_s = int((nocc + 1) * nocc / 2)
    oo_dim_t = int((nocc - 1) * nocc / 2)
    if exci_s is not None and exci_t is not None:
        print("both singlet and triplet results found.")
        if nelec == "n-2":
            exci0 = min(exci_s[oo_dim_s], exci_t[oo_dim_t])
        else:
            exci0 = max(exci_s[oo_dim_s-1], exci_t[oo_dim_t-1])
        _pprpa_print_eigenvector(
            multi="s", nocc=nocc, nvir=nvir, nocc_fro=nocc_fro, thresh=print_thresh,
            hh_state=hh_state, pp_state=pp_state, exci0=exci0, exci=exci_s, xy=xy_s)
        _pprpa_print_eigenvector(
            multi="t", nocc=nocc, nvir=nvir, nocc_fro=nocc_fro, thresh=print_thresh,
            hh_state=hh_state, pp_state=pp_state, exci0=exci0, exci=exci_t, xy=xy_t)
    else:
        if exci_s is not None:
            print("only singlet results found.")
            _pprpa_print_eigenvector(
                multi="s", nocc=nocc, nvir=nvir, nocc_fro=nocc_fro, thresh=print_thresh, hh_state=hh_state,
                pp_state=pp_state, exci0=exci_s[oo_dim_s], exci=exci_s, xy=xy_s)
        else:
            print("only triplet results found.")
            _pprpa_print_eigenvector(
                multi="t", nocc=nocc, nvir=nvir, nocc_fro=nocc_fro, thresh=print_thresh, hh_state=hh_state,
                pp_state=pp_state, exci0=exci_t[oo_dim_t], exci=exci_t, xy=xy_t)
    return


class ppRPA_direct():
    def __init__(
            self, nocc, mo_energy, Lpq, hh_state=5, pp_state=5, nelec="n-2", print_thresh=0.1):
        # necessary input
        self.nocc = nocc  # number of occupied orbitals
        self.mo_energy = numpy.asarray(mo_energy)  # orbital energy
        self.Lpq = numpy.asarray(Lpq)  # three-center density-fitting matrix in MO space

        # options
        self.hh_state = hh_state  # number of hole-hole states to print
        self.pp_state = pp_state  # number of particle-particle states to print
        self.nelec = nelec  # "n-2" or "n+2" for system is an N-2 or N+2 system
        self.print_thresh = print_thresh  # threshold to print component

        # internal flags
        self.multi = None  # multiplicity
        self.nmo = len(self.mo_energy)  # number of orbitals
        self.nvir = self.nmo - self.nocc  # number of virtual orbitals
        self.naux = Lpq.shape[0]  # number of auxiliary basis functions
        self.mu = None  # chemical potential

        # results
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
            self.mu = get_chemical_potential(nocc=self.nocc, mo_energy=self.mo_energy)

        return

    def dump_flags(self):
        print('\n******** %s ********' % self.__class__)
        if self.multi == "s":
            oo_dim = int((self.nocc + 1) * self.nocc / 2)
            vv_dim = int((self.nvir + 1) * self.nvir / 2)
        elif self.multi == "t":
            oo_dim = int((self.nocc - 1) * self.nocc / 2)
            vv_dim = int((self.nvir - 1) * self.nvir / 2)
        full_dim = oo_dim + vv_dim
        print('multiplicity = %s' % ("singlet" if self.multi == "s" else "triplet"))
        print('naux = %d' % self.naux)
        print('nmo = %d' % self.nmo)
        print('nocc = %d nvir = %d' % (self.nocc, self.nvir))
        print('occ-occ dimension = %d vir-vir dimension = %d' % (oo_dim, vv_dim))
        print('full dimension = %d' % full_dim)
        print('interested hh state = %d' % self.hh_state)
        print('interested pp state = %d' % self.pp_state)
        print('ground state = %s' % self.nelec)
        print('print threshold = %.2f%%' % (self.print_thresh*100))
        print('')
        return

    def check_memory(self):
        if self.multi == "s":
            oo_dim = int((self.nocc + 1) * self.nocc / 2)
            vv_dim = int((self.nvir + 1) * self.nvir / 2)
        elif self.multi == "t":
            oo_dim = int((self.nocc - 1) * self.nocc / 2)
            vv_dim = int((self.nvir - 1) * self.nvir / 2)
        full_dim = oo_dim + vv_dim

        # ppRPA matrix: A block and full matrix, eigenvector
        mem = (3 * full_dim * full_dim) * 8 / 1.0e6
        if mem < 1000:
            print("ppRPA needs at least %d MB memory." % mem)
        else:
            print("ppRPA needs at least %.1f GB memory." % (mem / 1.0e3))
        return

    def kernel(self, multi):
        self.multi = multi
        self.check_parameter()
        self.dump_flags()
        self.check_memory()
        start_clock("ppRPA direct: %s" % multi)
        if self.multi == "s":
            self.exci_s, self.xy_s, self.ec_s = diagonalize_pprpa_singlet(
                nocc=self.nocc, mo_energy=self.mo_energy, Lpq=self.Lpq, mu=self.mu)
        elif multi == "t":
            self.exci_t, self.xy_t, self.ec_t = diagonalize_pprpa_triplet(
                nocc=self.nocc, mo_energy=self.mo_energy, Lpq=self.Lpq, mu=self.mu)
        stop_clock("ppRPA direct: %s" % multi)
        return

    def save_pprpa(self, fn):
        assert self.exci_s is not None or self.exci_t is not None
        print("\nsave pprpa results to %s.\n" % fn)
        f = h5py.File(fn, "w")
        f["nocc"] = numpy.asarray(self.nocc)
        f["nvir"] = numpy.asarray(self.nvir)
        if self.exci_s is not None:
            f["exci_s"] = numpy.asarray(self.exci_s)
            f["xy_s"] = numpy.asarray(self.xy_s)
        if self.exci_t is not None:
            f["exci_t"] = numpy.asarray(self.exci_t)
            f["xy_t"] = numpy.asarray(self.xy_t)
        f.close()
        return

    def read_pprpa(self, fn, singlet=True, triplet=True):
        print("\nread pprpa results from %s.\n" % fn)
        f = h5py.File(fn, "r")
        if singlet is True:
            self.exci_s = numpy.asarray(f["exci_s"])
            self.xy_s = numpy.asarray(f["xy_s"])
        if triplet is True:
            self.exci_t = numpy.asarray(f["exci_t"])
            self.xy_t = numpy.asarray(f["xy_t"])
        f.close()
        return

    def analyze(self, nocc_fro=0):
        _analyze_pprpa_direct(exci_s=self.exci_s, xy_s=self.xy_s, exci_t=self.exci_t, xy_t=self.xy_t, nocc=self.nocc,
                              nvir=self.nvir, nelec=self.nelec, print_thresh=self.print_thresh, hh_state=self.hh_state,
                              pp_state=self.pp_state, nocc_fro=nocc_fro)
        return

    def get_correlation(self):
        self.check_parameter()
        start_clock("ppRPA correlation energy")
        if self.ec_s is None:
            self.exci_s, self.xy_s, self.ec_s = diagonalize_pprpa_singlet(
                nocc=self.nocc, mo_energy=self.mo_energy, Lpq=self.Lpq, mu=self.mu)

        if self.ec_t is None:
            self.exci_t, self.xy_t, self.ec_t = diagonalize_pprpa_triplet(
                nocc=self.nocc, mo_energy=self.mo_energy, Lpq=self.Lpq, mu=self.mu)
        stop_clock("ppRPA correlation energy")
        self.ec = self.ec_s + self.ec_t
        return self.ec
