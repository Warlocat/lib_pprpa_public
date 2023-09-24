import numpy
import scipy

from numpy import einsum

from lib_pprpa.pprpa_davidson import pprpa_orthonormalize_eigenvector, pprpa_print_a_pair
from lib_pprpa.pprpa_util import ij2index, start_clock, stop_clock


def diagonalize_pprpa_singlet(nocc, mo_energy, Lpq, nocc_act, nvir_act):
    """Diagonalize singlet ppRPA matrix.

    Args:
        nocc (int): number of occupied orbitals.
        mo_energy (double array): orbital energy.
        Lpq (double ndarray): three-center density-fitting matrix in MO space.
        nocc_act (int): number of active occupied orbitals.
        nvir_act (int): number of active virtual orbitals.

    Returns:
        exci (double array): ppRPA eigenvalue.
        xy (double ndarray): ppRPA eigenvector.
        ec (double): singlet correlation energy.
    """
    nmo = len(mo_energy)
    nvir = nmo - nocc
    mu = (mo_energy[nocc-1] + mo_energy[nocc]) * 0.5  # chemical potential

    nocc_fro = nocc - nocc_act  # number of frozen occupied orbitals
    nvir_fro = nvir - nvir_act  # number of frozen virtual orbitals

    oo_dim = int((nocc_act + 1) * nocc_act / 2)  # number of hole-hole pairs

    tri_row_o, tri_col_o = numpy.tril_indices(nocc_act)
    tri_row_v, tri_col_v = numpy.tril_indices(nvir_act)

    Lpq_act = numpy.asarray(Lpq[:, nocc_fro:(nmo-nvir_fro), nocc_fro:(nmo-nvir_fro)])

    # A matrix: particle-particle block
    A = einsum("Pac,Pbd->abcd", Lpq_act[:, nocc_act:, nocc_act:], Lpq_act[:, nocc_act:, nocc_act:], optimize=True)
    A += einsum("Pad,Pbc->abcd", Lpq_act[:, nocc_act:, nocc_act:], Lpq_act[:, nocc_act:, nocc_act:], optimize=True)

    A[numpy.diag_indices(nvir_act)] *= 1.0 / numpy.sqrt(2)
    A = A.transpose(2, 3, 0, 1)
    A[numpy.diag_indices(nvir_act)] *= 1.0 / numpy.sqrt(2)
    A = A.transpose(2, 3, 0, 1)

    A = A.reshape(nvir_act*nvir_act, nvir_act*nvir_act)
    orb_sum = numpy.asarray(mo_energy[nocc:(nmo-nvir_fro), None] + mo_energy[None, nocc:(nmo-nvir_fro)]).reshape(-1)
    orb_sum -= 2.0 * mu
    numpy.fill_diagonal(A, A.diagonal() + orb_sum)
    A = A.reshape(nvir_act, nvir_act, nvir_act, nvir_act)

    A = A[tri_row_v, tri_col_v, ...]
    A = A[..., tri_row_v, tri_col_v]

    # B matrix: particle-hole block
    B = einsum("Pai,Pbj->abij", Lpq_act[:, nocc_act:, :nocc_act], Lpq_act[:, nocc_act:, :nocc_act], optimize=True)
    B += einsum("Paj,Pbi->abij", Lpq_act[:, nocc_act:, :nocc_act], Lpq_act[:, nocc_act:, :nocc_act], optimize=True)

    B[numpy.diag_indices(nvir_act)] *= 1.0 / numpy.sqrt(2)
    B = B.transpose(2, 3, 0, 1)
    B[numpy.diag_indices(nocc_act)] *= 1.0 / numpy.sqrt(2)
    B = B.transpose(2, 3, 0, 1)

    B = B[tri_row_v, tri_col_v, ...]
    B = B[..., tri_row_o, tri_col_o]

    # C matrix: hole-hole block
    C = einsum("Pik,Pjl->ijkl", Lpq_act[:, :nocc_act, :nocc_act], Lpq_act[:, :nocc_act, :nocc_act], optimize=True)
    C += einsum("Pil,Pjk->ijkl", Lpq_act[:, :nocc_act, :nocc_act], Lpq_act[:, :nocc_act, :nocc_act], optimize=True)

    C[numpy.diag_indices(nocc_act)] *= 1.0 / numpy.sqrt(2)
    C = C.transpose(2, 3, 0, 1)
    C[numpy.diag_indices(nocc_act)] *= 1.0 / numpy.sqrt(2)
    C = C.transpose(2, 3, 0, 1)

    C = C.reshape(nocc_act*nocc_act, nocc_act*nocc_act)
    orb_sum = numpy.asarray(mo_energy[nocc_fro:nocc, None] + mo_energy[None, nocc_fro:nocc_act]).reshape(-1)
    orb_sum -= 2.0 * mu
    numpy.fill_diagonal(C, C.diagonal() - orb_sum)
    C = C.reshape(nocc_act, nocc_act, nocc_act, nocc_act)

    C = C[tri_row_o, tri_col_o, ...]
    C = C[..., tri_row_o, tri_col_o]

    # combine A, B and C matrix as
    # | C B^T |
    # |B A|
    M_upper = numpy.concatenate((C, B.T), axis=1)
    M_lower = numpy.concatenate((B, A), axis=1)
    M = numpy.concatenate((M_upper, M_lower), axis=0)

    # M = MW, W is the metric matrix [[I,0],[0,-I]]
    M[:oo_dim][:] *= -1.0

    exci, xy = scipy.linalg.eig(M)
    exci = exci.real
    xy = xy.T
    xy = numpy.asarray(list(x for _, x in sorted(zip(exci, xy), reverse=False)))
    exci = numpy.sort(exci)

    pprpa_orthonormalize_eigenvector(multi="s", nocc=nocc_act, TDA=None, exci=exci, xy=xy)

    trace_A = numpy.trace(A)
    sum_exci = numpy.sum(exci[oo_dim:])
    ec = sum_exci - trace_A

    return exci, xy, ec


def diagonalize_pprpa_triplet(nocc, mo_energy, Lpq, nocc_act, nvir_act):
    """Diagonalize triplet ppRPA matrix.

    Args:
        nocc (int): number of occupied orbitals.
        mo_energy (double array): orbital energy.
        Lpq (double ndarray): three-center density-fitting matrix in MO space.
        nocc_act (int): number of active occupied orbitals.
        nvir_act (int): number of active virtual orbitals.

    Returns:
        exci (double array): ppRPA eigenvalue.
        xy (double ndarray): ppRPA eigenvector.
        ec (double): triplet correlation energy, with a factor of 3.
    """
    nmo = len(mo_energy)
    nvir = nmo - nocc
    mu = (mo_energy[nocc-1] + mo_energy[nocc]) * 0.5  # chemical potential

    nocc_fro = nocc - nocc_act  # number of frozen occupied orbitals
    nvir_fro = nvir - nvir_act  # number of frozen virtual orbitals

    oo_dim = int((nocc_act - 1) * nocc_act / 2)  # number of hole-hole pairs

    tri_row_o, tri_col_o = numpy.tril_indices(nocc_act, -1)
    tri_row_v, tri_col_v = numpy.tril_indices(nvir_act, -1)

    Lpq_act = numpy.asarray(Lpq[:, nocc_fro:(nmo-nvir_fro), nocc_fro:(nmo-nvir_fro)])

    # A matrix: particle-particle block
    A = einsum("Pac,Pbd->abcd", Lpq_act[:, nocc_act:, nocc_act:], Lpq_act[:, nocc_act:, nocc_act:], optimize=True)
    A -= einsum("Pad,Pbc->abcd", Lpq_act[:, nocc_act:, nocc_act:], Lpq_act[:, nocc_act:, nocc_act:], optimize=True)

    A = A.reshape(nvir_act*nvir_act, nvir_act*nvir_act)
    orb_sum = numpy.asarray(mo_energy[nocc:(nmo-nvir_fro), None] + mo_energy[None, nocc:(nmo-nvir_fro)]).reshape(-1)
    orb_sum -= 2.0 * mu
    numpy.fill_diagonal(A, A.diagonal() + orb_sum)
    A = A.reshape(nvir_act, nvir_act, nvir_act, nvir_act)

    A = A[tri_row_v, tri_col_v, ...]
    A = A[..., tri_row_v, tri_col_v]

    # B matrix: particle-hole block
    B = einsum("Pai,Pbj->abij", Lpq_act[:, nocc_act:, :nocc_act], Lpq_act[:, nocc_act:, :nocc_act], optimize=True)
    B -= einsum("Paj,Pbi->abij", Lpq_act[:, nocc_act:, :nocc_act], Lpq_act[:, nocc_act:, :nocc_act], optimize=True)

    B = B[tri_row_v, tri_col_v, ...]
    B = B[..., tri_row_o, tri_col_o]

    # C matrix: hole-hole block
    C = einsum("Pik,Pjl->ijkl", Lpq_act[:, :nocc_act, :nocc_act], Lpq_act[:, :nocc_act, :nocc_act], optimize=True)
    C -= einsum("Pil,Pjk->ijkl", Lpq_act[:, :nocc_act, :nocc_act], Lpq_act[:, :nocc_act, :nocc_act], optimize=True)

    C = C.reshape(nocc_act*nocc_act, nocc_act*nocc_act)
    orb_sum = numpy.asarray(mo_energy[nocc_fro:nocc, None] + mo_energy[None, nocc_fro:nocc]).reshape(-1)
    orb_sum -= 2.0 * mu
    numpy.fill_diagonal(C, C.diagonal() - orb_sum)
    C = C.reshape(nocc_act, nocc_act, nocc_act, nocc_act)

    C = C[tri_row_o, tri_col_o, ...]
    C = C[..., tri_row_o, tri_col_o]

    # combine A, B and C matrix as
    # | C B^T |
    # |B A|
    M_upper = numpy.concatenate((C, B.T), axis=1)
    M_lower = numpy.concatenate((B, A), axis=1)
    M = numpy.concatenate((M_upper, M_lower), axis=0)

    # M = MW, W is the metric matrix [[I,0],[0,-I]]
    M[:oo_dim][:] *= -1.0

    exci, xy = scipy.linalg.eig(M)
    exci = exci.real
    xy = xy.T
    xy = numpy.asarray(list(x for _, x in sorted(zip(exci, xy), reverse=False)))
    exci = numpy.sort(exci)

    pprpa_orthonormalize_eigenvector(multi="t", nocc=nocc_act, TDA="None", exci=exci, xy=xy)

    trace_A = numpy.trace(A)
    sum_exci = numpy.sum(exci[oo_dim:])
    ec = (sum_exci - trace_A) * 3.0

    return exci, xy, ec


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
    nmo = nocc + nvir

    tri_row_o, tri_col_o = numpy.tril_indices(nocc, is_singlet-1)
    tri_row_v, tri_col_v = numpy.tril_indices(nvir, is_singlet-1)

    au2ev = 27.211386

    for istate in range(min(hh_state, oo_dim)):
        print("#%-d %s de-excitation:  exci= %-12.4f  eV   2e=  %-12.4f  eV" %
              (istate + 1, multi, (exci[oo_dim-istate-1] - exci0) * au2ev, exci[oo_dim-istate-1] * au2ev))
        for i in range(nocc):
            for j in range(i + is_singlet):
                ij = ij2index(i, j, tri_row_o, tri_col_o)
                percentage = numpy.power(xy[oo_dim-istate-1][ij], 2)
                if percentage > thresh:
                    pprpa_print_a_pair(is_pp=False, p=i+nocc_fro, q=j+nocc_fro, percentage=percentage)

        for a in range(nocc, nmo):
            for b in range(nocc, a + is_singlet):
                ab = ij2index(a - nocc, b - nocc, tri_row_v, tri_col_v)
                percentage = numpy.power(xy[oo_dim-istate-1][oo_dim + ab], 2)
                if percentage > thresh:
                    pprpa_print_a_pair(is_pp=True, p=a+nocc_fro, q=b+nocc_fro, percentage=percentage)

        print("")

    for istate in range(min(pp_state, vv_dim)):
        print("#%-d %s excitation:  exci= %-12.4f  eV   2e=  %-12.4f  eV" %
              (istate + 1, multi, (exci[oo_dim+istate] - exci0) * au2ev, exci[oo_dim+istate] * au2ev))
        for i in range(nocc):
            for j in range(i + is_singlet):
                ij = ij2index(i, j, tri_row_o, tri_col_o)
                percentage = numpy.power(xy[oo_dim+istate][ij], 2)
                if percentage > thresh:
                    pprpa_print_a_pair(is_pp=False, p=i+nocc_fro, q=j+nocc_fro, percentage=percentage)

        for a in range(nocc, nmo):
            for b in range(nocc, a + is_singlet):
                ab = ij2index(a - nocc, b - nocc, tri_row_v, tri_col_v)
                percentage = numpy.power(xy[oo_dim+istate][oo_dim + ab], 2)
                if percentage > thresh:
                    pprpa_print_a_pair(is_pp=True, p=a+nocc_fro, q=b+nocc_fro, percentage=percentage)

        print("")

    return


class ppRPA_direct():
    def __init__(
            self, nocc, mo_energy, Lpq, TDA=None, nocc_act=None, nvir_act=None, hh_state=5, pp_state=5,
            nelec="n-2", print_thresh=0.1):
        # necessary input
        self.nocc = nocc  # number of occupied orbitals
        self.mo_energy = numpy.asarray(mo_energy)  # orbital energy
        self.Lpq = numpy.asarray(Lpq)  # three-center density-fitting matrix in MO space

        # options
        self.TDA = TDA  # Tamm–Dancoff approximation, "pp" or "hh"
        self.nocc_act = nocc_act  # number of active occupied orbitals
        self.nvir_act = nvir_act  # number of active virtual orbitals
        self.hh_state = hh_state  # number of hole-hole states to print
        self.pp_state = pp_state  # number of particle-particle states to print
        self.nelec = nelec  #  "n-2" or "n+2" for system is an N-2 or N+2 system
        self.print_thresh = print_thresh  #  threshold to print component

        # internal flags
        self.multi = None  # multiplicity
        self.nmo = len(self.mo_energy)  # number of orbitals
        self.nvir = self.nmo - self.nocc  # number of virtual orbitals
        self.naux = Lpq.shape[0]  # number of auxiliary basis functions

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

        return

    def check_parameter(self):
        assert self.TDA in ["pp", "hh", None]
        if self.TDA == "pp":
            self.nocc_act = 0
            self.nvir_act = self.nvir
        if self.TDA == "hh":
            self.nocc_act = self.nocc
            self.nvir_act = 0

        self.nocc_act = self.nocc if self.nocc_act is None else min(self.nocc_act, self.nocc)
        self.nvir_act = self.nvir if self.nvir_act is None else min(self.nvir_act, self.nvir)

        assert 0.0 < self.print_thresh < 1.0
        assert self.nelec in ["n-2", "n+2"]

        return

    def dump_flags(self):
        print('\n******** %s ********' % self.__class__)
        if self.multi == "s":
            oo_dim = int((self.nocc_act + 1) * self.nocc_act / 2)
            vv_dim = int((self.nvir_act + 1) * self.nvir_act / 2)
        elif self.multi == "t":
            oo_dim = int((self.nocc_act - 1) * self.nocc_act / 2)
            vv_dim = int((self.nvir_act - 1) * self.nvir_act / 2)
        full_dim = oo_dim + vv_dim
        print('multiplicity = %s' % ("singlet" if self.multi == "s" else "triplet"))
        if self.TDA is not None:
            print('Tamm-Dancoff approximation = %s' % self.TDA)
        print('naux = %d' % self.naux)
        print('nmo = %d' % self.nmo)
        print('nocc = %d nvir = %d' % (self.nocc, self.nvir))
        print('nocc_act = %d nvir_act = %d' % (self.nocc_act, self.nvir_act))
        print('occ-occ dimension = %d vir-vir dimension = %d' % (oo_dim, vv_dim))
        print('full dimension = %d' % full_dim)
        print('interested hh state = %d' % self.hh_state)
        print('interested pp state = %d' % self.pp_state)
        print('ground state = %s' % self.nelec)
        print('print threshold = %.2f' % self.print_thresh)
        print('')
        return

    def check_memory(self):
        if self.multi == "s":
            oo_dim = int((self.nocc_act + 1) * self.nocc_act / 2)
            vv_dim = int((self.nvir_act + 1) * self.nvir_act / 2)
        elif self.multi == "t":
            oo_dim = int((self.nocc_act - 1) * self.nocc_act / 2)
            vv_dim = int((self.nvir_act - 1) * self.nvir_act / 2)
        full_dim = oo_dim + vv_dim

        # ppRPA matrix, singlet+triplet eigenvector
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
                nocc=self.nocc, mo_energy=self.mo_energy, Lpq=self.Lpq, nocc_act=self.nocc_act, nvir_act=self.nvir_act)
        else:
            self.exci_t, self.xy_t, self.ec_t = diagonalize_pprpa_triplet(
                nocc=self.nocc, mo_energy=self.mo_energy, Lpq=self.Lpq, nocc_act=self.nocc_act, nvir_act=self.nvir_act)
        stop_clock("ppRPA direct: %s" % multi)
        return

    def analyze(self, nocc_fro=None):
        print("\nanalyze ppRPA results.")
        print_thresh = self.print_thresh
        if nocc_fro is None:
            nocc_fro = self.nocc - self.nocc_act
        oo_dim_s = int((self.nocc_act + 1) * self.nocc_act / 2)
        oo_dim_t = int((self.nocc_act - 1) * self.nocc_act / 2)
        if self.exci_s is not None and self.exci_t is not None:
            print("both singlet and triplet results found.")
            if self.nelec == "n-2":
                exci0 = min(self.exci_s[oo_dim_s], self.exci_t[oo_dim_t])
            else:
                exci0 = max(self.exci_s[oo_dim_s-1], self.exci_t[oo_dim_t-1])
            _pprpa_print_eigenvector(
                multi="s", nocc=self.nocc_act, nvir=self.nvir_act, nocc_fro=nocc_fro, thresh=print_thresh,
                hh_state=self.hh_state, pp_state=self.pp_state, exci0=exci0, exci=self.exci_s, xy=self.xy_s)
            _pprpa_print_eigenvector(
                multi="t", nocc=self.nocc_act, nvir=self.nvir_act, nocc_fro=nocc_fro, thresh=print_thresh,
                hh_state=self.hh_state, pp_state=self.pp_state, exci0=exci0, exci=self.exci_t, xy=self.xy_t)
        else:
            if self.exci_s is not None:
                print("only singlet results found.")
                _pprpa_print_eigenvector(
                    multi="s", nocc=self.nocc_act, nvir=self.nvir_act, nocc_fro=nocc_fro, thresh=print_thresh,
                    hh_state=self.hh_state, pp_state=self.pp_state, exci0=self.exci_s[oo_dim_s],
                    exci=self.exci_s, xy=self.xy_s)
            else:
                print("only triplet results found.")
                _pprpa_print_eigenvector(
                    multi="t", nocc=self.nocc_act, nvir=self.nvir_act, nocc_fro=nocc_fro, thresh=print_thresh,
                    hh_state=self.hh_state, pp_state=self.pp_state, exci0=self.exci_t[oo_dim_t],
                    exci=self.exci_t, xy=self.xy_t)
        return

    def get_correlation(self):
        self.check_parameter()
        start_clock("ppRPA correlation energy")
        if self.ec_s is None:
            self.exci_s, self.xy_s, self.ec_s = diagonalize_pprpa_singlet(
                nocc=self.nocc, mo_energy=self.mo_energy, Lpq=self.Lpq, nocc_act=self.nocc_act, nvir_act=self.nvir_act)

        if self.ec_t is None:
            self.exci_t, self.xy_t, self.ec_t = diagonalize_pprpa_triplet(
                nocc=self.nocc, mo_energy=self.mo_energy, Lpq=self.Lpq, nocc_act=self.nocc_act, nvir_act=self.nvir_act)
        stop_clock("ppRPA correlation energy")
        self.ec = self.ec_s + self.ec_t
        return self.ec
