"""Direct solver for (complex-valued) generalized or spinor-based particle-particle random phase approximation.
Author: Jincheng Yu <pimetamon@gmail.com>
        Chaoqun Zhang <cq_zhang@outlook.com>
"""
import h5py
import numpy
import scipy
from lib_pprpa.pprpa_davidson import pprpa_print_a_pair
from lib_pprpa.pprpa_util import get_chemical_potential, start_clock, stop_clock, print_citation
from lib_pprpa.pprpa_direct import diagonalize_pprpa_triplet


def diagonalize_gpprpa(nocc, mo_energy, Lpq, mu=None):
    """Diagonalize G-ppRPA matrix.

    See function `lib_pprpa.pprpa_direct.diagonalize_pprpa_triplet`.

    """
    exci, xy, ec = diagonalize_pprpa_triplet(nocc, mo_energy, Lpq, mu=mu)

    return exci, xy, ec

def complex_matrix_norm(matrix):
    """Calculate the Complex norm of each matrix element.

    Args:
        matrix (complex ndarray): input matrix.

    Returns:
        out (double ndarray): norm square of each element.
    """
    out = numpy.zeros(matrix.shape, dtype=numpy.double)
    out = numpy.power(numpy.abs(matrix.real), 2) + numpy.power(numpy.abs(matrix.imag), 2)
    return out

def _pprpa_print_eigenvector(nocc, nvir, nocc_fro, thresh, hh_state,
                             pp_state, exci0, exci, xy):
    """Print components of an eigenvector.

    Args:
        nocc (int): number of occupied orbitals.
        nvir (int): number of virtual orbitals.
        nocc_fro (int): number of frozen occupied orbitals.
        thresh (double): threshold to print a pair.
        hh_state (int): number of interested hole-hole states.
        pp_state (int): number of interested particle-particle states.
        exci0 (double): lowest eigenvalue.
        exci (double array): ppRPA eigenvalue.
        xy (double/complex ndarray): ppRPA eigenvector.
    """
    oo_dim = int((nocc - 1) * nocc / 2)
    vv_dim = int((nvir - 1) * nvir / 2)
    print("\n     print G-ppRPA excitations\n")

    tri_row_o, tri_col_o = numpy.tril_indices(nocc, -1)
    tri_row_v, tri_col_v = numpy.tril_indices(nvir, -1)

    au2ev = 27.211386

    # =====================> two-electron removal <======================
    for istate in range(min(hh_state, oo_dim)):
        print("#%-d de-excitation:  exci= %-12.6f  eV   2e=  %-12.6f  eV" %
              (istate + 1, (exci[oo_dim-istate-1] - exci0) * au2ev,
               exci[oo_dim-istate-1] * au2ev))
        
        full = numpy.zeros(shape=[nocc, nocc], dtype=xy.dtype)
        full[tri_row_o, tri_col_o] = xy[oo_dim-istate-1][:oo_dim]
        full = complex_matrix_norm(full)
        pairs = numpy.argwhere(full > thresh)
        for i, j in pairs:
            pprpa_print_a_pair(is_pp=False, p=i+nocc_fro, q=j+nocc_fro,
                               percentage=full[i, j])

        full = numpy.zeros(shape=[nvir, nvir], dtype=xy.dtype)
        full[tri_row_v, tri_col_v] = xy[oo_dim-istate-1][oo_dim:]
        full = complex_matrix_norm(full)
        pairs = numpy.argwhere(full > thresh)
        for a, b in pairs:
            pprpa_print_a_pair(is_pp=True, p=a+nocc_fro+nocc,
                               q=b+nocc_fro+nocc, percentage=full[a, b])
        
        print("")

    # =====================> two-electron addition <=====================
    for istate in range(min(pp_state, vv_dim)):
        print("#%-d excitation:  exci= %-12.6f  eV   2e=  %-12.6f  eV" %
              (istate + 1, (exci[oo_dim+istate] - exci0) * au2ev,
               exci[oo_dim+istate] * au2ev))
        full = numpy.zeros(shape=[nocc, nocc], dtype=xy.dtype)
        full[tri_row_o, tri_col_o] = xy[oo_dim+istate][:oo_dim]
        full = complex_matrix_norm(full)
        pairs = numpy.argwhere(full > thresh)
        for i, j in pairs:
            pprpa_print_a_pair(is_pp=False, p=i+nocc_fro, q=j+nocc_fro,
                                   percentage=full[i, j])

        full = numpy.zeros(shape=[nvir, nvir], dtype=xy.dtype)
        full[tri_row_v, tri_col_v] = xy[oo_dim+istate][oo_dim:]
        full = complex_matrix_norm(full)
        pairs = numpy.argwhere(full > thresh)
        for a, b in pairs:
            pprpa_print_a_pair(is_pp=True, p=a+nocc_fro+nocc,
                               q=b+nocc_fro+nocc, percentage=full[a, b])

        print("")

    return


def _analyze_pprpa_direct(
        exci, xy, nocc, nvir, nelec='n-2', print_thresh=0.1, hh_state=5,
        pp_state=5, nocc_fro=0):
    print('\nanalyze G-ppRPA results.')
    oo_dim = round((nocc - 1) * nocc / 2)

    if nelec == 'n-2':
        exci0 = exci[oo_dim]
    else:
        exci0 = exci[oo_dim - 1]
    _pprpa_print_eigenvector(
        nocc, nvir, nocc_fro, print_thresh, hh_state,
        pp_state, exci0, exci, xy)

    pass


class GppRPA_direct():
    """Direct solver class for generalized ppRPA.

    Args:
        nocc (int): number of occupied orbitals
        mo_energy (double arrays): orbital energies
        Lpq (double/complex ndarrays): 
            three-center RI matrices in MO space

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
        self.nmo = len(self.mo_energy)
        # number of virtual orbitals
        self.nvir = self.nmo - self.nocc
        # number of auxiliary basis functions
        self.naux = self.Lpq.shape[0]
        # chemical potential
        self.mu = None
        # 'n-2' for ppRPA, 'n+2' for hhRPA
        self.nelec = nelec

        # =========================> results <===========================
        self.ec = None  # correlation energy
        self.exci = None  # two-electron addition energy
        self.xy = None  # ppRPA eigenvector

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
        vv_dim = int(self.nvir * (self.nvir - 1) / 2)
        oo_dim = int(self.nocc * (self.nocc - 1) / 2)
        full_dim = oo_dim + vv_dim

        print('\n******** %s ********' % self.__class__)
        print('naux = %d' % self.naux)
        print('naux = %d' % self.naux)
        print('nmo = %d' % self.nmo)
        print('nocc = %d nvir = %d' % (self.nocc, self.nvir))
        print('occ-occ dimension = %d vir-vir dimension = %d' %
              (oo_dim, vv_dim))
        print('full dimension = %d' % full_dim)
        print('interested hh state = %d' % self.hh_state)
        print('interested pp state = %d' % self.pp_state)
        print('ground state = %s' % self.nelec)
        print('print threshold = %.2f%%' % (self.print_thresh*100))
        print('')
        return

    def check_memory(self):
        # ====================> calculate dimensions <===================
        vv_dim = int(self.nvir * (self.nvir - 1) / 2)
        oo_dim = int(self.nocc * (self.nocc - 1) / 2)
        full_dim = oo_dim + vv_dim

        mem = (3 * full_dim * full_dim) * 8 / 1.0e6
        if self.Lpq.dtype == numpy.complex128:
            mem *= 2
        if mem < 1000:
            print("G-ppRPA needs at least %d MB memory." % mem)
        else:
            print("G-ppRPA needs at least %.1f GB memory." % (mem / 1.0e3))
        return

    def kernel(self):
        self.check_parameter()
        self.dump_flags()
        self.check_memory()

        start_clock("G-ppRPA direct")
        self.exci, self.xy, self.ec = diagonalize_gpprpa(
                self.nocc, self.mo_energy, self.Lpq, mu=self.mu
            )
        stop_clock("G-ppRPA direct")

        return

    def save_pprpa(self, fn):
        assert self.exci is not None
        print('\nsave G-ppRPA results to %s.\n' % fn)
        f = h5py.File(fn, 'w')
        f['nocc'] = numpy.asarray(self.nocc)
        f['nvir'] = numpy.asarray(self.nvir)

        f['exci'] = numpy.asarray(self.exci)
        f['xy'] = numpy.asarray(self.xy)

        f.close()
        return

    def read_pprpa(self, fn):
        print('\nread G-ppRPA results from %s.\n' % fn)
        f = h5py.File(fn, 'r')
        self.exci = numpy.asarray(f['exci'])
        self.xy = numpy.asarray(f['xy'])
        f.close()
        return

    def analyze(self, nocc_fro=0):
        _analyze_pprpa_direct(
            self.exci, self.xy, self.nocc, self.nvir, nelec=self.nelec,
            print_thresh=self.print_thresh, hh_state=self.hh_state,
            pp_state=self.pp_state, nocc_fro=nocc_fro)
        return
