import numpy
import scipy

from lib_pprpa.pprpa_direct import ppRPA_direct, pprpa_orthonormalize_eigenvector
from lib_pprpa.pprpa_util import start_clock, stop_clock, get_chemical_potential
from lib_pprpa.pyscf_util import Cholesky, get_pyscf_input_mol, get_fxc_r_st


def diagonalize_pprpaw_singlet(nocc, mo_energy, Lpq, w_mat, mu=None):
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
    A = w_mat[nocc:, nocc:, nocc:, nocc:]
    # orbital energy part
    A = A.reshape(nvir * nvir, nvir * nvir)
    orb_sum = (mo_energy[nocc:, None] + mo_energy[None, nocc:]).reshape(-1)
    orb_sum -= 2.0 * mu
    numpy.fill_diagonal(A, A.diagonal() + orb_sum)
    A = A.reshape(nvir, nvir, nvir, nvir)
    # take only low-triangular part
    A = A[tri_row_v, tri_col_v, ...]
    A = A[..., tri_row_v, tri_col_v]
    trace_A = numpy.trace(A)

    # B matrix: particle-hole block
    # two-electron integral part, <ab|ij>+<ab|ji>
    B = w_mat[nocc:, nocc:, :nocc, :nocc]
    # take only low-triangular part
    B = B[tri_row_v, tri_col_v, ...]
    B = B[..., tri_row_o, tri_col_o]

    # C matrix: hole-hole block
    # two-electron integral part, <ij|kl>+<ij|lk>
    C = w_mat[:nocc, :nocc, :nocc, :nocc]
    # orbital energy part
    C = C.reshape(nocc * nocc, nocc * nocc)
    orb_sum = (mo_energy[:nocc, None] + mo_energy[None, :nocc]).reshape(-1)
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
    # M to WM, W is the metric matrix [[-I,0],[0,I]]
    M[:oo_dim][:] *= -1.0

    # diagonalize ppRPA matrix
    exci, xy = scipy.linalg.eig(M)
    exci = exci.real
    xy = xy.T  # Fortran to Python order

    # sort eigenvalue and eigenvectors by ascending order
    idx = exci.argsort()
    exci = exci[idx]
    xy = xy[idx, :]

    pprpa_orthonormalize_eigenvector(multi="s", nocc=nocc, exci=exci, xy=xy)

    sum_exci = numpy.sum(exci[oo_dim:])
    ec = sum_exci - trace_A

    return exci, xy, ec


def diagonalize_pprpa_triplet(nocc, mo_energy, Lpq, w_mat, mu=None):
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
    A = w_mat[nocc:, nocc:, nocc:, nocc:]
    # orbital energy part
    A = A.reshape(nvir * nvir, nvir * nvir)
    orb_sum = (mo_energy[nocc:, None] + mo_energy[None, nocc:]).reshape(-1)
    orb_sum -= 2.0 * mu
    numpy.fill_diagonal(A, A.diagonal() + orb_sum)
    A = A.reshape(nvir, nvir, nvir, nvir)
    # take only low-triangular part
    A = A[tri_row_v, tri_col_v, ...]
    A = A[..., tri_row_v, tri_col_v]
    trace_A = numpy.trace(A)

    # B matrix: particle-hole block
    # two-electron integral part, <ab|ij>-<ab|ji>
    B = w_mat[nocc:, nocc:, :nocc, :nocc]
    # take only low-triangular part
    B = B[tri_row_v, tri_col_v, ...]
    B = B[..., tri_row_o, tri_col_o]

    # C matrix: hole-hole block
    # two-electron integral part, <ij|kl>-<ij|lk>
    C = w_mat[:nocc, :nocc, :nocc, :nocc]
    # orbital energy part
    C = C.reshape(nocc * nocc, nocc * nocc)
    orb_sum = (mo_energy[:nocc, None] + mo_energy[None, :nocc]).reshape(-1)
    orb_sum -= 2.0 * mu
    numpy.fill_diagonal(C, C.diagonal() - orb_sum)
    C = C.reshape(nocc, nocc, nocc, nocc)
    # take only low-triangular part
    C = C[tri_row_o, tri_col_o, ...]
    C = C[..., tri_row_o, tri_col_o]

    # combine A, B and C matrix as
    # | C B^T |
    # |B A|
    M_upper = numpy.concatenate((C, B.T.conj()), axis=1) # B.T.conj() for complex GHF
    M_lower = numpy.concatenate((B, A), axis=1)
    M = numpy.concatenate((M_upper, M_lower), axis=0)
    del A, B, C
    # M to WM, W is the metric matrix [[-I,0],[0,I]]
    M[:oo_dim][:] *= -1.0

    # diagonalize ppRPA matrix
    exci, xy = scipy.linalg.eig(M)
    exci = exci.real
    xy = xy.T  # Fortran to Python order

    # sort eigenvalue and eigenvectors by ascending order
    idx = exci.argsort()
    exci = exci[idx]
    xy = xy[idx, :]

    pprpa_orthonormalize_eigenvector(multi="t", nocc=nocc, exci=exci, xy=xy)

    sum_exci = numpy.sum(exci[oo_dim:])
    ec = (sum_exci - trace_A) * 3.0

    return exci, xy, ec


def get_K(multi, Lpq,fxc=None, rpa=False, hyb=0):
    """Get spin-adapted Hartree-exchange-correlation kernel matrix in MO basis.

    Parameters
    ----------
    multi : str
        multiplicity
    Lpq : numpy.ndarray
        three-center RI matrices in MO space

    fxc : numpy.ndarray, optional
        spin-adapted DFT exchange-correlation kernel, by default None
    rpa : bool, optional
        whether to include DFT exchange-correlation kernel, by default False
    hyb : double, optional
        HF exchange factor, by default 0

    Returns
    -------
    k_hxc : numpy.ndarray
        spin-adapted Hartree-exchange-correlation kernel
    """
    assert multi in ["s", "t"]
    nmo = Lpq.shape[-1]
    # Coulomb
    if multi == "s":
        eri_d = numpy.einsum("Lpq,Lsr->pqrs", Lpq, Lpq, optimize=True)
        k_hxc = eri_d * 2.0
    else:
        k_hxc = numpy.zeros(shape=[nmo, nmo, nmo, nmo], dtype=numpy.double)

    # HF exchange
    if hyb > 1e-10:
        eri_x = -numpy.einsum("Lps,Lqr->pqrs", Lpq, Lpq, optimize=True) * hyb
        k_hxc += eri_x

    # DFT exchange-correlation response
    if rpa is False:
        assert fxc is not None
        k_hxc += fxc

    return k_hxc


def get_M(nocc, mo_energy, k_hxc):
    """Get M matrix of Equation.19 in J. Phys. Chem. Lett. 2021, 12, 7236-7244.

    Parameters
    ----------
    nocc : int
        number of occupied orbitals
    mo_energy : numpy.ndarray
        orbital energy
    k_hxc : numpy.ndarray
        spin-adapted Hartree-exchange-correlation kernel

    Returns
    -------
    m_mat : numpy.ndarray
        spin-adapted M matrix
    """
    nmo = mo_energy.size
    nvir = nmo - nocc

    m_mat = k_hxc[:nocc, nocc:, :nocc, nocc:].reshape(nocc * nvir, nocc * nvir)
    m_mat = m_mat * 2.0

    # orbital energy contribution
    orb_diff = (mo_energy[None, nocc:] - mo_energy[:nocc, None]).reshape(-1)
    numpy.fill_diagonal(m_mat, m_mat.diagonal() + orb_diff)

    return m_mat


def get_W(nocc, k_hxc, m_mat, no_screening=False):
    """Get W matrix of Equation.14 in J. Phys. Chem. Lett. 2021, 12, 7236-7244.

    Parameters
    ----------
    nocc : int
        number of occupied orbitals
    k_hxc : numpy.ndarray
        spin-adapted Hartree-exchange-correlation kernel
    m_mat : numpy.ndarray
        spin-adapted M matrix of Equation.19
    no_screening : bool, optional
        turn off screening, by default False

    Returns
    -------
    w_mat : numpy.ndarray
        spin-adapted W matrix
    """
    nmo = k_hxc.shape[0]
    nvir = nmo - nocc

    k_pria = numpy.ascontiguousarray(k_hxc[:, :, :nocc, nocc:])
    K_mat1 = numpy.array(k_pria.reshape(nmo * nmo, nocc * nvir)) * 2.0

    k_jbsq = numpy.ascontiguousarray(k_pria.transpose(2, 3, 0, 1))
    K_mat2 = k_jbsq.reshape(nocc * nvir, nmo * nmo)

    cond_number = numpy.linalg.cond(m_mat)
    print(f"M matrix Condition number: {cond_number}")
    if cond_number < 1e12:  # Threshold for well-conditioned matrix
        print("Matrix is well-conditioned.")
    else:
        print("Matrix is ill-conditioned.")

    m_inv = numpy.linalg.inv(m_mat)

    wm = (K_mat1 @ m_inv @ K_mat2).reshape(nmo, nmo, nmo, nmo)

    w_mat = k_hxc if no_screening else k_hxc - wm

    return w_mat


def mo_energy_gsc2_r(mf, nocc_act=None, nvir_act=None):
    """Add GSC2 correction to original DFA eigenvalues.

    Parameters
    ----------
    mf : pyscf.dft.rks.RKS/pyscf.scf.rhf.RHF
        mean-field object
    nocc_act : int, optional
        number of active occupied orbitals, by default None
    nvir_act : int, optional
        number of active virtual orbitals, by default None

    Returns
    -------
    mo_energy_sc : numpy.ndarray
        GSC2 orbital energy
    """
    mo_energy = mf.mo_energy
    mo_occ = numpy.array(mf.mo_occ, copy=True) / 2.0  # avoid overwriting mf.mo_occ
    nocc = int(numpy.sum(mo_occ))
    nmo = mo_energy.size
    nvir = nmo - nocc

    # HF exchange factor
    hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, mf.mol.spin)[2]

    # determine active space
    nocc_act = nocc if nocc_act is None else min(nocc, nocc_act)
    nvir_act = nvir if nvir_act is None else min(nvir, nvir_act)
    nocc_act, mo_energy_act, Lpq_act = get_pyscf_input_mol(mf, nocc_act=nocc_act, nvir_act=nvir_act)

    # get spin-adapted DFT XC kernel
    fxc_s = get_fxc_r_st(mf, "s", nocc_act=nocc_act, nvir_act=nvir_act)
    fxc_t = get_fxc_r_st(mf, "t", nocc_act=nocc_act, nvir_act=nvir_act)

    # get spin-adapted Hxc kernel
    k_hxc_s = get_K("s", Lpq_act, fxc=fxc_s, rpa=False, hyb=hyb)
    k_hxc_t = get_K("t", Lpq_act, fxc=fxc_t, rpa=False, hyb=hyb)

    # get spin-adapted M matrix
    m_mat_s = get_M(nocc_act, mo_energy_act, k_hxc_s)
    m_mat_t = get_M(nocc_act, mo_energy_act, k_hxc_t)

    # get spin-adapted W matrix
    w_mat_s = get_W(nocc_act, k_hxc_s, m_mat_s, no_screening=False)
    w_mat_t = get_W(nocc_act, k_hxc_t, m_mat_t, no_screening=False)

    # add GSC2 correction
    idx_act = list(range(nocc - nocc_act, nocc + nvir_act))
    mo_energy_gsc = numpy.array(mo_energy, copy=True)
    # alpha-alpha = (singlet + triplet) / 2
    kappa = numpy.einsum("ppqq->pq", (w_mat_s + w_mat_t) / 2.0)
    mo_energy_gsc[idx_act] += numpy.einsum("pp,p->p", kappa, 0.5 - mo_occ[idx_act])
    return mo_energy_gsc


def get_Lpq_w(w_mat, antisym):
    """Get three-index density-fitting matrix for the W matrix.

    Parameters
    ----------
    w_mat : numpy.ndarray
        W matrix
    antisym : bool
        anti-symmetrize W

    Returns
    -------
    Lpq_w : numpy.ndarray
        density-fitting matrix for W
    """
    nmo = w_mat.shape[0]
    if antisym is True:
        w_mat = w_mat - w_mat.transpose(0, 1, 3, 2)
    print("\n*********\nw_mat shape = ", w_mat.shape)
    cd = Cholesky(w_mat, err_tol=1e-5, aosym='s1')
    Lpq_w = cd.kernel()
    Lpq_w = Lpq_w.reshape(-1, nmo, nmo)
    return Lpq_w


class RppRPAwDirect(ppRPA_direct):
    def __init__(
            self, nocc, mo_energy, Lpq, hh_state=5, pp_state=5, nelec="n-2",
            print_thresh=0.1, mo_dip=None):
        super().__init__(
            nocc, mo_energy, Lpq, hh_state=hh_state, pp_state=pp_state,
            nelec=nelec, print_thresh=print_thresh, mo_dip=mo_dip)
        self.fxc = [None, None]  # DFT exchange-correlation kernel, [singlet, triplet]
        self.Lpq_w = None  # Cholesky decomposition of the W matrix
        # screening option
        self.no_screening = False  # no screening
        self.rpa_response = False  # use RPA response
        self.antisym_w = False  # anti-symmetrize W
        self.hyb = 0  # HF exchange factor
        return

    def check_parameter(self):
        super().check_parameter()
        if self.rpa_response is False:
            assert self.fxc[0] is not None and self.fxc[1] is not None
        return

    def dump_flags(self):
        super().dump_flags()
        print("screening option")
        print("no screening = %-s" % self.no_screening)
        print("use RPA response = %-s" % self.rpa_response)
        print("anti-symmetrize W = %-s" % self.antisym_w)
        print("", flush=True)
        return

    def check_memory(self):
        if self.multi == "s":
            oo_dim = int((self.nocc + 1) * self.nocc / 2)
            vv_dim = int((self.nvir + 1) * self.nvir / 2)
        elif self.multi == "t":
            oo_dim = int((self.nocc - 1) * self.nocc / 2)
            vv_dim = int((self.nvir - 1) * self.nvir / 2)
        full_dim = oo_dim + vv_dim
        nmo = self.nocc + self.nvir

        # ppRPA matrix: A block and full matrix, eigenvector
        mem = (3 * full_dim * full_dim + nmo**4 * 2) * 8 / 1.0e6
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
        start_clock("ppRPA-w direct: %s" % multi)

        start_clock("calculate W matrix")
        fxc = self.fxc[0] if self.multi == "s" else self.fxc[1]
        k_hxc = get_K(multi, self.Lpq, fxc=fxc, rpa=self.rpa_response, hyb=self.hyb)
        m_mat = get_M(self.nocc, self.mo_energy, k_hxc)
        w_mat = get_W(self.nocc, k_hxc, m_mat, no_screening=self.no_screening)
        stop_clock("calculate W matrix")

        if self.multi == "s":
            self.exci_s, self.xy_s, self.ec_s = diagonalize_pprpaw_singlet(
                nocc=self.nocc, mo_energy=self.mo_energy, Lpq=self.Lpq, w_mat=w_mat, mu=self.mu
            )
        elif multi == "t":
            self.exci_t, self.xy_t, self.ec_t = diagonalize_pprpa_triplet(
                nocc=self.nocc, mo_energy=self.mo_energy, Lpq=self.Lpq, w_mat=w_mat, mu=self.mu
            )

        stop_clock("ppRPA-w direct: %s" % multi)
        return
