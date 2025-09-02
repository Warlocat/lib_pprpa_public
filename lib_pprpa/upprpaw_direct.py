import numpy
import scipy
import scipy.linalg

from lib_pprpa.upprpa_direct import UppRPA_direct
from lib_pprpa.pprpa_direct import pprpa_orthonormalize_eigenvector
from lib_pprpa.upprpa_direct import upprpa_orthonormalize_eigenvector, _pprpa_print_eigenvector
from lib_pprpa.pprpa_util import get_chemical_potential, start_clock, stop_clock
from lib_pprpa.gsc import mo_energy_gsc2


def diagonalize_pprpa_subspace_same_spin(nocc, mo_energy, w_mat, mu=None,
                                         active=[None, None]):
    """Diagonalize ppRPA matrix in subspace (alpha alpha, alpha, alpha)
    or (beta beta, beta beta).

    Args:
        nocc (int): number of occupied orbitals.
        mo_energy (double array): orbital energy.
        w_mat (double ndarray): W matrix in MO space.
        mu (double, optional): chemical potential. Defaults to None.

    Returns:
        exci (double array): ppRPA eigenvalue.
        xy (double ndarray): ppRPA eigenvector.
        ec (double): triplet correlation energy.
    """
    w_mat = w_mat.transpose(0,3,1,2)
    # truncate w_mat and mo_energy
    nmo = len(mo_energy)
    nvir = nmo - nocc
    nocc_act, nvir_act = active
    if nocc_act is None:
        nocc_act = nocc
    else:
        nocc_act = min(nocc, nocc_act)
    if nvir_act is None:
        nvir_act = nvir
    else:
        nvir_act = min(nvir, nvir_act)

    if mu is None:
        mu = get_chemical_potential(nocc=nocc, mo_energy=mo_energy)

    oo_dim = int((nocc_act - 1) * nocc_act / 2)
    tri_row_o, tri_col_o = numpy.tril_indices(nocc_act, -1)
    tri_row_v, tri_col_v = numpy.tril_indices(nvir_act, -1)

    # ===========================> A matrix <============================
    A = w_mat[
        nocc:(nocc+nvir_act), nocc:(nocc+nvir_act),\
            nocc:(nocc+nvir_act), nocc:(nocc+nvir_act)] - \
                w_mat[nocc:(nocc+nvir_act), nocc:(nocc+nvir_act),\
                      nocc:(nocc+nvir_act),\
                        nocc:(nocc+nvir_act)].transpose(0,1,3,2)
    A = A.reshape(nvir_act*nvir_act, nvir_act*nvir_act)
    orb_sum = numpy.asarray(mo_energy[nocc:(nocc+nvir_act), None] \
                            + mo_energy[None, nocc:(nocc+nvir_act)]).reshape(-1)
    orb_sum -= 2.0 * mu
    numpy.fill_diagonal(A, A.diagonal() + orb_sum)
    A = A.reshape(nvir, nvir, nvir, nvir)
    A = A[tri_row_v, tri_col_v, ...]
    A = A[..., tri_row_v, tri_col_v]
    trace_A = numpy.trace(A)

    # ===========================> B matrix <============================
    B = w_mat[nocc:(nocc+nvir_act), nocc:(nocc+nvir_act),\
              (nocc-nocc_act):nocc, (nocc-nocc_act):nocc] -\
                w_mat[nocc:(nocc+nvir_act), nocc:(nocc+nvir_act),\
                      (nocc-nocc_act):nocc,\
                        (nocc-nocc_act):nocc].transpose(0,1,3,2)
    B = B[tri_row_v, tri_col_v, ...]
    B = B[..., tri_row_o, tri_col_o]

    # ===========================> C matrix <============================
    C = w_mat[(nocc-nocc_act):nocc, (nocc-nocc_act):nocc, \
              (nocc-nocc_act):nocc, (nocc-nocc_act):nocc] - \
        w_mat[(nocc-nocc_act):nocc, (nocc-nocc_act):nocc, \
              (nocc-nocc_act):nocc, (nocc-nocc_act):nocc].transpose(0,1,3,2)
    C = C.reshape(nocc_act*nocc_act, nocc_act*nocc_act)
    orb_sum = numpy.asarray(mo_energy[(nocc-nocc_act):nocc, None] \
                            + mo_energy[None, (nocc-nocc_act):nocc]).reshape(-1)
    orb_sum -= 2.0 * mu
    numpy.fill_diagonal(C, C.diagonal() - orb_sum)
    C = C.reshape(nocc_act, nocc_act, nocc_act, nocc_act)
    C = C[tri_row_o, tri_col_o, ...]
    C = C[..., tri_row_o, tri_col_o]

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


def diagonalize_pprpa_subspace_diff_spin(nocc, mo_energy, w_mat, mu=None,
                                         active=[(None,None),(None,None)]):
    """Diagonalize ppRPA matrix in subspace (alpha beta, alpha, beta).

    Args:
        nocc(tuple of int): number of occupied orbitals, (nalpha, nbeta).
        mo_energy (list of double array): orbital energies.
        w_mat (list of double ndarray): W matrices in MO space.

    Kwarg:
        mu (double): chemical potential.

    Returns:
        exci (double array): ppRPA eigenvalue.
        xy (double ndarray): ppRPA eigenvector.
        ec (double): correlation energy from one subspace.
    """
    w_mat = w_mat.transpose(0,3,1,2)
    nmo = (len(mo_energy[0]), len(mo_energy[1]))
    nvir = (nmo[0]-nocc[0], nmo[1]-nocc[1])
    if mu is None:
        mu = get_chemical_potential(nocc, mo_energy)

    act_alpha, act_beta = active
    nocc_act = [act_alpha[0], act_beta[0]]
    nvir_act = [act_alpha[1], act_beta[1]]
    nocc_act[0] = nocc[0] if nocc_act[0] == None else min(nocc_act[0], nocc[0])
    nocc_act[1] = nocc[1] if nocc_act[1] == None else min(nocc_act[1], nocc[1])
    nvir_act[0] = nvir[0] if nvir_act[0] == None else min(nvir_act[0], nvir[0])
    nvir_act[1] = nvir[1] if nvir_act[1] == None else min(nvir_act[1], nvir[1])

    # ===========================> A matrix <============================
    A = w_mat[nocc[0]:(nocc[0]+nvir_act[0]), nocc[1]:(nocc[1]+nvir_act[1]),\
              nocc[0]:(nocc[0]+nvir_act[0]), nocc[1]:(nocc[1]+nvir_act[1])]
    A = A.reshape(nvir_act[0]*nvir_act[1], nvir_act[0]*nvir_act[1])
    orb_sum = numpy.asarray(
        mo_energy[0][nocc[0]:(nocc[0]+nvir_act[0]), None] +\
            mo_energy[1][None, nocc[1]:(nocc[1]+nvir_act[1])]
    ).reshape(-1)
    orb_sum -= 2.0 * mu
    numpy.fill_diagonal(A, A.diagonal() + orb_sum)
    trace_A = numpy.trace(A)

    # ===========================> B matrix <============================
    B = w_mat[nocc[0]:(nocc[0]+nvir_act[0]), nocc[1]:(nocc[1]+nvir_act[1]),\
              (nocc[0]-nocc_act[0]):nocc[0], (nocc[1]-nocc_act[1]):nocc[1]]
    B = B.reshape(nvir_act[0]*nvir_act[1], nocc_act[0]*nocc_act[1])

    # ===========================> C matrix <============================
    C = w_mat[(nocc[0]-nocc_act[0]):nocc[0], (nocc[1]-nocc_act[1]):nocc[1],\
              (nocc[0]-nocc_act[0]):nocc[0], (nocc[1]-nocc_act[1]):nocc[1]]
    C = C.reshape(nocc_act[0]*nocc_act[1], nocc_act[0]*nocc_act[1])
    orb_sum = numpy.asarray(
        mo_energy[0][(nocc[0]-nocc_act[0]):nocc[0], None] +\
            mo_energy[1][None, (nocc[1]-nocc_act[1]):nocc[1]]
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
    M[:nocc_act[0]*nocc_act[1]][:] *= -1.0

    # =====================> solve for eigenpairs <======================
    exci, xy = scipy.linalg.eig(M)
    exci = exci.real
    xy = xy.T  # Fortran to Python order

    # sort eigenpairs
    idx = exci.argsort()
    exci = exci[idx]
    xy = xy[idx, :]
    upprpa_orthonormalize_eigenvector('abab', nocc, exci, xy)

    sum_exci = numpy.sum(exci[nocc_act[0]*nocc_act[1]:])
    ec = sum_exci - trace_A

    return exci, xy, ec


def get_K(Lpq, fxc=None, rpa=False):
    """Construct Hartree-exchange-correlation kernel matrix in MO basis.

    Args:
        Lpq (list of double ndarray): three-center RI matrices in MO space.
        fxc (list of numpy.ndarray): exchange-correlation kernel,
            [aaaa, bbbb, aabb].
        rpa (bool): whether to include exchange-correlation kernel.

    Returns:
        k_hxc (list of numpy.ndarray): Hartree-exchange-correlation kernel
            matrix in MO basis, [aaaa, bbbb, aabb].
            bbaa = aabb.transpose(2,3,0,1)
    """
    if rpa is False:
        assert fxc is not None

    # aaaa
    pqrs = numpy.einsum('Lpq,Lsr->pqrs', Lpq[0], Lpq[0], optimize=True)
    kaa_hxc = pqrs if rpa else fxc[0] + pqrs

    # bbbb
    pqrs = numpy.einsum('Lpq,Lsr->pqrs', Lpq[1], Lpq[1], optimize=True)
    kbb_hxc = pqrs if rpa else fxc[1] + pqrs

    # aabb
    pqrs = numpy.einsum('Lpq,Lsr->pqrs', Lpq[0], Lpq[1], optimize=True)
    kab_hxc = pqrs if rpa else fxc[2] + pqrs

    return kaa_hxc, kbb_hxc, kab_hxc


def get_M(k_hxc, mo_energy, nmo, nocc):
    """Construct M matrix,
    as defined in J. Phys. Chem. Lett. 2021, 12, 7236-7244, Equation (19).

    Args:
        k_hxc (list of numpy.ndarray): Hartree-exchange-correlation kernel
            matrix in MO basis, [aaaa, bbbb, aabb].
            bbaa = aabb.transpose(2,3,0,1)
        mo_energy (list of numpy.ndarray): AO-to-MO coefficients, (alpha, beta).
        nmo (list of int): number of MOs.
        nocc (list of int): number of occupied orbitals, (alpha, beta).

    Returns:
        m_mat (numpy.ndarray): M matrix.
    """
    if isinstance(nmo, int):
        nmo = (nmo, nmo)
    nvir = (nmo[0] - nocc[0], nmo[1] - nocc[1])
    k_iajb = [k_hxc[0][:nocc[0], nocc[0]:, :nocc[0], nocc[0]:], # aaaa
              k_hxc[1][:nocc[1], nocc[1]:, :nocc[1], nocc[1]:], # bbbb
              k_hxc[2][:nocc[0], nocc[0]:, :nocc[1], nocc[1]:], # aabb
              k_hxc[2][:nocc[0], nocc[0]:, \
                :nocc[1], nocc[1]:].transpose(2,3,0,1) # bbaa
              ]
    k_iajb[0] = k_iajb[0].reshape(nocc[0] * nvir[0], nocc[0] * nvir[0]) * 2
    k_iajb[1] = k_iajb[1].reshape(nocc[1] * nvir[1], nocc[1] * nvir[1]) * 2
    k_iajb[2] = k_iajb[2].reshape(nocc[0] * nvir[0], nocc[1] * nvir[1]) * 2
    k_iajb[3] = k_iajb[3].reshape(nocc[1] * nvir[1], nocc[0] * nvir[0]) * 2

    # orbital energy contribution
    orb_diff = mo_energy[0][None, nocc[0] :] - mo_energy[0][: nocc[0], None]
    numpy.fill_diagonal(k_iajb[0], k_iajb[0].diagonal() + orb_diff.reshape(-1))

    orb_diff = mo_energy[1][None, nocc[1] :] - mo_energy[1][: nocc[1], None]
    numpy.fill_diagonal(k_iajb[1], k_iajb[1].diagonal() + orb_diff.reshape(-1))

    m_upper = numpy.concatenate((k_iajb[0], k_iajb[2]), axis=1)
    m_lower = numpy.concatenate((k_iajb[3], k_iajb[1]), axis=1)
    m_mat = numpy.concatenate((m_upper, m_lower), axis=0)

    return m_mat


def get_W(k_hxc, m_mat, nmo, nocc, no_screening=False):
    """Calculate W matrix as defined in
    J. Phys. Chem. Lett. 2021, 12, 7236-7244, Equation (14).

    Args:
        k_hxc (list of numpy.ndarray): Hartree-exchange-correlation kernel
            matrix in MO basis, [aaaa, bbbb, aabb].
            bbaa = aabb.transpose(2,3,0,1)
        m_mat (numpy.ndarray): M matrix.
        nmo (list of int): number of MOs.
        nocc (list of int): number of occupied orbitals, (alpha, beta).

    Returns:
        w_mat (list of numpy.ndarray): W matrices, [aaaa, bbbb, aabb].
    """
    if isinstance(nmo, int):
        nmo = (nmo, nmo)
    nvir = (nmo[0] - nocc[0], nmo[1] - nocc[1])
    cond_number = numpy.linalg.cond(m_mat)
    print(f"M matrix Condition number: {cond_number}")
    if cond_number < 1e12:  # Threshold for well-conditioned matrix
        print("Matrix is well-conditioned.")
    else:
        print("Matrix is ill-conditioned.")
    m_inv = numpy.linalg.inv(m_mat)

    k_pria = [
        k_hxc[0][:, :, : nocc[0], nocc[0] :],  # aaaa
        k_hxc[1][:, :, : nocc[1], nocc[1] :],  # bbbb
        k_hxc[2][:, :, : nocc[1], nocc[1] :],  # aabb
        k_hxc[2].transpose(2, 3, 0, 1)[:, :, : nocc[0], nocc[0] :],  # bbaa
    ]

    K_upper = numpy.concatenate((
        k_pria[0].reshape(nmo[0]*nmo[0], nocc[0]*nvir[0]),
        k_pria[2].reshape(nmo[0]*nmo[0], nocc[1]*nvir[1])), axis=1) * 2

    K_lower = numpy.concatenate((
        k_pria[3].reshape(nmo[1]*nmo[1], nocc[0]*nvir[0]),
        k_pria[1].reshape(nmo[1]*nmo[1], nocc[1]*nvir[1])), axis=1) * 2

    K_mat1 = numpy.concatenate((K_upper, K_lower), axis=0).copy()

    k_jbsq = []
    for mat in k_pria:
        k_jbsq.append(mat.transpose(2,3,0,1))
    K_upper = numpy.concatenate((
        k_jbsq[0].reshape(nocc[0]*nvir[0], nmo[0]*nmo[0]),
        k_jbsq[2].reshape(nocc[1]*nvir[1], nmo[0]*nmo[0])), axis=0)
    K_lower = numpy.concatenate((
        k_jbsq[3].reshape(nocc[0]*nvir[0], nmo[1]*nmo[1]),
        k_jbsq[1].reshape(nocc[1]*nvir[1], nmo[1]*nmo[1])), axis=0)
    K_mat2 = numpy.concatenate((K_upper, K_lower), axis=1).copy()

    wm = numpy.dot(K_mat1, m_inv)
    wm = numpy.dot(wm, K_mat2)

    wm_list = []
    wm_list.append(wm[:nmo[0]*nmo[0], \
        :nmo[0]*nmo[0]].reshape(nmo[0], nmo[0], nmo[0], nmo[0]))
    wm_list.append(wm[nmo[1]*nmo[1]:, \
        nmo[1]*nmo[1]:].reshape(nmo[1], nmo[1], nmo[1], nmo[1]))
    wm_list.append(wm[:nmo[0]*nmo[0], \
        nmo[1]*nmo[1]:].reshape(nmo[0], nmo[0], nmo[1], nmo[1]))

    w_mat = []
    for i in range(3):
        if no_screening:
            w_mat.append(k_hxc[i])
        else:
            w_mat.append(k_hxc[i] - wm_list[i])

    return w_mat


def _is_int_or_none(x):
    # Accept plain ints (not bools) or None
    return x is None or (isinstance(x, int) and not isinstance(x, bool))

def _as_pair(obj):
    """Convert an iterable to a validated 2-item list [nocc, nvir]."""
    try:
        a, b = list(obj)
    except Exception:
        raise ValueError("Each active-space pair must be an iterable of length 2.")
    if not (_is_int_or_none(a) and _is_int_or_none(b)):
        raise TypeError("Active-space entries must be integers or None.")
    if isinstance(a, int) and a < 0:
        raise ValueError("nocc_act must be >= 0 or None.")
    if isinstance(b, int) and b < 0:
        raise ValueError("nvir_act must be >= 0 or None.")
    return [a, b]

def process_active_space(spec):
    """
    Normalize user-defined active space specs for UHF/UKS into:
        [[nocc_act_alpha, nvir_act_alpha],
         [nocc_act_beta,  nvir_act_beta]]

    Accepted inputs:
      - [nocc_act, nvir_act]              (applies to both alpha and beta)
      - (nocc_act, nvir_act)
      - [[nocc_acta, nvir_acta], [nocc_actb, nvir_actb]]
      - ((nocc_acta, nvir_acta), (nocc_actb, nvir_actb))

    Integers must be >= 0; None means “all” for that category.
    """
    if isinstance(spec, (list, tuple)):
        # Single pair → apply to both spin channels
        if len(spec) == 2 and all(_is_int_or_none(x) for x in spec):
            pair = _as_pair(spec)
            return [pair[:], pair[:]]

        # Two pairs → alpha then beta
        if len(spec) == 2 and all(isinstance(s, (list, tuple)) for s in spec):
            alpha = _as_pair(spec[0])
            beta  = _as_pair(spec[1])
            return [alpha, beta]

    raise ValueError(
        "Active-space input must be either [nocc_act, nvir_act] or "
        "[[nocc_acta, nvir_acta], [nocc_actb, nvir_actb]], with ints or None."
    )

def _analyze_upprpaw_direct(exci, xy, nocc, nvir, nelec='n-2', print_thresh=0.1,
                            hh_state=5, pp_state=5, nocc_fro=0):
    print('\nanalyze U-ppRPAW results.')
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


class UppRPAwDirect(UppRPA_direct):
    def __init__(
            self, nocc, mo_energy, Lpq, fxc=None, hh_state=5, pp_state=5,
            nelec='n-2', active=[None, None], print_thresh=0.1,
            with_screening = True,
            use_rpa4='none' # avail: 'response', 'all', 'none'
            ):
        super().__init__(
            nocc, mo_energy, Lpq, hh_state=hh_state, pp_state=pp_state,
            nelec=nelec, print_thresh=print_thresh)
        self.fxc = fxc
        self.w_mat = None
        self.use_rpa4 = use_rpa4.lower()
        self.with_screening = with_screening
        self.active = process_active_space(active)

    def get_K(self, use_rpa=False):
        kHxc = get_K(self.Lpq, fxc=self.fxc, rpa=use_rpa)
        return kHxc

    def get_M(self, kHxc=None, use_rpa=False):
        if kHxc == None:
            kHxc=self.get_K(use_rpa=use_rpa)
        return get_M(kHxc, self.mo_energy, self.nmo, self.nocc)

    def get_W(self):
        use_rpa4 = self.use_rpa4
        if use_rpa4 == 'none':
            kHxc = self.get_K(use_rpa=False)
            m_mat = self.get_M(kHxc=kHxc, use_rpa=False)
        elif use_rpa4 == 'all':
            kHxc = self.get_K(use_rpa=True)
            m_mat = self.get_M(kHxc=kHxc, use_rpa=True)
        elif use_rpa4 == 'response':
            kHxc = self.get_K(use_rpa=False)
            m_mat = self.get_M(kHxc=None, use_rpa=True)
        else:
            raise NotImplementedError

        no_screening = not self.with_screening
        self.w_mat = get_W(kHxc, m_mat, self.nmo, self.nocc,
                           no_screening=no_screening)
        return self.w_mat

    def check_parameter(self, gsc2_e=True, mf=None):
        assert 0.0 < self.print_thresh < 1.0
        assert self.nelec in ["n-2", "n+2"]
        if self.mu is None:
            self.mu = get_chemical_potential(nocc=self.nocc,
                                             mo_energy=self.mo_energy)
        if self.w_mat == None:
            self.w_mat = self.get_W()
        if gsc2_e:
            assert mf is not None
            self.mo_energy = mo_energy_gsc2(mf, self.w_mat)
        return

    def kernel(self, subspace=['aa', 'bb', 'ab'], gsc2_e=False, mf=None):
        self.check_parameter(gsc2_e=gsc2_e, mf=mf)
        self.dump_flags()
        self.check_memory()

        if 'aa' in subspace:
            start_clock("U-ppRPAw direct: (alpha alpha, alpha alpha)")
            aa_exci, aa_xy, aa_ec = diagonalize_pprpa_subspace_same_spin(
                self.nocc[0], self.mo_energy[0], self.w_mat[0], mu=self.mu,
                active=self.active[0]
            )
            stop_clock("U-ppRPAw direct: (alpha alpha, alpha alpha)")
        else:
            aa_exci = aa_xy = aa_ec = None

        if 'bb' in subspace:
            start_clock("U-ppRPAw direct: (beta beta, beta beta)")
            bb_exci, bb_xy, bb_ec = diagonalize_pprpa_subspace_same_spin(
                self.nocc[1], self.mo_energy[1], self.w_mat[1], mu=self.mu,
                active=self.active[1]
            )
            stop_clock("U-ppRPAw direct: (beta beta, beta beta)")
        else:
            bb_exci = bb_xy = bb_ec = None

        if 'ab' in subspace:
            start_clock("U-ppRPAw direct: (alpha beta, alpha beta)")
            ab_exci, ab_xy, ab_ec = diagonalize_pprpa_subspace_diff_spin(
                self.nocc, self.mo_energy, self.w_mat[2], mu=self.mu,
                active=self.active
            )
            stop_clock("U-ppRPAw direct: (alpha beta, alpha beta)")
        else:
            ab_exci = ab_xy = ab_ec = None

        self.ec = [aa_ec, bb_ec, ab_ec]
        self.exci = [aa_exci, bb_exci, ab_exci]
        self.xy = [aa_xy, bb_xy, ab_xy]

        return

    def analyze(self, nocc_fro=(0, 0)):
        nocc = self.nocc
        nmo = (len(self.mo_energy[0]), len(self.mo_energy[1]))
        nvir = (nmo[0] - nocc[0], nmo[1] - nocc[1])
        act_alpha, act_beta = self.active
        nocc_act = [act_alpha[0], act_beta[0]]
        nvir_act = [act_alpha[1], act_beta[1]]
        nocc_act[0] = nocc[0] if nocc_act[0] == None else min(nocc_act[0], nocc[0])
        nocc_act[1] = nocc[1] if nocc_act[1] == None else min(nocc_act[1], nocc[1])
        nvir_act[0] = nvir[0] if nvir_act[0] == None else min(nvir_act[0], nvir[0])
        nvir_act[1] = nvir[1] if nvir_act[1] == None else min(nvir_act[1], nvir[1])
        nocc = nocc_act
        nvir = nvir_act
        nmo = [nocc[0] + nvir[0], nocc[1] + nvir[1]]
        _analyze_upprpaw_direct(
            self.exci, self.xy, nocc, nvir,
            nelec=self.nelec,
            print_thresh=self.print_thresh, hh_state=self.hh_state,
            pp_state=self.pp_state, nocc_fro=nocc_fro
        )

    def dump_flags(self):
        nocc = self.nocc
        nmo = (len(self.mo_energy[0]), len(self.mo_energy[1]))
        nvir = (nmo[0] - nocc[0], nmo[1] - nocc[1])
        act_alpha, act_beta = self.active
        nocc_act = [act_alpha[0], act_beta[0]]
        nvir_act = [act_alpha[1], act_beta[1]]
        nocc_act[0] = nocc[0] if nocc_act[0] == None else min(nocc_act[0], nocc[0])
        nocc_act[1] = nocc[1] if nocc_act[1] == None else min(nocc_act[1], nocc[1])
        nvir_act[0] = nvir[0] if nvir_act[0] == None else min(nvir_act[0], nvir[0])
        nvir_act[1] = nvir[1] if nvir_act[1] == None else min(nvir_act[1], nvir[1])
        nocc = nocc_act
        nvir = nvir_act
        nmo = [nocc[0] + nvir[0], nocc[1] + nvir[1]]
        # ====================> calculate dimensions <===================
        # (alpha, alpha) subspace
        aavv_dim = int(nvir[0] * (nvir[0] + 1) / 2)
        aaoo_dim = int(nocc[0] * (nocc[0] + 1) / 2)
        # (alpha, beta) subspace
        abvv_dim = int(nvir[0] * nvir[1])
        aboo_dim = int(nocc[0] * nocc[1])
        # (beta, beta) subspace
        bbvv_dim = int(nvir[1] * (nvir[1] + 1) / 2)
        bboo_dim = int(nocc[1] * (nocc[1] + 1) / 2)

        print('\n******** %s ********' % self.__class__)
        print('naux = %d' % self.naux)
        print('nmo = %d (%d alpha, %d beta)'
              % (nmo[0]+nmo[1], nmo[0], nmo[1]))
        print('nocc = %d (%d alpha, %d beta), nvir = %d (%d alpha, %d beta)'
              % (
                  nocc[0] + nocc[1], nocc[0], nocc[1],
                  nvir[0] + nvir[1], nvir[0], nvir[1]))
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
        nocc = self.nocc
        nmo = (len(self.mo_energy[0]), len(self.mo_energy[1]))
        nvir = (nmo[0] - nocc[0], nmo[1] - nocc[1])
        act_alpha, act_beta = self.active
        nocc_act = [act_alpha[0], act_beta[0]]
        nvir_act = [act_alpha[1], act_beta[1]]
        nocc_act[0] = nocc[0] if nocc_act[0] == None else min(nocc_act[0], nocc[0])
        nocc_act[1] = nocc[1] if nocc_act[1] == None else min(nocc_act[1], nocc[1])
        nvir_act[0] = nvir[0] if nvir_act[0] == None else min(nvir_act[0], nvir[0])
        nvir_act[1] = nvir[1] if nvir_act[1] == None else min(nvir_act[1], nvir[1])
        nocc = nocc_act
        nvir = nvir_act
        # ====================> calculate dimensions <===================
        # (alpha, alpha) subspace
        aavv_dim = int(nvir[0] * (nvir[0] + 1) / 2)
        aaoo_dim = int(nocc[0] * (nocc[0] + 1) / 2)
        aafull_dim = aavv_dim + aaoo_dim
        # (alpha, beta) subspace
        abvv_dim = int(nvir[0] * nvir[1])
        aboo_dim = int(nocc[0] * nocc[1])
        abfull_dim = abvv_dim + aboo_dim
        # (beta, beta) subspace
        bbvv_dim = int(nvir[1] * (nvir[1] + 1) / 2)
        bboo_dim = int(nocc[1] * (nocc[1] + 1) / 2)
        bbfull_dim = bbvv_dim + bboo_dim

        full_dim = max(aafull_dim, abfull_dim, bbfull_dim)

        mem = (3 * full_dim * full_dim) * 8 / 1.0e6
        if mem < 1000:
            print("U-ppRPAW needs at least %.1f MB memory." % mem)
        else:
            print("U-ppRPAW needs at least %.1f GB memory." % (mem / 1.0e3))
        return
