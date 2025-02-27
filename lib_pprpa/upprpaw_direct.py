import numpy
import scipy
import scipy.linalg
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix

from lib_pprpa.upprpa_direct import UppRPA_direct
from lib_pprpa.pprpa_direct import pprpa_orthonormalize_eigenvector
from lib_pprpa.upprpa_direct import upprpa_orthonormalize_eigenvector
from lib_pprpa.pprpa_util import inner_product, get_chemical_potential, start_clock, stop_clock


def diagonalize_pprpa_subspace_same_spin(nocc, mo_energy, w_mat, mu=None):
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
    nmo = len(mo_energy)
    nvir = nmo - nocc
    if mu is None:
        mu = get_chemical_potential(nocc=nocc, mo_energy=mo_energy)

    oo_dim = int((nocc - 1) * nocc / 2)
    tri_row_o, tri_col_o = numpy.tril_indices(nocc, -1)
    tri_row_v, tri_col_v = numpy.tril_indices(nvir, -1)

    # ===========================> A matrix <============================
    A = w_mat[nocc:, nocc:, nocc:, nocc:] - \
        w_mat[nocc:, nocc:, nocc:, nocc:].transpose(0,1,3,2)
    A = A.reshape(nvir*nvir, nvir*nvir)
    orb_sum = numpy.asarray(mo_energy[nocc:, None] \
                            + mo_energy[None, nocc:]).reshape(-1)
    orb_sum -= 2.0 * mu
    numpy.fill_diagonal(A, A.diagonal() + orb_sum)
    A = A.reshape(nvir, nvir, nvir, nvir)
    A = A[tri_row_v, tri_col_v, ...]
    A = A[..., tri_row_v, tri_col_v]
    trace_A = numpy.trace(A)

    # ===========================> B matrix <============================
    B = w_mat[nocc:, nocc:, :nocc, :nocc]
    B = B[tri_row_v, tri_col_v, ...]
    B = B[..., tri_row_o, tri_col_o]

    # ===========================> C matrix <============================
    C = w_mat[:nocc, :nocc, :nocc, :nocc] - \
        w_mat[:nocc, :nocc, :nocc, :nocc].transpose(0,1,3,2)
    C = C.reshape(nocc*nocc, nocc*nocc)
    orb_sum = numpy.asarray(mo_energy[:nocc, None] \
                            + mo_energy[None, :nocc]).reshape(-1)
    orb_sum -= 2.0 * mu
    numpy.fill_diagonal(C, C.diagonal() - orb_sum)
    C = C.reshape(nocc, nocc, nocc, nocc)
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


def diagonalize_pprpa_subspace_diff_spin(nocc, mo_energy, w_mat, mu=None):
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
    nmo = (len(mo_energy[0]), len(mo_energy[1]))
    nvir = (nmo[0]-nocc[0], nmo[1]-nocc[1])
    if mu is None:
        mu = get_chemical_potential(nocc, mo_energy)

    # ===========================> A matrix <============================
    A = w_mat[nocc[0]:, nocc[1]:, nocc[0]:, nocc[1]:]
    A = A.reshape(nvir[0]*nvir[1], nvir[0]*nvir[1])
    orb_sum = numpy.asarray(
        mo_energy[0][nocc[0]:, None] + mo_energy[1][None, nocc[1]:]
    ).reshape(-1)
    orb_sum -= 2.0 * mu
    numpy.fill_diagonal(A, A.diagonal() + orb_sum)
    trace_A = numpy.trace(A)

    # ===========================> B matrix <============================
    B = w_mat[nocc[0]:, nocc[1]:, :nocc[0], :nocc[1]]
    B = B.reshape(nvir[0]*nvir[1], nocc[0]*nocc[1])

    # ===========================> C matrix <============================
    C = w_mat[:nocc[0], :nocc[1], :nocc[0], :nocc[1]]
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


def get_K(fxc, Lpq, test=False):
    """Construct Hartree-exchange-correlation kernel matrix in MO basis.

    Args:
        fxc (list of numpy.ndarray): exchange-correlation kernel, 
            [aaaa, bbbb, aabb].
        Lpq (list of double ndarray): three-center RI matrices in MO space.

    Returns:
        k_hxc (list of numpy.ndarray): Hartree-exchange-correlation kernel
            matrix in MO basis, [aaaa, bbbb, aabb]. 
            bbaa = aabb.transpose(2,3,0,1)
    """
    # aaaa
    pqrs = numpy.einsum('Ppq,Psr->pqrs', Lpq[0], Lpq[0], optimize=True)
    kaa_hxc = fxc[0] + pqrs
    # bbbb
    pqrs = numpy.einsum('Ppq,Psr->pqrs', Lpq[1], Lpq[1], optimize=True)
    kbb_hxc = fxc[1] + pqrs
    # aabb
    pqrs = numpy.einsum('Ppq,Psr->pqrs', Lpq[0], Lpq[1], optimize=True)
    kab_hxc = fxc[2] + pqrs

    if test is True:
        kaa_hxc -= fxc[0]
        kbb_hxc -= fxc[1]
        kab_hxc -= fxc[2]

    return kaa_hxc, kbb_hxc, kab_hxc


def get_M(k_hxc, mo_energy, nmo, nocc):
    """Construct M matrix,
    as defined in J. Phys. Chem. Lett. 2021, 12, 7236−7244, Equation (19).

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
    nvir = (nmo[0]-nocc[0], nmo[1]-nocc[1])
    k_iajb = [k_hxc[0][:nocc[0], nocc[0]:, :nocc[0], nocc[0]:], # aaaa
              k_hxc[1][:nocc[1], nocc[1]:, :nocc[1], nocc[1]:], # bbbb
              k_hxc[2][:nocc[0], nocc[0]:, :nocc[1], nocc[1]:], # aabb
              k_hxc[2][:nocc[0], nocc[0]:, \
                :nocc[1], nocc[1]:].transpose(2,3,0,1) # bbaa
              ]
    k_iajb[0] = k_iajb[0].reshape(nocc[0]*nvir[0], nocc[0]*nvir[0]) + \
        k_iajb[0].transpose(0,1,3,2).reshape(nocc[0]*nvir[0], nocc[0]*nvir[0])
    k_iajb[1] = k_iajb[1].reshape(nocc[1]*nvir[1], nocc[1]*nvir[1]) + \
        k_iajb[1].transpose(0,1,3,2).reshape(nocc[1]*nvir[1], nocc[1]*nvir[1])
    k_iajb[2] = k_iajb[2].reshape(nocc[0]*nvir[0], nocc[1]*nvir[1]) + \
        k_iajb[2].transpose(0,1,3,2).reshape(nocc[0]*nvir[0], nocc[1]*nvir[1])
    k_iajb[3] = k_iajb[3].reshape(nocc[1]*nvir[1], nocc[0]*nvir[0]) + \
        k_iajb[3].transpose(0,1,3,2).reshape(nocc[1]*nvir[1], nocc[0]*nvir[0])

#    k_iajb = [numpy.zeros_like(k_iajb[0]),
#              numpy.zeros_like(k_iajb[1]),
#              numpy.zeros_like(k_iajb[2]),
#              numpy.zeros_like(k_iajb[3])
#              ]

    for i in range(nocc[0]):
        for a in range(nvir[0]):
            k_iajb[0][a + i*nvir[0], a + i*nvir[0]] +=\
                 mo_energy[0][a + nocc[0]] - mo_energy[0][i]

    for i in range(nocc[1]):
        for a in range(nvir[1]):
            k_iajb[1][a + i*nvir[1], a + i*nvir[1]] +=\
                 mo_energy[1][a + nocc[1]] - mo_energy[1][i]

    m_upper = numpy.concatenate((k_iajb[0], k_iajb[2]), axis=1)
    m_lower = numpy.concatenate((k_iajb[3], k_iajb[1]), axis=1)
    m_mat = numpy.concatenate((m_upper, m_lower), axis=0)

#    numpy.set_printoptions(threshold=100000000000)
#    print('====> m_mat')
#    print(m_mat)

    return m_mat


def get_W(k_hxc, m_mat, nmo, nocc, test=False):
    """Calculate W matrix as defined in 
    J. Phys. Chem. Lett. 2021, 12, 7236−7244, Equation (14).

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
    nvir = (nmo[0]-nocc[0], nmo[1]-nocc[1])
    cond_number = numpy.linalg.cond(m_mat)
    print(f"Condition number: {cond_number}")
    if cond_number < 1e12:  # Threshold for well-conditioned matrix
        print("Matrix is well-conditioned.")
    else:
        print("Matrix is ill-conditioned.")
    #sparse_matrix = csc_matrix(m_mat)
    #identity_matrix = csc_matrix(numpy.eye(m_mat.shape[0]))
    #m_inv = spsolve(sparse_matrix, identity_matrix).toarray()
    m_inv = numpy.linalg.inv(m_mat)
#    numpy.set_printoptions(threshold=100000000000)
#    print('====> m_inv')
#    print(m_inv)
    k_pria = [k_hxc[0][:, :, :nocc[0], nocc[0]:], # aaaa
              k_hxc[1][:, :, :nocc[1], nocc[1]:], # bbbb
              k_hxc[2][:, :, :nocc[1], nocc[1]:], # aabb
              k_hxc[2].transpose(2,3,0,1)[:, :, :nocc[0], nocc[0]:], # bbaa
             ]
    k_prai = [k_hxc[0][:, :, nocc[0]:, :nocc[0]], # aaaa
              k_hxc[1][:, :, nocc[1]:, :nocc[1]], # bbbb
              k_hxc[2][:, :, nocc[1]:, :nocc[1]], # aabb
              k_hxc[2].transpose(2,3,0,1)[:, :, nocc[0]:, :nocc[0]], # bbaa
             ]

    K_upper = numpy.concatenate((
        k_pria[0].reshape(nmo[0]*nmo[0], nocc[0]*nvir[0]), 
        k_pria[2].reshape(nmo[0]*nmo[0], nocc[1]*nvir[1])), axis=1)
    K_upper += numpy.concatenate((
        k_prai[0].reshape(nmo[0]*nmo[0], nocc[0]*nvir[0]), 
        k_prai[2].reshape(nmo[0]*nmo[0], nocc[1]*nvir[1])), axis=1)
    K_lower = numpy.concatenate((
        k_pria[3].reshape(nmo[1]*nmo[1], nocc[0]*nvir[0]), 
        k_pria[1].reshape(nmo[1]*nmo[1], nocc[1]*nvir[1])), axis=1)
    K_lower += numpy.concatenate((
        k_prai[3].reshape(nmo[1]*nmo[1], nocc[0]*nvir[0]), 
        k_prai[1].reshape(nmo[1]*nmo[1], nocc[1]*nvir[1])), axis=1)
    K_mat1 = numpy.concatenate((K_upper, K_lower), axis=0)

    k_jbsq = []
    for mat in k_pria:
        k_jbsq.append(mat.transpose(2,3,0,1))
    K_upper = numpy.concatenate((
        k_pria[0].reshape(nocc[0]*nvir[0], nmo[0]*nmo[0]), 
        k_pria[2].reshape(nocc[1]*nvir[1], nmo[0]*nmo[0])), axis=1)
    K_lower = numpy.concatenate((
        k_pria[3].reshape(nocc[0]*nvir[0], nmo[1]*nmo[1]), 
        k_pria[1].reshape(nocc[1]*nvir[1], nmo[1]*nmo[1])), axis=1)
    K_mat2 = numpy.concatenate((K_upper, K_lower), axis=0)

    wm = numpy.dot(K_mat1, m_inv)
    wm = numpy.dot(wm, K_mat2)
#    numpy.set_printoptions(threshold=100000000000)
#    print('====> wm')
#    print(wm)

    wm_list = []
    wm_list.append(wm[:nmo[0]*nmo[0], \
        :nmo[0]*nmo[0]].reshape(nmo[0], nmo[0], nmo[0], nmo[0]))
    wm_list.append(wm[nmo[1]*nmo[1]:, \
        nmo[1]*nmo[1]:].reshape(nmo[1], nmo[1], nmo[1], nmo[1]))
    wm_list.append(wm[:nmo[0]*nmo[0], \
        nmo[1]*nmo[1]:].reshape(nmo[0], nmo[0], nmo[1], nmo[1]))

    w_mat = []
    for i in range(3):
        if test:
            w_mat.append(k_hxc[i])
        else:
#            print('===> khxc ', i)
#            print(k_hxc[i])
#            print('===> wm ', i)
#            print(wm_list[i])
            w_mat.append(k_hxc[i] - wm_list[i])
#            print('===> wmat ', i)
#            print(w_mat[i])
        w_mat[i] = w_mat[i].transpose(0,3,1,2)

    return w_mat


class UppRPAwDirect(UppRPA_direct):
    def __init__(
            self, nocc, mo_energy, Lpq, fxc, hh_state=5, pp_state=5, 
            nelec='n-2', print_thresh=0.1):
        super().__init__(
            nocc, mo_energy, Lpq, hh_state=hh_state, pp_state=pp_state,
            nelec=nelec, print_thresh=print_thresh)
        self.fxc = fxc
        self.kHxc = None
        self.w_mat = None
    
    def get_K(self):
        self.kHxc = get_K(self.fxc, self.Lpq)
        return self.kHxc

    def get_M(self):
        if self.kHxc == None:
            self.get_K()
        return get_M(self.kHxc, self.mo_energy, self.nmo, self.nocc)
    
    def get_W(self):
        m_mat = self.get_M()
        self.w_mat = get_W(self.kHxc, m_mat, self.nmo, self.nocc)
        return self.w_mat

    def kernel(self, subspace=['aa', 'bb', 'ab']):
        self.check_parameter()
        self.dump_flags()
        self.check_memory()
        if self.w_mat == None:
            self.w_mat = self.get_W()

        if 'aa' in subspace:
            start_clock("U-ppRPAw direct: (alpha alpha, alpha alpha)")
            aa_exci, aa_xy, aa_ec = diagonalize_pprpa_subspace_same_spin(
                self.nocc[0], self.mo_energy[0], self.w_mat[0], mu=self.mu
            )
            stop_clock("U-ppRPAw direct: (alpha alpha, alpha alpha)")
        else:
            aa_exci = aa_xy = aa_ec = None

        if 'bb' in subspace:
            start_clock("U-ppRPAw direct: (beta beta, beta beta)")
            bb_exci, bb_xy, bb_ec = diagonalize_pprpa_subspace_same_spin(
                self.nocc[1], self.mo_energy[1], self.w_mat[1], mu=self.mu
            )
            stop_clock("U-ppRPAw direct: (beta beta, beta beta)")
        else:
            bb_exci = bb_xy = bb_ec = None

        if 'ab' in subspace:
            start_clock("U-ppRPAw direct: (alpha beta, alpha beta)")
            ab_exci, ab_xy, ab_ec = diagonalize_pprpa_subspace_diff_spin(
                self.nocc, self.mo_energy, self.w_mat[2], mu=self.mu
            )
            stop_clock("U-ppRPAw direct: (alpha beta, alpha beta)")
        else:
            ab_exci = ab_xy = ab_ec = None

        self.ec = [aa_ec, bb_ec, ab_ec]
        self.exci = [aa_exci, bb_exci, ab_exci]
        self.xy = [aa_xy, bb_xy, ab_xy]

        return
