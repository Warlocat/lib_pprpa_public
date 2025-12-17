"""Calculations of analytical energy gradients for particle-particle random phase approximation.
Author: Chaoqun Zhang <cq_zhang@outlook.com>
"""

import numpy as np
from functools import reduce
from lib_pprpa.pprpa_util import start_clock, stop_clock, GMRES_Pople, GMRES_wrapper

try:
    from socutils.scf import spinor_hf
    from socutils.dft import dft as spinor_dft

    with_socuils = True
except:
    with_socuils = False
    pass


def get_xy_full(xy, oo_dim, mult='t'):
    """Expand the lower triangular xy matrix to the full matrix."""
    vv_dim = len(xy) - oo_dim
    if mult == 't':
        ndim_v = round(0.5 * (np.sqrt(8 * vv_dim + 1) + 1)) if vv_dim > 0 else 0
        ndim_o = round(0.5 * (np.sqrt(8 * oo_dim + 1) + 1)) if oo_dim > 0 else 0
        occ_y_mat = np.zeros((ndim_o, ndim_o), dtype=xy.dtype)
        vir_x_mat = np.zeros((ndim_v, ndim_v), dtype=xy.dtype)
        occ_y_mat[np.tril_indices(ndim_o, -1)] = xy[:oo_dim]
        vir_x_mat[np.tril_indices(ndim_v, -1)] = xy[oo_dim:]
        occ_y_mat = occ_y_mat - occ_y_mat.T
        vir_x_mat = vir_x_mat - vir_x_mat.T
    else:
        ndim_v = round(0.5 * (np.sqrt(8 * vv_dim + 1) - 1)) if vv_dim > 0 else 0
        ndim_o = round(0.5 * (np.sqrt(8 * oo_dim + 1) - 1)) if oo_dim > 0 else 0
        occ_y_mat = np.zeros((ndim_o, ndim_o), dtype=xy.dtype)
        vir_x_mat = np.zeros((ndim_v, ndim_v), dtype=xy.dtype)
        occ_y_mat[np.tril_indices(ndim_o)] = xy[:oo_dim]
        vir_x_mat[np.tril_indices(ndim_v)] = xy[oo_dim:]
        occ_y_mat = occ_y_mat + occ_y_mat.T
        vir_x_mat = vir_x_mat + vir_x_mat.T
        np.fill_diagonal(occ_y_mat, 1.0 / np.sqrt(2.0) * occ_y_mat.diagonal())
        np.fill_diagonal(vir_x_mat, 1.0 / np.sqrt(2.0) * vir_x_mat.diagonal())

    return occ_y_mat, vir_x_mat


def make_rdm1_unrelaxed_from_xy_full(occ_y_mat, vir_x_mat, diag=True):
    """Make unrelaxed one-particle density matrix from the full X and Y matrices."""
    if diag:
        di = -np.einsum('ij,ij->i', occ_y_mat.conj(), occ_y_mat)
        da = np.einsum('ab,ab->a', vir_x_mat.conj(), vir_x_mat)
        # combine the two parts
        den = np.concatenate((di, da))
    else:
        print('Warning: non-diagonal 1e-RDM is not well-defined for pp-RPA with both DEA and DIP blocks.')
        den_v = np.einsum('ac,bc->ba', vir_x_mat.conj(), vir_x_mat)
        den_o = -np.einsum('ik,jk->ij', occ_y_mat.conj(), occ_y_mat)
        den = np.zeros((den_v.shape[0] + den_o.shape[0], den_v.shape[1] + den_o.shape[1]), dtype=den_v.dtype)
        den[: den_o.shape[0], : den_o.shape[1]] = den_o
        den[den_o.shape[0] :, den_o.shape[1] :] = den_v
    return den


def make_rdm1_unrelaxed(xy, oo_dim, mult='t', diag=True):
    """Make unrelaxed one-particle density matrix from the XY coefficients."""
    occ_y_mat, vir_x_mat = get_xy_full(xy, oo_dim, mult)
    return make_rdm1_unrelaxed_from_xy_full(occ_y_mat, vir_x_mat, diag=diag)


def make_tdm1(xy1, xy2, oo_dim, mult='t'):
    """Make transition density matrix from the XY coefficients."""
    assert len(xy1) == len(xy2)
    assert len(xy1) >= oo_dim
    occ_y_mat1, vir_x_mat1 = get_xy_full(xy1, oo_dim, mult)
    occ_y_mat2, vir_x_mat2 = get_xy_full(xy2, oo_dim, mult)
    vv_dim = len(xy1) - oo_dim

    if oo_dim == 0:
        tdm = np.einsum('ac,bc->ab', vir_x_mat1.conj(), vir_x_mat2).T
        diagonal_correction = np.einsum('ab,ab->', vir_x_mat1.conj(), vir_x_mat2)
    elif vv_dim == 0:
        tdm = -np.einsum('ik,jk->ij', occ_y_mat1.conj(), occ_y_mat2)
        diagonal_correction = np.einsum('ij,ij->', occ_y_mat1.conj(), occ_y_mat2)
    else:
        print('Warning: TDM is not well-defined for pp-RPA with both DEA and DIP blocks.')
        tdm_v = np.einsum('ac,bc->ab', vir_x_mat1.conj(), vir_x_mat2).T
        tdm_o = -np.einsum('ik,jk->ij', occ_y_mat1.conj(), occ_y_mat2)
        diagonal_correction = np.einsum('ab,ab->', vir_x_mat1.conj(), vir_x_mat2) + np.einsum(
            'ij,ij->', occ_y_mat1.conj(), occ_y_mat2
        )
        tdm = np.zeros((tdm_v.shape[0] + tdm_o.shape[0], tdm_v.shape[1] + tdm_o.shape[1]), dtype=tdm_v.dtype)
        tdm[: tdm_o.shape[0], : tdm_o.shape[1]] = tdm_o
        tdm[tdm_o.shape[0] :, tdm_o.shape[1] :] = tdm_v

    return tdm, diagonal_correction


def make_rdm2_from_xy_full(occ_y_mat, vir_x_mat):
    r"""DO NOT USE IT IN ANY PRACTICAL CALCULATIONS.
     FOR TESTING PURPOSE ONLY.

    Make two-particle density matrix from the XY coefficients.

    \Gamma_{ab,cd} = X_{ab}^* X_{cd}
    \Gamma_{ij,ab} = Y_{ij}^* X_{ab}
    \Gamma_{ab,ij} = X_{ab}^* Y_{ij}
    \Gamma_{ij,kl} = Y_{ij}^* Y_{kl}
    """
    dijkl = np.einsum('ij,kl->ijkl', occ_y_mat.conj(), occ_y_mat)
    dijab = np.einsum('ij,ab->ijab', occ_y_mat.conj(), vir_x_mat)
    dabij = np.einsum('ab,ij->abij', vir_x_mat.conj(), occ_y_mat)
    dabcd = np.einsum('ab,cd->abcd', vir_x_mat.conj(), vir_x_mat)
    o_size = len(occ_y_mat)
    v_size = len(vir_x_mat)
    size = o_size + v_size
    gamma2e = np.zeros((size, size, size, size), dtype=occ_y_mat.dtype)
    gamma2e[:o_size, :o_size, :o_size, :o_size] = dijkl
    gamma2e[:o_size, :o_size, o_size:, o_size:] = dijab
    gamma2e[o_size:, o_size:, :o_size, :o_size] = dabij
    gamma2e[o_size:, o_size:, o_size:, o_size:] = dabcd

    return gamma2e


def choose_slice(label, nfrozen_occ, nocc, nvir, nfrozen_vir):
    """Choose the slice for the given label.
    "i" for active occupied orbitals;
    "a" for active virtual orbitals;
    "p" for all active orbitals;
    "ip" for frozen occupied orbitals;
    "ap" for frozen virtual orbitals;
    "I" for all occupied orbitals;
    "A" for all virtual orbitals;
    "P" for all orbitals;

    In the energy ordering, frozen occ -> active occ -> active vir -> frozen vir.
    """
    if label == 'i':
        return slice(nfrozen_occ, nfrozen_occ + nocc)
    elif label == 'a':
        return slice(nfrozen_occ + nocc, nfrozen_occ + nocc + nvir)
    elif label == 'p':
        return slice(nfrozen_occ, nfrozen_occ + nocc + nvir)
    elif label == 'ip':
        return slice(0, nfrozen_occ)
    elif label == 'ap':
        return slice(nfrozen_occ + nocc + nvir, nfrozen_occ + nocc + nvir + nfrozen_vir)
    elif label == 'I':
        return slice(0, nfrozen_occ + nocc)
    elif label == 'A':
        return slice(nfrozen_occ + nocc, nfrozen_occ + nocc + nvir + nfrozen_vir)
    elif label == 'P':
        return slice(0, nfrozen_occ + nocc + nvir + nfrozen_vir)
    else:
        raise ValueError('label = {}. is not valid in choose_slice'.format(label))


def choose_range(label, nfrozen_occ, nocc, nvir, nfrozen_vir):
    """Choose the range list for the given label.
    "i" for active occupied orbitals;
    "a" for active virtual orbitals;
    "p" for all active orbitals;
    "ip" for frozen occupied orbitals;
    "ap" for frozen virtual orbitals;
    "I" for all occupied orbitals;
    "A" for all virtual orbitals;
    "P" for all orbitals;

    In the energy ordering, frozen occ -> active occ -> active vir -> frozen vir.
    """
    if label == 'i':
        return range(nfrozen_occ, nfrozen_occ + nocc)
    elif label == 'a':
        return range(nfrozen_occ + nocc, nfrozen_occ + nocc + nvir)
    elif label == 'p':
        return range(nfrozen_occ, nfrozen_occ + nocc + nvir)
    elif label == 'ip':
        return range(0, nfrozen_occ)
    elif label == 'ap':
        return range(nfrozen_occ + nocc + nvir, nfrozen_occ + nocc + nvir + nfrozen_vir)
    elif label == 'I':
        return range(0, nfrozen_occ + nocc)
    elif label == 'A':
        return range(nfrozen_occ + nocc, nfrozen_occ + nocc + nvir + nfrozen_vir)
    elif label == 'P':
        return range(0, nfrozen_occ + nocc + nvir + nfrozen_vir)
    else:
        raise ValueError('label = {}. is not valid in choose_slice'.format(label))


def contraction_1rdm_Lpq_diag(
    den, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, label1, label2, den_label='p', rhf=False
):
    r"""
    Contraction in the form of
        I_{ts} = \sum_{r} D_{rr} \langle rt||rs \rangle
               = \sum_{rP} D_{rr} (L^P_{rr} L^P_{ts} - L^P_{rs} L^P_{tr})

    Args:
        den: input density matrix.
        Lpq_full: full Lpq matrix including frozen orbitals.
        nocc: number of active occupied orbitals.
        nvir: number of active virtual orbitals.
        nfrozen_occ: number of frozen occupied orbitals.
        nfrozen_vir: number of frozen virtual orbitals.
        label1: label for the first index t
        label2: label for the second index s
        den_label: label for the density matrix index r
    Returns:
        out: contracted intermediates.
    """
    den_slice = choose_slice(den_label, nfrozen_occ, nocc, nvir, nfrozen_vir)
    slice1 = choose_slice(label1, nfrozen_occ, nocc, nvir, nfrozen_vir)
    slice2 = choose_slice(label2, nfrozen_occ, nocc, nvir, nfrozen_vir)
    out = np.einsum('r,Prr,Pts->ts', den, Lpq_full[:, den_slice, den_slice], Lpq_full[:, slice1, slice2], optimize=True)
    if rhf:
        out *= 2.0
    out -= np.einsum(
        'r,Prs,Ptr->ts', den, Lpq_full[:, den_slice, slice2], Lpq_full[:, slice1, den_slice], optimize=True
    )
    return out


def contraction_1rdm_Lpq(
    den, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, label1, label2, den_label1='P', den_label2='P', rhf=False
):
    r"""
    Contraction in the form of
        I_{ts} = \sum_{pq} D_{pq} \langle qt||ps \rangle
               = \sum_{pqP} D_{pq} (L^P_{qp} L^P_{ts} - L^P_{qs} L^P_{tp})

    Args:
        den: input density matrix.
        Lpq_full: full Lpq matrix including frozen orbitals.
        nocc: number of active occupied orbitals.
        nvir: number of active virtual orbitals.
        nfrozen_occ: number of frozen occupied orbitals.
        nfrozen_vir: number of frozen virtual orbitals.
        label1: label for the first index t
        label2: label for the second index s
        den_label1: label for the density matrix first index p
        den_label2: label for the density matrix second index q
    Returns:
        out: contracted intermediates.
    """
    den_slice1 = choose_slice(den_label1, nfrozen_occ, nocc, nvir, nfrozen_vir)
    den_slice2 = choose_slice(den_label2, nfrozen_occ, nocc, nvir, nfrozen_vir)
    slice1 = choose_slice(label1, nfrozen_occ, nocc, nvir, nfrozen_vir)
    slice2 = choose_slice(label2, nfrozen_occ, nocc, nvir, nfrozen_vir)
    out = np.einsum(
        'pq,Pqp,Pts->ts', den, Lpq_full[:, den_slice2, den_slice1], Lpq_full[:, slice1, slice2], optimize=True
    )
    if rhf:
        out *= 2.0
    out -= np.einsum(
        'pq,Pqs,Ptp->ts', den, Lpq_full[:, den_slice2, slice2], Lpq_full[:, slice1, den_slice1], optimize=True
    )
    return out


def contraction_2rdm_Lpq(occ_y_mat, vir_x_mat, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, label1, label2):
    r"""
    Contraction in the form of (anti-symmetrized or symmetrized)
        I_{tp} = \frac{1}{2} \sum_{qrs} \Gamma_{pq,rs} \langle tq||rs \rangle
               = \frac{1}{2} \sum_{qrsP} XY_{pq}^* XY_{rs}
                  (L^P_{tr} L^P_{qs} \pm L^P_{ts} L^P_{qr})
               = \sum_{qrsP} XY_{pq}^* XY_{rs} L^P_{tr} L^P_{qs}
        I_{ti} = \sum_{jklP} Y_{ij}^* Y_{kl} L^P_{tk} L^P_{jl}
               + \sum_{jcdP} Y_{ij}^* X_{cd} L^P_{tc} L^P_{jd}
        I_{ta} = \sum_{bklP} X_{ab}^* Y_{kl} L^P_{tk} L^P_{bl}
               + \sum_{bcdP} X_{ab}^* X_{cd} L^P_{tc} L^P_{bd}
    Args:
        occ_y_mat: coefficients for occupied orbitals Y
        vir_x_mat: coefficients for virtual orbitals X
        Lpq_full: full Lpq matrix including frozen orbitals.
        nocc: number of active occupied orbitals.
        nvir: number of active virtual orbitals.
        nfrozen_occ: number of frozen occupied orbitals.
        nfrozen_vir: number of frozen virtual orbitals.
        label1: label for the first index t
        label2: label for the second index p
    Returns:
        out: contracted intermediates.
    """
    # qrs are all active space indices.
    slice1 = choose_slice(label1, nfrozen_occ, nocc, nvir, nfrozen_vir)
    slice_i = choose_slice('i', nfrozen_occ, nocc, nvir, nfrozen_vir)
    slice_a = choose_slice('a', nfrozen_occ, nocc, nvir, nfrozen_vir)
    naux = Lpq_full.shape[0]
    
    # Special cases for TDA
    if label1 == 'p':
        if nocc == 0:
            label1 = 'a'
        elif nvir == 0:
            label1 = 'i'
    if label2 == 'p':
        if nocc == 0:
            label2 = 'a'
        elif nvir == 0:
            label2 = 'i'

    if label1 == 'i':
        n1 = nocc
    elif label1 == 'a':
        n1 = nvir
    elif label1 == 'ip':
        n1 = nfrozen_occ
    elif label1 == 'ap':
        n1 = nfrozen_vir
    else:
        n1 = nocc + nvir
    if label2 == 'i':
        # Slow but more readable version
        # out = np.einsum("ij,kl,Ptk,Pjl->ti", occ_y_mat.conj(), occ_y_mat,
        #                 Lpq_full[:,slice1,slice_i],
        #                 Lpq_full[:,slice_i,slice_i],
        #                 optimize=True)
        # out+= np.einsum("ij,cd,Ptc,Pjd->ti", occ_y_mat.conj(), vir_x_mat,
        #                 Lpq_full[:,slice1,slice_a],
        #                 Lpq_full[:,slice_i,slice_a],
        #                 optimize=True)
        L1i = np.ascontiguousarray(Lpq_full[:, slice1, slice_i]).reshape(-1, nocc)  # (Pt,k)
        Lij = np.ascontiguousarray(Lpq_full[:, slice_i, slice_i]).reshape(-1, nocc).conj()  # (P,j,l)*->(Pl,j)
        L1i = np.matmul(L1i, occ_y_mat).reshape(naux, n1, nocc)  # (Pt,k)(k,l)->(P,t,l)
        L1i = L1i.transpose(1, 0, 2).reshape(n1, -1)  # (P,t,l)->(t,Pl)
        tmp = np.matmul(L1i, Lij)  # (t,Pl)(Pl,j)->(t,j)
        out = np.matmul(tmp, occ_y_mat.T.conj())  # (t,j)(j,i) -> (t,i)

        if nvir > 0:
            L1a = np.ascontiguousarray(Lpq_full[:, slice1, slice_a]).reshape(-1, nvir)  # (Pt,c)
            Lai = np.ascontiguousarray(Lpq_full[:, slice_a, slice_i]).reshape(-1, nocc).conj()  # (P,d,j)*->(Pd,j)
            L1a = np.matmul(L1a, vir_x_mat).reshape(naux, n1, nvir)  # (Pt,c)(c,d)->(P,t,d)
            L1a = L1a.transpose(1, 0, 2).reshape(n1, -1)  # (P,t,d)->(t,Pd)
            tmp = np.matmul(L1a, Lai)  # (t,Pd)(Pd,j)->(t,j)
            out += np.matmul(tmp, occ_y_mat.T.conj())  # (t,j)(j,i) -> (t,i)
    elif label2 == 'a':
        # Slow but more readable version
        # out = np.einsum("ab,cd,Ptc,Pbd->ta", vir_x_mat.conj(), vir_x_mat,
        #                 Lpq_full[:,slice1,slice_a],
        #                 Lpq_full[:,slice_a,slice_a],
        #                 optimize=True)
        # out+= np.einsum("ab,kl,Ptk,Pbl->ta", vir_x_mat.conj(), occ_y_mat,
        #                 Lpq_full[:,slice1,slice_i],
        #                 Lpq_full[:,slice_a,slice_i],
        #                 optimize=True)
        L1a = np.ascontiguousarray(Lpq_full[:, slice1, slice_a]).reshape(-1, nvir)  # (Pt,c)
        Lab = np.ascontiguousarray(Lpq_full[:, slice_a, slice_a]).reshape(-1, nvir).conj()  # (P,b,d)*->(Pd,b)
        L1a = np.matmul(L1a, vir_x_mat).reshape(naux, n1, nvir)  # (Pt,c)(c,d)->(P,t,d)
        L1a = L1a.transpose(1, 0, 2).reshape(n1, -1)  # (P,t,d)->(t,Pd)
        tmp = np.matmul(L1a, Lab)  # (t,Pd)(Pd,b)->(t,b)
        out = np.matmul(tmp, vir_x_mat.T.conj())  # (t,b)(b,a) -> (t,a)

        if nocc > 0:
            L1i = np.ascontiguousarray(Lpq_full[:, slice1, slice_i]).reshape(-1, nocc)  # (Pt,k)
            Lia = np.ascontiguousarray(Lpq_full[:, slice_i, slice_a]).reshape(-1, nvir).conj()  # (P,l,b)*->(Pl,b)
            L1i = np.matmul(L1i, occ_y_mat).reshape(naux, n1, nocc)  # (Pt,k)(k,l)->(P,t,l)
            L1i = L1i.transpose(1, 0, 2).reshape(n1, -1)  # (P,t,l)->(t,Pl)
            tmp = np.matmul(L1i, Lia)  # (t,Pl)(Pl,b)->(t,b)
            out += np.matmul(tmp, vir_x_mat.T.conj())  # (t,b)(b,a) -> (t,a)
    elif label2 == 'p':
        # slow (more copies) but more readable version
        # out = np.concatenate((
        # contraction_2rdm_Lpq(occ_y_mat, vir_x_mat, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, label1, "i"),
        # contraction_2rdm_Lpq(occ_y_mat, vir_x_mat, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, label1, "a")), axis=1)
        out = np.zeros((n1, nocc + nvir), dtype=occ_y_mat.dtype)
        L1i = np.ascontiguousarray(Lpq_full[:, slice1, slice_i]).reshape(-1, nocc)  # (Pt,k)
        Lia = np.ascontiguousarray(Lpq_full[:, slice_i, slice_a]).reshape(-1, nvir).conj()  # (P,l,b)*->(Pl,b)
        Lij = np.ascontiguousarray(Lpq_full[:, slice_i, slice_i]).reshape(-1, nocc).conj()  # (P,j,l)*->(Pl,j)
        L1i = np.matmul(L1i, occ_y_mat).reshape(naux, n1, nocc)  # (Pt,k)(k,l)->(P,t,l)
        L1i = L1i.transpose(1, 0, 2).reshape(n1, -1)  # (P,t,l)->(t,Pl)
        tmp = np.matmul(L1i, Lia)  # (t,Pl)(Pl,b)->(t,b)
        out[:, nocc:] = np.matmul(tmp, vir_x_mat.T.conj())  # (t,b)(b,a) -> (t,a)
        tmp = np.matmul(L1i, Lij)  # (t,Pl)(Pl,j)->(t,j)
        out[:, :nocc] = np.matmul(tmp, occ_y_mat.T.conj())  # (t,j)(j,i) -> (t,i)

        L1a = np.ascontiguousarray(Lpq_full[:, slice1, slice_a]).reshape(-1, nvir)  # (Pt,c)
        Lab = np.ascontiguousarray(Lpq_full[:, slice_a, slice_a]).reshape(-1, nvir).conj()  # (P,b,d)*->(Pd,b)
        Lai = np.ascontiguousarray(Lpq_full[:, slice_a, slice_i]).reshape(-1, nocc).conj()  # (P,d,j)*->(Pd,j)
        L1a = np.matmul(L1a, vir_x_mat).reshape(naux, n1, nvir)  # (Pt,c)(c,d)->(P,t,d)
        L1a = L1a.transpose(1, 0, 2).reshape(n1, -1)  # (P,t,d)->(t,Pd)
        tmp = np.matmul(L1a, Lab)  # (t,Pd)(Pd,b)->(t,b)
        out[:, nocc:] += np.matmul(tmp, vir_x_mat.T.conj())  # (t,b)(b,a) -> (t,a)
        tmp = np.matmul(L1a, Lai)  # (t,Pd)(Pd,j)->(t,j)
        out[:, :nocc] += np.matmul(tmp, occ_y_mat.T.conj())  # (t,j)(j,i) -> (t,i)
    else:
        raise ValueError('label2 = {}. is not valid in contraction_2rdm_Lpq'.format(label2))

    return out

def contraction_2rdm_eri(occ_y_mat, vir_x_mat, eri_X, eri_Y, nocc, nvir, nfrozen_occ, nfrozen_vir, label1, label2):
    r"""
    Contraction in the form of (anti-symmetrized or symmetrized)
        I_{tp} = \sum_{qrsP} XY_{pq}^* XY_{rs} \langle tq|rs \rangle
        I_{ti} = \sum_{j} Y_{ij}^* \sum_{kl} Y_{kl} \langle tj|kl \rangle
               + \sum_{j} Y_{ij}^* \sum_{cd} X_{cd} \langle tj|cd \rangle
        I_{ta} = \sum_{b} X_{ab}^* \sum_{kl} Y_{kl} \langle tb|kl \rangle
               + \sum_{b} X_{ab}^* \sum_{cd} X_{cd} \langle tb|cd \rangle
    Args:
        occ_y_mat: coefficients for occupied orbitals Y
        vir_x_mat: coefficients for virtual orbitals X
        eri_X: \sum_{cd} X_{cd} \langle tq|cd \rangle (nall, nact)
        eri_Y: \sum_{kl} Y_{kl} \langle tq|kl \rangle (nall, nact)
        nocc: number of active occupied orbitals.
        nvir: number of active virtual orbitals.
        nfrozen_occ: number of frozen occupied orbitals.
        nfrozen_vir: number of frozen virtual orbitals.
        label1: label for the first index t
        label2: label for the second index p
    Returns:
        out: contracted intermediates.
    """
    # qrs are all active space indices.
    slice1 = choose_slice(label1, nfrozen_occ, nocc, nvir, nfrozen_vir)
    
    # Special cases for TDA
    if label1 == 'p':
        if nocc == 0:
            label1 = 'a'
        elif nvir == 0:
            label1 = 'i'
    if label2 == 'p':
        if nocc == 0:
            label2 = 'a'
        elif nvir == 0:
            label2 = 'i'

    if label1 == 'i':
        n1 = nocc
    elif label1 == 'a':
        n1 = nvir
    elif label1 == 'ip':
        n1 = nfrozen_occ
    elif label1 == 'ap':
        n1 = nfrozen_vir
    else:
        n1 = nocc + nvir
    if label2 == 'i':
        tmp = np.ascontiguousarray(eri_Y[slice1, :nocc] + eri_X[slice1, :nocc])  # (t,j)
        out = np.matmul(tmp, occ_y_mat.T.conj())  # (t,j)(i,j) -> (t,i)
    elif label2 == 'a':
        tmp = np.ascontiguousarray(eri_Y[slice1, nocc:] + eri_X[slice1, nocc:])  # (t,b)
        out = np.matmul(tmp, vir_x_mat.T.conj())  # (t,b)(a,b) -> (t,a)
    elif label2 == 'p':
        out = np.zeros((n1, nocc + nvir), dtype=occ_y_mat.dtype)
        tmp = np.ascontiguousarray(eri_Y[slice1, :nocc] + eri_X[slice1, :nocc])  # (t,j)
        out[:,:nocc] = np.matmul(tmp, occ_y_mat.T.conj())  # (t,j)(i,j) -> (t,i)
        tmp = np.ascontiguousarray(eri_Y[slice1, nocc:] + eri_X[slice1, nocc:])  # (t,b)
        out[:,nocc:] = np.matmul(tmp, vir_x_mat.T.conj())  # (t,b)(a,b) -> (t,a)
    else:
        raise ValueError('label2 = {}. is not valid in contraction_2rdm_Lpq'.format(label2))

    return out


def get_I_pp_int(den, occ_y_mat, vir_x_mat, mo_ene_full, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, rhf=False):
    """
    Calculate the I'' matrix.
    Args:
        den: unrelaxed density matrix.
        occ_y_mat: coefficients for occupied orbitals Y
        vir_x_mat: coefficients for virtual orbitals X
        mo_ene_full: full MO energies including frozen orbitals.
        Lpq_full: full Lpq matrix including frozen orbitals.
        nocc: number of active occupied orbitals.
        nvir: number of active virtual orbitals.
        nfrozen_occ: number of frozen occupied orbitals.
        nfrozen_vir: number of frozen virtual orbitals.
    Returns:
        i_prime: the I' intermediates (nmo_full, nmo_full)
        i_prime_prime: the I'' intermediates (nmo_full, nmo_full)
    """
    # create slices
    slice_p = choose_slice('p', nfrozen_occ, nocc, nvir, nfrozen_vir)  # all active
    slice_i = choose_slice('i', nfrozen_occ, nocc, nvir, nfrozen_vir)  # active occupied
    slice_a = choose_slice('a', nfrozen_occ, nocc, nvir, nfrozen_vir)  # active virtual
    slice_ip = choose_slice('ip', nfrozen_occ, nocc, nvir, nfrozen_vir)  # frozen occupied
    slice_ap = choose_slice('ap', nfrozen_occ, nocc, nvir, nfrozen_vir)  # frozen virtual
    slice_I = choose_slice('I', nfrozen_occ, nocc, nvir, nfrozen_vir)  # all occupied
    slice_A = choose_slice('A', nfrozen_occ, nocc, nvir, nfrozen_vir)  # all virtual

    # calculate I' first
    i_prime = np.zeros((len(mo_ene_full), len(mo_ene_full)), dtype=occ_y_mat.dtype)
    # I' active-active block
    i_prime[slice_p, slice_p] += contraction_2rdm_Lpq(
        occ_y_mat, vir_x_mat, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'p', 'p'
    )
    i_prime[slice_a, slice_i] += contraction_1rdm_Lpq_diag(
        den, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'a', 'i', rhf=rhf
    )
    for p in choose_range('p', nfrozen_occ, nocc, nvir, nfrozen_vir):
        i_prime[p, p] += mo_ene_full[p] * den[p - nfrozen_occ]

    if nfrozen_vir > 0:
        # I' frozen virtual-active block
        i_prime[slice_ap, slice_p] += contraction_2rdm_Lpq(
            occ_y_mat, vir_x_mat, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'ap', 'p'
        )
        i_prime[slice_ap, slice_i] += contraction_1rdm_Lpq_diag(
            den, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'ap', 'i', rhf=rhf
        )

    if nfrozen_occ > 0:
        # I' frozen occupied-active block
        i_prime[slice_ip, slice_p] += contraction_2rdm_Lpq(
            occ_y_mat, vir_x_mat, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'ip', 'p'
        )

        # I' all virtual-frozen occupied block
        i_prime[slice_A, slice_ip] += contraction_1rdm_Lpq_diag(
            den, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'A', 'ip', rhf=rhf
        )

    # calculate I'' next
    i_prime_prime = np.zeros_like(i_prime)
    # I'' active virtual-all occupied block
    i_prime_prime[slice_a, slice_I] = i_prime[slice_a, slice_I] - i_prime[slice_I, slice_a].T.conj()
    # I'' = I' blocks
    i_prime_prime[slice_A, slice_a] = i_prime[slice_A, slice_a]
    i_prime_prime[slice_I, slice_i] = i_prime[slice_I, slice_i]
    i_prime_prime[slice_ap, slice_I] = i_prime[slice_ap, slice_I]

    return i_prime, i_prime_prime


def contraction_2rdm_eri_chol(Gpq_chol, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, label1, label2):
    r"""
    Contraction in the form of
        I_{tp} = \frac{1}{2} \sum_{qrs} \Gamma_{pq,rs} \langle tq||rs \rangle
               = \sum_{qrsP} \Gamma_{pq,rs} L^P_{tr} L^P_{qs}
               = \sum_{rP} \Gamma^P_{pr} L^P_{tr}
    Args:
        Gpq_chol: dressed 2-RDM intermediates with CD/DF vectors.
                  dimension: (ncd, nmo_act, nmo_act)
        Lpq_full: full Lpq matrix including frozen orbitals.
        nocc: number of active occupied orbitals.
        nvir: number of active virtual orbitals.
        nfrozen_occ: number of frozen occupied orbitals.
        nfrozen_vir: number of frozen virtual orbitals.
        label1: label for the first index t
        label2: label for the second index p
    Returns:
        out: contracted intermediates.
    """
    # qrs are all active space indices.
    slice1 = choose_slice(label1, nfrozen_occ, nocc, nvir, nfrozen_vir)
    slice_p = choose_slice('p', nfrozen_occ, nocc, nvir, nfrozen_vir)
    if label2 == 'i':
        out = np.einsum('Ppr,Ptr->tp', Gpq_chol[:, :nocc, :], Lpq_full[:, slice1, slice_p], optimize=True)
    elif label2 == 'a':
        out = np.einsum('Ppr,Ptr->tp', Gpq_chol[:, nocc:, :], Lpq_full[:, slice1, slice_p], optimize=True)
    elif label2 == 'p':
        out = np.einsum('Ppr,Ptr->tp', Gpq_chol, Lpq_full[:, slice1, slice_p], optimize=True)
    else:
        raise ValueError('label2 = {}. is not valid in contraction_2rdm_eri_chol'.format(label2))
    return out


def get_I_pp_int_chol(den, Gpq_chol, mo_ene_full, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir):
    """
    Calculate the I'' matrix with Cholesky two-particle density matrices.
    It is not used in the ppRPA calculations since the ppRPA 2e-rdm is already sparse.
    Args:
        den: unrelaxed density matrix.
        Gpq_chol: dressed 2-RDM intermediates with CD/DF vectors.
                  dimension: (ncd, nmo_act, nmo_act)
        mo_ene_full: full MO energies including frozen orbitals.
        Lpq_full: full Lpq matrix including frozen orbitals.
        nocc: number of active occupied orbitals.
        nvir: number of active virtual orbitals.
        nfrozen_occ: number of frozen occupied orbitals.
        nfrozen_vir: number of frozen virtual orbitals.
    Returns:
        i_prime: the I' intermediates (nmo_full, nmo_full)
        i_prime_prime: the I'' intermediates (nmo_full, nmo_full)
    """
    # create slices
    slice_p = choose_slice('p', nfrozen_occ, nocc, nvir, nfrozen_vir)  # all active
    slice_i = choose_slice('i', nfrozen_occ, nocc, nvir, nfrozen_vir)  # active occupied
    slice_a = choose_slice('a', nfrozen_occ, nocc, nvir, nfrozen_vir)  # active virtual
    slice_ip = choose_slice('ip', nfrozen_occ, nocc, nvir, nfrozen_vir)  # frozen occupied
    slice_ap = choose_slice('ap', nfrozen_occ, nocc, nvir, nfrozen_vir)  # frozen virtual
    slice_I = choose_slice('I', nfrozen_occ, nocc, nvir, nfrozen_vir)  # all occupied
    slice_A = choose_slice('A', nfrozen_occ, nocc, nvir, nfrozen_vir)  # all virtual

    # calculate I' first
    i_prime = np.zeros((len(mo_ene_full), len(mo_ene_full)), dtype=Gpq_chol.dtype)
    # I' active-active block
    i_prime[slice_p, slice_p] += contraction_2rdm_eri_chol(
        Gpq_chol, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'p', 'p'
    )
    i_prime[slice_a, slice_i] += contraction_1rdm_Lpq_diag(
        den, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'a', 'i'
    )
    for p in choose_range('p', nfrozen_occ, nocc, nvir, nfrozen_vir):
        i_prime[p, p] += mo_ene_full[p] * den[p - nfrozen_occ]

    # I' frozen virtual-active block
    i_prime[slice_ap, slice_p] += contraction_2rdm_eri_chol(
        Gpq_chol, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'ap', 'p'
    )
    i_prime[slice_ap, slice_i] += contraction_1rdm_Lpq_diag(
        den, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'ap', 'i'
    )
    # I' frozen occupied-active block
    i_prime[slice_ip, slice_p] += contraction_2rdm_eri_chol(
        Gpq_chol, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'ip', 'p'
    )

    # I' all virtual-frozen occupied block
    i_prime[slice_A, slice_ip] += contraction_1rdm_Lpq_diag(
        den, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'A', 'ip'
    )

    # calculate I'' next
    i_prime_prime = np.zeros_like(i_prime)
    # I'' active virtual-all occupied block
    i_prime_prime[slice_a, slice_I] = i_prime[slice_a, slice_I] - i_prime[slice_I, slice_a].T.conj()
    # I'' = I' blocks
    i_prime_prime[slice_A, slice_a] = i_prime[slice_A, slice_a]
    i_prime_prime[slice_I, slice_i] = i_prime[slice_I, slice_i]
    i_prime_prime[slice_ap, slice_I] = i_prime[slice_ap, slice_I]

    return i_prime, i_prime_prime


def get_X_int(i_pp_A_I, d_p_I_i, d_p_A_a, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, rhf=False):
    """
    Calculate the X vector.
    Args:
        i_pp_A_I: the I'' intermediates (A-I block)
        d_p_I_i: the D' intermediates (I-i block)
        d_p_A_a: the D' intermediates (A-a block)
        Lpq_full: full Lpq matrix including frozen orbitals.
        nocc: number of active occupied orbitals.
        nvir: number of active virtual orbitals.
        nfrozen_occ: number of frozen occupied orbitals.
        nfrozen_vir: number of frozen virtual orbitals.
    Returns:
        x_int: the X vector (nvir_full*nocc_full)
    """
    x_vec = i_pp_A_I.copy()
    x_vec += contraction_1rdm_Lpq(d_p_I_i, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'A', 'I', 'I', 'i', rhf=rhf)
    x_vec += contraction_1rdm_Lpq(
        d_p_I_i.T.conj(), Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'A', 'I', 'i', 'I', rhf=rhf
    )
    x_vec += contraction_1rdm_Lpq(d_p_A_a, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'A', 'I', 'A', 'a', rhf=rhf)
    x_vec += contraction_1rdm_Lpq(
        d_p_A_a.T.conj(), Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'A', 'I', 'a', 'A', rhf=rhf
    )

    return x_vec.reshape(-1)


def z_vector_eqn_matvec(input_vec, mo_ene_full, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, rhf=False):
    """
    Calculate the Z-vector equation matrix-vector product.
    Args:
        input_vec: the D' intermediates A-I block
        mo_ene_full: full MO energies including frozen orbitals.
        Lpq_full: full Lpq matrix including frozen orbitals.
        nocc: number of active occupied orbitals.
        nvir: number of active virtual orbitals.
        nfrozen_occ: number of frozen occupied orbitals.
        nfrozen_vir: number of frozen virtual orbitals.
    Returns:
        hd_vec: the Z-vector equation matrix-vector product (nvir_full*nocc_full)
    """
    d_p_AI = input_vec.reshape(nvir + nfrozen_vir, nocc + nfrozen_occ)
    hd_vec = contraction_1rdm_Lpq(d_p_AI, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'A', 'I', 'A', 'I', rhf=rhf)
    hd_vec += contraction_1rdm_Lpq(
        d_p_AI.T.conj(), Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'A', 'I', 'I', 'A', rhf=rhf
    )
    for a in choose_range('A', nfrozen_occ, nocc, nvir, nfrozen_vir):
        for i in choose_range('I', nfrozen_occ, nocc, nvir, nfrozen_vir):
            hd_vec[a - nocc - nfrozen_occ, i] += d_p_AI[a - nocc - nfrozen_occ, i] * (mo_ene_full[a] - mo_ene_full[i])
    return hd_vec.reshape(-1)


def z_vector_eqn_solver(x_int, mo_ene_full, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, rhf=False, matvec=None, print_level=0):
    """
    Solve the Z-vector equation.
    Args:
        x_int: the X intermediates
        mo_ene_full: full MO energies including frozen orbitals.
        Lpq_full: full Lpq matrix including frozen orbitals.
        nocc: number of active occupied orbitals.
        nvir: number of active virtual orbitals.
        nfrozen_occ: number of frozen occupied orbitals.
        nfrozen_vir: number of frozen virtual orbitals.
    Returns:
        D' intermediates (A-I block)
    """

    if matvec is None:
        def matvec(input_vec):
            return z_vector_eqn_matvec(input_vec, mo_ene_full, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, rhf=rhf)

    mat_diag = np.zeros((nvir + nfrozen_vir, nocc + nfrozen_occ), dtype=np.double)  # diagonal elements must be real
    # It is not necessary to calculate the exact diagonal elements
    # since they are for preconditioning only.
    for a in choose_range('A', nfrozen_occ, nocc, nvir, nfrozen_vir):
        for i in choose_range('I', nfrozen_occ, nocc, nvir, nfrozen_vir):
            mat_diag[a - nfrozen_occ - nocc, i] = mo_ene_full[a] - mo_ene_full[i]
    mat_diag = mat_diag.reshape(-1)

    return GMRES_Pople(matvec, mat_diag, -x_int, printLevel=print_level).reshape(nvir + nfrozen_vir, nocc + nfrozen_occ)
    # return GMRES_wrapper(matvec, mat_diag, -x_int).reshape(nvir+nfrozen_vir, nocc+nfrozen_occ)


def get_D_p_int(i_pp, mo_ene_full, nocc, nvir, nfrozen_occ, nfrozen_vir, rhf=False):
    """
    Calculate the D' matrix.
    Args:
        i_pp: the I'' intermediates
        mo_ene_full: full MO energies including frozen orbitals.
        nocc: number of active occupied orbitals.
        nvir: number of active virtual orbitals.
        nfrozen_occ: number of frozen occupied orbitals.
        nfrozen_vir: number of frozen virtual orbitals.
    Returns:
        d_prime: the D' intermediates (nmo_full, nmo_full)
    """
    d_prime = np.zeros_like(i_pp)
    threshold = 1.0e-6
    # D' all occupied-active occupied block
    for i in choose_range('I', nfrozen_occ, nocc, nvir, nfrozen_vir):
        for j in choose_range('i', nfrozen_occ, nocc, nvir, nfrozen_vir):
            denorm = mo_ene_full[j] - mo_ene_full[i]
            factor = 1.0 / denorm if abs(denorm) >= threshold else 0.0
            d_prime[i, j] = factor * i_pp[i, j]

    # D' all virtual-active virtual block
    for a in choose_range('A', nfrozen_occ, nocc, nvir, nfrozen_vir):
        for b in choose_range('a', nfrozen_occ, nocc, nvir, nfrozen_vir):
            denorm = mo_ene_full[b] - mo_ene_full[a]
            factor = 1.0 / denorm if abs(denorm) >= threshold else 0.0
            d_prime[a, b] = factor * i_pp[a, b]

    return d_prime


def get_I_int(i_p, d_p, den, mo_ene_full, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, rhf=False):
    """
    Calculate the I matrix.
    Args:
        i_p: the I' intermediates
        d_p: the D' intermediates
        den: the unrelaxed density matrix
        mo_ene_full: full MO energies including frozen orbitals.
        Lpq_full: full Lpq matrix including frozen orbitals.
        nocc: number of active occupied orbitals.
        nvir: number of active virtual orbitals.
        nfrozen_occ: number of frozen occupied orbitals.
        nfrozen_vir: number of frozen virtual orbitals.
    Returns:
        i_int: the I intermediates (nmo_full, nmo_full)
    """
    i_int = -np.einsum('qp,p->pq', d_p, mo_ene_full)
    slice_I = choose_slice('I', nfrozen_occ, nocc, nvir, nfrozen_vir)
    slice_a = choose_slice('a', nfrozen_occ, nocc, nvir, nfrozen_vir)
    # I all occupied-all occupied block
    i_int[slice_I, slice_I] -= (
        0.5 * contraction_1rdm_Lpq_diag(den, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'I', 'I', 'p', rhf=rhf).T
    )
    i_int[slice_I, slice_I] -= contraction_1rdm_Lpq(
        d_p, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'I', 'I', 'P', 'P', rhf=rhf
    ).T
    # I active virtual-all occupied block
    i_int[slice_a, slice_I] -= i_p[slice_I, slice_a].T

    # I active-active block extra term
    threshold = 1.0e-6
    for i in choose_range('p', nfrozen_occ, nocc, nvir, nfrozen_vir):
        for j in choose_range('p', nfrozen_occ, nocc, nvir, nfrozen_vir):
            denorm = mo_ene_full[j] - mo_ene_full[i]
            if abs(denorm) < threshold:
                i_int[i, j] -= 0.5 * i_p[j, i]

    return i_int


def get_D_pp_int(d_p, den, nocc, nvir, nfrozen_occ, nfrozen_vir):
    """
    Make the relaxed one-particle density matrix.
    Args:
        d_p: the D' intermediates
        den: the unrelaxed density matrix
        nocc: number of active occupied orbitals.
        nvir: number of active virtual orbitals.
        nfrozen_occ: number of frozen occupied orbitals.
        nfrozen_vir: number of frozen virtual orbitals.
    Returns:
        d_prime_prime: the relaxed one-particle density matrix (nmo_full, nmo_full)
    """
    den_pp = d_p.T
    # active-active block
    for p in choose_range('p', nfrozen_occ, nocc, nvir, nfrozen_vir):
        den_pp[p, p] += 0.5 * den[p - nfrozen_occ]
    return den_pp


def make_rdm1_relaxed(xy, mo_ene_full, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, mult='t', rhf=False):
    r"""Calculate relaxed density matrix (and the I intermediates)
        for given pprpa roots.
    Args:
        xy: a pprpa eigenvector.
        mo_ene_full: full MO energies including frozen orbitals.
        Lpq_full: full Lpq matrix including frozen orbitals.
        nocc: number of active occupied orbitals.
        nvir: number of active virtual orbitals.
        nfrozen_occ: number of frozen occupied orbitals.
        nfrozen_vir: number of frozen virtual orbitals.
    Returns:
        den_relaxed: the relaxed one-particle density matrix (nmo_full, nmo_full)
        i_int: the I intermediates (nmo_full, nmo_full)
        Both are in the MO basis.
    """
    if mult == 's':
        oo_dim = (nocc + 1) * nocc // 2
    else:
        oo_dim = (nocc - 1) * nocc // 2
    occ_y_mat, vir_x_mat = get_xy_full(xy, oo_dim, mult)
    den_unrelaxed = make_rdm1_unrelaxed_from_xy_full(occ_y_mat, vir_x_mat)

    print('nocc = ', nocc)
    print('nvir = ', nvir)
    print('nfrozen_occ = ', nfrozen_occ)
    print('nfrozen_vir = ', nfrozen_vir)

    start_clock('Calculate i_prime and i_prime_prime')
    i_prime, i_prime_prime = get_I_pp_int(
        den_unrelaxed, occ_y_mat, vir_x_mat, mo_ene_full, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, rhf=rhf
    )
    stop_clock('Calculate i_prime and i_prime_prime')

    start_clock('Calculate d_prime')
    d_prime = get_D_p_int(i_prime_prime, mo_ene_full, nocc, nvir, nfrozen_occ, nfrozen_vir, rhf=rhf)
    # the other blocks are zero except for the all virtual-all occupied block
    # It is solved by the Z-vector equation
    slice_A = choose_slice('A', nfrozen_occ, nocc, nvir, nfrozen_vir)
    slice_I = choose_slice('I', nfrozen_occ, nocc, nvir, nfrozen_vir)
    slice_a = choose_slice('a', nfrozen_occ, nocc, nvir, nfrozen_vir)
    slice_i = choose_slice('i', nfrozen_occ, nocc, nvir, nfrozen_vir)
    x_int = get_X_int(
        i_prime_prime[slice_A, slice_I],
        d_prime[slice_I, slice_i],
        d_prime[slice_A, slice_a],
        Lpq_full,
        nocc,
        nvir,
        nfrozen_occ,
        nfrozen_vir,
        rhf=rhf,
    )
    d_prime[slice_A, slice_I] = z_vector_eqn_solver(
        x_int, mo_ene_full, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, rhf=rhf
    )
    stop_clock('Calculate d_prime')

    start_clock('Calculate I intermediates')
    i_int = get_I_int(
        i_prime, d_prime, den_unrelaxed, mo_ene_full, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, rhf=rhf
    )
    stop_clock('Calculate I intermediates')

    den_relaxed = get_D_pp_int(d_prime, den_unrelaxed, nocc, nvir, nfrozen_occ, nfrozen_vir)
    # In the derivation of the D' matrix, E = \sum_{pq} {D^pp_{pq} fx_{pq} + c.c.}
    # But the convention in the relaxed density matrix is usually E = \sum_{pq} D_{pq} fx_{qp}
    # Same for I intermediates
    den_relaxed = den_relaxed.T + den_relaxed.conj()
    i_int = i_int.T + i_int.conj()

    return den_relaxed, i_int


def make_rdm1_relaxed_pprpa(pprpa, mf, xy=None, mult='t', istate=0):
    r"""Calculate relaxed density matrix (and the I intermediates)
        for given pprpa and mean-field object.
    Args:
        pprpa: a pprpa object.
        mf: a mean-field object.
    Returns:
        den_relaxed: the relaxed one-particle density matrix (nmo_full, nmo_full)
        i_int: the I intermediates (nmo_full, nmo_full)
        Both are in the MO basis.
    """
    assert mult in ['t', 's'], 'mult = {}. is not valid in make_rdm1_relaxed_pprpa'.format(mult)
    from pyscf import scf, dft
    from lib_pprpa import pyscf_util

    if isinstance(mf, scf.hf.RHF) or isinstance(mf, dft.rks.RKS):
        from lib_pprpa import pprpa_direct, pprpa_davidson

        assert isinstance(pprpa, pprpa_direct.ppRPA_direct) or isinstance(pprpa, pprpa_davidson.ppRPA_Davidson)
        hf_var = 'RHF'
        if xy is None:
            if mult == 's':
                xy = pprpa.xy_s[istate]
            else:
                xy = pprpa.xy_t[istate]
        nocc_all = mf.mol.nelectron // 2
        nvir_all = mf.mol.nao - nocc_all
        _, mo_energy_full, Lpq_full = pyscf_util.get_pyscf_input_mol_r(mf)
    elif isinstance(mf, scf.uhf.UHF) or isinstance(mf, dft.uks.UKS):
        raise NotImplementedError('Unrestricted HF is not implemented.')
    elif (
        isinstance(mf, scf.ghf.GHF)
        or isinstance(mf, dft.gks.GKS)
        or (with_socuils and (isinstance(mf, spinor_hf.SpinorSCF) or isinstance(mf, spinor_dft.SpinorDFT)))
    ):
        from lib_pprpa import gpprpa_direct, gpprpa_davidson

        assert isinstance(pprpa, gpprpa_direct.GppRPA_direct) or isinstance(pprpa, gpprpa_davidson.GppRPA_Davidson)
        hf_var = 'GHF'
        if xy is None:
            xy = pprpa.xy[istate]
        nocc_all = mf.mol.nelectron
        nvir_all = mf.mol.nao * 2 - nocc_all
        _, mo_energy_full, Lpq_full = pyscf_util.get_pyscf_input_mol_g(mf)

    nocc = pprpa.nocc
    nvir = pprpa.nvir
    nfrozen_occ = nocc_all - nocc
    nfrozen_vir = nvir_all - nvir
    if hf_var == 'GHF' or hf_var == 'RHF':
        return make_rdm1_relaxed(
            xy, mo_energy_full, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, mult=mult, rhf=(hf_var == 'RHF')
        )
    else:
        raise NotImplementedError('hf_var = {}. is not implemented in make_rdm1_relaxed_pprpa'.format(hf_var))


from pyscf import lib
from pyscf.lib import logger
from pyscf.grad import rks as rks_grad


def get_veff_df_rks(ks_grad, mol=None, dm=None):
    """Coulomb + XC functional response
    Modified from pyscf.df.grad.rks.get_veff
    """
    if mol is None:
        mol = ks_grad.mol
    if dm is None:
        dm = ks_grad.base.make_rdm1()
    t0 = (logger.process_clock(), logger.perf_counter())

    mf = ks_grad.base
    ni = mf._numint
    grids, nlcgrids = rks_grad._initialize_grids(ks_grad)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory * 0.9 - mem_now)
    if ks_grad.grid_response:
        exc = []
        vxc = []
        for dmi in dm:
            exci, vxci = rks_grad.get_vxc_full_response(
                ni, mol, grids, mf.xc, dmi, max_memory=max_memory, verbose=ks_grad.verbose
            )
            exc.append(exci)
            vxc.append(vxci)
        exc = np.asarray(exc)
        vxc = np.asarray(vxc)
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = rks_grad.get_nlc_vxc_full_response(
                ni, mol, nlcgrids, xc, dm, max_memory=max_memory, verbose=ks_grad.verbose
            )
            exc += enlc
            vxc += vnlc
        logger.debug1(ks_grad, 'sum(grids response) %s', exc.sum(axis=0))
    else:
        exc, vxc = rks_grad.get_vxc(ni, mol, grids, mf.xc, dm, max_memory=max_memory, verbose=ks_grad.verbose)
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = rks_grad.get_nlc_vxc(ni, mol, nlcgrids, xc, dm, max_memory=max_memory, verbose=ks_grad.verbose)
            vxc += vnlc
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    vjk = np.zeros_like(vxc)
    if not ni.libxc.is_hybrid_xc(mf.xc):
        vj = ks_grad.get_j(mol, dm)
        vjk += vj
        if ks_grad.auxbasis_response:
            e1_aux = vj.aux
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        vj, vk = ks_grad.get_jk(mol, dm)
        if ks_grad.auxbasis_response:
            vk.aux *= hyb
        vk[:] *= hyb  # Don't erase the .aux tags!
        if omega != 0:  # For range separated Coulomb operator
            # TODO: replaced with vk_sr which is numerically more stable for
            # inv(int2c2e)
            vk_lr = ks_grad.get_k(mol, dm, omega=omega)
            vk[:] += vk_lr * (alpha - hyb)
            if ks_grad.auxbasis_response:
                vk.aux[:] += vk_lr.aux * (alpha - hyb)
        vjk += vj - vk * 0.5
        if ks_grad.auxbasis_response:
            e1_aux = vj.aux - vk.aux * 0.5

    if ks_grad.auxbasis_response:
        vjk = lib.tag_array(vjk, aux=e1_aux)
    if ks_grad.grid_response:
        vxc = lib.tag_array(vxc, exc1_grid=exc)
    return vxc, vjk


def get_veff_rks(ks_grad, mol=None, dm=None):
    """
    First order derivative of DFT effective potential matrix (wrt electron coordinates)

    Args:
        ks_grad : grad.uhf.Gradients or grad.uks.Gradients object
    """
    if mol is None:
        mol = ks_grad.mol
    if dm is None:
        dm = ks_grad.base.make_rdm1()
    t0 = (logger.process_clock(), logger.perf_counter())

    mf = ks_grad.base
    ni = mf._numint
    grids, nlcgrids = rks_grad._initialize_grids(ks_grad)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory * 0.9 - mem_now)
    if ks_grad.grid_response:
        exc, vxc = rks_grad.get_vxc_full_response(
            ni, mol, grids, mf.xc, dm, max_memory=max_memory, verbose=ks_grad.verbose
        )
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = rks_grad.get_nlc_vxc_full_response(
                ni, mol, nlcgrids, xc, dm, max_memory=max_memory, verbose=ks_grad.verbose
            )
            exc += enlc
            vxc += vnlc
        logger.debug1(ks_grad, 'sum(grids response) %s', exc.sum(axis=0))
    else:
        exc, vxc = rks_grad.get_vxc(ni, mol, grids, mf.xc, dm, max_memory=max_memory, verbose=ks_grad.verbose)
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = rks_grad.get_nlc_vxc(ni, mol, nlcgrids, xc, dm, max_memory=max_memory, verbose=ks_grad.verbose)
            vxc += vnlc
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    vjk = np.zeros_like(vxc)
    if not ni.libxc.is_hybrid_xc(mf.xc):
        vj = ks_grad.get_j(mol, dm)
        vjk += vj
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        vj, vk = ks_grad.get_jk(mol, dm)
        vk *= hyb
        if omega != 0:
            vk += ks_grad.get_k(mol, dm, omega=omega) * (alpha - hyb)
        vjk += vj - vk * 0.5

    vxc = lib.tag_array(vxc, exc1_grid=exc)
    return vxc, vjk


def _contract_xc_kernel(mf, xc_code, dmvo, dmoo=None, with_vxc=True, with_kxc=True, singlet=True, max_memory=2000):
    from pyscf import lib
    from pyscf.lib import logger
    from pyscf.grad.tdrks import _lda_eval_mat_, _gga_eval_mat_, _mgga_eval_mat_

    mol = mf.mol
    grids = mf.grids

    ni = mf._numint
    xctype = ni._xc_type(xc_code)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    # dmvo ~ reduce(np.dot, (orbv, Xai, orbo.T))
    dmvo = (dmvo + dmvo.T) * 0.5  # because K_{ia,jb} == K_{ia,bj}

    f1vo = np.zeros((4, nao, nao))  # 0th-order, d/dx, d/dy, d/dz
    deriv = 2
    if dmoo is not None:
        f1oo = np.zeros((4, nao, nao))
    else:
        f1oo = None
    if with_vxc:
        v1ao = np.zeros((4, nao, nao))
    else:
        v1ao = None
    if with_kxc:
        k1ao = np.zeros((4, nao, nao))
        deriv = 3
    else:
        k1ao = None

    if xctype == 'HF':
        return f1vo, f1oo, v1ao, k1ao
    elif xctype == 'LDA':
        fmat_, ao_deriv = _lda_eval_mat_, 1
    elif xctype == 'GGA':
        fmat_, ao_deriv = _gga_eval_mat_, 2
    elif xctype == 'MGGA':
        fmat_, ao_deriv = _mgga_eval_mat_, 2
        logger.warn(mf, 'PPRPA-MGGA Gradients may be inaccurate due to grids response')
    else:
        raise NotImplementedError(f'td-rks for functional {xc_code}')

    if singlet:
        for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            if xctype == 'LDA':
                ao0 = ao[0]
            else:
                ao0 = ao
            rho = ni.eval_rho2(mol, ao0, mo_coeff, mo_occ, mask, xctype, with_lapl=False)
            vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)[1:]

            rho1 = ni.eval_rho(mol, ao0, dmvo, mask, xctype, hermi=1, with_lapl=False) * 2  # *2 for alpha + beta
            if xctype == 'LDA':
                rho1 = rho1[np.newaxis]
            wv = np.einsum('yg,xyg,g->xg', rho1, fxc, weight)
            fmat_(mol, f1vo, ao, wv, mask, shls_slice, ao_loc)

            if dmoo is not None:
                rho2 = ni.eval_rho(mol, ao0, dmoo, mask, xctype, hermi=1, with_lapl=False) * 2
                if xctype == 'LDA':
                    rho2 = rho2[np.newaxis]
                wv = np.einsum('yg,xyg,g->xg', rho2, fxc, weight)
                fmat_(mol, f1oo, ao, wv, mask, shls_slice, ao_loc)
            if with_vxc:
                fmat_(mol, v1ao, ao, vxc * weight, mask, shls_slice, ao_loc)
            if with_kxc:
                wv = np.einsum('yg,zg,xyzg,g->xg', rho1, rho1, kxc, weight)
                fmat_(mol, k1ao, ao, wv, mask, shls_slice, ao_loc)
    else:
        for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            if xctype == 'LDA':
                ao0 = ao[0]
            else:
                ao0 = ao
            rho = ni.eval_rho2(mol, ao0, mo_coeff, mo_occ, mask, xctype, with_lapl=False)
            rho *= 0.5
            rho = np.repeat(rho[np.newaxis], 2, axis=0)
            vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)[1:]
            # fxc_t couples triplet excitation amplitudes
            # 1/2 int (tia - tIA) fxc (tjb - tJB) = tia fxc_t tjb
            fxc_t = fxc[:, :, 0] - fxc[:, :, 1]
            fxc_t = fxc_t[0] - fxc_t[1]

            rho1 = ni.eval_rho(mol, ao0, dmvo, mask, xctype, hermi=1, with_lapl=False)
            if xctype == 'LDA':
                rho1 = rho1[np.newaxis]
            wv = np.einsum('yg,xyg,g->xg', rho1, fxc_t, weight)
            fmat_(mol, f1vo, ao, wv, mask, shls_slice, ao_loc)

            if dmoo is not None:
                # fxc_s == 2 * fxc of spin restricted xc kernel
                # provides f1oo to couple the interaction between first order MO
                # and density response of tddft amplitudes, which is described by dmoo
                fxc_s = fxc[0, :, 0] + fxc[0, :, 1]
                rho2 = ni.eval_rho(mol, ao0, dmoo, mask, xctype, hermi=1, with_lapl=False)
                if xctype == 'LDA':
                    rho2 = rho2[np.newaxis]
                wv = np.einsum('yg,xyg,g->xg', rho2, fxc_s, weight)
                fmat_(mol, f1oo, ao, wv, mask, shls_slice, ao_loc)
            if with_vxc:
                vxc = vxc[0]
                fmat_(mol, v1ao, ao, vxc * weight, mask, shls_slice, ao_loc)
            if with_kxc:
                # kxc in terms of the triplet coupling
                # 1/2 int (tia - tIA) kxc (tjb - tJB) = tia kxc_t tjb
                kxc = kxc[0, :, 0] - kxc[0, :, 1]
                kxc = kxc[:, :, 0] - kxc[:, :, 1]
                wv = np.einsum('yg,zg,xyzg,g->xg', rho1, rho1, kxc, weight)
                fmat_(mol, k1ao, ao, wv, mask, shls_slice, ao_loc)

    f1vo[1:] *= -1
    if f1oo is not None:
        f1oo[1:] *= -1
    if v1ao is not None:
        v1ao[1:] *= -1
    if k1ao is not None:
        k1ao[1:] *= -1

    return f1vo, f1oo, v1ao, k1ao


def get_veff_krks(ks_grad, dm=None, kpts=None):
    from pyscf.pbc.grad.krks import get_vxc
    mf = ks_grad.base
    cell = ks_grad.cell
    if dm is None: dm = mf.make_rdm1()
    if kpts is None: kpts = mf.kpts
    t0 = (logger.process_clock(), logger.perf_counter())

    ni = mf._numint
    if ks_grad.grids is not None:
        grids = ks_grad.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
    if ks_grad.grid_response:
        raise NotImplementedError
    else:
        vxc = get_vxc(ni, cell, grids, mf.xc, dm, kpts,
                           max_memory=max_memory, verbose=ks_grad.verbose)

    t0 = logger.timer(ks_grad, 'vxc', *t0)
    if not ni.libxc.is_hybrid_xc(mf.xc):
        vjk = ks_grad.get_j(dm, kpts)
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=cell.spin)
        vj, vk = ks_grad.get_jk(dm, kpts)
        vk *= hyb
        if omega != 0:
            with cell.with_range_coulomb(omega):
                vk += ks_grad.get_k(dm, kpts) * (alpha - hyb)
        vjk = vj - vk * .5

    return vxc, vjk


def _contract_xc_kernel_krks(mf, xc_code, dmvo, max_memory=2000):
    from pyscf import lib
    from pyscf.lib import logger
    from pyscf.grad.tdrks import _lda_eval_mat_, _gga_eval_mat_, _mgga_eval_mat_
    from pyscf.dft.numint import eval_rho, eval_rho2

    mol = mf.mol
    grids = mf.grids

    ni = mf._numint
    xctype = ni._xc_type(xc_code)

    mo_coeff = mf.mo_coeff[0]
    mo_occ = mf.mo_occ[0]
    nao, nmo = mo_coeff.shape
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    # dmvo ~ reduce(np.dot, (orbv, Xai, orbo.T))
    dmvo = (dmvo + dmvo.T) * 0.5  # because K_{ia,jb} == K_{ia,bj}

    f1vo = np.zeros((4, nao, nao))  # 0th-order, d/dx, d/dy, d/dz
    deriv = 2
    f1oo = None
    v1ao = None
    k1ao = None

    if xctype == 'HF':
        return f1vo, f1oo, v1ao, k1ao
    elif xctype == 'LDA':
        fmat_, ao_deriv = _lda_eval_mat_, 1
    elif xctype == 'GGA':
        fmat_, ao_deriv = _gga_eval_mat_, 2
    elif xctype == 'MGGA':
        fmat_, ao_deriv = _mgga_eval_mat_, 2
        logger.warn(mf, 'PPRPA-MGGA Gradients may be inaccurate due to grids response')
    else:
        raise NotImplementedError(f'td-rks for functional {xc_code}')

    for aok0, aok1, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
        aok0 = aok0[0]
        if xctype == 'LDA':
            ao0 = aok0[0]
        else:
            ao0 = aok0
        rho = eval_rho2(mol, ao0, mo_coeff, mo_occ, mask, xctype, with_lapl=False)
        vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)[1:]

        rho1 = eval_rho(mol, ao0, dmvo, mask, xctype, hermi=1, with_lapl=False) * 2  # *2 for alpha + beta
        if xctype == 'LDA':
            rho1 = rho1[np.newaxis]
        wv = np.einsum('yg,xyg,g->xg', rho1, fxc, weight)
        fmat_(mol, f1vo, aok0, wv, mask, shls_slice, ao_loc)

    f1vo[1:] *= -1
    if f1oo is not None:
        f1oo[1:] *= -1
    if v1ao is not None:
        v1ao[1:] *= -1
    if k1ao is not None:
        k1ao[1:] *= -1
    return f1vo, f1oo, v1ao, k1ao
