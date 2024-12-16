import h5py
import numpy as np
import scipy

from lib_pprpa.analyze import pprpa_print_a_pair
from lib_pprpa.pprpa_direct import pprpa_orthonormalize_eigenvector, \
    diagonalize_pprpa_singlet, diagonalize_pprpa_triplet

from lib_pprpa.pprpa_util import ij2index, inner_product, start_clock, \
    stop_clock, print_citation, get_chemical_potential


def kernel(pprpa):
    # initialize trial vector and product matrix
    if pprpa._use_Lov:
        data_type = pprpa.Lpi.dtype
    else:
        data_type = pprpa.Lpq.dtype
    # the maximum size is max_vec + nroot for compacting
    tri_size = pprpa.max_vec + pprpa.nroot
    tri_vec = np.zeros(
        shape=[tri_size, pprpa.full_dim], dtype=data_type)
    tri_vec_sig = np.zeros(shape=[tri_size], dtype=data_type)
    if pprpa.channel == "pp":
        ntri = min(pprpa.nroot * 4, pprpa.vv_dim)
    else:
        ntri = min(pprpa.nroot * 4, pprpa.oo_dim)
    if pprpa.trial == "identity":
        tri_vec[:ntri], tri_vec_sig[:ntri] = get_identity_trial_vector(
            pprpa=pprpa, ntri=ntri)
    elif pprpa.trial == "subspace":
        tri_vec[:ntri], tri_vec_sig[:ntri] = get_subspace_trial_vector(
            pprpa=pprpa, ntri=ntri, channel=pprpa.channel,
            nocc_sub=pprpa.nocc_sub, nvir_sub=pprpa.nvir_sub)
    else:
        raise ValueError("trial vector method not recognized.")

    iter = 0
    nprod = 0  # number of contracted vectors
    mv_prod = np.zeros_like(tri_vec)  # ppRPA matrix vector product
    while iter < pprpa.max_iter:
        print(
            "\nppRPA Davidson %d-th iteration, ntri= %d , nprod= %d ." %
            (iter + 1, ntri, nprod), flush=True)
        mv_prod[nprod:ntri] = pprpa.contraction(tri_vec=tri_vec[nprod:ntri])
        nprod = ntri

        first_state, v_tri = _pprpa_subspace_diag(
            pprpa=pprpa, ntri=ntri, tri_vec=tri_vec,
            tri_vec_sig=tri_vec_sig, mv_prod=mv_prod)

        # If the subspace is too large, compact the subspace
        if ntri > pprpa.max_vec and pprpa._compact_subspace is True:
            ntri = _pprpa_compact_space(
            pprpa=pprpa, first_state=first_state, tri_vec=tri_vec,
            tri_vec_sig=tri_vec_sig, mv_prod=mv_prod, v_tri=v_tri)
            nprod = 0
            first_state, v_tri = _pprpa_subspace_diag(
                pprpa=pprpa, ntri=ntri, tri_vec=tri_vec,
                tri_vec_sig=tri_vec_sig, mv_prod=mv_prod)

        ntri_old = ntri
        conv, ntri = _pprpa_expand_space(
            pprpa=pprpa, first_state=first_state, tri_vec=tri_vec,
            tri_vec_sig=tri_vec_sig, mv_prod=mv_prod, v_tri=v_tri)
        print("add %d new trial vectors." % (ntri - ntri_old))

        iter += 1
        if conv is True:
            break

    assert conv is True, "ppRPA Davidson algorithm is not converged!"
    print("\nppRPA Davidson converged in %d iterations, final subspace size = %d" % (
        iter, nprod))

    pprpa_orthonormalize_eigenvector(
        multi=pprpa.multi, nocc=pprpa.nocc, exci=pprpa.exci, xy=pprpa.xy)

    return


# Davidson algorithm functions
def get_identity_trial_vector(pprpa, ntri):
    """Generate initial trial vectors in particle-particle or hole-hole channel.
    The order is determined by the pair orbital energy summation.
    The initial trial vectors are diagonal, and signatures are all 1 or -1.

    Args:
        pprpa (ppRPA_Davidson): ppRPA_Davidson object.
        ntri (int): the number of initial trial vectors.

    Returns:
        tri_vec (double ndarray): initial trial vectors.
        tri_vec_sig (double ndarray): signature of initial trial vectors.
    """

    is_singlet = 1 if pprpa.multi == "s" else 0

    max_orb_sum = 1.0e15

    class pair():
        def __init__(self):
            self.p = -1
            self.q = -1
            self.eig_sum = max_orb_sum

    pairs = []
    for r in range(ntri):
        t = pair()
        pairs.append(t)

    mo_energy = pprpa.mo_energy
    if pprpa.channel == "pp":
        # find particle-particle pairs with lowest orbital energy summation
        for r in range(ntri):
            for p in range(pprpa.nocc, pprpa.nmo):
                for q in range(pprpa.nocc, p + is_singlet):
                    valid = True
                    for rr in range(r):
                        if pairs[rr].p == p and pairs[rr].q == q:
                            valid = False
                            break
                    if (valid is True
                        and (mo_energy[p] + mo_energy[q]) < pairs[r].eig_sum):
                        pairs[r].p, pairs[r].q = p, q
                        pairs[r].eig_sum = mo_energy[p] + mo_energy[q]

        # sort pairs by ascending energy order
        for i in range(ntri-1):
            for j in range(i+1, ntri):
                if pairs[i].eig_sum > pairs[j].eig_sum:
                    p_tmp, q_tmp, eig_sum_tmp = \
                        pairs[i].p, pairs[i].q, pairs[i].eig_sum
                    pairs[i].p, pairs[i].q, pairs[i].eig_sum = \
                        pairs[j].p, pairs[j].q, pairs[j].eig_sum
                    pairs[j].p, pairs[j].q, pairs[j].eig_sum = \
                        p_tmp, q_tmp, eig_sum_tmp

        assert pairs[ntri-1].eig_sum < max_orb_sum, \
            "cannot find enough pairs for trial vectors"

        tri_vec = np.zeros(shape=[ntri, pprpa.full_dim], dtype=np.double)
        tri_vec_sig = np.zeros(shape=[ntri], dtype=np.double)
        tri_row_v, tri_col_v = np.tril_indices(pprpa.nvir, is_singlet-1)
        for r in range(ntri):
            p, q = pairs[r].p, pairs[r].q
            pq = ij2index(p - pprpa.nocc, q - pprpa.nocc, tri_row_v, tri_col_v)
            tri_vec[r, pprpa.oo_dim + pq] = 1.0
            tri_vec_sig[r] = 1.0
    else:
        # find hole-hole pairs with lowest orbital energy summation
        for r in range(ntri):
            for p in range(pprpa.nocc-1, -1, -1):
                for q in range(pprpa.nocc-1, p - is_singlet, -1):
                    valid = True
                    for rr in range(r):
                        if pairs[rr].p == q and pairs[rr].q == p:
                            valid = False
                            break
                    if (valid is True
                        and (mo_energy[p] + mo_energy[q]) < pairs[r].eig_sum):
                        pairs[r].p, pairs[r].q = q, p
                        pairs[r].eig_sum = mo_energy[p] + mo_energy[q]

        # sort pairs by descending energy order
        for i in range(ntri-1):
            for j in range(i+1, ntri):
                if pairs[i].eig_sum < pairs[j].eig_sum:
                    p_tmp, q_tmp, eig_sum_tmp = \
                        pairs[i].p, pairs[i].q, pairs[i].eig_sum
                    pairs[i].p, pairs[i].q, pairs[i].eig_sum = \
                        pairs[j].p, pairs[j].q, pairs[j].eig_sum
                    pairs[j].p, pairs[j].q, pairs[j].eig_sum = \
                        p_tmp, q_tmp, eig_sum_tmp

        assert pairs[ntri-1].eig_sum < max_orb_sum, \
            "cannot find enough pairs for trial vectors"

        tri_vec = np.zeros(shape=[ntri, pprpa.full_dim], dtype=np.double)
        tri_vec_sig = np.zeros(shape=[ntri], dtype=np.double)
        tri_row_o, tri_col_o = np.tril_indices(pprpa.nocc, is_singlet-1)
        for r in range(ntri):
            p, q = pairs[r].p, pairs[r].q
            pq = ij2index(p, q, tri_row_o, tri_col_o)
            tri_vec[r, pq] = 1.0
            tri_vec_sig[r] = -1.0

    return tri_vec, tri_vec_sig


def get_subspace_trial_vector(pprpa, ntri, channel=None, nocc_sub=40, nvir_sub=40):
    """Get trial vector from subspace diagonalization.

    Parameters
    ----------
    pprpa : ppRPA_Davidson
        ppRPA_Davidson object.
    ntri : int
        number of trial vectors.
    channel : str, optional
        channel to get ppRPA roots. "pp" or "hh", by default pprpa.channel
    nocc_sub : int, optional
        number of occupied orbitals in the subspace, by default 40
    nvir_sub : int, optional
        number of virtual orbitals in the subspace, by default 40

    Returns
    -------
    tri_vec: double ndarray
        initial trial vector.
    tri_vec_sig: double array
        signature of initial trial vector.
    """
    if channel is None:
        channel = pprpa.channel

    nocc_sub = min(pprpa.nocc, nocc_sub)
    nvir_sub = min(pprpa.nvir, nvir_sub)
    if pprpa.multi == "s":
        oo_dim_sub = int((nocc_sub + 1) * nocc_sub / 2)
        vv_dim_sub = int((nvir_sub + 1) * nvir_sub / 2)
        is_singlet = 1
    elif pprpa.multi == "t":
        oo_dim_sub = int((nocc_sub - 1) * nocc_sub / 2)
        vv_dim_sub = int((nvir_sub - 1) * nvir_sub / 2)
        is_singlet = 0

    if ntri > oo_dim_sub + vv_dim_sub:
        print("Number of trial vectors exceeds subspace size.")
        print("Use identity trial vectors instead.")
        return get_identity_trial_vector(pprpa, ntri)

    start, end = pprpa.nocc - nocc_sub, pprpa.nocc + nvir_sub
    mo_energy_sub = pprpa.mo_energy[start:end]
    if pprpa._use_Lov is True:
        Lpq_sub = np.concatenate(
            (pprpa.Lpi[:, :, start:], pprpa.Lpa[:, :, :nvir_sub]), axis=2)
        Lpq_sub = Lpq_sub[:, start:end, :]
    else:
        Lpq_sub = pprpa.Lpq[:, start:end, start:end]
    if pprpa.multi == "s":
        xy_sub = diagonalize_pprpa_singlet(nocc_sub, mo_energy_sub, Lpq_sub)[1]
    else:
        # GppRPA shares the same diagonalization function with triplet ppRPA
        xy_sub = diagonalize_pprpa_triplet(nocc_sub, mo_energy_sub, Lpq_sub)[1]

    tri_vec = np.zeros(shape=[ntri, pprpa.full_dim], dtype=xy_sub.dtype)
    if channel == "pp":
        xy_sub = xy_sub[oo_dim_sub: oo_dim_sub+ntri]
        tri_vec_sig = np.ones(shape=[ntri], dtype=np.double)
    else:
        xy_sub = np.flip(xy_sub[oo_dim_sub-ntri:oo_dim_sub], axis=0)
        tri_vec_sig = -np.ones(shape=[ntri], dtype=np.double)

    tri_row_o, tri_col_o = np.tril_indices(pprpa.nocc, is_singlet-1)
    tri_row_o_sub, tri_col_o_sub = np.tril_indices(nocc_sub, is_singlet-1)
    tri_row_v, tri_col_v = np.tril_indices(pprpa.nvir, is_singlet-1)
    tri_row_v_sub, tri_col_v_sub = np.tril_indices(nvir_sub, is_singlet-1)

    full_oo = np.zeros(shape=[ntri, pprpa.nocc, pprpa.nocc], dtype=xy_sub.dtype)
    full_vv = np.zeros(shape=[ntri, pprpa.nvir, pprpa.nvir], dtype=xy_sub.dtype)
    sub_oo = np.zeros(shape=[ntri, nocc_sub, nocc_sub], dtype=xy_sub.dtype)
    sub_vv = np.zeros(shape=[ntri, nvir_sub, nvir_sub], dtype=xy_sub.dtype)

    # Expand subspace lower triangle xy
    # Copy subspace xy to the correct positions in the full space xy
    # Compute full space xy to lower triangle
    sub_oo[:, tri_row_o_sub, tri_col_o_sub] = xy_sub[:, :oo_dim_sub]
    sub_vv[:, tri_row_v_sub, tri_col_v_sub] = xy_sub[:, oo_dim_sub:]
    full_oo[:, pprpa.nocc-nocc_sub:, pprpa.nocc-nocc_sub:] = sub_oo
    full_vv[:, :nvir_sub, :nvir_sub] = sub_vv
    tri_vec[:, :pprpa.oo_dim] = full_oo[:, tri_row_o, tri_col_o]
    tri_vec[:, pprpa.oo_dim:] = full_vv[:, tri_row_v, tri_col_v]

    return tri_vec, tri_vec_sig


def _pprpa_contraction(pprpa, tri_vec):
    """ppRPA contraction.

    Args:
        pprpa (ppRPA_Davidson): ppRPA_Davidson object.
        tri_vec (double ndarray): trial vector.

    Returns:
        mv_prod (double ndarray): product between ppRPA matrix and trial vectors.
    """
    nocc, nvir, nmo = pprpa.nocc, pprpa.nvir, pprpa.nmo
    naux = pprpa.naux
    mo_energy = pprpa.mo_energy
    Lpq = pprpa.Lpq
    Lpi = pprpa.Lpi
    Lpa = pprpa.Lpa

    ntri = tri_vec.shape[0]
    mv_prod = np.zeros(shape=[ntri, pprpa.full_dim], dtype=np.double)

    is_singlet = 1 if pprpa.multi == "s" else 0
    tri_row_o, tri_col_o = np.tril_indices(nocc, is_singlet - 1)
    tri_row_v, tri_col_v = np.tril_indices(nvir, is_singlet - 1)

    for ivec in range(ntri):
        # restore trial vector into full matrix
        z_oo = np.zeros(shape=[nocc, nocc], dtype=np.double)
        z_oo[tri_row_o, tri_col_o] = tri_vec[ivec][: pprpa.oo_dim]
        z_oo[np.diag_indices(nocc)] *= 1.0 / np.sqrt(2)
        z_oo = np.ascontiguousarray(z_oo.T)
        z_vv = np.zeros(shape=[nvir, nvir], dtype=np.double)
        z_vv[tri_row_v, tri_col_v] = tri_vec[ivec][pprpa.oo_dim :]
        z_vv[np.diag_indices(nvir)] *= 1.0 / np.sqrt(2)
        z_vv = np.ascontiguousarray(z_vv.T)

        # Lpqz_{L,pr} = \sum_s Lpq_{L,ps} z_{rs}
        Lpq_z = np.zeros(shape=[naux * nmo, nmo], dtype=np.double)
        if pprpa._use_Lov is True:
            Lpq_z[:, :nocc] = np.matmul(Lpi.reshape(naux * nmo, nocc), z_oo)
            Lpq_z[:, nocc:] = np.matmul(Lpa.reshape(naux * nmo, nvir), z_vv)
        else:
            Lpq_z[:, :nocc] = np.matmul(
                Lpq[:, :, :nocc].reshape(naux * nmo, nocc), z_oo)
            Lpq_z[:, nocc:] = np.matmul(
                Lpq[:, :, nocc:].reshape(naux * nmo, nvir), z_vv)

        # transpose and reshape for faster multiplication
        Lpq_z = Lpq_z.reshape(naux, nmo, nmo).transpose(1, 0, 2)
        Lpq_z = Lpq_z.reshape(nmo, naux * nmo)
        # NOTE: here assuming Lpq[L,p,q] = Lpq[L,q,p] for real orbitals
        if pprpa._use_Lov is True:
            prod_oo = np.matmul(Lpq_z[:nocc], Lpi.reshape(naux * nmo, nocc))
        else:
            prod_oo = np.matmul(
                Lpq_z[:nocc], Lpq[:, :, :nocc].reshape(naux * nmo, nocc))
        if pprpa.multi == "s":
            prod_oo += prod_oo.T
        else:
            prod_oo -= prod_oo.T
        # rotate upper-half to lower-half matrix
        prod_oo = prod_oo.T
        prod_oo[np.diag_indices(nocc)] *= 1.0 / np.sqrt(2)

        if pprpa._use_Lov is True:
            prod_vv = np.matmul(Lpq_z[nocc:], Lpa.reshape(naux * nmo, nvir))
        else:
            prod_vv = np.matmul(
                Lpq_z[nocc:], Lpq[:, :, nocc:].reshape(naux * nmo, nvir))
        if pprpa.multi == "s":
            prod_vv += prod_vv.T
        else:
            prod_vv -= prod_vv.T
        # rotate upper-half to lower-half matrix
        prod_vv = prod_vv.T
        prod_vv[np.diag_indices(nvir)] *= 1.0 / np.sqrt(2)

        mv_prod[ivec][: pprpa.oo_dim] = prod_oo[tri_row_o, tri_col_o]
        mv_prod[ivec][pprpa.oo_dim :] = prod_vv[tri_row_v, tri_col_v]

    # orbital energy contribution
    orb_sum_oo = np.array(mo_energy[None, :nocc] + mo_energy[:nocc, None])
    orb_sum_oo = orb_sum_oo[tri_row_o, tri_col_o]
    orb_sum_vv = np.array(mo_energy[None, nocc:] + mo_energy[nocc:, None])
    orb_sum_vv = orb_sum_vv[tri_row_v, tri_col_v]
    orb_sum = np.concatenate((orb_sum_oo, orb_sum_vv)) - 2.0 * pprpa.mu
    # hole-hole block has a factor -1
    orb_sum[: pprpa.oo_dim] *= -1.0
    mv_prod += orb_sum * tri_vec

    return mv_prod


def _pprpa_subspace_diag(pprpa, ntri, tri_vec, tri_vec_sig, mv_prod):
    """Diagonalize ppRPA matrix in the trial vector subspace.

    Parameters
    ----------
    pprpa : ppRPA_Davidson
        pprpa object
    ntri : int
        number of trial vectors
    tri_vec : double or complex ndarray
        trial vector
    tri_vec_sig : double ndarray
        signature of trial vectors
    mv_prod : double or complex ndarray
        product of ppRPA matrix and trial vector

    Returns
    -------
    first_state : int
        index of the first desired state
    v_tri : double or complex ndarray
        eigenvector in the trial vector subspace
    """
    data_type = tri_vec.dtype
    # get ppRPA matrix and metric matrix in subspace
    m_tilde = np.matmul(tri_vec[:ntri].conj(), mv_prod[:ntri].T)
    w_tilde = np.zeros_like(m_tilde)
    for i in range(ntri):
        if inner_product(tri_vec[i].conj(), tri_vec[i], pprpa.oo_dim).real > 0:
            w_tilde[i, i] = 1
        else:
            w_tilde[i, i] = -1

    # diagonalize subspace matrix
    if data_type == np.double:
        alphar, _, beta, _, v_tri, _, _ = scipy.linalg.lapack.dggev(
            m_tilde, w_tilde, compute_vl=0)
    elif data_type == np.complex128:
        alphar, beta, _, v_tri, _, _ = scipy.linalg.lapack.zggev(
            m_tilde, w_tilde, compute_vl=0)
    e_tri = (alphar / beta).real
    v_tri = v_tri.T  # Fortran matrix to Python order

    if pprpa.channel == "pp":
        # sort eigenvalues and eigenvectors by ascending order
        idx = e_tri.argsort()
        e_tri = e_tri[idx]
        v_tri = v_tri[idx, :]

        # re-order all states by signs, first hh then pp
        sig = np.zeros(shape=[ntri], dtype=int)
        for i in range(ntri):
            if np.sum(v_tri[i].conj() * tri_vec_sig[:ntri] * v_tri[i]).real > 0:
                sig[i] = 1
            else:
                sig[i] = -1

        hh_index = np.where(sig < 0)[0]
        pp_index = np.where(sig > 0)[0]
        e_tri_hh = e_tri[hh_index]
        e_tri_pp = e_tri[pp_index]
        e_tri[:len(hh_index)] = e_tri_hh
        e_tri[len(hh_index):] = e_tri_pp
        v_tri_hh = v_tri[hh_index]
        v_tri_pp = v_tri[pp_index]
        v_tri[:len(hh_index)] = v_tri_hh
        v_tri[len(hh_index):] = v_tri_pp

        # get only two-electron addition energy
        first_state=len(hh_index)
        pprpa.exci = e_tri[first_state:first_state+pprpa.nroot]
    else:
        # sort eigenvalues and eigenvectors by descending order
        idx = e_tri.argsort()[::-1]
        e_tri = e_tri[idx]
        v_tri = v_tri[idx, :]

        # re-order all states by signs, first pp then hh
        sig = np.zeros(shape=[ntri], dtype=int)
        for i in range(ntri):
            if np.sum(v_tri[i].conj() * tri_vec_sig[:ntri] * v_tri[i]).real > 0:
                sig[i] = 1
            else:
                sig[i] = -1

        hh_index = np.where(sig < 0)[0]
        pp_index = np.where(sig > 0)[0]
        e_tri_hh = e_tri[hh_index]
        e_tri_pp = e_tri[pp_index]
        e_tri[:len(pp_index)] = e_tri_pp
        e_tri[len(pp_index):] = e_tri_hh
        v_tri_hh = v_tri[hh_index]
        v_tri_pp = v_tri[pp_index]
        v_tri[:len(pp_index)] = v_tri_pp
        v_tri[len(pp_index):] = v_tri_hh

        # get only two-electron removal energy
        first_state=len(pp_index)
        pprpa.exci = e_tri[first_state:first_state+pprpa.nroot]

    return first_state, v_tri


def _pprpa_expand_space(
        pprpa, first_state, tri_vec, tri_vec_sig, mv_prod, v_tri):
    """Expand trial vector space in Davidson algorithm.

    Args:
        pprpa (ppRPA_Davidson): ppRPA_Davidson object.
        first_state (int): index of first particle-particle or hole-hole state.
        tri_vec (double ndarray): trial vector.
        tri_vec_sig (int array): signature of trial vector.
        mv_prod (double ndarray): product matrix of ppRPA matrix and trial vector.
        v_tri (double ndarray): eigenvector of subspace matrix.

    Returns:
        conv (bool): if Davidson algorithm is converged.
        ntri (int): updated number of trial vectors.
    """
    nocc, nvir = pprpa.nocc, pprpa.nvir
    mo_energy = pprpa.mo_energy
    nroot = pprpa.nroot
    exci = pprpa.exci
    max_vec = pprpa.max_vec
    residue_thresh = pprpa.residue_thresh

    is_singlet = 1 if pprpa.multi == "s" else 0

    tri_row_o, tri_col_o = np.tril_indices(nocc, is_singlet-1)
    tri_row_v, tri_col_v = np.tril_indices(nvir, is_singlet-1)

    # take only nRoot vectors, starting from first pp/hh channel
    tmp = v_tri[first_state:(first_state+nroot)]

    # get the eigenvectors in the original space
    ntri = v_tri.shape[0]
    pprpa.xy = np.matmul(tmp, tri_vec[:ntri])

    # compute residue vectors
    residue = np.matmul(tmp, mv_prod[:ntri])
    for i in range(nroot):
        residue[i][:pprpa.oo_dim] -= -exci[i] * pprpa.xy[i][:pprpa.oo_dim]
        residue[i][pprpa.oo_dim:] += -exci[i] * pprpa.xy[i][pprpa.oo_dim:]

    # check convergence
    conv_record = np.zeros(shape=[nroot], dtype=bool)
    max_residue = 0
    for i in range(nroot):
        max_residue = max(max_residue, abs(np.max(residue[i])))
        conv_record[i] = True if len(
            residue[i][abs(residue[i]) > residue_thresh]) == 0 else False
    nconv = len(conv_record[conv_record is True])
    print("max residue = %.6e" % max_residue)
    if nconv == nroot:
        return True, ntri

    orb_sum_oo = np.asarray(
        mo_energy[None, : nocc] + mo_energy[: nocc, None]) - 2.0 * pprpa.mu
    orb_sum_oo = orb_sum_oo[tri_row_o, tri_col_o]
    orb_sum_vv = np.asarray(
        mo_energy[None, nocc:] + mo_energy[nocc:, None]) - 2.0 * pprpa.mu
    orb_sum_vv = orb_sum_vv[tri_row_v, tri_col_v]

    # Schmidt orthogonalization
    ntri_old = ntri
    for iroot in range(nroot):
        if conv_record[iroot] is True:
            continue

        # convert residuals
        residue[iroot][:pprpa.oo_dim] /= -(exci[iroot] - orb_sum_oo)
        residue[iroot][pprpa.oo_dim:] /= (exci[iroot] - orb_sum_vv)

        for ivec in range(ntri):
            # compute product between new vector and old vector
            inp = -inner_product(residue[iroot], tri_vec[ivec].conj(), pprpa.oo_dim)
            # eliminate parallel part
            if tri_vec_sig[ivec] < 0:
                inp = -inp
            residue[iroot] += inp * tri_vec[ivec]

        # add a new trial vector
        if len(residue[iroot][abs(residue[iroot]) > residue_thresh]) > 0:
            if pprpa._compact_subspace is False:
                assert ntri < max_vec, "Davidson expansion failed!"
            inp = inner_product(residue[iroot].conj(), residue[iroot], pprpa.oo_dim).real
            tri_vec_sig[ntri] = 1 if inp > 0 else -1
            tri_vec[ntri] = residue[iroot] / np.sqrt(abs(inp))
            ntri = ntri + 1

    conv = True if ntri_old == ntri else False
    return conv, ntri


def _pprpa_compact_space(pprpa, first_state, tri_vec, tri_vec_sig, mv_prod, v_tri):
    """Generate new trial vectors from non-converged eigenvectors.

    Parameters
    ----------
    pprpa : ppRPA_Davidson
        ppRPA object
    first_state : int
        index of the first desired state
    tri_vec : double or complex ndarray
        trial vector, will be overwritten
    tri_vec_sig : double ndarray
        signature of trial vectors, will be overwritten
    mv_prod : double or complex ndarray
        product of ppRPA matrix and trial vector, will be overwritten
    v_tri : double or complex ndarray
        eigenvector in the trial vector subspace

    Returns
    -------
    ntri : ntri
        number of trail vectors
    """
    print("Compacting subspace...")
    if pprpa.channel == "pp":
        ntri = min(pprpa.nroot * 4, pprpa.vv_dim)
    else:
        ntri = min(pprpa.nroot * 4, pprpa.oo_dim)
    ntri_old = v_tri.shape[0]

    # recombines trial vector for the "best" ntri subspace vectors
    # v_tri is the coefficient from old to new trial space
    tmp = v_tri[first_state:(first_state+ntri)]
    tri_vec[:ntri] = np.matmul(tmp, tri_vec[:ntri_old])
    mv_prod[:ntri] = np.matmul(tmp, mv_prod[:ntri_old])

    tri_vec_sig[:ntri] = np.zeros(shape=[ntri], dtype=np.double)
    for i in range(ntri):
        if inner_product(tri_vec[i].conj(), tri_vec[i], pprpa.oo_dim).real > 0:
            tri_vec_sig[i] = 1
        else:
            tri_vec_sig[i] = -1

    # orthonormalize
    for i in range(ntri):
        for j in range(i):
            norm_j = inner_product(tri_vec[j].conj(), tri_vec[j], pprpa.oo_dim).real
            inp = inner_product(tri_vec[j].conj(), tri_vec[i], pprpa.oo_dim) / norm_j
            tri_vec[i] -= tri_vec[j] * inp
            mv_prod[i] -= mv_prod[j] * inp
        inp = inner_product(tri_vec[i].conj(), tri_vec[i], pprpa.oo_dim).real
        tri_vec[i] /= np.sqrt(abs(inp))
        mv_prod[i] /= np.sqrt(abs(inp))

    return ntri

# analysis functions
def _pprpa_print_eigenvector(
        multi, nocc, nvir, thresh, channel, exci0, exci, xy):
    """Print dominant components of an eigenvector.

    Args:
        multi (string): multiplicity.
        nocc (int): number of occupied orbitals.
        nvir (int): number of virtual orbitals.
        thresh (double): threshold to print a pair.
        channel (string): "pp" for particle-particle or "hh" for hole-hole.
        exci0 (double): lowest eigenvalue.
        exci (double array): ppRPA eigenvalue.
        xy (double ndarray): ppRPA eigenvector.
    """
    if multi == "s":
        oo_dim = int((nocc + 1) * nocc / 2)
        is_singlet = 1
        print("\n     print ppRPA excitations: singlet\n")
    elif multi == "t":
        oo_dim = int((nocc - 1) * nocc / 2)
        is_singlet = 0
        print("\n     print ppRPA excitations: triplet\n")

    tri_row_o, tri_col_o = np.tril_indices(nocc, is_singlet-1)
    tri_row_v, tri_col_v = np.tril_indices(nvir, is_singlet-1)

    nroot = len(exci)
    au2ev = 27.211386
    if channel == "pp":
        for iroot in range(nroot):
            print("#%-d %s excitation:  exci= %-12.6f  eV   2e=  %-12.6f  eV" %
                  (iroot + 1, multi,
                   (exci[iroot] - exci0) * au2ev, exci[iroot] * au2ev))
            if nocc > 0:
                full = np.zeros(shape=[nocc, nocc], dtype=np.double)
                full[tri_row_o, tri_col_o] = xy[iroot][:oo_dim]
                full = np.power(full, 2)
                pairs = np.argwhere(full > thresh)
                for i, j in pairs:
                    pprpa_print_a_pair(
                        is_pp=False, p=i, q=j, percentage=full[i, j])

            full = np.zeros(shape=[nvir, nvir], dtype=np.double)
            full[tri_row_v, tri_col_v] = xy[iroot][oo_dim:]
            full = np.power(full, 2)
            pairs = np.argwhere(full > thresh)
            for a, b in pairs:
                pprpa_print_a_pair(
                    is_pp=True, p=a+nocc, q=b+nocc, percentage=full[a, b])
            print("")
    else:
        for iroot in range(nroot):
            print("#%-d %s de-excitation:  exci= %-12.6f  eV   2e=  %-12.6f  eV" %
                  (iroot + 1, multi,
                   (exci[iroot] - exci0) * au2ev, exci[iroot] * au2ev))
            full = np.zeros(shape=[nocc, nocc], dtype=np.double)
            full[tri_row_o, tri_col_o] = xy[iroot][:oo_dim]
            full = np.power(full, 2)
            pairs = np.argwhere(full > thresh)
            for i, j in pairs:
                pprpa_print_a_pair(is_pp=False, p=i, q=j, percentage=full[i, j])

            if nvir > 0:
                full = np.zeros(shape=[nvir, nvir], dtype=np.double)
                full[tri_row_v, tri_col_v] = xy[iroot][oo_dim:]
                full = np.power(full, 2)
                pairs = np.argwhere(full > thresh)
                for a, b in pairs:
                    pprpa_print_a_pair(
                        is_pp=True, p=a+nocc, q=b+nocc, percentage=full[a, b])
            print("")

    return


def _analyze_pprpa_davidson(
        exci_s, xy_s, exci_t, xy_t, nocc, nvir, print_thresh=0.1, channel="pp"):
    print("\nanalyze ppRPA results.")

    if exci_s is not None and exci_t is not None:
        print("both singlet and triplet results found.")
        if channel == "pp":
            exci0 = min(exci_s[0], exci_t[0])
        else:
            exci0 = max(exci_s[0], exci_t[0])
        _pprpa_print_eigenvector(
            multi="s", nocc=nocc, nvir=nvir, thresh=print_thresh,
            channel=channel, exci0=exci0, exci=exci_s, xy=xy_s)
        _pprpa_print_eigenvector(
            multi="t", nocc=nocc, nvir=nvir, thresh=print_thresh,
            channel=channel, exci0=exci0, exci=exci_t, xy=xy_t)
    else:
        if exci_s is not None:
            print("only singlet results found.")
            _pprpa_print_eigenvector(
                multi="s", nocc=nocc, nvir=nvir, thresh=print_thresh,
                channel=channel, exci0=exci_s[0], exci=exci_s, xy=xy_s)
        else:
            print("only triplet results found.")
            _pprpa_print_eigenvector(
                multi="t", nocc=nocc, nvir=nvir, thresh=print_thresh,
                channel=channel, exci0=exci_t[0], exci=exci_t, xy=xy_t)
    return


class ppRPA_Davidson():
    def __init__(
            self, nocc, mo_energy, Lpq, channel="pp", nroot=5, max_vec=500,
            max_iter=100, trial="identity", residue_thresh=1.0e-7,
            print_thresh=0.1):
        # necessary input
        self.nocc = nocc  # number of occupied orbitals
        self.mo_energy = np.asarray(mo_energy)  # orbital energy
        # three-center density-fitting matrix in MO space
        self.Lpq = np.asarray(Lpq)
        self._use_Lov = False  # use C-contiguous Lpq block for better performance
        self.Lpi = None  # Lpi = Lpq[:, :, :nocc], C-contiguous
        self.Lpa = None  # Lpa = Lpq[:, :, nocc:], C-contiguous

        # options
        self.channel = channel  # channel of desired states, particle-particle or hole-hole
        self.nroot = nroot  # number of desired roots
        self.max_vec = max_vec  # max size of trial vectors
        self.max_iter = max_iter  # max iteration
        self.trial = trial  # mode to initialize trial vector
        self.nocc_sub = 40  # number of occpuied orbitals in the trial vector subspace
        self.nvir_sub = 40  # number of virtual orbitals in the trial vector subspace
        self.residue_thresh = residue_thresh  # residue threshold
        self.print_thresh = print_thresh  # threshold to print component
        self._compact_subspace = False  # compact large subspace

        # internal flags
        self.multi = None  # multiplicity
        self.is_singlet = None  # multiplicity is singlet
        self.mu = None  # chemical potential
        self.nmo = len(self.mo_energy)  # number of orbitals
        self.nvir = self.nmo - self.nocc  # number of virtual orbitals
        self.naux = Lpq.shape[0]  # number of auxiliary basis functions
        self.oo_dim = None  # particle-particle block dimension
        self.vv_dim = None  # hole-hole block dimension
        self.full_dim = None  # full matrix dimension

        # results
        self.exci = None  # two-electron addition energy
        self.xy = None  # ppRPA eigenvector
        self.exci_s = None  # singlet two-electron addition energy
        self.xy_s = None  # singlet two-electron addition eigenvector
        self.exci_t = None  # triplet two-electron addition energy
        self.xy_t = None  # triplet two-electron addition eigenvector

        print_citation()

        return

    def check_parameter(self):
        assert self.channel in ["pp", "hh"]

        assert self.multi in ["s", "t"]
        if self.multi == "s":
            self.oo_dim = int((self.nocc + 1) * self.nocc / 2)
            self.vv_dim = int((self.nvir + 1) * self.nvir / 2)
        elif self.multi == "t":
            self.oo_dim = int((self.nocc - 1) * self.nocc / 2)
            self.vv_dim = int((self.nvir - 1) * self.nvir / 2)
        self.full_dim = self.oo_dim + self.vv_dim

        self.max_vec = min(self.max_vec, self.full_dim)

        assert self.residue_thresh > 0
        assert 0.0 < self.print_thresh < 1.0

        if self.mu is None:
            self.mu = get_chemical_potential(
                nocc=self.nocc, mo_energy=self.mo_energy)

        return

    def dump_flags(self):
        print('\n******** %s ********' % self.__class__)
        print(
            'multiplicity = %s' %
            ("singlet" if self.multi == "s" else "triplet"))
        print('state channel = %s' % self.channel)
        print('naux = %d' % self.naux)
        print('nmo = %d' % self.nmo)
        print('nocc = %d nvir = %d' % (self.nocc, self.nvir))
        print("occ-occ dimension = %d vir-vir dimension = %d"
              % (self.oo_dim, self.vv_dim))
        print('full dimension = %d' % self.full_dim)
        print('number of roots = %d' % self.nroot)
        print('max subspace size = %d' % self.max_vec)
        print('max iteration = %d' % self.max_iter)
        print('trial vector = %s' % self.trial)
        if self.trial == "subspace":
            print("subspace nocc = %d nvir = %d" % (self.nocc_sub, self.nvir_sub))
        print('residue threshold = %.3e' % self.residue_thresh)
        print('print threshold = %.2f%%' % (self.print_thresh*100))
        # experiment features
        print("_use_Lov = %s" % self._use_Lov)
        print("_compact_subspace = %s" % self._compact_subspace)
        print('')
        return

    def check_memory(self):
        # intermediate in contraction; mv_prod, tri_vec, xy
        mem = (
            self.naux * self.nmo * self.nmo + 3 * self.max_vec * self.full_dim)\
                * 8 / 1.0e6
        if mem < 1000:
            print("ppRPA needs at least %d MB memory." % mem)
        else:
            print("ppRPA needs at least %.1f GB memory." % (mem / 1.0e3))
        return

    def kernel(self, multi):
        self.multi = multi
        self.check_parameter()

        # TODO: directly take Lpi and Lpa in the future
        if self._use_Lov is True and self.Lpq is not None:
            self.Lpi = np.ascontiguousarray(self.Lpq[:, :, :self.nocc])
            self.Lpa = np.ascontiguousarray(self.Lpq[:, :, self.nocc:])
            self.Lpq = None

        self.dump_flags()
        self.check_memory()
        start_clock("ppRPA Davidson: %s" % multi)
        kernel(pprpa=self)
        stop_clock("ppRPA Davidson: %s" % multi)
        if self.multi == "s":
            self.exci_s = self.exci.copy()
            self.xy_s = self.xy.copy()
        else:
            self.exci_t = self.exci.copy()
            self.xy_t = self.xy.copy()
        self.exci = self.xy = None
        return

    def save_pprpa(self, fn):
        assert self.exci_s is not None or self.exci_t is not None
        print("\nsave pprpa results to %s.\n" % fn)
        f = h5py.File(fn, "w")
        f["nocc"] = np.asarray(self.nocc)
        f["nvir"] = np.asarray(self.nvir)
        if self.exci_s is not None:
            f["exci_s"] = np.asarray(self.exci_s)
            f["xy_s"] = np.asarray(self.xy_s)
        if self.exci_t is not None:
            f["exci_t"] = np.asarray(self.exci_t)
            f["xy_t"] = np.asarray(self.xy_t)
        f.close()
        return

    def read_pprpa(self, fn, singlet=True, triplet=True):
        print("\nread pprpa results from %s.\n" % fn)
        f = h5py.File(fn, "r")
        if singlet is True:
            self.exci_s = np.asarray(f["exci_s"])
            self.xy_s = np.asarray(f["xy_s"])
        if triplet is True:
            self.exci_t = np.asarray(f["exci_t"])
            self.xy_t = np.asarray(f["xy_t"])
        f.close()
        return

    def analyze(self):
        _analyze_pprpa_davidson(
            exci_s=self.exci_s, xy_s=self.xy_s, exci_t=self.exci_t,
            xy_t=self.xy_t, nocc=self.nocc, nvir=self.nvir,
            print_thresh=self.print_thresh, channel=self.channel)
        return

    def contraction(self, tri_vec):
        return _pprpa_contraction(self, tri_vec)
