import h5py
import numpy
import scipy

from numpy import einsum

from lib_pprpa.pprpa_util import ij2index, inner_product, start_clock, stop_clock, print_citation, get_chemical_potential


def kernel(pprpa):
    # initialize trial vector and product matrix
    tri_vec = numpy.zeros(shape=[pprpa.max_vec, pprpa.full_dim], dtype=numpy.double)
    tri_vec_sig = numpy.zeros(shape=[pprpa.max_vec], dtype=numpy.double)
    ntri = min(pprpa.nroot * 4, pprpa.vv_dim) if pprpa.channel == "pp" else min(pprpa.nroot * 4, pprpa.oo_dim)
    tri_vec[:ntri], tri_vec_sig[:ntri] = _pprpa_get_trial_vector(pprpa=pprpa, ntri=ntri)
    mv_prod = numpy.zeros_like(tri_vec)

    iter = 0
    nprod = 0 # number of contracted vectors
    while iter < pprpa.max_iter:
        print("\nppRPA Davidson %d-th iteration, ntri= %d , nprod= %d ." % (iter+1, ntri, nprod))
        mv_prod[nprod:ntri] = _pprpa_contraction(pprpa=pprpa, tri_vec=tri_vec[nprod:ntri])
        nprod = ntri

        # get ppRPA matrix and metric matrix in subspace
        m_tilde = numpy.dot(tri_vec[:ntri], mv_prod[:ntri].T)
        w_tilde = numpy.zeros_like(m_tilde)
        for i in range(ntri):
            w_tilde[i, i] = 1 if inner_product(tri_vec[i], tri_vec[i], pprpa.oo_dim) > 0 else -1

        # diagonalize subspace matrix
        alphar, _, beta, _, v_tri, _, _ = scipy.linalg.lapack.dggev(m_tilde, w_tilde, compute_vl=0)
        e_tri = alphar / beta
        v_tri = v_tri.T  # Fortran matrix to Python order

        if pprpa.channel == "pp":
            # TODO: this sorting algorithm can be improved
            # sort eigenvalues and eigenvectors by ascending order
            v_tri = numpy.asarray(list(x for _, x in sorted(zip(e_tri, v_tri), reverse=False)))
            e_tri = numpy.sort(e_tri)

            # re-order all states by signs, first hh then pp
            sig = numpy.zeros(shape=[ntri], dtype=int)
            for i in range(ntri):
                sig[i] = 1 if numpy.sum((v_tri[i] ** 2) * tri_vec_sig[:ntri]) > 0 else -1

            hh_index = numpy.where(sig < 0)[0]
            pp_index = numpy.where(sig > 0)[0]
            e_tri_hh = e_tri[hh_index]
            e_tri_pp = e_tri[pp_index]
            e_tri[:len(hh_index)] = e_tri_hh
            e_tri[len(hh_index):] = e_tri_pp
            v_tri_hh = v_tri[hh_index]
            v_tri_pp = v_tri[pp_index]
            v_tri[:len(hh_index)] = v_tri_hh
            v_tri[len(hh_index):] = v_tri_pp

            # get only two-electron addition energy
            pprpa.exci = e_tri[len(hh_index):(len(hh_index)+pprpa.nroot)]
        else:
            # TODO: this sorting algorithm can be improved
            # sort eigenvalues and eigenvectors by ascending order
            v_tri = numpy.asarray(list(x for _, x in sorted(zip(e_tri, v_tri), reverse=True)))
            e_tri = numpy.sort(e_tri)
            e_tri = e_tri[::-1]

            # TODO: re-order all states by the signs
            # get first hh state by the sign of the eigenvector, not by the sign of the excitation energy
            for i in range(ntri):
                sum = numpy.sum((v_tri[i] ** 2) * tri_vec_sig[:ntri])
                if sum < 0:
                    first_state = i
                    break

            # get only two-electron addition energy
            pprpa.exci = e_tri[first_state:(first_state+pprpa.nroot)]

        ntri_old = ntri
        conv, ntri = _pprpa_expand_space(pprpa=pprpa, first_state=len(hh_index), tri_vec=tri_vec,
                                         tri_vec_sig=tri_vec_sig, mv_prod=mv_prod, v_tri=v_tri)
        print("add %d new trial vectors." % (ntri - ntri_old))

        iter += 1
        if conv is True:
            break

    assert conv is True, "ppRPA Davidson algorithm is not converged!"
    print("\nppRPA Davidson converged in %d iterations, final subspace size = %d" % (iter, nprod))

    pprpa_orthonormalize_eigenvector(multi=pprpa.multi, nocc=pprpa.nocc, exci=pprpa.exci, xy=pprpa.xy)

    return


# Davidson algorithm functions
def _pprpa_get_trial_vector(pprpa, ntri):
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
                    if valid is True and (pprpa.mo_energy[p] + pprpa.mo_energy[q]) < pairs[r].eig_sum:
                        pairs[r].p, pairs[r].q = p, q
                        pairs[r].eig_sum = pprpa.mo_energy[p] + pprpa.mo_energy[q]

        # sort pairs by ascending energy order
        for i in range(ntri-1):
            for j in range(i+1, ntri):
                if pairs[i].eig_sum > pairs[j].eig_sum:
                    p_tmp, q_tmp, eig_sum_tmp = pairs[i].p, pairs[i].q, pairs[i].eig_sum
                    pairs[i].p, pairs[i].q, pairs[i].eig_sum = pairs[j].p, pairs[j].q, pairs[j].eig_sum
                    pairs[j].p, pairs[j].q, pairs[j].eig_sum = p_tmp, q_tmp, eig_sum_tmp

        assert pairs[ntri-1].eig_sum < max_orb_sum, "cannot find enough pairs for trial vectors"

        tri_vec = numpy.zeros(shape=[ntri, pprpa.full_dim], dtype=numpy.double)
        tri_vec_sig = numpy.zeros(shape=[ntri], dtype=numpy.double)
        tri_row_v, tri_col_v = numpy.tril_indices(pprpa.nvir, is_singlet-1)
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
                    if valid is True and (pprpa.mo_energy[p] + pprpa.mo_energy[q]) < pairs[r].eig_sum:
                        pairs[r].p, pairs[r].q = q, p
                        pairs[r].eig_sum = pprpa.mo_energy[p] + pprpa.mo_energy[q]

        # sort pairs by descending energy order
        for i in range(ntri-1):
            for j in range(i+1, ntri):
                if pairs[i].eig_sum < pairs[j].eig_sum:
                    p_tmp, q_tmp, eig_sum_tmp = pairs[i].p, pairs[i].q, pairs[i].eig_sum
                    pairs[i].p, pairs[i].q, pairs[i].eig_sum = pairs[j].p, pairs[j].q, pairs[j].eig_sum
                    pairs[j].p, pairs[j].q, pairs[j].eig_sum = p_tmp, q_tmp, eig_sum_tmp

        assert pairs[ntri-1].eig_sum < max_orb_sum, "cannot find enough pairs for trial vectors"

        tri_vec = numpy.zeros(shape=[ntri, pprpa.full_dim], dtype=numpy.double)
        tri_vec_sig = numpy.zeros(shape=[ntri], dtype=numpy.double)
        tri_row_o, tri_col_o = numpy.tril_indices(pprpa.nocc, is_singlet-1)
        for r in range(ntri):
            p, q = pairs[r].p, pairs[r].q
            pq = ij2index(p, q, tri_row_o, tri_col_o)
            tri_vec[r, pq] = 1.0
            tri_vec_sig[r] = -1.0

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

    ntri = tri_vec.shape[0]
    mv_prod = numpy.zeros(shape=[ntri, pprpa.full_dim], dtype=numpy.double)

    pm = 1.0 if pprpa.multi == "s" else -1.0
    is_singlet = 1 if pprpa.multi == "s" else 0
    tri_row_o, tri_col_o = numpy.tril_indices(nocc, is_singlet-1)
    tri_row_v, tri_col_v = numpy.tril_indices(nvir, is_singlet-1)

    if nocc == 0:
        for ivec in range(ntri):
            z_vv = numpy.zeros(shape=[nvir, nvir], dtype=numpy.double)
            z_vv[tri_row_v, tri_col_v] = tri_vec[ivec]
            z_vv[numpy.diag_indices(nvir)] *= 1.0 / numpy.sqrt(2)

            # Lpqz_{L,pr} = \sum_s Lpq_{L,ps} z_{rs}
            Lpq_z = einsum("Lps,rs->Lpr", Lpq[:, nocc:, nocc:], z_vv, optimize=True)

            # MV_{pq} = \sum_{Lr} Lpq_{L,pr} Lpqz_{L,qr} \pm Lpq_{L,qr} Lpqz_{L,pr}
            mv_prod_full = einsum("Lpr,Lqr->pq", Lpq[:, nocc:, nocc:], Lpq_z, optimize=True)
            mv_prod_full += einsum("Lqr,Lpr->pq", Lpq[:, nocc:, nocc:], Lpq_z, optimize=True) * pm
            mv_prod_full[numpy.diag_indices(nvir)] *= 1.0 / numpy.sqrt(2)
            mv_prod[ivec] = mv_prod_full[tri_row_v, tri_col_v]

        orb_sum_vv = numpy.asarray(mo_energy[None, nocc:] + mo_energy[nocc:, None]) - 2.0 * pprpa.mu
        orb_sum_vv = orb_sum_vv[tri_row_v, tri_col_v]
        for ivec in range(ntri):
            oz_vv = orb_sum_vv * tri_vec[ivec]
            mv_prod[ivec] += oz_vv
    elif nvir == 0:
        for ivec in range(ntri):
            z_oo = numpy.zeros(shape=[nocc, nocc], dtype=numpy.double)
            z_oo[tri_row_o, tri_col_o] = tri_vec[ivec]
            z_oo[numpy.diag_indices(nocc)] *= 1.0 / numpy.sqrt(2)

            # Lpqz_{L,pr} = \sum_s Lpq_{L,ps} z_{rs}
            Lpq_z = einsum("Lps,rs->Lpr", Lpq[:, :nocc, :nocc], z_oo, optimize=True)

            # MV_{pq} = \sum_{Lr} Lpq_{L,pr} Lpqz_{L,qr} \pm Lpq_{L,qr} Lpqz_{L,pr}
            mv_prod_full = einsum("Lpr,Lqr->pq", Lpq[:, :nocc, :nocc], Lpq_z, optimize=True)
            mv_prod_full += einsum("Lqr,Lpr->pq", Lpq[:, :nocc, :nocc], Lpq_z, optimize=True) * pm
            mv_prod_full[numpy.diag_indices(nocc)] *= 1.0 / numpy.sqrt(2)
            mv_prod[ivec] = mv_prod_full[tri_row_o, tri_col_o]

        orb_sum_oo = numpy.asarray(mo_energy[None, :nocc] + mo_energy[:nocc, None]) - 2.0 * pprpa.mu
        orb_sum_oo = orb_sum_oo[tri_row_o, tri_col_o]
        for ivec in range(ntri):
            oz_oo = -orb_sum_oo * tri_vec[ivec]
            mv_prod[ivec] += oz_oo
    else:
        for ivec in range(ntri):
            z_oo = numpy.zeros(shape=[nocc, nocc], dtype=numpy.double)
            z_oo[tri_row_o, tri_col_o] = tri_vec[ivec][:pprpa.oo_dim]
            z_oo[numpy.diag_indices(nocc)] *= 1.0 / numpy.sqrt(2)
            z_vv = numpy.zeros(shape=[nvir, nvir], dtype=numpy.double)
            z_vv[tri_row_v, tri_col_v] = tri_vec[ivec][pprpa.oo_dim:]
            z_vv[numpy.diag_indices(nvir)] *= 1.0 / numpy.sqrt(2)

            # Lpqz_{L,pr} = \sum_s Lpq_{L,ps} z_{rs}
            Lpq_z = numpy.zeros(shape=[naux, nmo, nmo], dtype=numpy.double)
            Lpq_z[:, :nocc, :nocc] = einsum("Lps,rs->Lpr", Lpq[:, :nocc, :nocc], z_oo, optimize=True)
            Lpq_z[:, nocc:, :nocc] = einsum("Lps,rs->Lpr", Lpq[:, nocc:, :nocc], z_oo, optimize=True)
            Lpq_z[:, :nocc, nocc:] = einsum("Lps,rs->Lpr", Lpq[:, :nocc, nocc:], z_vv, optimize=True)
            Lpq_z[:, nocc:, nocc:] = einsum("Lps,rs->Lpr", Lpq[:, nocc:, nocc:], z_vv, optimize=True)

            # MV_{pq} = \sum_{Lr} Lpq_{L,pr} Lpqz_{L,qr} \pm Lpq_{L,qr} Lpqz_{L,pr}
            mv_prod_full = numpy.zeros(shape=[nmo, nmo], dtype=numpy.double)
            mv_prod_full[:nocc, :nocc] = einsum("Lpr,Lqr->pq", Lpq[:, :nocc], Lpq_z[:, :nocc], optimize=True)
            mv_prod_full[:nocc, :nocc] += einsum("Lqr,Lpr->pq", Lpq[:, :nocc], Lpq_z[:, :nocc], optimize=True) * pm
            mv_prod_full[nocc:, nocc:] = einsum("Lpr,Lqr->pq", Lpq[:, nocc:], Lpq_z[:, nocc:], optimize=True)
            mv_prod_full[nocc:, nocc:] += einsum("Lqr,Lpr->pq", Lpq[:, nocc:], Lpq_z[:, nocc:], optimize=True) * pm
            mv_prod_full[numpy.diag_indices(nmo)] *= 1.0 / numpy.sqrt(2)
            mv_prod[ivec][:pprpa.oo_dim] = mv_prod_full[:nocc, :nocc][tri_row_o, tri_col_o]
            mv_prod[ivec][pprpa.oo_dim:] = mv_prod_full[nocc:, nocc:][tri_row_v, tri_col_v]

        orb_sum_oo = numpy.asarray(mo_energy[None, :nocc] + mo_energy[:nocc, None]) - 2.0 * pprpa.mu
        orb_sum_oo = orb_sum_oo[tri_row_o, tri_col_o]
        orb_sum_vv = numpy.asarray(mo_energy[None, nocc:] + mo_energy[nocc:, None]) - 2.0 * pprpa.mu
        orb_sum_vv = orb_sum_vv[tri_row_v, tri_col_v]
        for ivec in range(ntri):
            oz_oo = -orb_sum_oo * tri_vec[ivec][:pprpa.oo_dim]
            mv_prod[ivec][:pprpa.oo_dim] += oz_oo
            oz_vv = orb_sum_vv * tri_vec[ivec][pprpa.oo_dim:]
            mv_prod[ivec][pprpa.oo_dim:] += oz_vv

    return mv_prod


def _pprpa_expand_space(pprpa, first_state, tri_vec, tri_vec_sig, mv_prod, v_tri):
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

    tri_row_o, tri_col_o = numpy.tril_indices(nocc, is_singlet-1)
    tri_row_v, tri_col_v = numpy.tril_indices(nvir, is_singlet-1)

    # take only nRoot vectors, starting from first pp channel
    tmp = v_tri[first_state:(first_state+nroot)]

    # get the eigenvectors in the original space
    ntri = v_tri.shape[0]
    pprpa.xy = numpy.dot(tmp, tri_vec[:ntri])

    # compute residue vectors
    residue = numpy.dot(tmp, mv_prod[:ntri])
    for i in range(nroot):
        residue[i][:pprpa.oo_dim] -= -exci[i] * pprpa.xy[i][:pprpa.oo_dim]
        residue[i][pprpa.oo_dim:] += -exci[i] * pprpa.xy[i][pprpa.oo_dim:]

    # check convergence
    conv_record = numpy.zeros(shape=[nroot], dtype=bool)
    max_residue = 0
    for i in range(nroot):
        max_residue = max(max_residue, abs(numpy.max(residue[i])))
        conv_record[i] = True if len(residue[i][abs(residue[i]) > residue_thresh]) == 0 else False
    nconv = len(conv_record[conv_record is True])
    print("max residue = %.6e" % max_residue)
    if nconv == nroot:
        return True, ntri

    orb_sum_oo = numpy.asarray(mo_energy[None, :nocc] + mo_energy[:nocc, None]) - 2.0 * pprpa.mu
    orb_sum_oo = orb_sum_oo[tri_row_o, tri_col_o]
    orb_sum_vv = numpy.asarray(mo_energy[None, nocc:] + mo_energy[nocc:, None]) - 2.0 * pprpa.mu
    orb_sum_vv = orb_sum_vv[tri_row_v, tri_col_v]

    # Schmidt orthogonalization
    ntri_old = ntri
    for iroot in range(nroot):
        if conv_record[iroot] is True:
            continue

        # convert residuals
        #if pprpa.TDA == "pp":
        #    residue[iroot][pprpa.oo_dim:] /= (exci[iroot] - orb_sum_vv)
        #elif pprpa.TDA == "hh":
        #    residue[iroot][:pprpa.oo_dim] /= -(exci[iroot] - orb_sum_oo)
        #else:
        residue[iroot][:pprpa.oo_dim] /= -(exci[iroot] - orb_sum_oo)
        residue[iroot][pprpa.oo_dim:] /= (exci[iroot] - orb_sum_vv)

        for ivec in range(ntri):
            # compute product between new vector and old vector
            inp = -inner_product(residue[iroot], tri_vec[ivec], pprpa.oo_dim)
            # eliminate parallel part
            if tri_vec_sig[ivec] < 0:
                inp = -inp
            residue[iroot] += inp * tri_vec[ivec]

        # add a new trial vector
        if len(residue[iroot][abs(residue[iroot]) > residue_thresh]) > 0:
            assert ntri < max_vec, ("ppRPA Davidson expansion failed! ntri %d exceeds max_vec %d!" % (ntri, max_vec))
            inp = inner_product(residue[iroot], residue[iroot], pprpa.oo_dim)
            tri_vec_sig[ntri] = 1 if inp > 0 else -1
            tri_vec[ntri] = residue[iroot] / numpy.sqrt(abs(inp))
            ntri = ntri + 1

    conv = True if ntri_old == ntri else False
    return conv, ntri


def pprpa_orthonormalize_eigenvector(multi, nocc, exci, xy):
    """Orthonormalize ppRPA eigenvector.
    The eigenvector is normalized as Y^2 - X^2 = 1.
    This function will rewrite input exci and xy, after calling this function,
    exci and xy will be re-ordered as [hole-hole, particle-particle].

    Args:
        multi (string): multiplicity.
        nocc (int): number of occupied orbitals.
        exci (double array): ppRPA eigenvalue.
        xy (double ndarray): ppRPA eigenvector.
    """
    nroot = xy.shape[0]

    if multi == "s":
        oo_dim = int((nocc + 1) * nocc / 2)
    elif multi == "t":
        oo_dim = int((nocc - 1) * nocc / 2)

    # determine the vector is pp or hh
    sig = numpy.zeros(shape=[nroot], dtype=numpy.double)
    for i in range(nroot):
        sig[i] = 1 if inner_product(xy[i], xy[i], oo_dim) > 0 else -1

    # eliminate parallel component
    for i in range(nroot):
        for j in range(i):
            if abs(exci[i] - exci[j]) < 1.0e-7:
                inp = inner_product(xy[i], xy[j], oo_dim)
                xy[i] -= sig[j] * xy[j] * inp

    # normalize
    for i in range(nroot):
        inp = inner_product(xy[i], xy[i], oo_dim)
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

    return


# analysis functions
def _pprpa_print_eigenvector(multi, nocc, nvir, thresh, channel, exci0, exci, xy):
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

    tri_row_o, tri_col_o = numpy.tril_indices(nocc, is_singlet-1)
    tri_row_v, tri_col_v = numpy.tril_indices(nvir, is_singlet-1)

    nroot = len(exci)
    au2ev = 27.211386
    if channel == "pp":
        for iroot in range(nroot):
            print("#%-d %s excitation:  exci= %-12.4f  eV   2e=  %-12.4f  eV" %
                  (iroot + 1, multi, (exci[iroot] - exci0) * au2ev, exci[iroot] * au2ev))
            if nocc > 0:
                full = numpy.zeros(shape=[nocc, nocc], dtype=numpy.double)
                full[tri_row_o, tri_col_o] = xy[iroot][:oo_dim]
                full = numpy.power(full, 2)
                pairs = numpy.argwhere(full > thresh)
                for i, j in pairs:
                    pprpa_print_a_pair(is_pp=False, p=i, q=j, percentage=full[i, j])

            full = numpy.zeros(shape=[nvir, nvir], dtype=numpy.double)
            full[tri_row_v, tri_col_v] = xy[iroot][oo_dim:]
            full = numpy.power(full, 2)
            pairs = numpy.argwhere(full > thresh)
            for a, b in pairs:
                pprpa_print_a_pair(is_pp=True, p=a+nocc, q=b+nocc, percentage=full[a, b])
            print("")
    else:
        for iroot in range(nroot):
            print("#%-d %s de-excitation:  exci= %-12.4f  eV   2e=  %-12.4f  eV" %
                  (iroot + 1, multi, (exci[iroot] - exci0) * au2ev, exci[iroot] * au2ev))
            full = numpy.zeros(shape=[nocc, nocc], dtype=numpy.double)
            full[tri_row_o, tri_col_o] = xy[iroot][:oo_dim]
            full = numpy.power(full, 2)
            pairs = numpy.argwhere(full > thresh)
            for i, j in pairs:
                pprpa_print_a_pair(is_pp=False, p=i, q=j, percentage=full[i, j])

            if nvir > 0:
                full = numpy.zeros(shape=[nvir, nvir], dtype=numpy.double)
                full[tri_row_v, tri_col_v] = xy[iroot][oo_dim:]
                full = numpy.power(full, 2)
                pairs = numpy.argwhere(full > thresh)
                for a, b in pairs:
                    pprpa_print_a_pair(is_pp=True, p=a+nocc, q=b+nocc, percentage=full[a, b])
            print("")

    return


def pprpa_print_a_pair(is_pp, p, q, percentage):
    """Print the percentage of a pair in the eigenvector.

    Args:
        is_pp (bool): the eigenvector is in particle-particle channel.
        p (int): MO index of the first orbital.
        q (int): MO index of the second orbital.
        percentage (double): the percentage of this pair.
    """
    if is_pp:
      print("    particle-particle pair: %5d %5d   %5.2f%%" % (p + 1, q + 1, percentage * 100))
    else:
      print("    hole-hole pair:         %5d %5d   %5.2f%%" % (p + 1, q + 1, percentage * 100))
    return


def _analyze_pprpa_davidson(exci_s, xy_s, exci_t, xy_t, nocc, nvir, print_thresh=0.1, channel="pp"):
    print("\nanalyze ppRPA results.")

    if exci_s is not None and exci_t is not None:
        print("both singlet and triplet results found.")
        exci0 = min(exci_s[0], exci_t[0]) if channel == "pp" else max(exci_s[0], exci_t[0])
        _pprpa_print_eigenvector(multi="s", nocc=nocc, nvir=nvir, thresh=print_thresh, channel=channel,
                                 exci0=exci0, exci=exci_s, xy=xy_s)
        _pprpa_print_eigenvector(multi="t", nocc=nocc, nvir=nvir, thresh=print_thresh, channel=channel,
                                 exci0=exci0, exci=exci_t, xy=xy_t)
    else:
        if exci_s is not None:
            print("only singlet results found.")
            _pprpa_print_eigenvector(multi="s", nocc=nocc, nvir=nvir, thresh=print_thresh, channel=channel,
                                     exci0=exci_s[0], exci=exci_s, xy=xy_s)
        else:
            print("only triplet results found.")
            _pprpa_print_eigenvector(multi="t", nocc=nocc, nvir=nvir, thresh=print_thresh, channel=channel,
                                     exci0=exci_t[0], exci=exci_t, xy=xy_t)
    return


class ppRPA_Davidson():
    def __init__(self, nocc, mo_energy, Lpq, channel="pp", nroot=5, max_vec=200, max_iter=100, residue_thresh=1.0e-7,
                 print_thresh=0.1):
        # necessary input
        self.nocc = nocc  # number of occupied orbitals
        self.mo_energy = numpy.asarray(mo_energy)  # orbital energy
        self.Lpq = numpy.asarray(Lpq)  # three-center density-fitting matrix in MO space

        # options
        self.channel = channel  # channel of desired states, particle-particle or hole-hole
        self.nroot = nroot  # number of desired roots
        self.max_vec = max_vec  # max size of trial vectors
        self.max_iter = max_iter  # max iteration
        self.residue_thresh = residue_thresh  # residue threshold
        self.print_thresh = print_thresh  #  threshold to print component

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
        assert self.Lpq.shape[1] == self.Lpq.shape[2] == self.nmo

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
            self.mu = get_chemical_potential(nocc=self.nocc, mo_energy=self.mo_energy)

        return

    def dump_flags(self):
        print('\n******** %s ********' % self.__class__)
        print('multiplicity = %s' % ("singlet" if self.multi == "s" else "triplet"))
        print('state channel = %s' % self.channel)
        print('naux = %d' % self.naux)
        print('nmo = %d' % self.nmo)
        print('nocc = %d nvir = %d' % (self.nocc, self.nvir))
        print('occ-occ dimension = %d vir-vir dimension = %d' % (self.oo_dim, self.vv_dim))
        print('full dimension = %d' % self.full_dim)
        print('number of roots = %d' % self.nroot)
        print('max subspace size = %d' % self.max_vec)
        print('max iteration = %d' % self.max_iter)
        print('residue threshold = %.3e' % self.residue_thresh)
        print('print threshold = %.2f%%' % (self.print_thresh*100))
        print('')
        return

    def check_memory(self):
        # intermediate in contraction; mv_prod, tri_vec, xy
        mem = (self.naux * self.nmo * self.nmo + 3 * self.max_vec * self.full_dim) * 8 / 1.0e6
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

    def analyze(self):
        _analyze_pprpa_davidson(exci_s=self.exci_s, xy_s=self.xy_s, exci_t=self.exci_t, xy_t=self.xy_t, nocc=self.nocc,
                                nvir=self.nvir, print_thresh=self.print_thresh, channel=self.channel)
        return
