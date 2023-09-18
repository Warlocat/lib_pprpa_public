import numpy
import scipy

from numpy import einsum


def kernel(pprpa):
    # local variables for convenience
    nocc = pprpa.nocc
    mo_energy = pprpa.mo_energy
    nroot = pprpa.nroot
    max_vec = pprpa.max_vec
    max_iter = pprpa.max_iter
    TDA = pprpa.TDA

    # determine dimension
    nmo = len(mo_energy)
    nvir = nmo - nocc
    if pprpa.multi == "s":
        oo_dim = int((nocc + 1) * nocc / 2)
        vv_dim = int((nvir + 1) * nvir / 2)
    elif pprpa.multi == "t":
        oo_dim = int((nocc - 1) * nocc / 2)
        vv_dim = int((nvir - 1) * nvir / 2)
    if TDA is True:
        oo_dim = 0
    full_dim = oo_dim + vv_dim

    # initialize trial vector and product matrix
    tri_vec = numpy.zeros(shape=[max_vec, full_dim], dtype=numpy.double)
    tri_vec_sig = numpy.zeros(shape=[max_vec], dtype=numpy.double)
    ntri = min(nroot * 4, vv_dim)  # initial guess size should be larger than nroot
    tri_vec[:ntri], tri_vec_sig[:ntri] = _pprpa_get_trial_vector(pprpa=pprpa, ntri=ntri)
    mv_prod = numpy.zeros_like(tri_vec)

    iter = 0
    nprod = 0 # number of contracted vectors
    while iter < max_iter:
        print("\nppRPA Davidson #%d iteration, ntri= %d , nprod= %d ." % (iter+1, ntri, nprod))
        mv_prod[nprod:ntri] = _pprpa_contraction(pprpa=pprpa, tri_vec=tri_vec[nprod:ntri])
        nprod = ntri

        m_tilde = numpy.dot(tri_vec[:ntri], mv_prod[:ntri].T)
        w_tilde = numpy.zeros_like(m_tilde)
        for i in range(ntri):
            w_tilde[i, i] = 1 if inner_product(tri_vec[i], tri_vec[i], oo_dim) > 0 else -1

        alphar, _, beta, _, v_tri, _, _ = scipy.linalg.lapack.dggev(m_tilde, w_tilde, compute_vl=0)
        e_tri = alphar / beta
        v_tri = v_tri.T  # Fortran matrix to Python order

        # sort eigenvalues and eigenvectors by ascending order
        v_tri = numpy.asarray(list(x for _, x in sorted(zip(e_tri, v_tri), reverse=False)))
        e_tri = numpy.sort(e_tri)

        # get first pp state by the sign of the eigenvector, not by the sign of the excitation energy
        for i in range(ntri):
            sum = numpy.sum((v_tri[i] ** 2) * tri_vec_sig[:ntri])
            if sum > 0:
                first_pp = i
                break

        # get only two-electron addition energy
        pprpa.exci = e_tri[first_pp:(first_pp+nroot)]

        ntri_old = ntri
        conv, ntri = _pprpa_expand_space(pprpa=pprpa, first_pp=first_pp, tri_vec=tri_vec, tri_vec_sig=tri_vec_sig,
                                         mv_prod=mv_prod, v_tri=v_tri)
        print("add %d new trial vectors." % (ntri - ntri_old))

        iter += 1
        if conv is True:
            break

    assert conv is True, "ppRPA Davidson algorithm is not converged!"
    print("ppRPA Davidson converged in %d iterations, final subspace size = %d" % (iter, nprod))

    pprpa_orthonormalize_eigenvector(multi=pprpa.multi, nocc=nocc, TDA=pprpa.TDA, exci=pprpa.exci, xy=pprpa.xy)

    return


# utility functions
def ij2index(r, c, row, col):
    """Get index of a row and column in a square matrix in a lower triangular matrix.

    Args:
        r (int): row index in s square matrix.
        c (int): column index in s square matrix.
        row (int array): row index array of a lower triangular matrix.
        col (int array): column index array of a lower triangular matrix.

    Returns:
        i (int): index in the lower triangular matrix.
    """
    for i in range(len(row)):
        if r == row[i] and c == col[i]:
            return i

    raise ValueError("cannot find the index!")


def inner_product(u, v, oo_dim):
    """Calculate ppRPA inner product.
    product = <Y1,Y2> - <X1,X2>, where X is occ-occ block, Y is vir-vir block.

    Args:
        u (double array): first vector.
        v (double array): second vector
        oo_dim (int): occ-occ block dimension

    Returns:
        inp (double): inner product.
    """
    inp = -numpy.sum(u[:oo_dim] * v[:oo_dim]) + numpy.sum(u[oo_dim:] * v[oo_dim:])
    return inp


# Davidson algorithm functions
def _pprpa_get_trial_vector(pprpa, ntri):
    """Generate initial trial vectors for particle-particle excitations.
    The order is determined by the vir-vir pair orbital energy summation.
    The initial trial vectors are diagonal and signatures are all 1.

    Args:
        pprpa (ppRPA_Davidson): ppRPA_Davidson object.
        ntri (int): the number of initial trial vectors.

    Returns:
        tri_vec (double ndarray): initial trial vectors.
        tri_vec_sig (double ndarray): signature of initial trial vectors.
    """
    nocc = pprpa.nocc
    mo_energy = pprpa.mo_energy
    TDA = pprpa.TDA

    nmo = len(mo_energy)
    nvir = nmo - nocc
    if pprpa.multi == "s":
        oo_dim = int((nocc + 1) * nocc / 2)
        vv_dim = int((nvir + 1) * nvir / 2)
        is_singlet = 1
    elif pprpa.multi == "t":
        oo_dim = int((nocc - 1) * nocc / 2)
        vv_dim = int((nvir - 1) * nvir / 2)
        is_singlet = 0
    if TDA is True:
        oo_dim = 0
    full_dim = oo_dim + vv_dim

    max_orb_sum = 1.0e15

    class pp_pair():
        def __init__(self):
            self.a = -1
            self.b = -1
            self.eig_sum = max_orb_sum

    pairs = []
    for r in range(ntri):
        t = pp_pair()
        pairs.append(t)

    # find particle-particle pairs with lowest orbital energy summation
    for r in range(ntri):
        for a in range(nocc, nmo):
            for b in range(nocc, a + is_singlet):
                valid = True
                for rr in range(r):
                    if pairs[rr].a == a and pairs[rr].b == b:
                        valid = False
                        break
                if valid is True and (mo_energy[a] + mo_energy[b]) < pairs[r].eig_sum:
                    pairs[r].a, pairs[r].b = a, b
                    pairs[r].eig_sum = mo_energy[a] + mo_energy[b]

    # sort pairs by ascending energy order
    for i in range(ntri-1):
        for j in range(i+1, ntri):
            if pairs[i].eig_sum > pairs[j].eig_sum:
                a_tmp, b_tmp, eig_sum_tmp = pairs[i].a, pairs[i].b, pairs[i].eig_sum
                pairs[i].a, pairs[i].b, pairs[i].eig_sum = pairs[j].a, pairs[j].b, pairs[j].eig_sum
                pairs[j].a, pairs[j].b, pairs[j].eig_sum = a_tmp, b_tmp, eig_sum_tmp

    assert pairs[ntri-1].eig_sum < max_orb_sum, "cannot find enough pairs for trial vectors"

    tri_vec = numpy.zeros(shape=[ntri, full_dim], dtype=numpy.double)
    tri_vec_sig = numpy.zeros(shape=[ntri], dtype=numpy.double)
    tri_row_v, tri_col_v = numpy.tril_indices(nvir, is_singlet-1)
    for r in range(ntri):
        a, b = pairs[r].a, pairs[r].b
        ab = ij2index(a - nocc, b - nocc, tri_row_v, tri_col_v)
        tri_vec[r, oo_dim + ab] = 1.0
        tri_vec_sig[r] = 1.0

    return tri_vec, tri_vec_sig


def _pprpa_contraction(pprpa, tri_vec):
    """ppRPA contraction.

    Args:
        pprpa (ppRPA_Davidson): ppRPA_Davidson object.
        tri_vec (double ndarray): trial vector.

    Returns:
        mv_prod (double ndarray): product between ppRPA matrix and trial vectors.
    """
    nocc = pprpa.nocc
    mo_energy = pprpa.mo_energy
    Lpq = pprpa.Lpq
    TDA = pprpa.TDA

    naux, nmo, _ = Lpq.shape
    nvir = nmo - nocc
    ntri = tri_vec.shape[0]

    mu = (mo_energy[nocc-1] + mo_energy[nocc]) * 0.5

    if pprpa.multi == "s":
        oo_dim = int((nocc + 1) * nocc / 2)
        vv_dim = int((nvir + 1) * nvir / 2)
        is_singlet = 1
        pm = 1.0
    elif pprpa.multi == "t":
        oo_dim = int((nocc - 1) * nocc / 2)
        vv_dim = int((nvir - 1) * nvir / 2)
        is_singlet = 0
        pm = -1.0
    if TDA is True:
        oo_dim = 0
    full_dim = oo_dim + vv_dim

    mv_prod = numpy.zeros(shape=[ntri, full_dim], dtype=numpy.double)

    tri_row_o, tri_col_o = numpy.tril_indices(nocc, is_singlet-1)
    tri_row_v, tri_col_v = numpy.tril_indices(nvir, is_singlet-1)

    for ivec in range(ntri):
        if TDA is False:
            z_oo = numpy.zeros(shape=[nocc, nocc], dtype=numpy.double)
            z_oo[tri_row_o, tri_col_o] = tri_vec[ivec][:oo_dim]
            z_oo[numpy.diag_indices(nocc)] *= 1.0 / numpy.sqrt(2)
        z_vv = numpy.zeros(shape=[nvir, nvir], dtype=numpy.double)
        z_vv[tri_row_v, tri_col_v] = tri_vec[ivec][oo_dim:]
        z_vv[numpy.diag_indices(nvir)] *= 1.0 / numpy.sqrt(2)
        Lpq_z = numpy.zeros(shape=[naux, nmo, nmo], dtype=numpy.double)

        # Lpqz_{L,pr} = \sum_s Lpq_{L,ps} z_{rs}
        if TDA is False:
            Lpq_z[:, :nocc, :nocc] = einsum("Lps,rs->Lpr", Lpq[:, :nocc, :nocc], z_oo, optimize=True)
            Lpq_z[:, nocc:, :nocc] = einsum("Lps,rs->Lpr", Lpq[:, nocc:, :nocc], z_oo, optimize=True)
        Lpq_z[:, :nocc, nocc:] = einsum("Lps,rs->Lpr", Lpq[:, :nocc, nocc:], z_vv, optimize=True)
        Lpq_z[:, nocc:, nocc:] = einsum("Lps,rs->Lpr", Lpq[:, nocc:, nocc:], z_vv, optimize=True)

        # MV_{pq} = \sum_{Lr} Lpq_{L,pr} Lpqz_{L,qr} \pm Lpq_{L,qr} Lpqz_{L,pr}
        mv_prod_full = numpy.zeros(shape=[nmo, nmo], dtype=numpy.double)
        if TDA is False:
            mv_prod_full[:nocc, :nocc] = einsum("Lpr,Lqr->pq", Lpq[:, :nocc, :], Lpq_z[:, :nocc, :], optimize=True)
            mv_prod_full[:nocc, :nocc] += einsum("Lqr,Lpr->pq", Lpq[:, :nocc, :],
                                                 Lpq_z[:, :nocc, :], optimize=True) * pm
        mv_prod_full[nocc:, nocc:] = einsum("Lpr,Lqr->pq", Lpq[:, nocc:, :], Lpq_z[:, nocc:, :], optimize=True)
        mv_prod_full[nocc:, nocc:] += einsum("Lqr,Lpr->pq", Lpq[:, nocc:, :], Lpq_z[:, nocc:, :], optimize=True) * pm

        mv_prod_full[numpy.diag_indices(nmo)] *= 1.0 / numpy.sqrt(2)
        if TDA is False:
            mv_prod[ivec][:oo_dim] = mv_prod_full[:nocc, :nocc][tri_row_o, tri_col_o]
        mv_prod[ivec][oo_dim:] = mv_prod_full[nocc:, nocc:][tri_row_v, tri_col_v]

    orb_sum_oo = numpy.asarray(mo_energy[None, :nocc] + mo_energy[:nocc, None]) - 2.0 * mu
    orb_sum_oo = orb_sum_oo[tri_row_o, tri_col_o]
    orb_sum_vv = numpy.asarray(mo_energy[None, nocc:] + mo_energy[nocc:, None]) - 2.0 * mu
    orb_sum_vv = orb_sum_vv[tri_row_v, tri_col_v]
    for ivec in range(ntri):
        if TDA is False:
            oz_oo = -orb_sum_oo * tri_vec[ivec][:oo_dim]
            mv_prod[ivec][:oo_dim] += oz_oo
        oz_vv = orb_sum_vv * tri_vec[ivec][oo_dim:]
        mv_prod[ivec][oo_dim:] += oz_vv

    return mv_prod


def _pprpa_expand_space(pprpa, first_pp, tri_vec, tri_vec_sig, mv_prod, v_tri):
    """Expand trial vector space in Davidson algorithm.

    Args:
        pprpa (ppRPA_Davidson): ppRPA_Davidson object.
        first_pp (int): index of first particle-particle state.
        tri_vec (double ndarray): trial vector.
        tri_vec_sig (int array): signature of trial vector.
        mv_prod (double ndarray): product matrix of ppRPA matrix and trial vector.
        v_tri (double ndarray): eigenvector of subspace matrix.

    Returns:
        conv (bool): if Davidson algorithm is converged.
        ntri (int): updated number of trial vectors.
    """
    nocc = pprpa.nocc
    mo_energy = pprpa.mo_energy
    nroot = pprpa.nroot
    exci = pprpa.exci
    max_vec = pprpa.max_vec
    residue_thresh = pprpa.residue_thresh
    TDA = pprpa.TDA

    nmo = len(mo_energy)
    nvir = nmo - nocc
    ntri = v_tri.shape[0]
    if pprpa.multi == "s":
        oo_dim = int((nocc + 1) * nocc / 2)
        is_singlet = 1
    elif pprpa.multi == "t":
        oo_dim = int((nocc - 1) * nocc / 2)
        is_singlet = 0
    if TDA is True:
        oo_dim = 0
    mu = (mo_energy[nocc-1] + mo_energy[nocc]) * 0.5

    tri_row_o, tri_col_o = numpy.tril_indices(nocc, is_singlet-1)
    tri_row_v, tri_col_v = numpy.tril_indices(nvir, is_singlet-1)

    # take only nRoot vectors, starting from first pp channel
    tmp = v_tri[first_pp:(first_pp+nroot)]

    # get the eigenvectors in the original space
    pprpa.xy = numpy.dot(tmp, tri_vec[:ntri])

    # compute residue vectors
    residue = numpy.dot(tmp, mv_prod[:ntri])
    for i in range(nroot):
        residue[i][:oo_dim] -= -exci[i] * pprpa.xy[i][:oo_dim]
        residue[i][oo_dim:] += -exci[i] * pprpa.xy[i][oo_dim:]

    # check convergence
    conv_record = numpy.zeros(shape=[nroot], dtype=bool)
    for i in range(nroot):
        conv_record[i] = True if len(residue[i][abs(residue[i]) > residue_thresh]) == 0 else False
    nconv = len(conv_record[conv_record is True])
    if nconv == nroot:
        return True, ntri

    orb_sum_oo = numpy.asarray(mo_energy[None, :nocc] + mo_energy[:nocc, None]) - 2.0 * mu
    orb_sum_oo = orb_sum_oo[tri_row_o, tri_col_o]
    orb_sum_vv = numpy.asarray(mo_energy[None, nocc:] + mo_energy[nocc:, None]) - 2.0 * mu
    orb_sum_vv = orb_sum_vv[tri_row_v, tri_col_v]

    # Schmidt orthogonalization
    ntri_old = ntri
    for iroot in range(nroot):
        if conv_record[iroot] is True:
            continue

        # convert residuals
        if TDA is False:
            residue[iroot][:oo_dim] /= -(exci[iroot] - orb_sum_oo)
        residue[iroot][oo_dim:] /= (exci[iroot] - orb_sum_vv)

        for ivec in range(ntri):
            # compute product between new vector and old vector
            inp = -inner_product(residue[iroot], tri_vec[ivec], oo_dim)
            # eliminate parallel part
            if tri_vec_sig[ivec] < 0:
                inp = -inp
            residue[iroot] += inp * tri_vec[ivec]

            # add a new trial vector
        if len(residue[iroot][abs(residue[iroot]) > residue_thresh]) > 0:
            assert ntri < max_vec, ("ppRPA Davidson expansion failed! ntri %d exceeds max_vec %d!" % (ntri, max_vec))
            inp = inner_product(residue[iroot], residue[iroot], oo_dim)
            tri_vec_sig[ntri] = 1 if inp > 0 else -1
            tri_vec[ntri] = residue[iroot] / numpy.sqrt(abs(inp))
            ntri = ntri + 1

    conv = True if ntri_old == ntri else False
    return conv, ntri


def pprpa_orthonormalize_eigenvector(multi, nocc, TDA, exci, xy):
    """Orthonormalize ppRPA eigenvector.
    The eigenvector is normalized as Y^2 - X^2 = 1.

    Args:
        multi (string): multiplicity.
        nocc (int): number of occupied orbitals.
        TDA (bool): use TDA.
        exci (double array): ppRPA eigenvalue.
        xy (double ndarray): ppRPA eigenvector.
    """
    nroot = xy.shape[0]

    if multi == "s":
        oo_dim = int((nocc + 1) * nocc / 2)
    elif multi == "t":
        oo_dim = int((nocc - 1) * nocc / 2)
    if TDA is True:
        oo_dim = 0

    # determine the vector is pp or hh
    sig = numpy.zeros(shape=[nroot], dtype=numpy.double)
    for i in range(nroot):
        sig[i] = 1 if inner_product(xy[i], xy[i], oo_dim) > 0 else -1

    # eliminate parallel component
    for i in range(nroot):
        for j in range(i):
            if abs(exci[i] - exci[j]) < 1.0e-7:
                inp = inner_product(xy[i], xy[j], oo_dim)
                xy[i] -= sig[i] * xy[j] * inp

    # normalize
    for i in range(nroot):
        inp = inner_product(xy[i], xy[i], oo_dim)
        inp = numpy.sqrt(abs(inp))
        xy[i] /= inp

    # change |X -Y> to |X Y>
    xy[:][:oo_dim] *= -1

    return


# analysis functions
def _pprpa_print_eigenvector(multi, nocc, nvir, thresh, TDA, exci0, exci, xy):
    """Print dominant components of an eigenvector.

    Args:
        multi (string): multiplicity.
        nocc (int): number of occupied orbitals.
        nvir (int): number of virtual orbitals.
        thresh (double): threshold to print a pair.
        TDA (bool): if TDA is used.
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
    if TDA is True:
        oo_dim = 0
    nmo = nocc + nvir

    tri_row_o, tri_col_o = numpy.tril_indices(nocc, is_singlet-1)
    tri_row_v, tri_col_v = numpy.tril_indices(nvir, is_singlet-1)

    nroot = len(exci)
    au2ev = 27.211386
    for iroot in range(nroot):
        print("#%-d %s excitation:  exci= %-12.4f  eV   2e=  %-12.4f  eV" %
              (iroot + 1, multi, (exci[iroot] - exci0) * au2ev, exci[iroot] * au2ev))
        if TDA is False:
            for i in range(nocc):
                for j in range(i + is_singlet):
                    ij = ij2index(i, j, tri_row_o, tri_col_o)
                    percentage = numpy.power(xy[iroot][ij], 2)
                    if percentage > thresh:
                        pprpa_print_a_pair(is_pp=False, p=i, q=j, percentage=percentage)

        for a in range(nocc, nmo):
            for b in range(nocc, a + is_singlet):
                ab = ij2index(a - nocc, b - nocc, tri_row_v, tri_col_v)
                percentage = numpy.power(xy[iroot][oo_dim + ab], 2)
                if percentage > thresh:
                    pprpa_print_a_pair(is_pp=True, p=a, q=b, percentage=percentage)

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


class ppRPA_Davidson():
    def __init__(self, nocc, mo_energy, Lpq, TDA=False, nroot=5, max_vec=200, max_iter=100,
                 residue_thresh=1.0e-7, print_thresh=0.1):
        # necessary input
        self.nocc = nocc  # number of occupied orbitals
        self.mo_energy = numpy.asarray(mo_energy)  # orbital energy
        self.Lpq = numpy.asarray(Lpq)  # three-center density-fitting matrix in MO space

        # options
        self.multi = None  # multiplicity
        self.TDA = TDA  # Tamm–Dancoff approximation, only use A matrix
        self.nroot = nroot  # number of desired roots
        self.max_vec = max_vec  # max size of trial vectors
        self.max_iter = max_iter  # max iteration
        self.residue_thresh = residue_thresh  # residue threshold
        self.print_thresh = print_thresh  #  threshold to print component

        # results
        self.exci = None  # two-electron addition energy
        self.xy = None  # ppRPA eigenvector
        self.exci_s = None  # singlet two-electron addition energy
        self.xy_s = None  # singlet two-electron addition eigenvector
        self.exci_t = None  # triplet two-electron addition energy
        self.xy_t = None  # triplet two-electron addition eigenvector

        return

    def dump_flags(self):
        print('\n******** %s ********' % self.__class__)
        nocc = self.nocc
        nmo = len(self.mo_energy)
        nvir = nmo - self.nocc
        if self.multi == "s":
            oo_dim = int((nocc + 1) * nocc / 2)
            vv_dim = int((nvir + 1) * nvir / 2)
        elif self.multi == "t":
            oo_dim = int((nocc - 1) * nocc / 2)
            vv_dim = int((nvir - 1) * nvir / 2)
        full_dim = oo_dim + vv_dim
        print('multiplicity = %s' % ("singlet" if self.multi == "s" else "triplet"))
        print('Tamm-Dancoff approximation = %s' % self.TDA)
        print('nmo = %d' % nmo)
        print('nocc = %d' % self.nocc)
        print('nvir = %d' % nvir)
        print('occ-occ dimension = %d' % oo_dim)
        print('vir-vir dimension = %d' % vv_dim)
        print('full dimension = %d' % full_dim)
        print('number of roots = %d' % self.nroot)
        print('max subspace size = %d' % self.max_vec)
        print('max iteration = %d' % self.max_iter)
        print('residue threshold = %.3e' % self.residue_thresh)
        print('print threshold = %.2f' % self.print_thresh)
        print('')
        return

    def check_memory(self):
        nocc = self.nocc
        nmo = len(self.mo_energy)
        nvir = nmo - self.nocc
        naux = self.Lpq.shape[0]
        max_vec = self.max_vec
        if self.multi == "s":
            oo_dim = int((nocc + 1) * nocc / 2)
            vv_dim = int((nvir + 1) * nvir / 2)
        elif self.multi == "t":
            oo_dim = int((nocc - 1) * nocc / 2)
            vv_dim = int((nvir - 1) * nvir / 2)
        full_dim = oo_dim + vv_dim

        # intermediate in contraction; mv_prod, tri_vec
        mem = (naux * nmo * nmo + 2 * max_vec * full_dim) * 64 / 1.0e6
        if mem < 1000:
            print("ppRPA needs at least %d MB memory." % mem)
        else:
            print("ppRPA needs at least %.1f GB memory." % mem / 1.0e3)
        return

    def kernel(self, multi):
        self.multi = multi
        self.dump_flags()
        self.check_memory()
        kernel(pprpa=self)
        if self.multi == "s":
            self.exci_s = self.exci
            self.xy_s = self.xy
        else:
            self.exci_t = self.exci
            self.xy_t = self.xy
        return

    def analyze(self):
        print("\nanalyze ppRPA results.")
        print_thresh = self.print_thresh
        nocc = self.nocc
        nmo = len(self.mo_energy)
        nvir = nmo - nocc
        if self.exci_s is not None and self.exci_t is not None:
            print("both singlet and triplet results found.")
            if self.exci_s[0] < self.exci_t[0]:
                exci0 = self.exci_s[0]
                print("lowest 2e addition energy is singlet: %-12.4f eV\n" % (exci0*27.211386))
            else:
                exci0 = self.exci_s[0]
                print("lowest 2e addition energy is triplet: %-12.4f eV\n" % (exci0*27.211386))

            _pprpa_print_eigenvector(multi="s", nocc=nocc, nvir=nvir, thresh=print_thresh, TDA=self.TDA,
                                     exci0=exci0, exci=self.exci_s, xy=self.xy_s)
            _pprpa_print_eigenvector(multi="t", nocc=nocc, nvir=nvir, thresh=print_thresh, TDA=self.TDA,
                                     exci0=exci0, exci=self.exci_t, xy=self.xy_t)
        else:
            if self.exci_s is not None:
                print("only singlet results found.")
                _pprpa_print_eigenvector(multi="s", nocc=nocc, nvir=nvir, thresh=print_thresh, TDA=self.TDA,
                                         exci0=self.exci_s[0], exci=self.exci_s, xy=self.xy_s)
            else:
                print("only triplet results found.")
                _pprpa_print_eigenvector(multi="t", nocc=nocc, nvir=nvir, thresh=print_thresh, TDA=self.TDA,
                                         exci0=self.exci_t[0], exci=self.exci_t, xy=self.xy_t)
        return
