
from functools import reduce
import numpy
from numpy import einsum


def get_pprpa_nto(multi, state, xy, nocc, nvir, mo_coeff, nocc_full):
    """Get ppRPA natural transition orbital coefficient and weight.

    Args:
        multi (char): multiplicity.
        state (int): index of the desired state.
        xy (double ndarray): ppRPA eigenvector.
        nocc (int or int array): number of (active) occupied orbitals.
        nvir (int or int array): number of (active) virtual orbitals.
        mo_coeff (double ndarray): coefficient from AO to MO.
        nocc_full (int or int array): number of occupied orbitals of the full system.

    Returns:
        weight_o (double array): weight of occupied NTOs.
        nto_coeff_o1 (double ndarray): coefficient from AO to the first hole orbital in occupied NTOs.
        nto_coeff_o2 (double ndarray): coefficient from AO to the second hole orbital in occupied NTOs.
        weight_v (double array): weight of virtual NTOs.
        nto_coeff_v1 (double ndarray): coefficient from AO to the first particle orbital in virtual NTOs.
        nto_coeff_v2 (double ndarray): coefficient from AO to the second particle orbital in virtual NTOs.
    """
    print("get NTO for multi=%s state=%d" % (multi, state))
    orbo = mo_coeff[:, nocc_full-nocc:nocc_full]
    orbv = mo_coeff[:, nocc_full:nocc_full+nvir]

    if multi == "s":
        oo_dim = int((nocc + 1) * nocc / 2)
        vv_dim = int((nvir + 1) * nvir / 2)
    elif multi == "t":
        oo_dim = int((nocc - 1) * nocc / 2)
        vv_dim = int((nvir - 1) * nvir / 2)
    assert oo_dim > 0 or vv_dim > 0

    is_singlet = 1 if multi == "s" else 0
    tril_row_o, tril_col_o = numpy.tril_indices(nocc, is_singlet-1)
    tril_row_v, tril_col_v = numpy.tril_indices(nvir, is_singlet-1)

    if oo_dim > 0:
        y_full = numpy.zeros(shape=[nocc, nocc], dtype=numpy.double)
        y_full[tril_row_o, tril_col_o] = xy[state][:oo_dim]

    if vv_dim > 0:
        x_full = numpy.zeros(shape=[nvir, nvir], dtype=numpy.double)
        x_full[tril_row_v, tril_col_v] = xy[state][oo_dim:]

    # do SVD decomposition then get AO->NTO coefficient
    if oo_dim > 0:
        nto_o1, wo, nto_o2T = numpy.linalg.svd(y_full)
        nto_o2 = nto_o2T.conj().T
        weight_o = wo**2
        nto_coeff_o1 = numpy.dot(orbo, nto_o1)
        nto_coeff_o2 = numpy.dot(orbo, nto_o2)

        # analyze hole-hole NTO
        idx_large_wt = numpy.where(weight_o > 0.1)[0]
        if len(idx_large_wt) > 0:
            print("largest hole-hole NTO component")
        for i in range(len(idx_large_wt)):
            idx_o1 = numpy.where(abs(nto_o1[:, i]) > 0.3)[0]
            idx_o2 = numpy.where(abs(nto_o2[:, i]) > 0.3)[0]
            print("weight: %-12.3f" % weight_o[i])
            print("  hole 1:")
            for j in idx_o1:
                print("  orb=%-4d percent=%.2f%%" %
                      (j+1, numpy.square(nto_o1[j, i])*100))
            print("  hole 2:")
            for j in idx_o2:
                print("  orb=%-4d percent=%.2f%%" %
                      (j+1, numpy.square(nto_o2[j, i])*100))

    if vv_dim > 0:
        nto_v1, wv, nto_v2T = numpy.linalg.svd(x_full)
        nto_v2 = nto_v2T.conj().T
        weight_v = wv**2

        idx = numpy.argmax(abs(nto_v1.real), axis=0)
        nto_v1[:, nto_v1[idx, numpy.arange(nvir)].real < 0] *= -1
        idx = numpy.argmax(abs(nto_v2.real), axis=0)
        nto_v2[:, nto_v2[idx, numpy.arange(nvir)].real < 0] *= -1

        nto_coeff_v1 = numpy.dot(orbv, nto_v1)
        nto_coeff_v2 = numpy.dot(orbv, nto_v2)

        # analyze particle-particle NTO
        idx_large_wt = numpy.where(weight_v > 0.1)[0]
        if len(idx_large_wt) > 0:
            print("largest particle-particle NTO component")
        for i in idx_large_wt:
            idx_v1 = numpy.where(abs(nto_v1[:, i]) > 0.3)[0]
            idx_v2 = numpy.where(abs(nto_v2[:, i]) > 0.3)[0]
            print("weight: %-12.3f" % weight_v[i])
            print("  particle 1:")
            for j in idx_v1:
                print("  orb=%-4d percent=%.2f%%" %
                      (nocc+j+1, numpy.square(nto_v1[j, i])*100))
            print("  particle 2:")
            for j in idx_v2:
                print("  orb=%-4d percent=%.2f%%" %
                      (nocc+j+1, numpy.square(nto_v2[j, i])*100))

    print("NTO analysis finished.\n")

    if oo_dim > 0 and vv_dim > 0:
        return weight_o, nto_coeff_o1, nto_coeff_o2, weight_v, nto_coeff_v1, \
            nto_coeff_v2
    elif oo_dim > 0:
        return weight_o, nto_coeff_o1, nto_coeff_o2
    elif vv_dim > 0:
        return weight_v, nto_coeff_v1, nto_coeff_v2


def get_pprpa_dm(multi, state, xy, nocc, nvir, mo_coeff, nocc_full,
                 full_return=False):
    """Get the ppRPA density matrix of the desired state.

    Args:
        multi (char): multiplicity.
        state (int): index of the desired state.
        xy (double ndarray): ppRPA eigenvector.
        nocc (int or int array): number of (active) occupied orbitals.
        nvir (int or int array): number of (active) virtual orbitals.
        mo_coeff (double ndarray): coefficient from AO to MO.
        nocc_full (int or int array): number of occupied orbitals of the full system.
        full_return (bool): return all density matrixes.

    Returns:
        dm (double ndarray): [nspin * nmo_full * nmo_full], density matrix of two spin channels.
        dm1h (double ndarray): density matrix of the first hole.
        dm1p (double ndarray): density matrix of the first particle.
        dm2h (double ndarray): density matrix of the second hole.
        dm2p (double ndarray): density matrix of the second particle.
    """
    print("get ppRPA density matrix for multi=%s state=%d" % (multi, state))
    nmo_full = mo_coeff.shape[0]

    if multi == "s":
        oo_dim = int((nocc + 1) * nocc / 2)
        vv_dim = int((nvir + 1) * nvir / 2)
    elif multi == "t":
        oo_dim = int((nocc - 1) * nocc / 2)
        vv_dim = int((nvir - 1) * nvir / 2)
    assert oo_dim > 0 or vv_dim > 0

    is_singlet = 1 if multi == "s" else 0
    tril_row_o, tril_col_o = numpy.tril_indices(nocc, is_singlet-1)
    tril_row_v, tril_col_v = numpy.tril_indices(nvir, is_singlet-1)

    if oo_dim > 0:
        y_full = numpy.zeros(shape=[nocc, nocc], dtype=numpy.double)
        y_full[tril_row_o, tril_col_o] = xy[state][:oo_dim]

    if vv_dim > 0:
        x_full = numpy.zeros(shape=[nvir, nvir], dtype=numpy.double)
        x_full[tril_row_v, tril_col_v] = xy[state][oo_dim:]

    if oo_dim > 0:
        dm_oo1 = -einsum('ik,jk->ij', y_full, y_full)
        dm_oo2 = -einsum('ki,kj->ij', y_full, y_full)

    if vv_dim > 0:
        dm_vv1 = einsum('ac,bc->ab', x_full, x_full)
        dm_vv2 = einsum('ca,cb->ab', x_full, x_full)

    dm = numpy.zeros(shape=[2, nmo_full, nmo_full], dtype=numpy.double)
    dm[0, :nocc_full, :nocc_full] = numpy.eye(nocc_full)
    dm[1, :nocc_full, :nocc_full] = numpy.eye(nocc_full)
    if multi == "s":
        if oo_dim > 0:
            dm[0, nocc_full-nocc:nocc_full, nocc_full-nocc:nocc_full] += dm_oo1
            dm[1, nocc_full-nocc:nocc_full, nocc_full-nocc:nocc_full] += dm_oo2
        if vv_dim > 0:
            dm[0, nocc_full:nocc_full+nvir, nocc_full:nocc_full+nvir] += dm_vv1
            dm[1, nocc_full:nocc_full+nvir, nocc_full:nocc_full+nvir] += dm_vv2
    else:
        if oo_dim > 0:
            dm[0, nocc_full-nocc:nocc_full, nocc_full-nocc:nocc_full] += dm_oo1
            dm[0, nocc_full-nocc:nocc_full, nocc_full-nocc:nocc_full] += dm_oo2
        if vv_dim > 0:
            dm[0, nocc_full:nocc_full+nvir, nocc_full:nocc_full+nvir] += dm_vv1
            dm[0, nocc_full:nocc_full+nvir, nocc_full:nocc_full+nvir] += dm_vv2

    print("  trace of alpha dm = %.6f" % numpy.trace(dm[0]))
    print("  trace of beta dm = %.6f" % numpy.trace(dm[1]))

    dm[0] = reduce(numpy.dot, (mo_coeff, dm[0], mo_coeff.T))
    dm[1] = reduce(numpy.dot, (mo_coeff, dm[1], mo_coeff.T))

    print("density matrix generation finished.\n")

    if full_return is False:
        return dm
    else:
        dm1h = numpy.zeros(shape=[nmo_full, nmo_full], dtype=numpy.double)
        dm2h = numpy.zeros(shape=[nmo_full, nmo_full], dtype=numpy.double)
        dm1p = numpy.zeros(shape=[nmo_full, nmo_full], dtype=numpy.double)
        dm2p = numpy.zeros(shape=[nmo_full, nmo_full], dtype=numpy.double)

        if oo_dim > 0:
            dm1h[nocc_full-nocc:nocc_full, nocc_full-nocc:nocc_full] += dm_oo1
            dm1h = reduce(numpy.dot, (mo_coeff, dm1h, mo_coeff.T))

            dm2h[nocc_full-nocc:nocc_full, nocc_full-nocc:nocc_full] += dm_oo2
            dm2h = reduce(numpy.dot, (mo_coeff, dm2h, mo_coeff.T))

        if oo_dim > 0:
            dm1p[nocc_full:nocc_full+nvir, nocc_full:nocc_full+nvir] += dm_vv1
            dm1p = reduce(numpy.dot, (mo_coeff, dm1p, mo_coeff.T))

            dm2p[nocc_full:nocc_full+nvir, nocc_full:nocc_full+nvir] += dm_vv2
            dm2p = reduce(numpy.dot, (mo_coeff, dm2p, mo_coeff.T))

        return dm, dm1h, dm1p, dm2h, dm2p
