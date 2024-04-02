import numpy

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
    triu_row_o, triu_col_o = numpy.triu_indices(nocc, is_singlet-1)
    triu_row_v, triu_col_v = numpy.triu_indices(nvir, is_singlet-1)

    # 1. remove the index restrictions as equation 17 and 18 in doi.org/10.1039/C4CP04109G
    # 2. renormalize eigenvector as PySCF TDDFT NTO implementation:
    # https://github.com/pyscf/pyscf/blob/0a17e425e3c3dc28cfba0b54613194909db20548/pyscf/tdscf/rhf.py#L223
    norm = 0.0
    if oo_dim > 0:
        y_full = numpy.zeros(shape=[nocc, nocc], dtype=numpy.double)
        y_full[tril_row_o, tril_col_o] = xy[state][:oo_dim]
        y_full[triu_row_o, triu_col_o] = -y_full[tril_row_o, tril_col_o]
        norm -= numpy.sum(y_full**2)

    if vv_dim > 0:
        x_full = numpy.zeros(shape=[nvir, nvir], dtype=numpy.double)
        x_full[tril_row_v, tril_col_v] = xy[state][oo_dim:]
        x_full[triu_row_v, triu_col_v] = -x_full[tril_row_v, tril_col_v]
        norm += numpy.sum(x_full**2)
    norm = numpy.sqrt(numpy.abs(norm))

    # do SVD decomposition then get AO->NTO coefficient
    if oo_dim > 0:
        nto_o1, wo, nto_o2T = numpy.linalg.svd(y_full)
        nto_o2 = nto_o2T.conj().T
        weight_o = wo**2
        nto_coeff_o1 = numpy.dot(orbo, nto_o1)
        nto_coeff_o2 = numpy.dot(orbo, nto_o2)

    if vv_dim > 0:
        x_full *= 1. / norm
        nto_v1, wv, nto_v2T = numpy.linalg.svd(x_full)
        nto_v2 = nto_v2T.conj().T
        weight_v = wv**2
        nto_coeff_v1 = numpy.dot(orbv, nto_v1)
        nto_coeff_v2 = numpy.dot(orbv, nto_v2)

    if oo_dim > 0 and vv_dim > 0:
        return weight_o, nto_coeff_o1, nto_coeff_o2, weight_v, nto_coeff_v1, nto_coeff_v2
    elif oo_dim > 0:
        return weight_o, nto_coeff_o1, nto_coeff_o2
    elif vv_dim > 0:
        return weight_v, nto_coeff_v1, nto_coeff_v2
