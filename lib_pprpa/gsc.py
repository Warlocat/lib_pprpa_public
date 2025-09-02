import numpy

def mo_energy_gsc2(mf, w_mat, nocc_act=None, nvir_act=None, rpa=False):
    """Add GSC2 correction to original DFA eigenvalues.

    Args:
        mf (pyscf.dft.UKS): molecular mean-field object.

    Kwargs:
        w_mat (list of numpy.ndarray): W matrices, [aaaa, bbbb, aabb].
    """
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy

    nmo = len(mo_energy[0])
    nocc = mf.nelec
    nvir = (nmo - nocc[0], nmo - nocc[1])

    if nocc_act is None:
        nocc_act = nocc
    elif isinstance(nocc_act, (int, numpy.int64)):
        nocc_act = [nocc_act, nocc_act]
        nocc_act = (min(nocc[0], nocc_act[0]), min(nocc[1], nocc_act[1]))
    else:
        nocc_act = (min(nocc[0], nocc_act[0]), min(nocc[1], nocc_act[1]))

    if nvir_act is None:
        nvir_act = nvir
    elif isinstance(nvir_act, (int, numpy.int64)):
        nvir_act = [nvir_act, nvir_act]
        nvir_act = (min(nvir[0], nvir_act[0]), min(nvir[1], nvir_act[1]))
    else:
        nvir_act = (min(nvir[0], nvir_act[0]), min(nvir[1], nvir_act[1]))

    mo_energy_act = [
        mo_energy[0, (nocc[0]-nocc_act[0]):(nocc[0]+nvir_act[0])],
        mo_energy[1, (nocc[1]-nocc_act[1]):(nocc[1]+nvir_act[1])]]

    mo_occ_act = [
        mo_occ[0, (nocc[0]-nocc_act[0]):(nocc[0]+nvir_act[0])],
        mo_occ[1, (nocc[1]-nocc_act[1]):(nocc[1]+nvir_act[1])]]

    kappa = []
    for i in range(2):
        mat = numpy.einsum('ppqq->pq', w_mat[i])
        kappa.append(mat)

    mo_occ_act = [0.5 - mo_occ_act[0], 0.5 - mo_occ_act[1]]
    delta_mo_ea = numpy.einsum(
        'pp,p->p', kappa[0], mo_occ_act[0], optimize=True)
    delta_mo_eb = numpy.einsum(
        'pp,p->p', kappa[1], mo_occ_act[1], optimize=True)

    mo_energy_act = [mo_energy_act[0] + delta_mo_ea,
                     mo_energy_act[1] + delta_mo_eb]

    return mo_energy_act



