import numpy as np
from functools import reduce
from pyscf import lib, scf
from pyscf.df.df_jk import _DFHF
from pyscf.lib import logger
from pyscf.df.grad import rhf as rhf_grad
from lib_pprpa.grad import grad_utils
from lib_pprpa.pprpa_util import start_clock, stop_clock


def grad_elec(pprpa_grad, xy, mult, atmlst=None):
    mf = pprpa_grad.mf
    pprpa = pprpa_grad.base
    mol = mf.mol
    mf_grad = mf.nuc_grad_method()
    if atmlst is None:
        atmlst = range(mol.natm)
    assert mult in ['t', 's'], 'mult = {}. is not valid in grad_elec'.format(mult)

    nocc_all = mf.mol.nelectron // 2
    nocc = pprpa.nocc
    nvir = pprpa.nvir
    nfrozen_occ = nocc_all - nocc

    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)
    dm0, i_int = make_rdm1_relaxed_rhf_pprpa(
        pprpa, mf, xy=xy, mult=mult, cphf_max_cycle=pprpa_grad.cphf_max_cycle, cphf_conv_tol=pprpa_grad.cphf_conv_tol
    )
    dm0 = lib.einsum('pi,ij,qj->pq', mf.mo_coeff, dm0, mf.mo_coeff, optimize=True)
    pprpa_grad.rdm1e = dm0
    dm0_hf = mf.make_rdm1()
    i_int = lib.einsum('pi,ij,qj->pq', mf.mo_coeff, i_int, mf.mo_coeff, optimize=True)
    i_int -= mf_grad.make_rdm1e(mf.mo_energy, mf.mo_coeff, mf.mo_occ)

    occ_y_mat, vir_x_mat = grad_utils.get_xy_full(xy, pprpa.oo_dim, mult)
    coeff_occ = mf.mo_coeff[:, nfrozen_occ : nfrozen_occ + nocc]
    coeff_vir = mf.mo_coeff[:, nfrozen_occ + nocc : nfrozen_occ + nocc + nvir]
    xy_ao = lib.einsum('pi,ij,qj->pq', coeff_vir, vir_x_mat, coeff_vir, optimize=True) + lib.einsum(
        'pi,ij,qj->pq', coeff_occ, occ_y_mat, coeff_occ, optimize=True
    )

    aux_response = False
    if isinstance(mf, _DFHF):
        # aux_response is Ture by default in DFHF
        # To my opinion, aux_response should always be True for DFHF
        aux_response = mf_grad.auxbasis_response
    elif not pprpa._use_eri and not pprpa._ao_direct:
        print(
            'Warning:   The analytical gradient of the ppRPA must be used with the density\n\
            fitting mean-field method. The calculation will proceed but the analytical\n\
            gradient is no longer exact (does NOT agree with numerical gradients).'
        )

    if not hasattr(mf, 'xc'):  # RHF
        vj, vk = mf_grad.get_jk(mol, (dm0_hf, dm0, xy_ao), hermi=0)
        vhf = np.zeros_like(vj)

        vhf[:2] = vj[:2] - 0.5 * vk[:2]
        vhf[2] = vk[2]
        if aux_response:
            vhf_aux = np.zeros_like(vj.aux)
            vhf_aux[:2, :2] = vj.aux[:2, :2] - 0.5 * vk.aux[:2, :2]
            if mult == 's':
                vhf_aux[2, 2] = vk.aux[2, 2]
            else:
                vhf_aux[2, 2] = -vk.aux[2, 2]
            vhf = lib.tag_array(vhf, aux=vhf_aux)

        aoslices = mol.aoslice_by_atom()
        de = np.zeros((len(atmlst), 3))
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            h1ao = hcore_deriv(ia)
            h1ao[:,p0:p1]   += vhf[0,:,p0:p1]
            h1ao[:,:,p0:p1] += vhf[0,:,p0:p1].transpose(0,2,1)
            de[k] += lib.einsum('xij,ij->x', h1ao, dm0+dm0_hf)
            # nabla was applied on bra in s1, *2 for the contributions of nabla|ket>
            de[k] += lib.einsum('xij,ij->x', vhf[1, :, p0:p1], dm0_hf[p0:p1, :]) * 2
            de[k] += lib.einsum('xij,ij->x', vhf[2, :, p0:p1], xy_ao[p0:p1, :]) * 2

            de[k] += lib.einsum('xij,ji->x', s1[:, p0:p1], i_int[:, p0:p1]) * 2

            if aux_response:
                de[k] += vhf.aux[0, 1, ia] + 0.5*vhf.aux[0, 0, ia]
                de[k] += vhf.aux[1, 0, ia] + 0.5*vhf.aux[0, 0, ia]
                de[k] += vhf.aux[2, 2, ia]
    else:  # RKS
        # The grid response by default is not included.
        # Even if grid_response is set to True, the response is not complete.
        # It will include the response of the Vxc but NOT the fxc.
        # For benchmarking, one can use high grid level to avoid this error.
        # mf_grad.grid_response = True
        from lib_pprpa.grad.grad_utils import get_veff_df_rks, get_veff_rks, _contract_xc_kernel

        vj, vk = mf_grad.get_jk(mol, xy_ao, hermi=0)
        vhf = vk
        if aux_response:
            vxc, vjk = get_veff_df_rks(mf_grad, mol, (dm0_hf, dm0))
            if mult == 's':
                vhf_aux = vk.aux[0, 0]
            else:
                vhf_aux = -vk.aux[0, 0]
            vhf = lib.tag_array(vhf, aux=vhf_aux)
        else:
            vxc, vjk = get_veff_rks(mf_grad, mol, (dm0_hf, dm0))
        
        vjk[1] += _contract_xc_kernel(mf, mf.xc, dm0, None, False, False, True)[0][1:]*0.5

        aoslices = mol.aoslice_by_atom()
        de = np.zeros((len(atmlst), 3))
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            h1ao = hcore_deriv(ia)
            h1ao[:, p0:p1] += vxc[0, :, p0:p1] + vjk[0, :, p0:p1]
            h1ao[:, :, p0:p1] += vxc[0, :, p0:p1].transpose(0, 2, 1) + vjk[0, :, p0:p1].transpose(0, 2, 1)
            de[k] += lib.einsum('xij,ij->x', h1ao, dm0 + dm0_hf)
            # nabla was applied on bra in s1, *2 for the contributions of nabla|ket>
            de[k] += lib.einsum('xij,ij->x', vjk[1, :, p0:p1], dm0_hf[p0:p1, :]) * 2
            de[k] += lib.einsum('xij,ij->x', vhf[:, p0:p1], xy_ao[p0:p1, :]) * 2

            de[k] += lib.einsum('xij,ji->x', s1[:, p0:p1], i_int[:, p0:p1]) * 2

            if aux_response:
                de[k] += vjk.aux[0, 1, ia] + 0.5*vjk.aux[0, 0, ia]
                de[k] += vjk.aux[1, 0, ia] + 0.5*vjk.aux[0, 0, ia]
                de[k] += vhf.aux[ia]
            if mf_grad.grid_response:
                de[k] += vxc.exc1_grid[0, ia]

    return de


def grad_elec_mf(mf, atmlst=None):
    # for test purpose
    mf_grad = mf.nuc_grad_method()
    dm0_hf = mf.make_rdm1()
    mol = mf.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)
    i_int = -mf_grad.make_rdm1e(mf.mo_energy, mf.mo_coeff, mf.mo_occ)

    # mf_grad.grid_response = True
    veff = mf_grad.get_veff(mol, dm0_hf)

    aoslices = mol.aoslice_by_atom()
    de = np.zeros((len(atmlst), 3))
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        h1ao = hcore_deriv(ia)
        de[k] += lib.einsum('xij,ij->x', h1ao, dm0_hf)

        # nabla was applied on bra in s1, *2 for the contributions of nabla|ket>
        de[k] += lib.einsum('xij,ij->x', veff[:, p0:p1], dm0_hf[p0:p1]) * 2
        # de[k] += lib.einsum('xij,ij->x', fxcz1[:,p0:p1], dm0_2e[p0:p1])*2
        if mf_grad.auxbasis_response:
            de[k] += veff.aux[ia]

        de[k] += lib.einsum('xij,ji->x', s1[:, p0:p1], i_int[:, p0:p1]).real * 2

    return de


def make_rdm1_relaxed_rhf_pprpa(
        pprpa, mf, xy=None, mult='t', istate=0, cphf_max_cycle=20,
        cphf_conv_tol=1.0e-8, vresp=None, pair_get_k=None,
        cphf_x0=None, cphf_x0_out=None):
    r"""Calculate relaxed density matrix (and the I intermediates)
        for given pprpa and mean-field object.
    Args:
        pprpa: a pprpa object.
        mf: a mean-field RHF/RKS object.
        pair_get_k: optional callback ``pair_get_k(dms, hermi)`` for direct
            exchange contractions of the two active pair densities.  This
            avoids reconstructing full-system three-index factors when the
            solver uses a compact active space.  A callback with a true
            ``accepts_ket`` attribute is called as ``pair_get_k(dms, hermi,
            ket=orbp)`` and returns ``K @ orbp`` (nao, nact): only those
            columns are ever used.
    Args:
        cphf_x0: optional AO-basis initial guess for the CPHF solution, as
            returned through ``cphf_x0_out`` by an earlier call.  Along a
            geometry optimization the previous step's solution is a good
            guess and the solver then iterates only on the correction.
        cphf_x0_out: optional dict.  When given, the converged CPHF
            solution is stored in it under ``'cphf_x0_ao'`` in the AO
            basis, ready to be handed back as ``cphf_x0``.

    Returns:
        den_relaxed: the relaxed one-particle density matrix (nmo_full, nmo_full)
        i_int: the I intermediates (nmo_full, nmo_full)
        Both are in the MO basis.
    """
    assert mult in ['t', 's'], 'mult = {}. is not valid in make_rdm1_relaxed_pprpa'.format(mult)
    from lib_pprpa import pyscf_util
    from lib_pprpa.grad.grad_utils import choose_slice, choose_range, contraction_2rdm_Lpq, \
                           contraction_2rdm_eri, get_xy_full, make_rdm1_unrelaxed_from_xy_full

    if xy is None:
        if mult == 's':
            xy = pprpa.xy_s[istate]
        else:
            xy = pprpa.xy_t[istate]
    nocc_all = mf.mol.nelectron // 2
    nvir_all = mf.mol.nao - nocc_all
    nocc = pprpa.nocc
    nvir = pprpa.nvir
    nfrozen_occ = nocc_all - nocc
    nfrozen_vir = nvir_all - nvir
    if mult == 's':
        oo_dim = (nocc + 1) * nocc // 2
    else:
        oo_dim = (nocc - 1) * nocc // 2

    # create slices
    slice_p = choose_slice('p', nfrozen_occ, nocc, nvir, nfrozen_vir)  # all active
    slice_i = choose_slice('i', nfrozen_occ, nocc, nvir, nfrozen_vir)  # active occupied
    slice_a = choose_slice('a', nfrozen_occ, nocc, nvir, nfrozen_vir)  # active virtual
    slice_ip = choose_slice('ip', nfrozen_occ, nocc, nvir, nfrozen_vir)  # frozen occupied
    slice_ap = choose_slice('ap', nfrozen_occ, nocc, nvir, nfrozen_vir)  # frozen virtual
    slice_I = choose_slice('I', nfrozen_occ, nocc, nvir, nfrozen_vir)  # all occupied
    slice_A = choose_slice('A', nfrozen_occ, nocc, nvir, nfrozen_vir)  # all virtual

    orbA = mf.mo_coeff[:, slice_A]
    orbI = mf.mo_coeff[:, slice_I]
    orbp = mf.mo_coeff[:, slice_p]
    orbi = mf.mo_coeff[:, slice_i]
    orba = mf.mo_coeff[:, slice_a]
    occ_y_mat, vir_x_mat = get_xy_full(
        xy, oo_dim, mult, nocc=nocc, nvir=nvir)
    use_pair_eri = pprpa._use_eri or pprpa._ao_direct or pair_get_k is not None
    if pprpa._use_eri and not hasattr(mf, 'cell'):
        # molecular fast path: contract 2-RDM with stored MO ERIs directly
        _, mo_ene_full, eri_full = pyscf_util.get_pyscf_input_mol_eri_r(mf, return_raw=True)
        eri_oo = np.ascontiguousarray(eri_full[:, slice_p, slice_i, slice_i])
        eri_vv = np.ascontiguousarray(eri_full[:, slice_p, slice_a, slice_a])
        eri_full = None
        X_eri = np.matmul(eri_vv.reshape(-1, nvir*nvir), vir_x_mat.reshape(-1)).reshape(-1, nocc+nvir)
        Y_eri = np.matmul(eri_oo.reshape(-1, nocc*nocc), occ_y_mat.reshape(-1)).reshape(-1, nocc+nvir)
    elif use_pair_eri:
        hermi = 1 if mult == 's' else 2
        mo_ene_full = mf.mo_energy
        X_ao = orba @ vir_x_mat @ orba.T
        Y_ao = orbi @ occ_y_mat @ orbi.T
        pair_dms = np.stack((X_ao, Y_ao))
        if pair_get_k is None:
            X_eri, Y_eri = mf.get_k(dm=pair_dms, hermi=hermi)
            X_eri, Y_eri = X_eri @ orbp, Y_eri @ orbp
        elif getattr(pair_get_k, 'accepts_ket', False):
            X_eri, Y_eri = pair_get_k(pair_dms, hermi=hermi, ket=orbp)
        else:
            X_eri, Y_eri = pair_get_k(pair_dms, hermi=hermi)
            X_eri, Y_eri = X_eri @ orbp, Y_eri @ orbp
        X_eri = mf.mo_coeff.T @ X_eri
        Y_eri = mf.mo_coeff.T @ Y_eri
    else:
        if nfrozen_occ > 0 or nfrozen_vir > 0 or pprpa.Lpq is None:
            # A periodic GDF solver has the same Lpq algebra as molecular DF,
            # but its factors must be reconstructed with the periodic metric.
            # Falling through to get_pyscf_input_mol used to silently replace
            # the periodic operator whenever a frozen orbital was present (or
            # after Davidson released the full Lpq tensor).
            if hasattr(mf, 'cell'):
                _, mo_ene_full, Lpq_full = pyscf_util.get_pyscf_input_sc(mf)
            else:
                _, mo_ene_full, Lpq_full = pyscf_util.get_pyscf_input_mol(mf)
        else:
            mo_ene_full = pprpa.mo_energy
            Lpq_full = pprpa.Lpq

    # set singlet=None, generate function for CPHF type response kernel
    if vresp is None:
        vresp = mf.gen_response(singlet=None, hermi=1)
    den_u = make_rdm1_unrelaxed_from_xy_full(occ_y_mat, vir_x_mat)
    den_u_ao = lib.einsum('pi,i,qi->pq', orbp, den_u, orbp, optimize=True)
    veff_den_u = reduce(np.dot, (mf.mo_coeff.T, vresp(den_u_ao) * 2, mf.mo_coeff))

    start_clock('Calculate i_prime and i_prime_prime')
    # calculate I' first
    i_prime = np.zeros((len(mo_ene_full), len(mo_ene_full)), dtype=occ_y_mat.dtype)
    # I' active-active block
    if not use_pair_eri:
        i_prime[slice_p, slice_p] += contraction_2rdm_Lpq(
            occ_y_mat, vir_x_mat, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'p', 'p'
        )
    else:
        i_prime[slice_p, slice_p] += contraction_2rdm_eri(
            occ_y_mat, vir_x_mat, X_eri, Y_eri, nocc, nvir, nfrozen_occ, nfrozen_vir, 'p', 'p'
        )
    i_prime[slice_a, slice_i] += veff_den_u[slice_a, slice_i]
    for p in choose_range('p', nfrozen_occ, nocc, nvir, nfrozen_vir):
        i_prime[p, p] += mo_ene_full[p] * den_u[p - nfrozen_occ]

    if nfrozen_vir > 0:
        # I' frozen virtual-active block
        if not use_pair_eri:
            i_prime[slice_ap, slice_p] += contraction_2rdm_Lpq(
                occ_y_mat, vir_x_mat, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'ap', 'p'
            )
        else:
            i_prime[slice_ap, slice_p] += contraction_2rdm_eri(
                occ_y_mat, vir_x_mat, X_eri, Y_eri, nocc, nvir, nfrozen_occ, nfrozen_vir, 'ap', 'p'
            )
        i_prime[slice_ap, slice_i] += veff_den_u[slice_ap, slice_i]
    if nfrozen_occ > 0:
        # I' frozen occupied-active block
        if not use_pair_eri:
            i_prime[slice_ip, slice_p] += contraction_2rdm_Lpq(
                occ_y_mat, vir_x_mat, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'ip', 'p'
            )
        else:
            i_prime[slice_ip, slice_p] += contraction_2rdm_eri(
                occ_y_mat, vir_x_mat, X_eri, Y_eri, nocc, nvir, nfrozen_occ, nfrozen_vir, 'ip', 'p'
            )
        # I' all virtual-frozen occupied block
        i_prime[slice_A, slice_ip] += veff_den_u[slice_A, slice_ip]

    # calculate I'' next
    i_prime_prime = np.zeros_like(i_prime)
    # I'' active virtual-all occupied block
    i_prime_prime[slice_a, slice_I] = i_prime[slice_a, slice_I] - i_prime[slice_I, slice_a].T
    # I'' = I' blocks
    i_prime_prime[slice_A, slice_a] = i_prime[slice_A, slice_a]
    i_prime_prime[slice_I, slice_i] = i_prime[slice_I, slice_i]
    i_prime_prime[slice_ap, slice_I] = i_prime[slice_ap, slice_I]
    stop_clock('Calculate i_prime and i_prime_prime')

    start_clock('Calculate d_prime')
    d_prime = np.zeros_like(i_prime_prime)
    threshold = 1.0e-6
    # D' all occupied-active occupied block
    for i in choose_range('I', nfrozen_occ, nocc, nvir, nfrozen_vir):
        for j in choose_range('i', nfrozen_occ, nocc, nvir, nfrozen_vir):
            denorm = mo_ene_full[j] - mo_ene_full[i]
            factor = 1.0 / denorm if abs(denorm) >= threshold else 0.0
            d_prime[i, j] = factor * i_prime_prime[i, j]

    # D' all virtual-active virtual block
    for a in choose_range('A', nfrozen_occ, nocc, nvir, nfrozen_vir):
        for b in choose_range('a', nfrozen_occ, nocc, nvir, nfrozen_vir):
            denorm = mo_ene_full[b] - mo_ene_full[a]
            factor = 1.0 / denorm if abs(denorm) >= threshold else 0.0
            d_prime[a, b] = factor * i_prime_prime[a, b]

    x_int = i_prime_prime[slice_A, slice_I].copy()
    d_ao = reduce(np.dot, (orbI, d_prime[slice_I, slice_i], orbi.T))
    d_ao += reduce(np.dot, (orbA, d_prime[slice_A, slice_a], orba.T))
    d_ao += d_ao.T
    x_int += reduce(np.dot, (orbA.T, vresp(d_ao) * 2, orbI))

    log = logger.new_logger(mf)

    # The Krylov vectors are small, (nvir_all, nocc_all), but the intermediates
    # are (nao, nao).  On the periodic GPU path that is a large array crossing
    # the bus twice per iteration, so when the response accepts device arrays
    # keep the orbitals resident and do both transforms there; only the small
    # vectors move.  The NumPy path below is the original one and the
    # arithmetic is identical either way.
    use_device = getattr(vresp, 'accepts_device_arrays', False)
    if use_device:
        import cupy
        orbA_gpu = cupy.asarray(orbA)
        orbI_gpu = cupy.asarray(orbI)
    nresponse = [0]

    def fvind(x):
        nresponse[0] += 1
        shape = (nvir + nfrozen_vir, nocc + nfrozen_occ)
        if use_device:
            xd = cupy.asarray(x.reshape(shape)) * 2
            dm = orbA_gpu @ xd @ orbI_gpu.T
            v1ao = vresp(dm + dm.T)
            return cupy.asnumpy(orbA_gpu.T @ v1ao @ orbI_gpu).ravel()
        dm = reduce(np.dot, (orbA, x.reshape(shape) * 2, orbI.T))
        v1ao = vresp(dm + dm.T)
        return reduce(np.dot, (orbA.T, v1ao, orbI)).ravel()

    # pyscf.scf.cphf.solve does not forward an initial guess to lib.krylov, so
    # solve_nos1 is inlined here to pass one.  With x0 None the arithmetic is
    # identical to cphf.solve.
    x0 = None
    if cphf_x0 is not None and cphf_x0.shape == (mf.mol.nao, mf.mol.nao):
        # The AO basis is not orthogonal, so C^T D C is not the inverse of
        # C d C^T.  The MO coefficients of an AO matrix are C^T S D S C,
        # because C^T S C = 1.  Without S the guess is distorted by the AO
        # metric even when the geometry has not moved.
        ovlp = np.asarray(mf.get_ovlp())
        if ovlp.ndim == 3:
            ovlp = ovlp[0]
        x0 = reduce(np.dot, (orbA.T, ovlp.real, cphf_x0, ovlp.real, orbI))

    e_a = mo_ene_full[mf.mo_occ == 0]
    e_i = mo_ene_full[mf.mo_occ > 0]
    e_ai = 1 / (e_a[:, None] - e_i)
    nvir_all, nocc_all_ = e_ai.shape
    mo1base = x_int * -e_ai

    def vind_vo(mo1):
        v = fvind(mo1.reshape(-1, nvir_all, nocc_all_))
        v = v.reshape(-1, nvir_all, nocc_all_) * e_ai
        return v.reshape(-1, nvir_all * nocc_all_)

    mo1 = lib.krylov(vind_vo, mo1base.reshape(-1, nvir_all * nocc_all_),
                     x0=None if x0 is None else x0.reshape(-1, nvir_all * nocc_all_),
                     tol=cphf_conv_tol, max_cycle=cphf_max_cycle, hermi=False)
    d_prime[slice_A, slice_I] = mo1.reshape(nvir + nfrozen_vir, nocc + nfrozen_occ)

    # Hand the solution back in the AO basis: between geometry steps the MO
    # basis changes (new SCF, possible phase flips or reordering), and the AO
    # matrix survives that.
    if cphf_x0_out is not None:
        cphf_x0_out['cphf_x0_ao'] = reduce(
            np.dot, (orbA, d_prime[slice_A, slice_I], orbI.T))
    log.debug('CPHF: %d response evaluations, device=%s, warm start=%s',
              nresponse[0], use_device, x0 is not None)
    stop_clock('Calculate d_prime')

    start_clock('Calculate I intermediates')
    i_int = -lib.einsum('qp,p->qp', d_prime, mo_ene_full)
    # I all occupied-all occupied block
    dp_ao = reduce(np.dot, (mf.mo_coeff, d_prime, mf.mo_coeff.T))
    veff_dp_II = reduce(np.dot, (orbI.T, vresp(dp_ao + dp_ao.T), orbI))
    i_int[slice_I, slice_I] -= 0.5 * veff_den_u[slice_I, slice_I]
    i_int[slice_I, slice_I] -= veff_dp_II
    # I active virtual-all occupied block
    i_int[slice_I, slice_a] -= i_prime[slice_I, slice_a]

    # I active-active block extra term
    for i in choose_range('p', nfrozen_occ, nocc, nvir, nfrozen_vir):
        for j in choose_range('p', nfrozen_occ, nocc, nvir, nfrozen_vir):
            denorm = mo_ene_full[j] - mo_ene_full[i]
            if abs(denorm) < threshold:
                i_int[i, j] -= 0.5 * i_prime[i, j]
    stop_clock('Calculate I intermediates')

    den_relaxed = d_prime
    # active-active block
    for p in choose_range('p', nfrozen_occ, nocc, nvir, nfrozen_vir):
        den_relaxed[p, p] += 0.5 * den_u[p - nfrozen_occ]
    den_relaxed = den_relaxed + den_relaxed.T
    i_int = i_int + i_int.T

    return den_relaxed, i_int


class Gradients(rhf_grad.Gradients):
    cphf_max_cycle = 20
    cphf_conv_tol = 1e-8

    _keys = {
        'cphf_max_cycle',
        'cphf_conv_tol',
        'mol',
        'base',
        'chkfile',
        'state',
        'atmlst',
        'de',
    }

    def __init__(self, pprpa, mf, mult='t', state=0):
        self.mf = mf
        assert isinstance(mf, scf.hf.RHF)
        self.base = pprpa
        self.mol = mf.mol
        self.state = state
        self.verbose = self.mol.verbose
        self.mult = mult

        self.rdm1e = None
        self.atmlst = None
        self.de = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s gradients for %s ********', self.base.__class__, self.mf.__class__)
        log.info('cphf_conv_tol = %g', self.cphf_conv_tol)
        log.info('cphf_max_cycle = %d', self.cphf_max_cycle)
        log.info('State ID = %d', self.state)
        log.info('max_memory %d MB (current use %d MB)', self.mf.max_memory, lib.current_memory()[0])
        log.info('\n')
        return self

    def grad_elec(self, xy, mult, atmlst):
        return grad_elec(self, xy, mult, atmlst)

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(
                self, '--------- %s gradients for state %d ----------', self.base.__class__.__name__, self.state
            )
            self._write(self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')

    def kernel(self, xy=None, state=None, mult=None, atmlst=None):
        if mult is None:
            mult = self.mult
        if xy is None:
            if state is None:
                state = self.state
            else:
                self.state = state
            if mult == 't':
                xy = self.base.xy_t[state]
            else:
                xy = self.base.xy_s[state]
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.INFO:
            self.dump_flags()

        de = self.grad_elec(xy, mult, atmlst)
        self.de = de = de + self.grad_nuc(atmlst=atmlst)
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        self._finalize()
        return self.de


Grad = Gradients

from lib_pprpa.pprpa_direct import ppRPA_direct
from lib_pprpa.pprpa_davidson import ppRPA_Davidson

ppRPA_direct.Gradients = ppRPA_Davidson.Gradients = lib.class_as_method(Gradients)
