import numpy as np
from functools import reduce

from pyscf import lib
from pyscf.lib import logger
from lib_pprpa.grad.pprpa import Gradients as pprpa_grad
from pyscf.pbc.gto.pseudo import pp_int
from lib_pprpa.pprpa_util import start_clock, stop_clock
from lib_pprpa.grad.grad_utils import _contract_xc_kernel_krks, get_veff_krks, get_xy_full


def grad_elec(pprpa_grad, xy, mult, atmlst=None):
    mf = pprpa_grad.mf
    pprpa = pprpa_grad.base
    cell = mf.mol
    kmf = rhf_to_krhf(mf)
    kmf_grad = kmf.nuc_grad_method()
    if atmlst is None:
        atmlst = range(cell.natm)
    assert mult in ['t', 's'], 'mult = {}. is not valid in grad_elec'.format(mult)

    nocc_all = mf.mol.nelectron // 2
    nocc = pprpa.nocc
    nvir = pprpa.nvir
    nfrozen_occ = nocc_all - nocc
    kpts = mf.kpts
    mo_coeff = mf.mo_coeff
    log = logger.Logger(kmf_grad.stdout, kmf_grad.verbose)

    if hasattr(mf, 'xc') and kmf_grad.grid_response:
        raise NotImplementedError('Grid response is not implemented in pprpa yet.')

    dm0, i_int = make_rdm1_relaxed_rhf_pprpa(
        pprpa, mf, xy=xy, mult=mult, cphf_max_cycle=pprpa_grad.cphf_max_cycle, cphf_conv_tol=pprpa_grad.cphf_conv_tol
    )
    i_int = mo_coeff @ i_int @ mo_coeff.T
    i_int -= kmf_grad.make_rdm1e(kmf.mo_energy, kmf.mo_coeff, kmf.mo_occ)[0]

    dm0 = mo_coeff @ dm0 @ mo_coeff.T
    pprpa_grad.rdm1e = dm0
    dm0_hf = kmf.make_rdm1()[0] # (nband,3,nao,nao)

    occ_y_mat, vir_x_mat = get_xy_full(xy, pprpa.oo_dim, mult)
    coeff_occ = mo_coeff[:, nfrozen_occ : nfrozen_occ + nocc]
    coeff_vir = mo_coeff[:, nfrozen_occ + nocc : nfrozen_occ + nocc + nvir]
    xy_ao = coeff_vir @ vir_x_mat @ coeff_vir.T + coeff_occ @ occ_y_mat @ coeff_occ.T

    hcore_deriv = kmf_grad.hcore_generator(cell, kpts)
    s1 = kmf_grad.get_ovlp(cell, kpts)[0]

    if not hasattr(mf, 'xc'):  # HF
        t0 = (logger.process_clock(), logger.perf_counter())
        log.debug('Computing Gradients of NR-HF Coulomb repulsion')
        vhf = kmf_grad.get_veff([np.array([dm0_hf]), np.array([dm0])]) # (3,nset,nband,nao,nao)
        vhf = vhf[:,:,0,:,:].transpose(1,0,2,3)
        vk = kmf_grad.get_k(np.array([xy_ao])) # (3,nband,nao,nao)
        vk = vk[:,0,:,:]
        log.timer('gradients of 2e part', *t0)

        aoslices = cell.aoslice_by_atom()
        de = np.zeros((len(atmlst), 3))
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            h1ao = hcore_deriv(ia)[:,0] # (3,nband,nao,nao)
            h1ao[:,p0:p1]   += vhf[0,:,p0:p1]
            h1ao[:,:,p0:p1] += vhf[0,:,p0:p1].transpose(0,2,1)
            de[k] += np.einsum('xij,ij->x', h1ao, dm0+dm0_hf)
            # nabla was applied on bra in s1, *2 for the contributions of nabla|ket>
            de[k] += np.einsum('xij,ij->x', vhf[1, :, p0:p1], dm0_hf[p0:p1, :]) * 2
            de[k] += np.einsum('xij,ij->x', vk[:, p0:p1], xy_ao[p0:p1, :]) * 2

            de[k] += np.einsum('xij,ji->x', s1[:, p0:p1], i_int[:, p0:p1]) * 2
    else:  # KS
        vk = kmf_grad.get_k(np.array([xy_ao])) # (3,nband,nao,nao)
        vk = vk[:,0,:,:]
        vxc, vjk = get_veff_krks(kmf_grad, np.array([[dm0_hf], [dm0]]))
        vxc = vxc[:,:,0,:,:].transpose(1,0,2,3)
        vjk = vjk[:,:,0,:,:].transpose(1,0,2,3)
        
        vjk[1] += _contract_xc_kernel_krks(kmf, kmf.xc, dm0)[0][1:]*0.5

        aoslices = cell.aoslice_by_atom()
        de = np.zeros((len(atmlst), 3))
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            h1ao = hcore_deriv(ia)[:,0] # (3,nband,nao,nao)
            h1ao[:, p0:p1] += vxc[0, :, p0:p1] + vjk[0, :, p0:p1]
            h1ao[:, :, p0:p1] += vxc[0, :, p0:p1].transpose(0, 2, 1) + vjk[0, :, p0:p1].transpose(0, 2, 1)
            de[k] += np.einsum('xij,ij->x', h1ao, dm0 + dm0_hf)
            # nabla was applied on bra in s1, *2 for the contributions of nabla|ket>
            de[k] += np.einsum('xij,ij->x', vjk[1, :, p0:p1], dm0_hf[p0:p1, :]) * 2
            de[k] += np.einsum('xij,ij->x', vk[:, p0:p1], xy_ao[p0:p1, :]) * 2

            de[k] += np.einsum('xij,ji->x', s1[:, p0:p1], i_int[:, p0:p1]) * 2

    de += pp_int.vppnl_nuc_grad(cell, np.array([dm0+dm0_hf]), kpts)

    return de


def rhf_to_krhf(myrhf):
    from pyscf.pbc import scf, dft
    if hasattr(myrhf, 'xc'):
        mykrhf = dft.KRKS(myrhf.mol, kpts = np.array([np.zeros(3)]))
        mykrhf.xc = myrhf.xc
    else:
        mykrhf = scf.KRHF(myrhf.mol, kpts = np.array([np.zeros(3)]))
    mykrhf.mo_coeff = [myrhf.mo_coeff]
    mykrhf.mo_energy = [myrhf.mo_energy]
    mykrhf.mo_occ = [myrhf.mo_occ]
    mykrhf.exxdiv = myrhf.exxdiv
    mykrhf.converged = myrhf.converged
    mykrhf.e_tot = myrhf.e_tot
    return mykrhf


def make_rdm1_relaxed_rhf_pprpa(pprpa, mf, xy=None, mult='t', istate=0, cphf_max_cycle=20, cphf_conv_tol=1.0e-8):
    r"""Calculate relaxed density matrix (and the I intermediates)
        for given pprpa and mean-field object.
    Args:
        pprpa: a pprpa object.
        mf: a mean-field RHF/RKS object.
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
    occ_y_mat, vir_x_mat = get_xy_full(xy, oo_dim, mult)
    if pprpa._use_eri or pprpa._ao_direct:
        hermi = 1 if mult == 's' else 2
        mo_ene_full = mf.mo_energy
        X_ao = orba @ vir_x_mat @ orba.T
        X_eri = mf.get_k(dm=X_ao, hermi=hermi)
        X_eri = mf.mo_coeff.T @ X_eri @ orbp
        Y_ao = orbi @ occ_y_mat @ orbi.T
        Y_eri = mf.get_k(dm=Y_ao, hermi=hermi)
        Y_eri = mf.mo_coeff.T @ Y_eri @ orbp
    else:
        raise NotImplementedError("Lpq based contraction is not supported yet in pprpa_gamma gradient.")

    # set singlet=None, generate function for CPHF type response kernel
    vresp = mf.gen_response(singlet=None, hermi=1)
    den_u = make_rdm1_unrelaxed_from_xy_full(occ_y_mat, vir_x_mat)
    den_u_ao = np.einsum('pi,i,qi->pq', orbp, den_u, orbp, optimize=True)
    veff_den_u = reduce(np.dot, (mf.mo_coeff.T, vresp(den_u_ao) * 2, mf.mo_coeff))

    start_clock('Calculate i_prime and i_prime_prime')
    # calculate I' first
    i_prime = np.zeros((len(mo_ene_full), len(mo_ene_full)), dtype=occ_y_mat.dtype)
    # I' active-active block
    if not pprpa._use_eri and not pprpa._ao_direct:
        raise NotImplementedError("Lpq based contraction is not supported yet in pprpa_gamma gradient.")
    else:
        i_prime[slice_p, slice_p] += contraction_2rdm_eri(
            occ_y_mat, vir_x_mat, X_eri, Y_eri, nocc, nvir, nfrozen_occ, nfrozen_vir, 'p', 'p'
        )
    i_prime[slice_a, slice_i] += veff_den_u[slice_a, slice_i]
    for p in choose_range('p', nfrozen_occ, nocc, nvir, nfrozen_vir):
        i_prime[p, p] += mo_ene_full[p] * den_u[p - nfrozen_occ]

    if nfrozen_vir > 0:
        # I' frozen virtual-active block
        if not pprpa._use_eri and not pprpa._ao_direct:
            raise NotImplementedError("Lpq based contraction is not supported yet in pprpa_gamma gradient.")
        else:
            i_prime[slice_ap, slice_p] += contraction_2rdm_eri(
                occ_y_mat, vir_x_mat, X_eri, Y_eri, nocc, nvir, nfrozen_occ, nfrozen_vir, 'ap', 'p'
            )
        i_prime[slice_ap, slice_i] += veff_den_u[slice_ap, slice_i]
    if nfrozen_occ > 0:
        # I' frozen occupied-active block
        if not pprpa._use_eri and not pprpa._ao_direct:
            raise NotImplementedError("Lpq based contraction is not supported yet in pprpa_gamma gradient.")
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

    def fvind(x):
        dm = reduce(np.dot, (orbA, x.reshape(nvir + nfrozen_vir, nocc + nfrozen_occ) * 2, orbI.T))
        dm = dm + dm.T
        v1ao = vresp(dm)
        return reduce(np.dot, (orbA.T, v1ao, orbI)).ravel()

    from pyscf.scf import cphf

    d_prime[slice_A, slice_I] = cphf.solve(
        fvind, mo_ene_full, mf.mo_occ, x_int, max_cycle=cphf_max_cycle, tol=cphf_conv_tol
    )[0].reshape(nvir + nfrozen_vir, nocc + nfrozen_occ)
    stop_clock('Calculate d_prime')

    start_clock('Calculate I intermediates')
    i_int = -np.einsum('qp,p->qp', d_prime, mo_ene_full)
    # I all occupied-all occupied block
    dp_ao = reduce(np.dot, (mf.mo_coeff, d_prime, mf.mo_coeff.T))
    dp_ao = dp_ao + dp_ao.T
    veff_dp_II = reduce(np.dot, (orbI.T, vresp(dp_ao), orbI))
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


class Gradients(pprpa_grad):
    def __init__(self, pprpa, mf, mult='t', state=0):
        from pyscf.pbc import scf
        self.mf = mf
        assert isinstance(mf, scf.rhf.SCF)
        assert len(mf.kpts) == 1 and np.allclose(mf.kpts[0], np.zeros(3)), "Only Gamma-point KSCF is supported in ppRPA gradients."
        self.base = pprpa
        self.mol = mf.mol
        self.state = state
        self.verbose = self.mol.verbose
        self.mult = mult

        self.rdm1e = None
        self.atmlst = None
        self.de = None

    def grad_nuc(self, cell=None, atmlst=None):
        if cell is None: cell = self.mol
        from pyscf.pbc.grad.krhf import grad_nuc
        return grad_nuc(cell, atmlst)

    def grad_elec(self, xy, mult, atmlst):
        return grad_elec(self, xy, mult, atmlst)
    
    def optimizer(self, solver='ase'):
        '''Geometry optimization solver
        '''
        solver = solver.lower()
        if solver == 'ase':
            from pyscf.geomopt import ase_solver
            return ase_solver.GeometryOptimizer(self.base)
        else:
            raise RuntimeError(f'Optimization solver {solver} not supported')

Grad = Gradients

from lib_pprpa.pprpa_davidson import ppRPA_Davidson

ppRPA_Davidson.Gradients = lib.class_as_method(Gradients)
