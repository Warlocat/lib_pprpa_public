import scipy
from functools import reduce
import numpy as np
from pyscf import lib, scf
from pyscf.df.df_jk import _DFHF
from pyscf.lib import logger
from lib_pprpa.grad import grad_utils
from lib_pprpa.grad.pprpa import Gradients as rhf_pprpa_grad

try:
    from socutils.scf import spinor_hf
    from socutils.grad import ghf_grad
    from socutils.grad import df_ghf_grad

    with_socuils = True
except ImportError:
    raise ImportError('socutils is not installed. The GHF gradient is not available.')

def grad_elec(pprpa_grad, xy, atmlst=None, correlation_only=False):
    mf = pprpa_grad.mf
    pprpa = pprpa_grad.base
    mol = mf.mol
    nao = mol.nao_nr()
    mf_grad = pprpa_grad.mf.Gradients()

    if atmlst is None:
        atmlst = range(mol.natm)

    nocc_all = mf.mol.nelectron
    nocc = pprpa.nocc
    nvir = pprpa.nvir
    nfrozen_occ = nocc_all - nocc

    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)
    # dm0, i_int = grad_utils.make_rdm1_relaxed_pprpa(
    #     pprpa, mf, xy = xy
    # )
    dm0, i_int = make_rdm1_relaxed_ghf_pprpa(pprpa, mf, xy)
    dm0 = np.einsum('pi,ij,qj->pq', mf.mo_coeff, dm0, mf.mo_coeff.conj(), optimize=True)
    pprpa_grad.rdm1e = dm0
    dm0_hf = mf.make_rdm1()
    i_int = np.einsum('pi,ij,qj->pq', mf.mo_coeff, i_int, mf.mo_coeff.conj(), optimize=True)
    if not correlation_only:
        i_int -= mf_grad.make_rdm1e(mf.mo_energy, mf.mo_coeff, mf.mo_occ)

    occ_y_mat, vir_x_mat = grad_utils.get_xy_full(xy, pprpa.oo_dim)
    coeff_occ = mf.mo_coeff[:, nfrozen_occ : nfrozen_occ + nocc]
    coeff_vir = mf.mo_coeff[:, nfrozen_occ + nocc : nfrozen_occ + nocc + nvir]
    xy_ao = np.einsum('pi,ij,qj->pq', coeff_vir, vir_x_mat, coeff_vir, optimize=True)  \
          + np.einsum('pi,ij,qj->pq', coeff_occ, occ_y_mat, coeff_occ, optimize=True)

    if correlation_only:
        dm0_1e = dm0
        dm0_2e = dm0
    else:
        dm0_1e = dm0 + dm0_hf
        dm0_2e = dm0 + 0.5 * dm0_hf

    if isinstance(mf, _DFHF):
        # aux_response is Ture by default in DFHF
        # To my opinion, aux_response should always be True for DFHF
        aux_response = True # ghf.auxbasis_response not available in PySCF
    else:
        print(
            'Warning:   The analytical gradient of the ppRPA must be used with the density\n\
            fitting mean-field method. The calculation will proceed but the analytical\n\
            gradient is no longer exact (does NOT agree with numerical gradients).'
        )

    if not hasattr(mf, 'xc'):  # GHF
        if xy.dtype == np.float64:
            vj, vk = mf_grad.get_jk(mol, (dm0_hf, dm0_2e, xy_ao))
        else:
            vj, vk = mf_grad.get_jk(mol, (dm0_hf, dm0_2e, xy_ao, xy_ao.conj()))
        vhf = np.zeros_like(vj)

        vhf[:2] = vj[:2] - vk[:2]
        vhf[2] = -vk[2]
        if aux_response:
            vhf_aux = np.zeros_like(vj.aux)
            vhf_aux[:2, :2] = vj.aux[:2, :2] - vk.aux[:2, :2]
            vhf_aux[2, 2] = -vk.aux[2, 2] if xy.dtype == np.float64 else -vk.aux[2, 3]
            vhf_aux = vhf_aux[0,1] + vhf_aux[1,0] + vhf_aux[2,2]
            vhf = lib.tag_array(vhf, aux=vhf_aux)

        aoslices = mol.aoslice_by_atom()
        de = np.zeros((len(atmlst), 3), dtype=xy.dtype)
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            h1ao = hcore_deriv(ia)
            if h1ao.shape[-1] == dm0_1e.shape[-1]//2: # for spin-free case
                h1ao = np.asarray([scipy.linalg.block_diag(h1ao[i],h1ao[i]) for i in range(3)])
            de[k] += np.einsum('xij,ji->x', h1ao, dm0_1e)
            # nabla was applied on bra in s1, *2 for the contributions of nabla|ket>
            de[k] += np.einsum('xij,ji->x', vhf[0, :, p0:p1], dm0_2e[:,p0:p1]).real * 2
            de[k] += np.einsum('xij,ji->x', vhf[1, :, p0:p1], dm0_hf[:,p0:p1]).real * 2
            de[k] += np.einsum('xij,ji->x', vhf[0, :, nao+p0:nao+p1], dm0_2e[:,nao+p0:nao+p1]).real * 2
            de[k] += np.einsum('xij,ji->x', vhf[1, :, nao+p0:nao+p1], dm0_hf[:,nao+p0:nao+p1]).real * 2

            de[k] += np.einsum('xij,ji->x', vhf[2, :, p0:p1], xy_ao[:,p0:p1].conj()) * 2
            de[k] += np.einsum('xij,ji->x', vhf[2, :, nao+p0:nao+p1], xy_ao[:,nao+p0:nao+p1].conj()) * 2

            de[k] += np.einsum('xij,ji->x', s1[:, p0:p1], i_int[:nao, p0:p1]).real * 2
            de[k] += np.einsum('xij,ji->x', s1[:, p0:p1], i_int[nao:, nao+p0:nao+p1]).real * 2
            if aux_response:
                de[k] += vhf.aux[ia]
    else:  # GKS
        raise NotImplementedError('Gradient for GKS reference is not implemented yet')

    if np.abs(de.imag).max() > 1e-5:
        logger.warn(pprpa_grad, 'Gradient has large imaginary part %s', np.abs(de.imag).max())
    return de.real


def grad_elec_mf(mf, atmlst=None):
    mol = mf.mol
    nao = mol.nao_nr()
    mf_grad = mf.Gradients()

    if atmlst is None:
        atmlst = range(mol.natm)

    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)
    dm0_hf = mf.make_rdm1()
    i_int = -mf_grad.make_rdm1e(mf.mo_energy, mf.mo_coeff, mf.mo_occ)

    if isinstance(mf, _DFHF):
        # aux_response is Ture by default in DFHF
        # To my opinion, aux_response should always be True for DFHF
        aux_response = True # ghf.auxbasis_response not available in PySCF
    else:
        print(
            'Warning:   The analytical gradient of the ppRPA must be used with the density\n\
            fitting mean-field method. The calculation will proceed but the analytical\n\
            gradient is no longer exact (does NOT agree with numerical gradients).'
        )

    if not hasattr(mf, 'xc'):  # GHF
        vhf = mf_grad.get_veff(mol, dm0_hf)

        aoslices = mol.aoslice_by_atom()
        de = np.zeros((len(atmlst), 3), dtype=dm0_hf.dtype)
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            h1ao = hcore_deriv(ia)
            if h1ao.shape[-1] == dm0_hf.shape[-1]//2:
                h1ao = np.asarray([scipy.linalg.block_diag(h1ao[i],h1ao[i]) for i in range(3)])
            de[k] += np.einsum('xij,ji->x', h1ao, dm0_hf)
            # nabla was applied on bra in s1, *2 for the contributions of nabla|ket>
            de[k] += np.einsum('xij,ji->x', vhf[:, p0:p1], dm0_hf[:,p0:p1]) * 2
            de[k] += np.einsum('xij,ji->x', vhf[:, nao+p0:nao+p1], dm0_hf[:,nao+p0:nao+p1]) * 2

            de[k] += np.einsum('xij,ji->x', s1[:, p0:p1], i_int[:nao, p0:p1]) * 2
            de[k] += np.einsum('xij,ji->x', s1[:, p0:p1], i_int[nao:, nao+p0:nao+p1]) * 2
            if aux_response:
                de[k] += vhf.aux[ia]
    else:  # GKS
        raise NotImplementedError('Gradient for GKS reference is not implemented yet')

    return de.real


def make_rdm1_relaxed_ghf_pprpa(pprpa, mf, xy=None, istate=0, cphf_max_cycle=20, cphf_conv_tol=1.0e-8):
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
    from lib_pprpa import pyscf_util
    from lib_pprpa.grad.grad_utils import choose_slice, choose_range, contraction_2rdm_Lpq, \
                           get_xy_full, make_rdm1_unrelaxed_from_xy_full
    from lib_pprpa.pprpa_util import start_clock, stop_clock

    if xy is None:
        xy = pprpa.xy[istate]
    nocc_all = mf.mol.nelectron
    nvir_all = mf.mol.nao*2 - nocc_all
    nocc = pprpa.nocc
    nvir = pprpa.nvir
    nfrozen_occ = nocc_all - nocc
    nfrozen_vir = nvir_all - nvir
    if nfrozen_occ > 0 or nfrozen_vir > 0:
        _, mo_ene_full, Lpq_full = pyscf_util.get_pyscf_input_mol_g(mf)
    else:
        mo_ene_full = pprpa.mo_energy
        Lpq_full = pprpa.Lpq

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

    # set singlet=None, generate function for CPHF type response kernel
    vresp = mf.gen_response(hermi=0)

    occ_y_mat, vir_x_mat = get_xy_full(xy, oo_dim)
    den_u = make_rdm1_unrelaxed_from_xy_full(occ_y_mat, vir_x_mat)
    den_u_ao = np.einsum('pi,i,qi->pq', orbp, den_u, orbp.conj(), optimize=True)
    veff_den_u = reduce(np.dot, (mf.mo_coeff.T.conj(), vresp(den_u_ao), mf.mo_coeff))

    start_clock('Calculate i_prime and i_prime_prime')
    # calculate I' first
    i_prime = np.zeros((len(mo_ene_full), len(mo_ene_full)), dtype=occ_y_mat.dtype)
    # I' active-active block
    i_prime[slice_p, slice_p] += contraction_2rdm_Lpq(
        occ_y_mat, vir_x_mat, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'p', 'p'
    )
    i_prime[slice_a, slice_i] += veff_den_u[slice_a, slice_i]
    for p in choose_range('p', nfrozen_occ, nocc, nvir, nfrozen_vir):
        i_prime[p, p] += mo_ene_full[p] * den_u[p - nfrozen_occ]

    if nfrozen_vir > 0:
        # I' frozen virtual-active block
        i_prime[slice_ap, slice_p] += contraction_2rdm_Lpq(
            occ_y_mat, vir_x_mat, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'ap', 'p'
        )
        i_prime[slice_ap, slice_i] += veff_den_u[slice_ap, slice_i]
    if nfrozen_occ > 0:
        # I' frozen occupied-active block
        i_prime[slice_ip, slice_p] += contraction_2rdm_Lpq(
            occ_y_mat, vir_x_mat, Lpq_full, nocc, nvir, nfrozen_occ, nfrozen_vir, 'ip', 'p'
        )
        # I' all virtual-frozen occupied block
        i_prime[slice_A, slice_ip] += veff_den_u[slice_A, slice_ip]

    # calculate I'' next
    i_prime_prime = np.zeros_like(i_prime)
    # I'' active virtual-all occupied block
    i_prime_prime[slice_a, slice_I] = i_prime[slice_a, slice_I] - i_prime[slice_I, slice_a].T.conj()
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
    d_ao = reduce(np.dot, (orbI, d_prime[slice_I, slice_i], orbi.T.conj()))
    d_ao += reduce(np.dot, (orbA, d_prime[slice_A, slice_a], orba.T.conj()))
    d_ao += d_ao.T.conj()
    x_int += reduce(np.dot, (orbA.T.conj(), vresp(d_ao), orbI))

    def fvind(x):
        dm = reduce(np.dot, (orbA, x.reshape(nvir + nfrozen_vir, nocc + nfrozen_occ), orbI.T.conj()))
        v1ao = vresp(dm + dm.T.conj())
        return reduce(np.dot, (orbA.T.conj(), v1ao, orbI)).ravel()

    from pyscf.scf import cphf

    d_prime[slice_A, slice_I] = cphf.solve(
        fvind, mo_ene_full, mf.mo_occ, x_int, max_cycle=cphf_max_cycle, tol=cphf_conv_tol, verbose=10
    )[0].reshape(nvir + nfrozen_vir, nocc + nfrozen_occ)
    stop_clock('Calculate d_prime')

    start_clock('Calculate I intermediates')
    i_int = -np.einsum('qp,p->qp', d_prime, mo_ene_full)
    # I all occupied-all occupied block
    dp_ao = reduce(np.dot, (mf.mo_coeff, d_prime, mf.mo_coeff.T.conj()))
    veff_dp_II = reduce(np.dot, (orbI.T.conj(), vresp(dp_ao), orbI))
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
    den_relaxed = den_relaxed + den_relaxed.T.conj()
    i_int = i_int + i_int.T.conj()

    return den_relaxed, i_int


class Gradients(rhf_pprpa_grad):
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

    def __init__(self, pprpa, mf, state=0):
        self.mf = mf
        assert isinstance(mf, scf.ghf.GHF) or isinstance(mf, spinor_hf.SpinorSCF)
        self.spinor = isinstance(mf, spinor_hf.SpinorSCF)
        self.base = pprpa
        self.mol = mf.mol
        self.state = state
        self.verbose = self.mol.verbose
        self.mult = "t"

        self.rdm1e = None
        self.atmlst = None
        self.de = None

    def grad_elec(self, xy, atmlst):
        return grad_elec(self, xy, atmlst)


    def kernel(self, xy=None, state=None, atmlst=None):
        if xy is None:
            if state is None:
                state = self.state
            else:
                self.state = state
            xy = self.base.xy[state]
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.INFO:
            self.dump_flags()

        de = self.grad_elec(xy, atmlst)
        self.de = de = de + self.grad_nuc(atmlst=atmlst)
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        self._finalize()
        return self.de


Grad = Gradients

from lib_pprpa.gpprpa_direct import GppRPA_direct
from lib_pprpa.gpprpa_davidson import GppRPA_Davidson

GppRPA_direct.Gradients = GppRPA_Davidson.Gradients = lib.class_as_method(Gradients)