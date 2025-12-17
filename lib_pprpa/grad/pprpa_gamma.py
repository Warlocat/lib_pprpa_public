import numpy as np
from functools import reduce

from pyscf import lib
from pyscf.lib import logger
from lib_pprpa.grad.pprpa import Gradients as pprpa_grad
from lib_pprpa.grad.pprpa import make_rdm1_relaxed_rhf_pprpa
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



class Gradients(pprpa_grad):
    def __init__(self, pprpa, mf, mult='t', state=0):
        from pyscf.pbc import scf
        self.mf = mf
        assert isinstance(mf, scf.rhf.SCF)
        assert len(mf.kpts) == 1 and np.allclose(mf.kpts[0], np.zeros(3)), "Only Gamma-point KSCF is supported in ppRPA gradients."
        assert pprpa._ao_direct or pprpa._use_eri, "PBC ppRPA gradients require either MO eri or AO direct approach."
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
