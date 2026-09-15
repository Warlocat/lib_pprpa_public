"""GPU PBC XC-gradient helper for ppRPA Gamma-point gradients.

Current GPU4PySCF provides the CPHF XC response through
``KRKS.gen_response``.  The ppRPA-specific nuclear derivative skeleton is not
part of that API, so ``_contract_xc_kernel`` remains here for Gamma-point RKS
LDA/GGA calculations.

gpu4pyscf pbc AO layout from KNumInt.block_loop is (ngrid, nao) for LDA and
(ncomp, ngrid, nao) for GGA; eval_rho returns (ngrid,) / (nvar, ngrid).
"""
import numpy as np
import cupy as cp


def _contract_xc_kernel(mf, xc_code, dmvo, dm0=None, with_vxc=False, kpts=None):
    """GPU pbc fxc gradient skeleton (Gamma), mirroring CPU
    _contract_xc_kernel_krks.  Returns (f1vo, v1ao):
      f1vo (4,nao,nao): [0]=fxc.dmvo response matrix, [1:]=x/y/z skeleton.
      v1ao (4,nao,nao) or None: same for the Vxc potential (if with_vxc).
    """
    from gpu4pyscf.pbc.grad import krks as pbc_krks_grad
    ni = mf._numint
    cell = mf.mol
    grids = mf.grids
    if grids.coords is None:
        grids.build()
    if kpts is None:
        kpts = np.asarray(mf.kpts).reshape(-1, 3)
    is_gamma = abs(np.asarray(kpts)).max() < 1e-9

    xctype = ni._xc_type(xc_code)
    if xctype == "LDA":
        ao_deriv = 1
    elif xctype == "GGA":
        ao_deriv = 2
    else:
        raise NotImplementedError(f"GPU pbc _contract_xc_kernel for {xc_code}")

    if dm0 is None:
        dm0 = mf.make_rdm1()
    dm0 = cp.asarray(dm0)
    if dm0.ndim == 2:
        dm0 = dm0[None]
    dmvo = cp.asarray(dmvo)
    dmvo = (dmvo + dmvo.T) * 0.5
    nao = dmvo.shape[-1]
    f1vo = cp.zeros((4, nao, nao), dtype=cp.complex128)
    v1ao = cp.zeros((4, nao, nao), dtype=cp.complex128) if with_vxc else None

    def _build(ao, wv):
        """(4,nao,nao): [0]=value matrix, [1:]=nuclear-deriv skeleton."""
        out = cp.zeros((4, nao, nao), dtype=cp.complex128)
        if xctype == "LDA":
            aow = ao[0] * wv[0][:, None]
            for k in range(4):
                out[k] = ao[k].conj().T.dot(aow)
        else:  # GGA
            wv = wv.copy()
            wv[0] *= .5
            aow = cp.einsum("cgi,cg->gi", ao[:4], wv[:4])
            tmp = ao[0].conj().T.dot(aow)
            out[0] = tmp + tmp.conj().T
            out[1:] = pbc_krks_grad._gga_grad_sum_(ao, wv)
        out[1:] *= -1
        return out

    for ao_ks, weight, coords in ni.block_loop(cell, grids, ao_deriv, kpts,
                                               sort_grids=True):
        ao = ao_ks[0]
        ao0 = ao_ks[:, 0] if xctype == "LDA" else ao_ks[:, :4]
        rho0 = cp.asarray(ni.eval_rho(cell, ao0, dm0, xctype=xctype, hermi=1))
        if rho0.ndim == 1:
            rho0 = rho0[None]
        vxc, _fxc = ni.eval_xc_eff(xc_code, rho0, deriv=2, xctype=xctype)[1:3]
        rho1 = cp.asarray(ni.eval_rho(cell, ao0, dmvo[None],
                                      xctype=xctype, hermi=1)) * 2.0
        if rho1.ndim == 1:
            rho1 = rho1[None]
        wv_f = cp.einsum("yg,xyg->xg", rho1, _fxc) * weight
        f1vo += _build(ao, wv_f)
        if with_vxc:
            v1ao += _build(ao, vxc * weight)
    if is_gamma:
        f1vo = f1vo.real
        if with_vxc:
            v1ao = v1ao.real
    return f1vo, v1ao
