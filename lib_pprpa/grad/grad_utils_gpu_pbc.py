"""GPU (cupy) PBC XC-kernel helpers for ppRPA Gamma-point gradients.

gpu4pyscf leaves nr_rks_fxc / _contract_xc_kernel NotImplemented for PBC.
These implement them for the Gamma point, RKS, LDA/GGA, mirroring the CPU
pyscf.pbc.dft.numint routines (validated bit-for-bit against them).

gpu4pyscf pbc AO layout from KNumInt.block_loop is (ngrid, nao) for LDA and
(ncomp, ngrid, nao) for GGA; eval_rho returns (ngrid,) / (nvar, ngrid).
"""
import numpy as np
import cupy as cp


def nr_rks_fxc(ni, cell, grids, xc_code, dm0, dms, kpts=None, hermi=1, fxc=None):
    """Contract the RKS XC kernel with perturbing density matrices (Gamma).

    Mirrors pyscf.pbc.dft.numint.nr_rks_fxc for real orbitals (v_hermi=1):
    the response operator is symmetrized.

    Args:
        ni  : gpu4pyscf.pbc.dft.numint.KNumInt
        dm0 : ground-state DM, (nao,nao) or (1,nao,nao)
        dms : perturbing DM(s), (nao,nao) or (nset,nao,nao)
    Returns:
        cupy array, same leading shape as ``dms``.
    """
    if kpts is None:
        kpts = np.zeros((1, 3))
    kpts = np.asarray(kpts).reshape(-1, 3)
    assert kpts.shape[0] == 1, "GPU pbc nr_rks_fxc: Gamma point only"
    is_gamma = abs(kpts).max() < 1e-9

    xctype = ni._xc_type(xc_code)
    if xctype == "LDA":
        ao_deriv = 0
    elif xctype == "GGA":
        ao_deriv = 1
    else:
        raise NotImplementedError(f"GPU pbc nr_rks_fxc for {xc_code} (LDA/GGA only)")

    dm0 = cp.asarray(dm0)
    if dm0.ndim == 2:
        dm0 = dm0[None]
    dms = cp.asarray(dms)
    single = dms.ndim == 2
    if single:
        dms = dms[None]
    nset = dms.shape[0]
    nao = dms.shape[-1]
    vmat = cp.zeros((nset, nao, nao), dtype=cp.complex128)

    for ao_ks, weight, coords in ni.block_loop(cell, grids, ao_deriv, kpts,
                                               sort_grids=True):
        ao = ao_ks[0]  # Gamma: single k-point, (ngrid,nao) or (4,ngrid,nao)
        rho0 = cp.asarray(ni.eval_rho(cell, ao_ks, dm0, xctype=xctype, hermi=1))
        if rho0.ndim == 1:
            rho0 = rho0[None]
        _fxc = ni.eval_xc_eff(xc_code, rho0, deriv=2, xctype=xctype)[2]
        for i in range(nset):
            rho1 = cp.asarray(ni.eval_rho(cell, ao_ks, dms[i:i+1],
                                          xctype=xctype, hermi=hermi))
            if rho1.ndim == 1:
                rho1 = rho1[None]
            wv = cp.einsum("xg,xyg->yg", rho1, _fxc) * weight
            if xctype == "LDA":
                aow = ao * wv[0][:, None]
                vmat[i] += ao.conj().T.dot(aow)
            else:  # GGA
                wv = wv.copy()
                wv[0] *= .5
                aow = cp.einsum("cgi,cg->gi", ao[:4], wv[:4])
                vmat[i] += ao[0].conj().T.dot(aow)
    if xctype != "LDA":
        vmat = vmat + vmat.transpose(0, 2, 1).conj()
    if is_gamma:
        vmat = vmat.real
    if single:
        vmat = vmat[0]
    return vmat


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
