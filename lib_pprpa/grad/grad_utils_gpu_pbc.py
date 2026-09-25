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


def relaxed_xc_gradient(mf, xc_code, dmvo, dm0=None, kpts=None,
                        density_epsilon=None):
    """Nuclear XC gradient for a reference plus relaxed density.

    For a reference AO density ``D`` and relaxed density ``P``, the required
    ppRPA XC term is the nuclear derivative of

    ``E_xc[D] + dE_xc[D]/dD[P]``.

    Uniform grids use the existing analytical AO-skeleton contraction.  For
    atom-centered periodic Becke grids, the derivative of the complete
    ground-state XC gradient is taken along ``P``.  This retains both Becke
    weight and grid-coordinate response without a numerical nuclear gradient.
    The one-dimensional density derivative is central and can be converged
    independently through ``density_epsilon``.
    """
    from gpu4pyscf.pbc.dft import BeckeGrids
    from gpu4pyscf.pbc.grad import krks as pbc_krks_grad

    from lib_pprpa.grad.pbc_xc_response import get_vxc_full_response_multi

    ni = mf._numint
    cell = mf.mol
    grids = mf.grids
    if grids.coords is None:
        grids.build()
    if kpts is None:
        kpts = np.asarray(mf.kpts).reshape(-1, 3)
    else:
        kpts = np.asarray(kpts).reshape(-1, 3)
    if len(kpts) != 1:
        raise NotImplementedError(
            "relaxed_xc_gradient currently supports Gamma-point densities only"
        )
    if dm0 is None:
        dm0 = mf.make_rdm1()
    dm0 = cp.asarray(dm0)
    if dm0.ndim == 2:
        dm0 = dm0[None]
    dmvo = cp.asarray(dmvo)
    if dm0.ndim != 3 or dm0.shape[0] != 1 or dmvo.ndim != 2:
        raise ValueError("expected one reference k-point density and one AO response")
    dmvo = (dmvo + dmvo.T.conj()) * 0.5

    if isinstance(grids, BeckeGrids):
        if density_epsilon is None:
            density_epsilon = getattr(
                mf, "pprpa_xc_density_epsilon", 1e-3)
        density_epsilon = float(density_epsilon)
        if not np.isfinite(density_epsilon) or density_epsilon <= 0:
            raise ValueError("density_epsilon must be finite and positive")
        p = dmvo[None]

        def full_gradient(dm):
            value = pbc_krks_grad.get_vxc_full_response(
                ni, cell, grids, xc_code, dm, kpts, hermi=1)
            return cp.asnumpy(value) if isinstance(value, cp.ndarray) else np.asarray(value)

        # One grid pass for all three densities instead of three passes each
        # re-evaluating the same AOs and Becke weights.  The batched routine is
        # vendored in pbc_xc_response because it is not part of any released
        # gpu4pyscf; see that module for how it differs from the upstream
        # single-density krks.get_vxc_full_response it is adapted from.
        g0, gp, gm = (np.asarray(g) for g in
                      get_vxc_full_response_multi(
                          ni, cell, grids, xc_code,
                          [dm0, dm0 + density_epsilon * p, dm0 - density_epsilon * p],
                          kpts, hermi=1))
        return g0 + (gp - gm) / (2 * density_epsilon)

    f1vo, v1ao = _contract_xc_kernel(
        mf, xc_code, dmvo, dm0=dm0, with_vxc=True, kpts=kpts)
    total = dm0[0] + dmvo
    reference = dm0[0]
    de = cp.zeros((cell.natm, 3), dtype=cp.float64)
    aoslices = cell.aoslice_by_atom()
    for ia in range(cell.natm):
        p0, p1 = aoslices[ia, 2:]
        de[ia] += cp.einsum(
            "xij,ij->x", v1ao[1:, p0:p1], total[p0:p1]).real * 2
        de[ia] += cp.einsum(
            "xij,ij->x", f1vo[1:, p0:p1], reference[p0:p1]).real
    return cp.asnumpy(de)


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
