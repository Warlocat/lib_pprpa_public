"""Batched Becke-grid XC nuclear gradient for periodic pp-RPA gradients.

Adapted from ``gpu4pyscf.pbc.grad.krks`` under the Apache License 2.0.

``get_vxc_full_response_multi`` below is a batched counterpart of the released
``gpu4pyscf.pbc.grad.krks.get_vxc_full_response``.  It lives here rather than
being imported from gpu4pyscf because the batched form is not part of any
released gpu4pyscf, whereas every routine it calls is.  The per-grid-point
arithmetic is copied from upstream unchanged, so each returned gradient equals
the corresponding single-density ``get_vxc_full_response`` result.

Why lib_pprpa needs a batched form
----------------------------------
``grad_utils_gpu_pbc.relaxed_xc_gradient`` differentiates the ground-state XC
gradient along the relaxed density by central differences, so it needs the full
response for three densities: ``D``, ``D + eps*P`` and ``D - eps*P``.  With the
upstream routine that is six full grid passes, five of which repeat work that
does not depend on the density at all.

Differences from upstream ``get_vxc_full_response``
---------------------------------------------------
1. Signature.  Upstream takes a single ``dm_kpts`` and returns one
   ``(natm, 3)`` array.  This takes a sequence ``dms`` and returns a list of
   ``(natm, 3)`` arrays in the same order.

2. One grid pass instead of two.  Upstream loops over the grid twice: once at
   ``ao_deriv`` to accumulate the Becke weight-derivative term, and again at
   ``ao_deriv + 1`` for the orbital response and the density part of the grid
   response.  Here there is a single loop at ``ao_deriv + 1``.  The components
   ``ao_ks[:, :4]`` (``ao_ks[:, 0]`` for LDA) already present in that block are
   exactly what upstream's first pass consumed, so the weight term is
   accumulated in the same loop.  This halves the number of passes even for a
   single density.

3. One ``eval_xc_eff`` call instead of two.  Upstream evaluates the functional
   with ``deriv=0`` in the first pass to obtain ``exc`` and again with
   ``deriv=1`` in the second to obtain ``vxc``.  Here a single ``deriv=1`` call
   returns both.

4. The Becke weight derivative does not depend on the density, so
   ``get_becke_weight_derivative`` is evaluated once per grid block and shared
   across every density in ``dms``, instead of once per density.

5. Upstream's separate ``GGA`` and ``MGGA`` branches are merged into one, with
   the ``wv[4]`` tau scaling and the ``_tau_grad_dot_`` term applied only for
   MGGA.  The arithmetic in each case is upstream's.

Nothing else differs.  ``BeckeGrids.block_loop`` partitions the grid with a
fixed block size, so the blocks - and hence the reduction order of every
accumulated term - are identical to upstream's second pass.
"""

# Copyright 2025 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cupy as cp
from gpu4pyscf.pbc.dft import BeckeGrids
from gpu4pyscf.pbc.dft.gen_grid import get_becke_weight_derivative
from gpu4pyscf.pbc.grad import krhf as krhf_grad
from gpu4pyscf.pbc.grad.krks import _d1_dot_, _gga_grad_sum_, _tau_grad_dot_

__all__ = ['get_vxc_full_response_multi']


def get_vxc_full_response_multi(ni, cell, grids, xc_code, dms, kpts, hermi=1):
    '''dExc/dR (with Becke grid response) for several densities in one grid pass.

    dms : sequence of (nkpts, nao, nao) arrays.  Returns a list of
    (natm, 3) numpy arrays in the same order, each equal to
    gpu4pyscf.pbc.grad.krks.get_vxc_full_response(ni, cell, grids, xc_code,
    dm, kpts, hermi).
    '''
    assert isinstance(grids, BeckeGrids)
    assert hermi == 1, "Only hermitian dm_kpts is supported"
    dms = [cp.asarray(dm) for dm in dms]
    for dm in dms:
        assert dm.ndim == 3
    ndm = len(dms)
    xctype = ni._xc_type(xc_code)
    nao, natm, nkpts = cell.nao, cell.natm, len(kpts)
    ngrids = grids.coords.shape[0]
    if xctype in ('LDA',):
        ao_deriv = 0
    elif xctype in ('GGA', 'MGGA'):
        ao_deriv = 1
    else:
        raise NotImplementedError(f"Unrecognized xctype = {xctype}")

    de_w = [cp.zeros((natm, 3), dtype=cp.float64) for _ in range(ndm)]
    de_rho = [cp.zeros((natm, 3), dtype=dm.dtype) for dm in dms]
    dvmat = [cp.zeros((nkpts, 3, nao, nao), dtype=dm.dtype) for dm in dms]

    g1 = 0
    # Difference 2: a single pass at ao_deriv+1.  ao_ks[:, :4] supplies
    # everything upstream's separate first pass needed, and the higher
    # components feed the orbital response.
    for ao_ks, weight, coords in ni.block_loop(cell, grids, ao_deriv + 1, kpts):
        g0, g1 = g1, g1 + weight.size
        i_atom = int(grids.supatm_to_atm_idx[grids.supatm_idx[g0]])
        assert cp.max(cp.abs(grids.supatm_to_atm_idx[grids.supatm_idx[g0:g1]] - i_atom)) == 0
        # Difference 4: density-independent, so computed once per block and
        # shared by every dm rather than once per dm.
        dweight_dA = get_becke_weight_derivative(grids, natm, (g0, g1))

        for n, dm in enumerate(dms):
            if xctype == 'LDA':
                rho = ni.eval_rho(cell, ao_ks[:, 0], dm, xctype=xctype, hermi=hermi)
                # Difference 3: one deriv=1 call returns exc and vxc together.
                exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=0)[:2]
                rho0 = rho if rho.ndim == 1 else rho[0]
                de_w[n] += cp.einsum("Adg->Ad", dweight_dA * (rho0 * exc))
                wv = weight * vxc[0]
                aow = cp.einsum('kpi,p->kpi', ao_ks[:, 0], wv)
                for kn in range(nkpts):
                    vtmp = _d1_dot_(ao_ks[kn, 1:4], aow[kn])
                    dvmat[n][kn] += vtmp
                    de_rho[n][i_atom] += cp.einsum('xij,ji->x', vtmp, dm[kn]) * 2
            else:  # Difference 5: upstream's GGA and MGGA branches merged
                rho = ni.eval_rho(cell, ao_ks[:, :4], dm, xctype=xctype, hermi=hermi)
                exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=0)[:2]
                rho0 = rho[0]
                de_w[n] += cp.einsum("Adg->Ad", dweight_dA * (rho0 * exc))
                wv = weight * vxc
                wv[0] *= .5
                if xctype == 'MGGA':
                    wv[4] *= .5  # for the factor 1/2 in tau
                for kn in range(nkpts):
                    vtmp = _gga_grad_sum_(ao_ks[kn], wv[:4])
                    if xctype == 'MGGA':
                        vtmp = vtmp + _tau_grad_dot_(ao_ks[kn], wv[4])
                    dvmat[n][kn] += vtmp
                    de_rho[n][i_atom] += cp.einsum('xij,ji->x', vtmp, dm[kn]) * 2
        dweight_dA = None
    assert g1 == ngrids

    out = []
    for n, dm in enumerate(dms):
        exc = de_rho[n].get().real
        exc -= krhf_grad.contract_h1e_dm(cell, dvmat[n], dm, hermi=1)
        exc *= 1.0 / nkpts
        exc += de_w[n].get()
        out.append(exc)
    return out
