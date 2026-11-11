"""Density-contracted one-electron nuclear derivative at the Gamma point.

``pyscf.pbc.grad`` builds the one-electron derivative one atom at a time:
``hcore_generator(ia)`` evaluates every AO on the full uniform mesh and runs
three inverse FFTs to form a ``(3, nao, nao)`` matrix, which the caller then
traces against the density.  That AO pass is identical for every atom, so the
cost grows as ``natm`` times the mesh.

The trace only needs the density on the mesh,

    de[A, x] = sum_g dV_loc,A^x(g) rho(g) + h1-slice terms,

so this module puts the density on the mesh once and turns each atom into a
reciprocal-space dot product by Parseval's theorem.  The remaining h1 slice
terms, the AO-centre derivative of the kinetic and local pseudopotential
integrals, are cheap slices of the derivative integrals of ``get_hcore``.

The local pseudopotential terms come from gpu4pyscf's reciprocal-space kernels
(``multigrid.eval_vpplocG`` and ``multigrid_v3._pploc_derivatives``), which never form
``(natm, ngrids)`` arrays; for all-electron cells the nuclear Coulomb sum is blocked
over plane waves instead.  The total local potential feeds the AO-centre terms, so
the derivative integrals are built here rather than by
``gpu4pyscf.pbc.grad.krhf.get_hcore``.

Gamma point only.
"""

import numpy as np
import cupy as cp
from lib_pprpa.gpu_mem import free_bytes, plan_budget
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.pbc import tools
from gpu4pyscf.pbc.dft import numint as pbc_numint
from gpu4pyscf.pbc.dft.gen_grid import UniformGrids
from gpu4pyscf.pbc.dft.multigrid import eval_vpplocG
from gpu4pyscf.pbc.dft.multigrid_v3 import _get_Gv_bases, _pploc_derivatives
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.tools.pbc import get_coulG


def hcore_deriv_contracted(cell, kpts, dm):
    '''sum_ia einsum('kxij,kji->x', hcore_deriv(ia), dm) for all atoms at once.

    dm : (nao, nao) real symmetric AO density at Gamma.
    Returns (natm, 3) numpy array identical to the per-atom loop.
    '''
    kpts = np.asarray(kpts).reshape(-1, 3)
    if len(kpts) != 1 or abs(kpts).max() > 0:
        raise NotImplementedError("hcore_deriv_contracted is Gamma-point only")
    natm = cell.natm
    mesh = cell.mesh
    ngrids = int(np.prod(mesh))
    dm = cp.asarray(dm)
    dm_k = dm[None]

    # --- density on the uniform mesh, evaluated once ---------------------
    ni = pbc_numint.KNumInt()
    grids = UniformGrids(cell)
    rho = cp.empty(ngrids, dtype=cp.float64)
    g0 = g1 = 0
    for ao_ks, weight, coords in ni.block_loop(cell, grids, 0, kpts, sort_grids=True):
        g0, g1 = g1, g1 + len(weight)
        rho[g0:g1] = ni.eval_rho(cell, ao_ks, dm_k, xctype='LDA', hermi=1).real
    assert g1 == ngrids
    # block_loop(sort_grids=True) delivered points in argsort order; undo it
    rho_unsorted = cp.empty_like(rho)
    rho_unsorted[cp.asarray(grids.argsort())] = rho
    rho_G = tools.fft(rho_unsorted, mesh)            # unnormalised, as numpy

    # --- per-atom reciprocal-space dot: sum_g vloc_R rho = (1/N) Re sum_G vloc_g^* rho_G
    if cell._pseudo:
        # gpu4pyscf's local-pseudopotential kernels: one Gaussian factor per atom from separable
        # structure factors on the device.  The gradient takes rho(G) scaled to the cell
        # volume and returns the (natm, 3) atom gradient and the strain; vG, G = 0 included,
        # is the total local potential of get_hcore.
        Gv_bases = _get_Gv_bases(mesh, cell.reciprocal_vectors())
        de = _pploc_derivatives(cell, rho_G * (cell.vol / ngrids), Gv_bases)[0]
        vG = eval_vpplocG(cell, mesh)
    else:
        de, vG = _nuclear_coulomb(cell, mesh, rho_G)

    # --- AO-centre (h1 slice) terms ---------------------------------------
    # (nk, 3, nao, nao) derivative integrals, as gpu4pyscf.pbc.grad.krhf.get_hcore
    h1 = int1e.int1e_ipkin(cell, kpts)
    vlocR = tools.ifft(vG, mesh).real[grids.argsort()]
    g0 = g1 = 0
    for ao_ks, weight, coords in ni.block_loop(cell, grids, 1, kpts, sort_grids=True):
        ao_ks = ao_ks.transpose(0, 1, 3, 2)           # (nk, comp, nao, blk)
        g0, g1 = g1, g1 + len(weight)
        aow = ao_ks[:, 0] * vlocR[g0:g1]
        contract('kxig,kjg->kxij', ao_ks[:, 1:].conj(), aow, beta=1, out=h1)
    aoslices = cell.aoslice_by_atom()
    for ia in range(natm):
        p0, p1 = aoslices[ia, 2:]
        # hcore[:,:,p0:p1] -= h1[:,:,p0:p1]  -> -sum_{i in A} h1[x,i,j] dm[j,i]
        de[ia] -= cp.einsum('xij,ji->x', h1[0, :, p0:p1, :], dm[:, p0:p1]).real
        # hcore[:,:,:,p0:p1] -= h1[:,:,p0:p1].T.conj() -> -sum_{j in A} conj(h1[x,j,i]) dm[j,i]
        de[ia] -= cp.einsum('xji,ji->x', h1[0, :, p0:p1, :].conj(), dm[p0:p1, :]).real
    return cp.asnumpy(de)


def _nuclear_coulomb(cell, mesh, rho_G):
    '''All-electron analogue of the local-pseudopotential terms: the per-atom gradient
    (natm, 3) and the total nuclear potential -sum_a Z_a SI_a coulG in G space.

    Blocked over plane waves: the unblocked form holds several (natm, nG) complex arrays at
    once.  Blocking changes no arithmetic and no reduction order per element.
    '''
    natm = cell.natm
    ngrids = int(np.prod(mesh))
    Gv = cp.asarray(cell.get_Gv(mesh))                 # (nG, 3)
    atom_coords = cp.asarray(cell.atom_coords())       # (natm, 3)
    Z = cp.asarray(cell.atom_charges(), dtype=cp.float64)
    # get_coulG zeroes the first G it is given as G = 0, so it takes the whole mesh
    coulG = get_coulG(cell, mesh=mesh, Gv=Gv)
    vG = cp.empty(ngrids, dtype=cp.complex128)
    # at most 6 (natm, blk) complex arrays live at once (SI_b, coef_b and the
    # temporaries of its product with rho_G)
    blk = int(max(1, min(ngrids, plan_budget(free_bytes()) // max(1, natm * 16 * 6))))
    de = cp.zeros((natm, 3))
    for g0 in range(0, ngrids, blk):
        g1 = min(g0 + blk, ngrids)
        Gv_b = Gv[g0:g1]
        # SI[a,g] = exp(-i G_g . R_a), same convention as aft.get_SI
        SI_b = cp.exp(-1j * (atom_coords @ Gv_b.T))   # (natm, blk)
        coef_b = 1j * Z[:, None] * SI_b * coulG[g0:g1]
        # vloc_g[A,x,G] = Gv[G,x] * coef[A,G]
        de += cp.einsum('gx,ag->ax', Gv_b, (coef_b.conj() * rho_G[g0:g1])).real
        vG[g0:g1] = 1j * coef_b.sum(axis=0)            # -sum_a Z SI coulG
        SI_b = coef_b = None
    return de / ngrids, vG
