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
integrals, are cheap slices of the derivative integrals ``get_hcore`` already
returns.

The plane-wave sum is blocked.  The unblocked form holds several
``(natm, ngrids)`` complex arrays at once, which for a 215-atom cell with
1.7e7 plane waves is about 200 GB and exceeds any single GPU; blocking bounds
peak memory without changing the arithmetic or the per-element reduction
order.

Gamma point only.
"""

import numpy as np
import cupy as cp
from gpu4pyscf.lib.cupy_helper import get_avail_mem
from gpu4pyscf.pbc import tools
from gpu4pyscf.pbc.df.aft import get_SI
from gpu4pyscf.pbc.dft import numint as pbc_numint
from gpu4pyscf.pbc.dft.gen_grid import UniformGrids
from gpu4pyscf.pbc.grad.krhf import get_hcore
from pyscf.pbc.gto.pseudo.pp import get_vlocG
from gpu4pyscf.pbc.tools.pbc import get_coulG


def hcore_deriv_contracted(cell, kpts, dm):
    '''sum_ia einsum('kxij,kji->x', hcore_deriv(ia), dm) for all atoms at once.

    dm : (nao, nao) real symmetric AO density at Gamma.
    Returns (natm, 3) numpy array identical to the per-atom loop.
    '''
    kpts = np.asarray(kpts).reshape(-1, 3)
    if len(kpts) != 1 or abs(kpts).max() > 0:
        raise NotImplementedError("hcore_deriv_contracted is Gamma-point only")
    natm, nao = cell.natm, cell.nao
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
    Gv = cp.asarray(cell.Gv)                          # (nG, 3)
    # Blocked over plane waves.  The unblocked form holds several (natm, nG)
    # complex arrays simultaneously -- for NV215 (215 atoms, 17,373,979 plane
    # waves) each is 60 GB, so it asks for ~200 GB and dies on any single GPU.
    # Blocking changes no arithmetic and no reduction order per element; it
    # only bounds peak memory.  Verified bit-identical on NV63.
    atom_coords = cp.asarray(cell.atom_coords())      # (natm, 3)
    Z = None if cell._pseudo else cp.asarray(cell.atom_charges(), dtype=cp.float64)
    # ~6 (natm, blk) complex temporaries live at once; keep them under a fifth
    # of free device memory.
    bytes_per_g = max(1, natm * 16 * 6)
    blk = int(max(1, min(ngrids, int(get_avail_mem() * 0.2) // bytes_per_g)))
    de = cp.zeros((natm, 3))
    for g0 in range(0, ngrids, blk):
        g1 = min(g0 + blk, ngrids)
        Gv_b = Gv[g0:g1]
        # SI[a,g] = exp(-i G_g . R_a), same convention as aft.get_SI
        SI_b = cp.exp(-1j * (atom_coords @ Gv_b.T))   # (natm, blk)
        if cell._pseudo:
            vlocG_b = cp.asarray(get_vlocG(cell, cp.asnumpy(Gv_b)))
            coef_b = 1j * SI_b * vlocG_b
        else:
            coulG_b = get_coulG(cell, mesh=mesh, Gv=Gv_b)
            coef_b = 1j * Z[:, None] * SI_b * coulG_b
        # vloc_g[A,x,G] = Gv[G,x] * coef[A,G]
        de += cp.einsum('gx,ag->ax', Gv_b, (coef_b.conj() * rho_G[g0:g1])).real
        SI_b = coef_b = None
    de /= ngrids

    # --- AO-centre (h1 slice) terms ---------------------------------------
    h1 = get_hcore(cell, kpts)                        # (nk, 3, nao, nao) derivative ints
    aoslices = cell.aoslice_by_atom()
    for ia in range(natm):
        p0, p1 = aoslices[ia, 2:]
        # hcore[:,:,p0:p1] -= h1[:,:,p0:p1]  -> -sum_{i in A} h1[x,i,j] dm[j,i]
        de[ia] -= cp.einsum('xij,ji->x', h1[0, :, p0:p1, :], dm[:, p0:p1]).real
        # hcore[:,:,:,p0:p1] -= h1[:,:,p0:p1].T.conj() -> -sum_{j in A} conj(h1[x,j,i]) dm[j,i]
        de[ia] -= cp.einsum('xji,ji->x', h1[0, :, p0:p1, :].conj(), dm[p0:p1, :]).real
    return cp.asnumpy(de)
