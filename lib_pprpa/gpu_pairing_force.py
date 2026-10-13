'''Pairing-exchange force of the Gamma-point pp-RPA gradient for X = C M C^T.

The CPU reference (``grad.pprpa_gamma``) forms vk[x]_il = -sum_jk (d_x i j|k l) X_jk
and adds 2 sum_{i in A} sum_l vk[x]_il X_il.  The amplitude density
X = C_v x C_v^T + C_o y C_o^T is C M C^T with C = [C_v, C_o] and M = diag(x, y),
so X = L C^T with L = C M.  With the orbitals on the uniform grid,

    de[A, x]   = -2 sum_{i in A} sum_m L_im G[x, i, m]
    G[x, i, m] = (vol/ngrid) sum_g d_x ao_i(g) W_m(g)
    W_m        = sum_n phiL_n W[phiC_n phiC_m]          (bare Coulomb, G=0 dropped)

one FFT pass over the r(r+1)/2 codensity pairs (the potential is symmetric in
n, m) and one chunked pass over the gradient AOs.  phiL = M^T phiC is kept on
the device when it fits and otherwise rebuilt from phiC block by block.  Exact
for a general M, so singlet and triplet amplitudes need no symmetric/antisymmetric
split.  The sign: ``eval_ao(deriv=1)`` differentiates with respect to the electron
coordinate; the nuclear derivative flips it.

W is also the ket grid of the exchange of X itself, K[X] C = (vol/ngrid) sum_g ao W^T
with K_pq = sum_jk (pj|kq) X_jk, so one ``pair_potential_lowrank`` serves the relaxed
density's exchange (``pair_exchange``) and the force.  With several
``devices`` the FFT batches of pairs are dispatched over them (``gpu_multi``) and the
partial W are added on the first; the AO passes run on W's device.
'''

import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.lib import logger
from gpu4pyscf.pbc import tools as gtools

from lib_pprpa.gpu_coulomb import ao_loop, codensity, coulomb_potential, half_kernel, mo_on_grid
from lib_pprpa.gpu_mem import FFT_BATCH_MIN, free_bytes, max_fft_batch, plan_budget
from lib_pprpa.gpu_multi import default_devices, each, run

__all__ = ['pairing_k_force_lowrank', 'pair_potential_lowrank', 'pair_exchange']

# Bytes per (codensity, grid point) in one FFT batch: one coulomb_potential call
# peaks at 56.4 (codensities, half spectrum, real output, two cuFFT work areas,
# measured on the 159^3 mesh, see gpu_ao2mo) plus 8 for the scratch of the W updates.
_FFT_BYTES = 64


def _potential(cell, mesh, C, M, keep, store_phil, devices, log):
    '''W (len(keep), ngrid) CuPy on devices[0] for the columns ``keep`` of C, over the pairs
    of kept columns.  The FFT batches of pairs are dispatched over ``devices``; each rank
    accumulates its own W and the partials are added on devices[0].'''
    t0 = log.init_timer()
    r = len(keep)
    ngrid = int(np.prod(mesh))
    devices = default_devices(devices)
    nshare = max(devices.count(d) for d in devices)
    Mk = M[:, keep]                                         # (nC, r)
    # pairs (n, m) with m <= n, row n contiguous; codensities index phiC
    n_idx, m_idx = np.tril_indices(r)
    npair = len(n_idx)
    with cp.cuda.Device(devices[0]):
        # planned before setup: each rank on a device holds phiC and its W
        free = free_bytes() - nshare * (C.shape[1] + r) * ngrid * 8
    phil_bytes = nshare * r * ngrid * 8
    if store_phil is None:
        store_phil = phil_bytes + FFT_BATCH_MIN * _FFT_BYTES * ngrid <= plan_budget(free)
    if store_phil:
        free -= phil_bytes
        fft_bytes = _FFT_BYTES
    else:
        fft_bytes = _FFT_BYTES + 8                           # rebuilt phiL rows
    # the rebuilt rows take one more row than the batch (rows_buf)
    blk = (plan_budget(free) - (0 if store_phil else 8 * ngrid)) // (fft_bytes * ngrid)
    blk = max(1, min(npair, blk, max_fft_batch(ngrid)))
    log.debug('pairing potential: rank %d, store_phil %s, %d pairs per FFT batch, %d device(s)',
              r, store_phil, blk, len(devices))

    def setup(rank):
        phiC, = mo_on_grid(cell, [C], mesh)
        Mk_d = cp.asarray(Mk)
        return dict(phiC=phiC, Mk=Mk_d, phiL=Mk_d.T @ phiC if store_phil else None,
                    n=cp.asarray(keep[n_idx], dtype=np.int32),
                    m=cp.asarray(keep[m_idx], dtype=np.int32),
                    w_half=half_kernel(gtools.get_coulG(cell, mesh=mesh), mesh),
                    W=cp.zeros((r, ngrid)))

    def work(rank, st, task):
        p0, p1 = task
        phiC, phiL, W = st['phiC'], st['phiL'], st['W']
        V = coulomb_potential(codensity(phiC, phiC, st['n'], st['m'], p0, p1), mesh,
                              st['w_half'])
        scratch = cp.empty((p1 - p0, ngrid))
        rows_buf = None if store_phil else cp.empty((p1 - p0 + 1, ngrid))
        rows, starts = np.unique(n_idx[p0:p1], return_index=True)
        for n, s0 in zip(rows, starts):
            s1 = s0 + 1
            while s1 < p1 - p0 and n_idx[p0 + s1] == n:
                s1 += 1
            m0, m1 = int(m_idx[p0 + s0]), int(m_idx[p0 + s1 - 1]) + 1
            Vs = V[s0:s1]                                   # pairs (n, m0..m1-1)
            k = m1 - m0 - (1 if m1 - 1 == n else 0)         # the diagonal pair once
            if store_phil:
                phiL_n, phiL_m = phiL[n], phiL[m0:m0 + k]
            else:
                cols = np.append(np.arange(m0, m0 + k), n)
                cp.matmul(st['Mk'][:, cols].T, phiC, out=rows_buf[:k + 1])
                phiL_n, phiL_m = rows_buf[k], rows_buf[:k]
            cp.multiply(phiL_n[None], Vs, out=scratch[:s1 - s0])
            W[m0:m1] += scratch[:s1 - s0]                   # W_m += phiL_n V_nm
            if k > 0:
                cp.multiply(phiL_m, Vs[:k], out=scratch[:k])
                W[n] += scratch[:k].sum(axis=0)             # W_n += phiL_m V_nm

    _, states = run(devices, setup, work, list(lib.prange(0, npair, blk)))
    parts = [st['W'] for st in states]
    states = None
    W = parts[0]
    with cp.cuda.Device(devices[0]):
        for rank in range(1, len(parts)):
            for n in range(r):                              # row by row from the rank's device
                W[n] += cp.asarray(parts[rank][n])
            parts[rank] = None
    each(devices, lambda rank: cp.get_default_memory_pool().free_all_blocks())
    log.timer(f'pairing potential: {npair} codensity pairs (rank {r}, '
              f'{len(devices)} device(s))', *t0)
    return W


def pair_potential_lowrank(cell, mesh, C, M, store_phil=None, devices=None, verbose=None):
    '''W_m = sum_n phiL_n W[phiC_n phiC_m] for every column m of C, CuPy (nC, ngrid) on the
    first of ``devices`` (default: current), for ``pairing_k_force_lowrank(..., W=W)`` and
    ``pair_exchange``.'''
    log = logger.new_logger(cell, verbose)
    C = np.asarray(C, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    mesh = np.asarray(mesh, dtype=int)
    return _potential(cell, mesh, C, M, np.arange(C.shape[1]), store_phil, devices, log)


def pair_exchange(cell, mesh, W):
    '''K[C M C^T] @ C, NumPy (nao, nC), from the ``pair_potential_lowrank`` grids W of C and M
    (K_pq = sum_jk (pj|kq) D_jk, gpu4pyscf's fft_jk.get_k with hermi=0).'''
    mesh = np.asarray(mesh, dtype=int)
    ngrid = int(np.prod(mesh))
    with W.device:
        K = cp.zeros((cell.nao, W.shape[0]))
        for g0, g1, aoT in ao_loop(cell, mesh, extra_per_point=8 * W.shape[0]):
            K += aoT @ W[:, g0:g1].T
        return (K * (cell.vol / ngrid)).get()


def pairing_k_force_lowrank(cell, mesh, C, M, W=None, store_phil=None, devices=None,
                            verbose=None):
    '''Pairing-exchange contribution to dE/dR, NumPy (natm, 3), for X = C M C^T.

    Args:
        C : (nao, r) NumPy orbitals.
        M : (r, r) NumPy mixing matrix; its zero columns are dropped.

    Kwargs:
        W : ``pair_potential_lowrank(cell, mesh, C, M)``, when already built.
        store_phil : keep phiL = M^T phiC on the device (True) or rebuild its rows
            from phiC per block (False).  None keeps it when an FFT batch of
            gpu_mem.FFT_BATCH_MIN codensities still fits beside it.
        devices : CUDA device ids the FFT pass is dispatched over when W is built here
            (default: current).
    '''
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()
    C = np.asarray(C, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    nao = C.shape[0]
    mesh = np.asarray(mesh, dtype=int)
    ngrid = int(np.prod(mesh))
    if W is None:
        keep = np.flatnonzero(np.abs(M).max(axis=0) > 0)
        if len(keep) == 0:
            return np.zeros((cell.natm, 3))
        W = _potential(cell, mesh, C, M, keep, store_phil, devices, log)
    else:
        keep = np.arange(C.shape[1])
    r = len(keep)
    L = C @ M[:, keep]

    with W.device:
        G = cp.zeros((3, nao, r))
        for g0, g1, ao in ao_loop(cell, mesh, deriv=1, extra_per_point=8 * r):
            Wc = cp.ascontiguousarray(W[:, g0:g1])
            for x in range(3):
                G[x] += ao[1 + x].T @ Wc.T
        G *= cell.vol / ngrid
        t = cp.einsum('xim,im->xi', G, cp.asarray(L)).get()
    de = np.zeros((cell.natm, 3))
    for ia, (_, _, p0, p1) in enumerate(cell.aoslice_by_atom()):
        de[ia] = -2.0 * t[:, p0:p1].sum(axis=1)
    log.timer('pairing force: gradient-AO pass', *t0)
    return de
