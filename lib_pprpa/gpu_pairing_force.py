'''Pairing-exchange force of the Gamma-point pp-RPA gradient for X = L R^T.

The CPU reference (``grad.pprpa_gamma``) forms vk[x]_il = -sum_jk (d_x i j|k l) X_jk
and adds 2 sum_{i in A} sum_l vk[x]_il X_il.  The amplitude density
X = C_v x C_v^T + C_o y C_o^T has rank at most nocc + nvir, X = L R^T with
L = [C_v x, C_o y] and R = [C_v, C_o].  With the factors on the uniform grid,

    de[A, x]   = -2 sum_{i in A} sum_m L_im G[x, i, m]
    G[x, i, m] = (vol/ngrid) sum_g d_x ao_i(g) W_m(g)
    W_m        = sum_n phiL_n W[phiR_n phiR_m]          (bare Coulomb, G=0 dropped)

one FFT pass over the r(r+1)/2 codensity pairs (the potential is symmetric in
n, m) and one chunked pass over the gradient AOs.  Exact for a general X, so
singlet and triplet amplitudes need no symmetric/antisymmetric split.  The
sign: ``eval_ao(deriv=1)`` differentiates with respect to the electron
coordinate; the nuclear derivative flips it.
'''

import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.lib import logger
from gpu4pyscf.pbc import tools as gtools

from lib_pprpa.gpu_coulomb import ao_loop, codensity, coulomb_potential, half_kernel, mo_on_grid
from lib_pprpa.gpu_mem import free_bytes, max_fft_batch

__all__ = ['pairing_k_force_lowrank']

# Bytes per (codensity, grid point) in one FFT batch: rho, half spectrum, real
# potential, the cuFFT work area and the scratch for the W updates.
_FFT_BYTES = 48
_FFT_BUDGET_FRAC = 0.3


def pairing_k_force_lowrank(cell, mesh, L, R, verbose=None):
    '''Pairing-exchange contribution to dE/dR, NumPy (natm, 3), for X = L R^T.

    Args:
        L, R : (nao, r) NumPy factors; zero columns of L are dropped.
    '''
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()
    L = np.asarray(L, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    keep = np.abs(L).max(axis=0) > 0
    L, R = L[:, keep], R[:, keep]
    nao, r = L.shape
    mesh = np.asarray(mesh, dtype=int)
    ngrid = int(np.prod(mesh))
    if r == 0:
        return np.zeros((cell.natm, 3))

    w_half = half_kernel(gtools.get_coulG(cell, mesh=mesh), mesh)
    phiL, phiR = mo_on_grid(cell, [L, R], mesh)
    # pairs (n, m) with m <= n, row n contiguous
    n_idx, m_idx = np.tril_indices(r)
    npair = len(n_idx)
    n_dev, m_dev = cp.asarray(n_idx, dtype=np.int32), cp.asarray(m_idx, dtype=np.int32)
    blk = int(_FFT_BUDGET_FRAC * free_bytes()) // (_FFT_BYTES * ngrid)
    blk = max(1, min(npair, blk, max_fft_batch(ngrid, mesh)))
    scratch = cp.empty((blk, ngrid))
    W = cp.zeros((r, ngrid))
    for p0, p1 in lib.prange(0, npair, blk):
        V = coulomb_potential(codensity(phiR, phiR, n_dev, m_dev, p0, p1), mesh, w_half)
        rows, starts = np.unique(n_idx[p0:p1], return_index=True)
        for n, s0 in zip(rows, starts):
            s1 = s0 + 1
            while s1 < p1 - p0 and n_idx[p0 + s1] == n:
                s1 += 1
            m0, m1 = int(m_idx[p0 + s0]), int(m_idx[p0 + s1 - 1]) + 1
            Vs = V[s0:s1]                                   # pairs (n, m0..m1-1)
            cp.multiply(phiL[n][None], Vs, out=scratch[:s1 - s0])
            W[m0:m1] += scratch[:s1 - s0]                   # W_m += phiL_n V_nm
            k = m1 - m0 - (1 if m1 - 1 == n else 0)         # the diagonal pair once
            if k > 0:
                cp.multiply(phiL[m0:m0 + k], Vs[:k], out=scratch[:k])
                W[n] += scratch[:k].sum(axis=0)             # W_n += phiL_m V_nm
    V = scratch = phiR = None
    t0 = log.timer(f'pairing force: {npair} codensity pairs (rank {r})', *t0)

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
