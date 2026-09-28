'''Shared pieces of the batched-FFT Coulomb kernels on the uniform grid (Gamma point).

For a batch of real codensities rho_P(g) = A[a_P](g) B[b_P](g) the kernels form

    vR = ifft(w(G) fft(rho)),   w = (vol / ngrid) 4 pi / |G|^2

and contract vR with other codensities.  The transform runs real-to-complex on
the half mesh; the kernel is symmetrised, (w(G) + w(-G)) / 2, because the C2R
transform assumes a Hermitian spectrum while pyscf's ``ifft(...).real`` only
ever sees the even part of w.  On the Nyquist planes of an even mesh in a
non-orthogonal cell w is not even, and the plain half kernel is off by 1e-8.
'''

import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
from pyscf import lib
from gpu4pyscf.pbc.dft import numint as gnumint

from lib_pprpa.gpu_mem import free_bytes

__all__ = ['mo_on_grid', 'codensity', 'half_kernel', 'coulomb_potential']

_KPTS0 = np.zeros((1, 3))
_GRID_BLK_MIN = 4096


def mo_on_grid(cell, mo_coeffs, mesh, blksize=None):
    '''MO values on the uniform grid, one (nmo, ngrid) CuPy array per coefficient set.

    The AOs are evaluated in grid chunks, so the (ngrid, nao) AO array is never
    resident and one evaluation serves every coefficient set.
    '''
    coords = cell.gen_uniform_grids(mesh)
    ngrid = len(coords)
    mos = [cp.asarray(m, dtype=np.float64) for m in mo_coeffs]
    outs = [cp.empty((m.shape[1], ngrid)) for m in mos]
    if blksize is None:
        # the (g, nao) AO block, a like-sized allowance for its evaluation, one
        # output column per MO
        per_point = 16 * cell.nao + 8 * sum(m.shape[1] for m in mos)
        blksize = int(0.4 * free_bytes()) // per_point
    blksize = max(1, min(ngrid, max(_GRID_BLK_MIN, blksize)))
    for g0, g1 in lib.prange(0, ngrid, blksize):
        ao = gnumint.eval_ao_kpts(cell, coords[g0:g1], kpts=_KPTS0, deriv=0)[0]
        aoT = cp.asarray(ao).T                  # (nao, g), real at Gamma
        for out, m in zip(outs, mos):
            out[:, g0:g1] = m.T @ aoT
    return outs


# rho[P, g] = A[aidx[P], g] * B[bidx[P], g] in one pass: two reads and one
# write per element, no gathered temporaries.
_CODENSITY = cp.ElementwiseKernel(
    'raw float64 A, raw float64 B, raw int32 aidx, raw int32 bidx, int64 ngrid, int64 P0',
    'float64 rho',
    '''
    const long long row = i / ngrid;
    const long long g = i - row * ngrid;
    const long long P = P0 + row;
    rho = A[(long long)aidx[P] * ngrid + g] * B[(long long)bidx[P] * ngrid + g];
    ''',
    'pprpa_codensity')


def codensity(A, B, aidx, bidx, P0, P1, out=None):
    '''Codensities of pairs [P0, P1) as (P1-P0, ngrid); A, B are (n, ngrid) grids
    and aidx, bidx int32 device arrays mapping a pair to its rows.'''
    if out is None:
        out = cp.empty((P1 - P0, A.shape[1]))
    _CODENSITY(A, B, aidx, bidx, A.shape[1], P0, out)
    return out


def half_kernel(w_flat, mesh):
    '''The kernel w(G) symmetrised and cut to the R2C half mesh (nx, ny, nz//2 + 1).'''
    nx, ny, nz = (int(m) for m in mesh)
    w = cp.asarray(w_flat).reshape(nx, ny, nz)
    ix, iy, iz = (cp.asarray((-np.arange(n)) % n) for n in (nx, ny, nz))   # G -> -G
    ws = 0.5 * (w + w[ix][:, iy][:, :, iz])
    return cp.ascontiguousarray(ws[:, :, :nz // 2 + 1])


def coulomb_potential(rho, mesh, w_half):
    '''ifft(fft(rho) * w) for a batch of real functions, (f, ngrid) -> (f, ngrid),
    through the real-to-complex half mesh with the symmetrised kernel.'''
    nx, ny, nz = (int(m) for m in mesh)
    vG = cufft.rfftn(rho.reshape(-1, nx, ny, nz), axes=(1, 2, 3))
    vG *= w_half
    vR = cufft.irfftn(vG, s=(nx, ny, nz), axes=(1, 2, 3), overwrite_x=True)
    return vR.reshape(rho.shape[0], -1)
