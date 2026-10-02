'''Gamma-point FFT exchange of low-rank AO densities D = L R^T (pp-RPA relaxed density).

The pp-RPA pair densities X_ao = C_a x C_a^T and Y_ao = C_i y C_i^T have rank at
most the active space, and the relaxed density only uses mo_coeff^T K orbp.
With phiL = L^T ao, phiR = R^T ao and phiK = ket^T ao on the uniform grid,

    (K ket)_pk = sum_jkq (pj|kq) D_jk ket_qk = (vol/ngrid) sum_g ao_p(g) U_k(g),
    U_k = sum_m phiL_m W[phiR_m phiK_k],

the hermi=0 convention of ``gpu4pyscf.pbc.df.fft_jk.get_k``.  This takes
rank x nk transforms in place of the nao x nao codensities of the dense kernel
and never holds the (nao, ngrid) AO grid, which the dense kernel cannot fit at
nao ~ 3000.  Exact for a general D; the bra-to-ket move of the Coulomb operator
changes only the rounding (1e-11 relative against the dense kernel).  With
several ``devices`` the ket rows are dispatched over them (``gpu_multi``).
'''

import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.lib import logger
from gpu4pyscf.pbc import tools as gtools
from gpu4pyscf.lib.cupy_helper import contract

from lib_pprpa.gpu_coulomb import ao_loop, coulomb_potential, half_kernel, mo_on_grid
from lib_pprpa.gpu_mem import free_bytes, max_fft_batch
from lib_pprpa.gpu_multi import default_devices, each, run

__all__ = ['get_k_lowrank', 'pair_get_k_lowrank']

# Bytes per (codensity, grid point) in one FFT batch: rho, half spectrum, real
# potential, the cuFFT work area and the contraction with phiL.
_FFT_BYTES = 48
_FFT_BUDGET_FRAC = 0.3


def _fft_batch(free, ngrid, mesh, r, nk):
    '''(ket rows, codensities per row) of one FFT batch within ``free`` bytes.

    Whole rows of r codensities while one fits; below that one row is split
    into codensity chunks, since a single row alone is r * ngrid * _FFT_BYTES.
    '''
    nb = int(_FFT_BUDGET_FRAC * free) // (_FFT_BYTES * ngrid)
    nb = max(1, min(nb, max_fft_batch(ngrid, mesh)))
    return max(1, min(nk, nb // r)), min(r, nb)


def get_k_lowrank(cell, mesh, factors, ket=None, devices=None, verbose=None):
    '''K[D] @ ket for the low-rank densities D = L R^T.

    Args:
        factors : sequence of (L, R) pairs, each (nao, r), NumPy or CuPy.
        ket : (nao, nk) coefficients; None returns the full K (ket = identity).
        devices : CUDA device ids to dispatch the ket rows over (default: current).
    Returns:
        NumPy (nset, nao, nk) with K_pq = sum_jk (pj|kq) D_jk.
    '''
    log = logger.new_logger(cell, verbose)
    mesh = np.asarray(mesh, dtype=int)
    ngrid = int(np.prod(mesh))
    nao = cell.nao
    ket = np.eye(nao) if ket is None else np.asarray(ket)
    nk = ket.shape[1]
    devices = default_devices(devices)
    nshare = max(devices.count(d) for d in devices)
    weight = cell.vol / ngrid
    out = np.empty((len(factors), nao, nk))
    for iset, (L, R) in enumerate(factors):
        t0 = log.init_timer()
        r = np.asarray(L).shape[1]
        with cp.cuda.Device(devices[0]):
            # planned before setup: each rank on a device holds its three MO grids,
            # and the device holds at most all nk rows of U
            resident = (nshare * (2 * r + nk) + nk) * ngrid * 8
            kb, mb = _fft_batch(free_bytes() - resident, ngrid, mesh, r, nk)
        log.debug('low-rank exchange: %d ket rows x %d of %d codensities per FFT batch', kb, mb, r)
        tasks = list(lib.prange(0, nk, kb))

        def setup(rank):
            phiL, phiR, phiK = mo_on_grid(cell, [L, R, ket], mesh)
            w_half = half_kernel(gtools.get_coulG(cell, mesh=mesh), mesh)
            return dict(phiL=phiL, phiR=phiR, phiK=phiK, w_half=w_half, U={})

        def work(rank, st, task):
            k0, k1 = task
            for m0, m1 in lib.prange(0, r, mb):
                rho = (st['phiR'][m0:m1][None] * st['phiK'][k0:k1, None]).reshape(-1, ngrid)
                vR = coulomb_potential(rho, mesh, st['w_half']).reshape(k1 - k0, m1 - m0, ngrid)
                rho = None
                Um = contract('mg,kmg->kg', st['phiL'][m0:m1], vR)
                vR = None
                if m0 == 0:
                    st['U'][k0] = Um
                else:
                    st['U'][k0] += Um
                Um = None

        _, states = run(devices, setup, work, tasks)

        def finish(rank):
            # this device's ket columns: K[:, k] = weight sum_g ao(g) U_k(g)
            st = states[rank]
            st['phiL'] = st['phiR'] = st['phiK'] = None
            K = cp.zeros((nao, nk))
            for g0, g1, aoT in ao_loop(cell, mesh, extra_per_point=8 * nk):
                for k0, U in st['U'].items():
                    K[:, k0:k0 + U.shape[0]] += aoT @ U[:, g0:g1].T
            st['U'] = None
            return (K * weight).get()

        out[iset] = np.sum(each(devices, finish), axis=0)
        log.timer(f'low-rank exchange: rank {r}, {nk} ket columns, {len(devices)} device(s)', *t0)
    return out


def pair_get_k_lowrank(cell, mf, orbp, mesh=None, devices=None, verbose=None):
    '''``pair_get_k`` callback for ``make_rdm1_relaxed_rhf_pprpa`` built on the
    low-rank exchange.

    The pair densities lie in the span of the active orbitals ``orbp``, so
    D = (D S orbp) orbp^T exactly, and the columns of the left factor that are
    zero (the occupied ones for X, the virtual ones for Y) are dropped.  The
    callback accepts ``ket`` and returns K @ ket (``accepts_ket``); ``hermi``
    is accepted and not needed, the result never assumes symmetry.
    '''
    mesh = cell.mesh if mesh is None else mesh
    ovlp = np.asarray(mf.get_ovlp()).real
    ovlp = ovlp[0] if ovlp.ndim == 3 else ovlp
    orbp = np.asarray(orbp)
    nao = orbp.shape[0]
    s_orbp = ovlp @ orbp

    def pair_get_k(dms, hermi=0, ket=None):
        factors = []
        for dm in np.asarray(dms).reshape(-1, nao, nao):
            L = dm @ s_orbp
            keep = np.abs(L).max(axis=0) > 1e-12 * np.abs(L).max()
            factors.append((L[:, keep], orbp[:, keep]))
        return get_k_lowrank(cell, mesh, factors, ket=ket, devices=devices, verbose=verbose)

    pair_get_k.accepts_ket = True
    return pair_get_k
