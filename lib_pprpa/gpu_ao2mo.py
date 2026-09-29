'''GPU (cupy) Gamma-point FFT ao2mo for pp-RPA: the active-space vvvv / oovv /
oooo blocks as packed Gram matrices over MO pairs, built on the GPU.

Same FFT MO-integral transform as pyscf.pbc.df.fft_ao2mo, in cupy:
  rho_pq(r) = mo_p(r) mo_q(r)  ->  fft  ->  E[(pq),(rs)] = sum_G rho_pq(G)* wG rho_rs(G)

Each block is the symmetric Gram matrix E = vR rho^T of one pair set, stored as
its lower triangle only (``pack_offset``): vvvv and oooo run over the compact
pairs p >= q, P = p(p+1)/2 + q, since rho_pq = rho_qp for real orbitals; oovv
runs over all pairs P = i nvir + a.  In the physicist convention pp-RPA uses,
vvvv[a,b,c,d] = <ab|cd> = (ac|bd) = E[P(a,c), Q(b,d)] and oovv[i,j,a,b] = <ij|ab>
= (ia|jb) = E[P(i,a), Q(j,b)]; ``pprpa_eri_gpu`` contracts the packed form
directly, so the 8x (2x for oovv) larger physicist tensors are never formed.

E is formed in row strips of the pair index: the potential of an outer strip is
built once (FFT sub-batches into a fixed buffer) and contracted, one GEMM per
strip pair, against every inner strip at or below it.  Each tile is scattered
into the packed rows of its outer strip on the device and the finished strip is
copied to pinned host memory in one transfer, so with several ``devices`` the
outer strips are simply dispatched over them (``gpu_multi``).  The strip width
is chosen from the free device memory, whole MO rows at a time, and capped at
``PAIR_BLK_CAP``.
'''

import math

import numpy as np
import cupy as cp
import cupyx
from pyscf.lib import logger
from gpu4pyscf.pbc import tools as gtools

from lib_pprpa.gpu_coulomb import codensity, coulomb_potential, half_kernel, mo_on_grid
from lib_pprpa.gpu_mem import eri_bytes, free_bytes, max_fft_batch, pack_offset
from lib_pprpa.gpu_multi import default_devices, each, run

__all__ = ['gpu_ao2mo_blocks', 'PAIR_BLK_CAP']

# Widest strip the planner chooses.  The fp64 tile GEMM on a B200 peaks near
# b = 4800-5000 pairs and loses ~20% at 6000-7600, where the two operand strips
# also pass 70 GB each.
PAIR_BLK_CAP = 5000
# Bytes per pair-gridpoint: the R2C potential chain (rho, half spectrum, real
# vR and the two cuFFT work areas, measured at 56.4 on a 159^3 mesh) and the two
# GEMM operand strips (vR and the inner codensity).
_FFT_BYTES = 56
_GEMM_BYTES = 16
# Share of the strip budget for the FFT sub-batch; the transform is ~0.1% of the
# flops and only needs to keep cuFFT busy.
_FFT_BUDGET_FRAC = 0.15
_FFT_BLK_MIN = 16
_RESERVE_FRAC = 0.06


class _PairLayout:
    '''Row-aligned strips over the pair index P = (p, q) of one block.

    Row p holds p + 1 pairs (q <= p) when ``compact`` and nB pairs otherwise.
    Strips are whole rows, so the packed rows of a strip are one contiguous range.
    '''

    def __init__(self, nA, nB, compact):
        self.nA, self.nB, self.compact = int(nA), int(nB), bool(compact)
        row_len = np.arange(1, nA + 1) if compact else np.full(nA, nB)
        self.row_off = np.concatenate([[0], np.cumsum(row_len)])
        self.npair = int(self.row_off[-1])
        self.min_blk = int(row_len.max())            # a strip holds at least one row

    def pairs(self, r0, r1):
        '''Pair offsets [P0, P1) of rows [r0, r1).'''
        return int(self.row_off[r0]), int(self.row_off[r1])

    def split(self, r0, r1, blk):
        '''Whole-row strips of rows [r0, r1) with at most ``blk`` pairs each
        (a single row is always emitted, even above ``blk``).'''
        out, a = [], r0
        while a < r1:
            b = a + 1
            while b < r1 and self.row_off[b + 1] - self.row_off[a] <= blk:
                b += 1
            out.append((a, b))
            a = b
        return out

    def index_arrays(self):
        '''int32 device arrays: pair P -> (p, q).'''
        if self.compact:
            p = np.repeat(np.arange(self.nA), np.arange(1, self.nA + 1))
            q = np.concatenate([np.arange(k + 1) for k in range(self.nA)])
        else:
            p = np.repeat(np.arange(self.nA), self.nB)
            q = np.tile(np.arange(self.nB), self.nA)
        return cp.asarray(p, dtype=np.int32), cp.asarray(q, dtype=np.int32)


# Scatter the Gram tile of pair rows [P0, P0 + nP) and columns [Q0, Q0 + nQ)
# into the packed rows of its strip: element (P, Q <= P) lands at
# P(P+1)/2 + Q - base.  Elements above the diagonal are the transposes of
# stored ones and are dropped.
_PACK_TILE = cp.ElementwiseKernel(
    'raw float64 tile, int64 nQ, int64 P0, int64 Q0, int64 base',
    'raw float64 buf',
    '''
    const long long P = P0 + i / nQ;
    const long long Q = Q0 + i - (i / nQ) * nQ;
    if (Q <= P) buf[P * (P + 1) / 2 + Q - base] = tile[i];
    ''',
    'pprpa_pack_tile')


def _pack_tile(buf, tile, P0, Q0, base):
    '''Write the tile E[P0:, Q0:] into ``buf``, the packed rows starting at offset ``base``.'''
    tile = cp.ascontiguousarray(tile)
    _PACK_TILE(tile, tile.shape[1], P0, Q0, base, buf, size=tile.size)


def _plan_strips(layout, ngrid, mesh, pair_blk=None, nshare=1):
    '''(pair_blk, fft_blk): GEMM strip width and FFT sub-batch from free memory.

    The FFT sub-batch takes ``_FFT_BUDGET_FRAC`` of the budget within the cuFFT
    plan cap; the strip then solves 8 b^2 + (16 ngrid + 8 npair) b <= rest (a
    b x b tile, the two operand strips and the packed rows of the strip),
    rounded to whole rows and capped.  ``nshare`` workers share this device's memory.
    '''
    free = free_bytes() / nshare
    budget = max(0, free - max(512 * 1024**2, int(_RESERVE_FRAC * free)))
    fblk = max(_FFT_BLK_MIN, int(_FFT_BUDGET_FRAC * budget) // (_FFT_BYTES * ngrid))
    fblk = max(1, min(fblk, max_fft_batch(ngrid, mesh), layout.npair))
    if pair_blk is None:
        rest = max(0, budget - _FFT_BYTES * ngrid * fblk)
        lin = _GEMM_BYTES * ngrid + 8 * layout.npair
        pair_blk = int((-lin + math.sqrt(lin * lin + 32 * rest)) / 16)
        pair_blk = min(pair_blk, PAIR_BLK_CAP)
    blk = max(layout.min_blk, int(pair_blk))
    if not layout.compact:
        blk = (blk // layout.nB) * layout.nB
    blk = min(blk, layout.npair)
    return blk, min(fblk, blk)


def _strip(st, task, layout, mesh, out, blk, fblk):
    '''Outer row strip [pa, pb): its potential once (fixed-shape FFT batches),
    then one GEMM per inner strip at or below it, each tile packed into the
    strip's rows on the device; the finished rows go to the host in one copy.'''
    pa, pb = task
    moA, moB, aidx, bidx = st['moA'], st['moB'], st['aidx'], st['bidx']
    P0, P1 = layout.pairs(pa, pb)
    base = pack_offset(P0)
    vR = st['vR'][:P1 - P0]
    rho_f = st['rho_f']
    buf = st['buf'][:pack_offset(P1) - base]
    for f0 in range(P0, P1, fblk):
        f1 = min(P1, f0 + fblk)
        n = f1 - f0
        if n < fblk:
            rho_f[n:] = 0.0                         # keep the FFT batch shape fixed
        codensity(moA, moB, aidx, bidx, f0, f1, out=rho_f[:n])
        vR[f0 - P0:f1 - P0] = coulomb_potential(rho_f, mesh, st['w_half'])[:n]
    for ra, rb in layout.split(0, pb, blk):         # lower triangle of E
        Q0, Q1 = layout.pairs(ra, rb)
        rho_q = codensity(moA, moB, aidx, bidx, Q0, Q1, out=st['rho'][:Q1 - Q0])
        _pack_tile(buf, vR.dot(rho_q.T), P0, Q0, base)
    buf.get(out=out[base:base + buf.size])


def _block(per_dev, sel, mesh, out, layout, pair_blk, devices, log):
    '''Fill the packed block ``out``; ``per_dev[rank]`` holds that device's
    [moO, moV, w_half] and ``sel`` picks the two MO sets of the block.'''
    ngrid = int(np.prod(mesh))
    nshare = max(devices.count(d) for d in devices)
    with cp.cuda.Device(devices[0]):
        blk, fblk = _plan_strips(layout, ngrid, mesh, pair_blk, nshare)
    tasks = layout.split(0, layout.nA, blk)[::-1]  # high rows own more tiles: first
    nbuf = max(pack_offset(layout.pairs(a, b)[1]) - pack_offset(layout.pairs(a, b)[0])
               for a, b in tasks)
    log.debug('ao2mo block %s: npair=%d pair_blk=%d fft_blk=%d strips=%d on %d device(s)',
              'compact' if layout.compact else 'full', layout.npair, blk, fblk,
              len(tasks), len(devices))

    def setup(rank):
        grids = per_dev[rank]
        aidx, bidx = layout.index_arrays()
        # fixed buffers for the whole block: one cuFFT plan, no pool churn
        return dict(moA=grids[sel[0]], moB=grids[sel[1]], w_half=grids[2], aidx=aidx, bidx=bidx,
                    vR=cp.empty((blk, ngrid)), rho=cp.empty((blk, ngrid)),
                    rho_f=cp.empty((fblk, ngrid)), buf=cp.empty(nbuf))

    run(devices, setup, lambda rank, st, task: _strip(st, task, layout, mesh, out, blk, fblk),
        tasks)
    return out


def gpu_ao2mo_blocks(cell, cocc, cvir, mesh, pair_blk=None, devices=None, verbose=None):
    '''Return the packed vvvv, oovv, oooo Gram matrices for the active orbitals cocc, cvir.

    Kwargs:
        pair_blk : force the GEMM strip width in pairs (default: from free memory).
        devices : CUDA device ids to dispatch the pair strips over (default:
            the current device).
    Returns:
        Three pinned NumPy arrays, the lower triangles of the pair Gram matrices
        (see the module docstring); ``pprpa_eri_gpu.attach_gpu_eri_contraction``
        takes them as they are.
    '''
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()
    devices = default_devices(devices)
    mesh = np.asarray(mesh, dtype=int)
    ng = int(np.prod(mesh))
    no, nv = np.asarray(cocc).shape[1], np.asarray(cvir).shape[1]
    per_dev = each(devices, lambda rank: mo_on_grid(cell, [cocc, cvir], mesh)
                   + [half_kernel(gtools.get_coulG(cell, mesh=mesh) * (cell.vol / ng), mesh)])
    log.info('GPU ao2mo: nocc=%d nvir=%d ngrid=%d packed eri=%.2f GB on %d device(s)',
             no, nv, ng, eri_bytes(no, nv) / 1e9, len(devices))
    t0 = log.timer('ao2mo MO grids', *t0)

    blocks = []
    for name, sel in (('vvvv', (1, 1)), ('oooo', (0, 0)), ('oovv', (0, 1))):
        nA, nB = per_dev[0][sel[0]].shape[0], per_dev[0][sel[1]].shape[0]
        layout = _PairLayout(nA, nB, compact=sel[0] == sel[1])
        out = cupyx.empty_pinned(pack_offset(layout.npair))
        blocks.append(_block(per_dev, sel, mesh, out, layout, pair_blk, devices, log))
        t0 = log.timer(f'ao2mo {name}', *t0)
    vvvv, oooo, oovv = blocks
    per_dev = None
    each(devices, lambda rank: cp.get_default_memory_pool().free_all_blocks())
    return vvvv, oovv, oooo
