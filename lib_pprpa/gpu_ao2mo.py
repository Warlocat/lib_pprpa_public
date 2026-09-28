'''GPU (cupy) Gamma-point FFT ao2mo for pp-RPA: the active-space vvvv / oovv /
oooo blocks in physicist layout, built directly on the GPU (no CPU ao2mo, no GDF).

Same FFT MO-integral transform as pyscf.pbc.df.fft_ao2mo, in cupy:
  rho_pq(r) = mo_p(r) mo_q(r)  ->  fft  ->  eri_chem[(pq),(rs)] = sum_G rho_pq(G)* wG rho_rs(G)
then permuted to the physicist convention pp-RPA expects (matches pprpaobj):
  vvvv[a,b,c,d] = <ab|cd>,  oovv[i,j,a,b] = <ij|ab>,  oooo[i,j,k,l] = <ij|kl>.

The chemist block of one MO-pair set is the Gram matrix E = vR rho^T over the
pair index P = (p, q), with vR the Coulomb potential of the codensities.  It is
formed in row strips of the pair index: the potential of an outer strip is
built once (FFT sub-batches into a fixed buffer) and contracted, one GEMM per
strip pair, against every inner strip at or below it -- E is symmetric, so only
its lower triangle is computed and each tile is scattered together with its
transpose.  When the same MO set stands on both sides (vvvv, oooo) the pair
index runs over p >= q only, since rho_pq = rho_qp for real orbitals.  Both are
exact bookkeeping.  The strip width is chosen from the free device memory,
whole MO rows at a time, and capped at ``PAIR_BLK_CAP``.

The three finals stay on the device when they fit ``gpu_mem.RESIDENT_VRAM_FRAC``
of its memory and are otherwise assembled in pinned host memory, which the
streamed Davidson contraction (``pprpa_eri_gpu``) then uploads at the link rate.
'''

import math

import numpy as np
import cupy as cp
import cupyx
from pyscf.lib import logger
from gpu4pyscf.pbc import tools as gtools

from lib_pprpa.gpu_coulomb import codensity, coulomb_potential, half_kernel, mo_on_grid
from lib_pprpa.gpu_mem import eri_bytes, fits_resident, free_bytes, max_fft_batch

__all__ = ['gpu_ao2mo_blocks', 'PAIR_BLK_CAP']

# Widest strip the planner chooses.  The fp64 tile GEMM on a B200 peaks near
# b = 4800-5000 pairs and loses ~20% at 6000-7600, where the two operand strips
# also pass 70 GB each.
PAIR_BLK_CAP = 5000
# Bytes per pair-gridpoint: the R2C potential chain (rho, half spectrum, real
# vR, cuFFT work area) and the two GEMM operand strips (vR and the inner codensity).
_FFT_BYTES = 40
_GEMM_BYTES = 16
# Share of the strip budget for the FFT sub-batch; the transform is ~0.1% of the
# flops and only needs to keep cuFFT busy.
_FFT_BUDGET_FRAC = 0.15
_FFT_BLK_MIN = 16
_RESERVE_FRAC = 0.06


class _PairLayout:
    '''Row-aligned strips over the pair index P = (p, q) of one block.

    Row p holds p + 1 pairs (q <= p) when ``compact`` and nB pairs otherwise.
    Strips are whole rows, so a tile lands in the physicist tensor as slices.
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


def _write_tile(out, tile, layout, pa, pb, ra, rb):
    '''Scatter the Gram tile E[I, J] of outer rows [pa, pb) and inner rows
    [ra, rb) into the physicist tensor ``out``, together with every element
    fixed by symmetry, for the canonical row pairs r <= p only.

    out[a, b, c, d] = <ab|cd> = (ac|bd); tile[(p, q), (r, s)] = (pq|rs).
    ``out`` and ``tile`` are both NumPy or both CuPy.
    '''
    ro = layout.row_off
    if not layout.compact:
        nB = layout.nB
        phys = tile.reshape(pb - pa, nB, rb - ra, nB).transpose(0, 2, 1, 3)
        if rb <= pa:                                # entirely below the diagonal
            out[pa:pb, ra:rb] = phys
            out[ra:rb, pa:pb] = phys.transpose(1, 0, 3, 2)
            return
        for p in range(pa, pb):                     # straddles it: keep r <= p
            r1 = min(rb, p + 1)
            if r1 > ra:
                blk = phys[p - pa, :r1 - ra]
                out[p, ra:r1] = blk
                out[ra:r1, p] = blk.transpose(0, 2, 1)
        return
    # compact: T[q, s] = (pq|rs) for q <= p, s <= r, and its eight permutation
    # images (pq|rs) (qp|rs) (qp|sr) (rs|qp) (pq|sr) (rs|pq) (sr|pq) (sr|qp),
    # each landing on out[x, z, y, w] for (xy|zw)
    for p in range(pa, pb):
        rows = slice(int(ro[p] - ro[pa]), int(ro[p + 1] - ro[pa]))
        for r in range(ra, min(rb, p + 1)):
            T = tile[rows, int(ro[r] - ro[ra]):int(ro[r + 1] - ro[ra])]
            TT = T.T
            out[p, r, :p + 1, :r + 1] = T
            out[:p + 1, r, p, :r + 1] = T
            out[:p + 1, :r + 1, p, r] = T
            out[r, :p + 1, :r + 1, p] = T
            out[p, :r + 1, :p + 1, r] = TT
            out[r, p, :r + 1, :p + 1] = TT
            out[:r + 1, p, r, :p + 1] = TT
            out[:r + 1, :p + 1, r, p] = TT


def _plan_strips(layout, ngrid, mesh, pair_blk=None):
    '''(pair_blk, fft_blk): GEMM strip width and FFT sub-batch from free memory.

    The FFT sub-batch takes ``_FFT_BUDGET_FRAC`` of the budget within the cuFFT
    plan cap; the strip then solves 8 b^2 + 16 ngrid b <= rest (a b x b tile
    plus the two operand strips), rounded to whole rows and capped.
    '''
    free = free_bytes()
    budget = max(0, free - max(512 * 1024**2, int(_RESERVE_FRAC * free)))
    fblk = max(_FFT_BLK_MIN, int(_FFT_BUDGET_FRAC * budget) // (_FFT_BYTES * ngrid))
    fblk = max(1, min(fblk, max_fft_batch(ngrid, mesh), layout.npair))
    if pair_blk is None:
        rest = max(0, budget - _FFT_BYTES * ngrid * fblk)
        lin = _GEMM_BYTES * ngrid
        pair_blk = int((-lin + math.sqrt(lin * lin + 32 * rest)) / 16)
        pair_blk = min(pair_blk, PAIR_BLK_CAP)
    blk = max(layout.min_blk, int(pair_blk))
    if not layout.compact:
        blk = (blk // layout.nB) * layout.nB
    blk = min(blk, layout.npair)
    return blk, min(fblk, blk)


def _block(moA, moB, w_half, mesh, out, layout, pair_blk, log):
    '''Fill one physicist block ``out`` from the MO grids moA, moB (n, ngrid).'''
    ngrid = moA.shape[1]
    blk, fblk = _plan_strips(layout, ngrid, mesh, pair_blk)
    aidx, bidx = layout.index_arrays()
    # fixed buffers for the whole block: one cuFFT plan, no pool churn
    vR_buf = cp.empty((blk, ngrid))
    rho_buf = cp.empty((blk, ngrid))
    rho_f = cp.empty((fblk, ngrid))
    on_gpu = isinstance(out, cp.ndarray)
    strips = layout.split(0, layout.nA, blk)
    log.debug('ao2mo block %s: npair=%d pair_blk=%d fft_blk=%d strips=%d',
              'compact' if layout.compact else 'full', layout.npair, blk, fblk, len(strips))
    for pa, pb in strips:
        P0, P1 = layout.pairs(pa, pb)
        vR = vR_buf[:P1 - P0]
        for f0 in range(P0, P1, fblk):
            f1 = min(P1, f0 + fblk)
            n = f1 - f0
            if n < fblk:
                rho_f[n:] = 0.0                     # keep the FFT batch shape fixed
            codensity(moA, moB, aidx, bidx, f0, f1, out=rho_f[:n])
            vR[f0 - P0:f1 - P0] = coulomb_potential(rho_f, mesh, w_half)[:n]
        for ra, rb in layout.split(0, pb, blk):     # lower triangle of E
            Q0, Q1 = layout.pairs(ra, rb)
            rho_q = codensity(moA, moB, aidx, bidx, Q0, Q1, out=rho_buf[:Q1 - Q0])
            tile = vR.dot(rho_q.T)
            _write_tile(out, tile if on_gpu else tile.get(), layout, pa, pb, ra, rb)
    return out


def gpu_ao2mo_blocks(cell, cocc, cvir, mesh, pair_blk=None, return_gpu=False,
                     stage_host=None, verbose=None):
    '''Return vvvv, oovv, oooo (physicist, real) for the active orbitals cocc, cvir.

    Kwargs:
        pair_blk : force the GEMM strip width in pairs (default: from free memory).
        stage_host : None decides from ``gpu_mem.fits_resident``; True assembles
            the finals in pinned host memory, False on the device.
        return_gpu : with device-resident finals, return CuPy arrays instead of
            NumPy copies.
    Returns:
        CuPy arrays when the finals are resident (and ``return_gpu``), otherwise
        NumPy arrays, pinned when host-staged.
    '''
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()
    moO, moV = mo_on_grid(cell, [cocc, cvir], mesh)
    no, nv, ng = moO.shape[0], moV.shape[0], moO.shape[1]
    w_half = half_kernel(gtools.get_coulG(cell, mesh=mesh) * (cell.vol / ng), mesh)
    if stage_host is None:
        stage_host = not fits_resident(no, nv, extra_bytes=(no + nv) * ng * 8)
    log.info('GPU ao2mo: nocc=%d nvir=%d ngrid=%d eri=%.2f GB %s', no, nv, ng,
             eri_bytes(no, nv) / 1e9, 'staged in pinned host memory' if stage_host else 'on device')
    t0 = log.timer('ao2mo MO grids', *t0)

    def alloc(shape):
        return cupyx.empty_pinned(shape) if stage_host else cp.empty(shape)

    blocks = []
    for name, (A, B) in (('vvvv', (moV, moV)), ('oooo', (moO, moO)), ('oovv', (moO, moV))):
        layout = _PairLayout(A.shape[0], B.shape[0], compact=A is B)
        out = alloc((A.shape[0], A.shape[0], B.shape[0], B.shape[0]))
        blocks.append(_block(A, B, w_half, mesh, out, layout, pair_blk, log))
        t0 = log.timer(f'ao2mo {name}', *t0)
    vvvv, oooo, oovv = blocks
    moO = moV = w_half = None
    cp.get_default_memory_pool().free_all_blocks()
    if stage_host and fits_resident(no, nv):
        vvvv, oovv, oooo = (cp.asarray(x) for x in (vvvv, oovv, oooo))
        stage_host = False
    if stage_host or return_gpu:
        return vvvv, oovv, oooo
    return vvvv.get(), oovv.get(), oooo.get()
