'''Device- and host-memory bookkeeping shared by the GPU ao2mo and Davidson ERI paths.

The active-space MO ERIs are three float64 blocks stored as the lower triangles
of their pair Gram matrices (``gpu_ao2mo``): vvvv and oooo over the compact
pairs, oovv over all occupied-virtual pairs.  Host arrays live in page-locked
(pinned) memory so that streaming them to the device runs at the link rate instead
of through the driver's bounce buffer (about 6x on a B200).
'''

import contextlib
import threading

import numpy as np
import cupy as cp

__all__ = ['FFT_BATCH_MIN', 'pack_offset', 'eri_bytes', 'free_bytes', 'plan_budget',
           'pooled', 'is_pooled', 'empty_pinned', 'is_pinned', 'max_fft_batch']


# Elements of one batched cuFFT plan.  Past 2^31 - 1 a plan either fails (Bluestein
# path, a prime factor above 127: CUFFT_INVALID_SIZE on the 151^3 NV63 ke 600 mesh) or
# goes 64-bit with a work area that triples the R2C chain's device memory (107^3 on an
# H200: 16.1 -> 48.1 B per element at 2^31); throughput saturates at FFT_BATCH_MIN.
CUFFT_MAX_PLAN_ELEMENTS = 2**31 - 1
# Left out of every planned budget: cuBLAS / cuFFT workspaces and the pool's rounding.
_RESERVE_FRAC = 0.06
_RESERVE_MIN = 512 * 1024**2
# Transforms per cuFFT batch at which the R2C Coulomb chain saturates: within 1.5%
# of the time per transform of batch 128 on 107^3 and 159^3 meshes (H200); batch 1
# is 16% / 52% slower.
FFT_BATCH_MIN = 16
# Largest block pooled() keeps in cupy's pool.  The per-iteration blocks it is for are
# 0.1-1.5 GB (numint grid blocks, FFT batch outputs); one-per-pass blocks above it (AO
# chunks of 20+ GB) gain nothing from caching, and a cached one that a smaller array
# splits is no longer reusable: an NV215 pairing pass then asked an 80 GB card for a
# third 22 GB chunk.
_POOL_BLOCK_MAX = 4 * 1024**3
# this thread's pooled() scope
_POOLED = threading.local()


def is_pooled():
    '''True inside a ``pooled`` scope of this thread.'''
    return getattr(_POOLED, 'on', False)


@contextlib.contextmanager
def pooled():
    '''Serve this thread's device allocations up to ``_POOL_BLOCK_MAX`` from cupy's pool.

    gpu4pyscf's allocator sends blocks over 100 MiB straight to cudaMalloc and
    cudaFree, and cudaFree synchronises the device; loops that make such a block per
    iteration (numint grid blocks, batched FFT outputs) stall on it.  The pool keeps
    them for reuse within its limit (gpu4pyscf: 90% of the device), and its free
    blocks are released on exit; larger blocks still go to cudaMalloc.  Nests;
    ``gpu_multi`` workers inherit it.
    '''
    if is_pooled():
        yield
        return
    pool = cp.get_default_memory_pool()

    def malloc(size):
        if size <= _POOL_BLOCK_MAX:
            return pool.malloc(size)
        return cp.cuda.MemoryPointer(cp.cuda.Memory(size), 0)     # cudaMalloc, freed with it

    _POOLED.on = True
    try:
        with cp.cuda.using_allocator(malloc):
            yield
    finally:
        _POOLED.on = False
        pool.free_all_blocks()


def free_bytes():
    '''Device memory a planner may allocate, after releasing what cupy caches.

    The driver's figure: gpu4pyscf's ``get_avail_mem`` is 90% of the total less the
    memory pool's usage, which misses the large blocks its allocator sends straight to
    cudaMalloc.  This thread's cached cuFFT plans (each keeps its work area) and the
    pool's free blocks are released first.  Inside ``pooled`` it is at most the pool's
    limit less what the pool holds.  Call it where a block size is planned, not
    inside an FFT loop.
    '''
    pool = cp.get_default_memory_pool()
    cp.fft.config.get_plan_cache().clear()
    pool.free_all_blocks()
    free = cp.cuda.runtime.memGetInfo()[0]
    limit = pool.get_limit()
    if is_pooled() and limit:
        free = min(free, max(0, limit - pool.total_bytes()))
    return free


def plan_budget(free):
    '''Bytes a planner may fill out of ``free`` device bytes, after the workspace reserve.'''
    free = int(free)
    return max(0, free - max(_RESERVE_MIN, int(_RESERVE_FRAC * free)))


def pack_offset(n):
    '''Offset of row ``n`` of a packed lower triangle, n (n + 1) / 2: the number
    of pairs p >= q of n orbitals, and the length of an n-row triangle.'''
    n = int(n)
    return n * (n + 1) // 2


def eri_bytes(nocc, nvir):
    '''Bytes of the packed vvvv + oooo + oovv as float64.'''
    nocc, nvir = int(nocc), int(nvir)
    return 8 * (pack_offset(pack_offset(nvir)) + pack_offset(pack_offset(nocc))
                + pack_offset(nocc * nvir))


def empty_pinned(shape, dtype=np.float64):
    '''Page-locked NumPy array of exactly its size, freed with the array.

    ``cupyx.empty_pinned`` draws from cupy's pinned pool, which rounds every block
    up to a power of two, all of it resident (a 5.9 GB request holds 8.6 GB); this
    calls cudaHostAlloc for the bytes asked for.
    '''
    shape = (int(shape),) if np.isscalar(shape) else tuple(int(n) for n in shape)
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape)) * dtype.itemsize
    if nbytes == 0:
        return np.empty(shape, dtype)
    mem = cp.cuda.PinnedMemoryPointer(cp.cuda.PinnedMemory(nbytes), 0)
    return np.ndarray(shape, dtype, buffer=mem)


def is_pinned(array):
    '''True if a NumPy array lives in pinned host memory (``empty_pinned``,
    ``gpu4pyscf.lib.cupy_helper.pin_memory`` or any view of them).'''
    attrs = cp.cuda.runtime.pointerGetAttributes(array.ctypes.data)
    return attrs.type == cp.cuda.runtime.memoryTypeHost


def max_fft_batch(ngrid):
    '''Largest number of ``ngrid``-point transforms one cuFFT plan may batch.'''
    return max(1, CUFFT_MAX_PLAN_ELEMENTS // int(ngrid))
