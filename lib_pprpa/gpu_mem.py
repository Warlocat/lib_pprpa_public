'''Device- and host-memory bookkeeping shared by the GPU ao2mo and Davidson ERI paths.

The active-space MO ERIs are three float64 blocks stored as the lower triangles
of their pair Gram matrices (``gpu_ao2mo``): vvvv and oooo over the compact
pairs, oovv over all occupied-virtual pairs.  Host arrays live in page-locked
(pinned) memory so that streaming them to the device runs at the link rate instead
of through the driver's bounce buffer (about 6x on a B200).
'''

import numpy as np
import cupy as cp

from lib_pprpa.pair_operator import pack_offset

__all__ = ['FFT_BATCH_MIN', 'pack_offset', 'eri_bytes', 'free_bytes', 'plan_budget',
           'empty_pinned', 'is_pinned', 'max_fft_batch']


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


def free_bytes():
    '''Free device memory as the driver reports it, after releasing what cupy caches.

    gpu4pyscf's ``get_avail_mem`` is 90% of the total less the memory pool's
    usage; its allocator sends large blocks straight to cudaMalloc, so that
    figure does not move when the ERI tensors fill the device.  The pool's free
    blocks and this thread's cached cuFFT plans (each keeps its work area) are
    released first: neither can serve a block over the allocator's 100 MB
    threshold.  Call it where a block size is planned, not inside an FFT loop.
    '''
    cp.get_default_memory_pool().free_all_blocks()
    cp.fft.config.get_plan_cache().clear()
    return cp.cuda.runtime.memGetInfo()[0]


def plan_budget(free):
    '''Bytes a planner may fill out of ``free`` device bytes, after the workspace reserve.'''
    free = int(free)
    return max(0, free - max(_RESERVE_MIN, int(_RESERVE_FRAC * free)))


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
