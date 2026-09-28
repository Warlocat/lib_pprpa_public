'''Device- and host-memory bookkeeping shared by the GPU ao2mo and Davidson ERI paths.

The active-space MO ERIs are three float64 blocks, vvvv (nvir^4), oooo (nocc^4)
and oovv (nocc^2 nvir^2).  They are kept resident on the device when they fit
``RESIDENT_VRAM_FRAC`` of its total memory and staged on the host otherwise.
Host-staged blocks live in page-locked (pinned) memory so that streaming them
to the device runs at the link rate instead of through the driver's bounce
buffer (about 6x on a B200).
'''

import cupy as cp

__all__ = ['RESIDENT_VRAM_FRAC', 'eri_bytes', 'fits_resident', 'free_bytes', 'is_pinned',
           'needs_bluestein', 'max_fft_batch']

# Leaves room for the trial vectors, the SCF arrays and the gradient intermediates.
RESIDENT_VRAM_FRAC = 0.75

# cuFFT rejects a batched plan of more than 2^31 - 1 elements when a transform
# dimension has a prime factor above 127 and takes the Bluestein path
# (CUFFT_INVALID_SIZE, e.g. the 151^3 mesh of the NV63 cell at ke = 600 Ha).
# Direct-path meshes ran 1.5 x 2^31 on a B200, so they get that relaxed cap.
CUFFT_MAX_PLAN_ELEMENTS = 2**31 - 1
CUFFT_MAX_PLAN_ELEMENTS_DIRECT = int(1.5 * 2**31)
_CUFFT_MAX_DIRECT_PRIME = 127


def free_bytes():
    '''Free device memory as the driver reports it.

    gpu4pyscf's ``get_avail_mem`` is 90% of the total less the memory pool's
    usage; its allocator sends large blocks straight to cudaMalloc, so that
    figure does not move when the ERI tensors fill the device.
    '''
    return cp.cuda.runtime.memGetInfo()[0]


def eri_bytes(nocc, nvir):
    '''Bytes of vvvv + oooo + oovv as float64.'''
    nocc, nvir = int(nocc), int(nvir)
    return (nvir**4 + nocc**4 + nocc**2 * nvir**2) * 8


def fits_resident(nocc, nvir, extra_bytes=0, total_bytes=None, frac=RESIDENT_VRAM_FRAC):
    '''True if the three ERI blocks plus ``extra_bytes`` fit ``frac`` of the device memory.

    ``total_bytes`` overrides the device total (tests).
    '''
    if total_bytes is None:
        total_bytes = cp.cuda.runtime.memGetInfo()[1]
    return eri_bytes(nocc, nvir) + int(extra_bytes) <= int(frac * int(total_bytes))


def is_pinned(array):
    '''True if a NumPy array lives in pinned host memory (``cupyx.empty_pinned``,
    ``gpu4pyscf.lib.cupy_helper.pin_memory`` or any view of them).'''
    attrs = cp.cuda.runtime.pointerGetAttributes(array.ctypes.data)
    return attrs.type == cp.cuda.runtime.memoryTypeHost


def _largest_prime_factor(n):
    n = int(n)
    largest, p = 1, 2
    while p * p <= n:
        while n % p == 0:
            largest, n = p, n // p
        p += 1
    return max(largest, n) if n > 1 else largest


def needs_bluestein(mesh):
    '''True if a mesh dimension has a prime factor cuFFT cannot handle directly.'''
    return any(_largest_prime_factor(n) > _CUFFT_MAX_DIRECT_PRIME for n in mesh)


def max_fft_batch(ngrid, mesh):
    '''Largest number of ``ngrid``-point transforms one cuFFT plan may batch.'''
    limit = CUFFT_MAX_PLAN_ELEMENTS if needs_bluestein(mesh) else CUFFT_MAX_PLAN_ELEMENTS_DIRECT
    return max(1, limit // int(ngrid))
