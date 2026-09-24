'''Device- and host-memory bookkeeping shared by the GPU ao2mo and Davidson ERI paths.

The active-space MO ERIs are three float64 blocks, vvvv (nvir^4), oooo (nocc^4)
and oovv (nocc^2 nvir^2).  They are kept resident on the device when they fit
``RESIDENT_VRAM_FRAC`` of its total memory and staged on the host otherwise.
Host-staged blocks live in page-locked (pinned) memory so that streaming them
to the device runs at the link rate instead of through the driver's bounce
buffer (about 6x on a B200).
'''

import cupy as cp

__all__ = ['RESIDENT_VRAM_FRAC', 'eri_bytes', 'fits_resident', 'free_bytes', 'is_pinned']

# Leaves room for the trial vectors, the SCF arrays and the gradient intermediates.
RESIDENT_VRAM_FRAC = 0.75


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
