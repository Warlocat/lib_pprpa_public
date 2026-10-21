'''cuBLAS fp64 emulation, switched on per call where it pays.

cuBLAS 13 can run a DGEMM as fixed-point slices on the integer tensor cores
(math mode CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH).  Slicing the operands costs
~(m + n) k and the product ~m n k, so for C (m, n) = A (m, k) B (k, n) the gain
depends on the harmonic width m n / (m + n): wide ao2mo tiles run up to 3.8x
faster than native fp64 on a B200 and more accurately, small GEMMs ~10x slower.
Enabled for a whole process (CUBLAS_EMULATE_DOUBLE_PRECISION) it is paid on
every SCF and CPHF call, so ``fp64_emulation`` switches the current thread's
cuBLAS handle for one block of calls: ``eager`` for a GEMM whose shape
``emulation_pays``, ``performant`` (cuBLAS decides per call) around library
code whose GEMM shapes are not ours.
'''

import contextlib
import ctypes
import functools
import os

import cupy as cp
from cupy.cuda import cublas

__all__ = ['EMULATION_MIN_WIDTH', 'emulation_available', 'emulation_pays', 'fp64_emulation']

# Harmonic width m n / (m + n) above which an eager emulated DGEMM beats native
# fp64, per compute-capability major, measured with cuBLAS 13.8 for k = 3e5-4e6:
# B200 crosses over at ~380 (square b ~ 770); H200 only gains from ~1200
# (1.16x at 2400^2, 1.36x at 5000^2) and its ``performant`` heuristic loses.
EMULATION_MIN_WIDTH = {9: 1200, 10: 400}
# Architectures where ``performant`` never loses to native in our measurements.
_PERFORMANT_ARCH = (10,)

_FP64_EMULATED_FIXEDPOINT_MATH = 8      # cublasMath_t
_STRATEGY = {'default': 0, 'performant': 1, 'eager': 2}


@functools.cache
def _set_strategy():
    '''cublasSetEmulationStrategy from the libcublas cupy already loaded, or None
    (cuBLAS < 13); cupy does not wrap it.'''
    cp.cuda.device.get_cublas_handle()
    try:
        lib = ctypes.CDLL(f'libcublas.so.{cp.cuda.runtime.runtimeGetVersion() // 1000}',
                          mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
        fn = lib.cublasSetEmulationStrategy
    except (OSError, AttributeError):
        return None
    fn.argtypes = [ctypes.c_void_p, ctypes.c_int]
    fn.restype = ctypes.c_int
    return fn


@functools.cache
def _arch(device_id):
    '''Compute-capability major of the device if cuBLAS can emulate on it, else None.'''
    major = int(cp.cuda.Device(device_id).compute_capability[:-1])
    if major in EMULATION_MIN_WIDTH and _set_strategy() is not None:
        return major
    return None


def emulation_available(strategy='eager'):
    '''True if the current device gains from emulated fp64 GEMMs under ``strategy``.'''
    major = _arch(cp.cuda.Device().id)
    if strategy == 'performant':
        return major in _PERFORMANT_ARCH
    return major is not None


def emulation_pays(m, n):
    '''True if an (m, n) output DGEMM is faster emulated (``eager``) on the current device.'''
    m, n = int(m), int(n)
    major = _arch(cp.cuda.Device().id)
    return (major is not None and m > 0 and n > 0
            and m * n >= EMULATION_MIN_WIDTH[major] * (m + n))


@contextlib.contextmanager
def fp64_emulation(strategy):
    '''Emulate the DGEMMs launched inside the block on this thread's cuBLAS handle.

    ``strategy`` is 'eager' (every DGEMM), 'performant' (cuBLAS decides per
    call) or None (no change).  cupy keeps one handle per device and thread, so
    the switch never leaks to other threads; the math mode is read at launch,
    so restoring it right after an asynchronous call is safe.
    '''
    if strategy is None:
        yield
        return
    handle = cp.cuda.device.get_cublas_handle()
    mode = cublas.getMathMode(handle)
    cublas.setMathMode(handle, _FP64_EMULATED_FIXEDPOINT_MATH)
    _set_strategy()(handle, _STRATEGY[strategy])
    try:
        yield
    finally:
        _set_strategy()(handle, _STRATEGY['default'])
        cublas.setMathMode(handle, mode)
