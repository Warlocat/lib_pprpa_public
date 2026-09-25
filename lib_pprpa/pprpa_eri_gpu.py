"""GPU (cupy) batched ERI contraction for the pp-RPA Davidson use_eri path.

Replaces the CPU per-trial-vector loop in lib_pprpa.pprpa_davidson._pprpa_contraction
(the `_use_eri` branch) with a single batched cupy contraction that keeps the
active-space ERI (vvvv/oovv/oooo) resident on the GPU.  The whole MVP

    prod_vv = vvvv . z_vv  + oovv^T . z_oo
    prod_oo = oooo . z_oo  + oovv   . z_vv
    (+ symmetrize, + orbital-energy diagonal)

is done for ALL trial vectors at once, so there is no host<->device copy of the
ERI and no Python loop over vectors.  Algebra is element-for-element identical to
the CPU routine (same z.T flattening, same 1/sqrt(2) diagonal scaling, same
hh-block sign, same physicist->matmul reshapes).

When the three blocks do not fit the device (mode='streamed', the default when
they exceed gpu_mem.RESIDENT_VRAM_FRAC of its memory) they stay on the host in
pinned memory and are streamed once per MVP in contiguous row strips for the
same GEMMs.  Blocks that are not already pinned (gpu_mem.is_pinned) are copied
into pinned memory once at attach time.

attach_gpu_eri_contraction(pprpa, vvvv, oovv, oooo) accepts numpy OR cupy blocks,
sets up the use_eri dims, and overrides pprpa.contraction.

Validate:  python pprpa_eri_gpu.py   (compares GPU mv_prod vs CPU mv_prod to ~1e-12)
"""
import math
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import pin_memory

from lib_pprpa.gpu_mem import eri_bytes, fits_resident, free_bytes, is_pinned

_INV_SQRT2 = 1.0 / math.sqrt(2.0)
# Share of the free device memory one streamed row strip may occupy.
_STRIP_MEM_FRAC = 0.25


def attach_gpu_eri_contraction(pprpa, vvvv, oovv, oooo, mode="auto", strip=None,
                               pin=True, verbose=None):
    """Keep the active-space ERI on GPU (or stream it) and route Davidson MVP through cupy.

    Kwargs:
        mode : 'auto', 'resident' or 'streamed'.  'auto' keeps the blocks on the
            device when they are already cupy arrays or fit
            gpu_mem.RESIDENT_VRAM_FRAC of its memory, and streams them otherwise.
        strip : rows of a block uploaded per GEMM in streamed mode; default from
            the free device memory.
        pin : in streamed mode, copy host blocks that are not already pinned into
            pinned memory (one host copy per block); uploads then run at the
            link rate.  Blocks allocated with cupyx.empty_pinned are used as is.
    """
    log = logger.new_logger(pprpa, logger.INFO if verbose is None else verbose)
    pprpa._use_eri = True
    nvir = pprpa.nvir
    nocc = pprpa.nocc
    if mode == "auto":
        on_device = all(isinstance(x, cp.ndarray) for x in (vvvv, oovv, oooo))
        mode = "resident" if on_device or fits_resident(nocc, nvir) else "streamed"
    if mode not in ("resident", "streamed"):
        raise ValueError(f"unknown mode {mode!r}")
    # reshape to matmul form once: (nv^2,nv^2),(no^2,no^2),(no^2,nv^2); on the
    # device when resident, C-contiguous on the host when streamed
    pprpa._gpu_vvvv = _flat(vvvv, nvir, nvir, mode, pin)
    pprpa._gpu_oooo = _flat(oooo, nocc, nocc, mode, pin)
    pprpa._gpu_oovv = _flat(oovv, nocc, nvir, mode, pin)
    if mode == "streamed" and strip is None:
        n2 = max(nocc * nocc, nvir * nvir)
        strip = max(1, min(n2, int(_STRIP_MEM_FRAC * free_bytes()) // (8 * n2)))
    pprpa._gpu_eri_strip = strip if mode == "streamed" else None
    pprpa._gpu_mo_energy = cp.asarray(pprpa.mo_energy)
    # a tiny host array so kernel()'s `data_type = pprpa.vvvv.dtype` still works
    pprpa.vvvv = np.empty(0, dtype=np.float64)
    pprpa.oovv = None
    pprpa.oooo = None
    pprpa.contraction = lambda tri_vec: _gpu_eri_contraction(pprpa, tri_vec)
    log.info("GPU ERI contraction: mode=%s strip=%s pinned=%s eri=%.2f GB nocc=%d nvir=%d",
             mode, pprpa._gpu_eri_strip,
             mode == "streamed" and is_pinned(pprpa._gpu_vvvv),
             eri_bytes(nocc, nvir) / 1e9, nocc, nvir)
    return pprpa


def release_gpu_eri(pprpa):
    """Drop the ERI blocks held by attach_gpu_eri_contraction and free the cupy pools.

    The solver keeps its eigenvalues and eigenvectors; its contraction is no
    longer usable.
    """
    for attr in ("_gpu_vvvv", "_gpu_oooo", "_gpu_oovv", "_gpu_mo_energy",
                 "_gpu_eri_strip", "contraction"):
        pprpa.__dict__.pop(attr, None)
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _flat(block, n0, n1, mode, pin):
    """C-contiguous (n0*n0, n1*n1) block on the device (resident) or the host (streamed)."""
    shape = (n0 * n0, n1 * n1)
    if mode == "resident":
        return cp.ascontiguousarray(cp.asarray(block).reshape(shape))
    block = block.get() if isinstance(block, cp.ndarray) else np.asarray(block)
    block = np.ascontiguousarray(block.reshape(shape))
    if pin and not is_pinned(block):
        block = pin_memory(block)
    return block


def _streamed_products(zvvT, zooT, V, O, OV, strip):
    """The four GEMMs of the MVP with the host blocks streamed in row strips.

    vvvv and oooo are symmetric in their pair indices, so the row strip V[P]
    stands in for V[:, P].T, and a row strip of oovv serves both of its
    products: each block crosses the bus exactly once per call.
    """
    ntri = zvvT.shape[0]
    nv2, no2 = V.shape[0], O.shape[0]
    prod_vv = cp.zeros((ntri, nv2))
    prod_oo = cp.zeros((ntri, no2))
    for p0, p1 in lib.prange(0, nv2, strip):
        prod_vv += zvvT[:, p0:p1] @ cp.asarray(V[p0:p1])
    for p0, p1 in lib.prange(0, no2, strip):
        prod_oo += zooT[:, p0:p1] @ cp.asarray(O[p0:p1])
    for p0, p1 in lib.prange(0, no2, strip):
        ov = cp.asarray(OV[p0:p1])
        prod_vv += zooT[:, p0:p1] @ ov
        prod_oo[:, p0:p1] += zvvT @ ov.T
    return prod_vv, prod_oo


def _gpu_eri_contraction(pprpa, tri_vec):
    nocc, nvir = pprpa.nocc, pprpa.nvir
    no2, nv2 = nocc * nocc, nvir * nvir
    oo_dim = pprpa.oo_dim
    k = (1 if pprpa.multi == "s" else 0) - 1     # 0 keeps diagonal (singlet), -1 drops it
    tro, tco = cp.tril_indices(nocc, k)
    trv, tcv = cp.tril_indices(nvir, k)
    di_o = cp.arange(nocc)
    di_v = cp.arange(nvir)

    T = cp.asarray(tri_vec)                       # (ntri, full_dim)
    ntri = T.shape[0]

    # restore packed trial vectors into full (lower-triangle) matrices
    z_oo = cp.zeros((ntri, nocc, nocc))
    z_vv = cp.zeros((ntri, nvir, nvir))
    z_oo[:, tro, tco] = T[:, :oo_dim]
    z_oo[:, di_o, di_o] *= _INV_SQRT2
    z_vv[:, trv, tcv] = T[:, oo_dim:]
    z_vv[:, di_v, di_v] *= _INV_SQRT2

    # CPU code contracts against z.T -> flatten the transpose
    zooT = z_oo.transpose(0, 2, 1).reshape(ntri, no2)
    zvvT = z_vv.transpose(0, 2, 1).reshape(ntri, nv2)

    if pprpa._gpu_eri_strip is None:
        prod_vv = zvvT @ pprpa._gpu_vvvv.T            # vvvv . z_vv
        prod_oo = zooT @ pprpa._gpu_oooo.T            # oooo . z_oo
        prod_vv = prod_vv + zooT @ pprpa._gpu_oovv    # + oovv^T . z_oo
        prod_oo = prod_oo + zvvT @ pprpa._gpu_oovv.T  # + oovv   . z_vv
    else:
        prod_vv, prod_oo = _streamed_products(
            zvvT, zooT, pprpa._gpu_vvvv, pprpa._gpu_oooo, pprpa._gpu_oovv, pprpa._gpu_eri_strip)

    prod_vv = prod_vv.reshape(ntri, nvir, nvir)
    prod_oo = prod_oo.reshape(ntri, nocc, nocc)

    if pprpa.multi == "s":
        prod_vv = prod_vv + prod_vv.transpose(0, 2, 1)
        prod_oo = prod_oo + prod_oo.transpose(0, 2, 1)
    else:
        prod_vv = prod_vv - prod_vv.transpose(0, 2, 1)
        prod_oo = prod_oo - prod_oo.transpose(0, 2, 1)

    # rotate upper-half to lower-half (the .T in the CPU routine)
    prod_oo = cp.ascontiguousarray(prod_oo.transpose(0, 2, 1))
    prod_oo[:, di_o, di_o] *= _INV_SQRT2
    prod_vv = cp.ascontiguousarray(prod_vv.transpose(0, 2, 1))
    prod_vv[:, di_v, di_v] *= _INV_SQRT2

    mv = cp.empty((ntri, pprpa.full_dim))
    mv[:, :oo_dim] = prod_oo[:, tro, tco]
    mv[:, oo_dim:] = prod_vv[:, trv, tcv]

    # orbital-energy diagonal term: (e_p + e_q - 2 mu), hh block negated
    me = pprpa._gpu_mo_energy
    orb_oo = (me[None, :nocc] + me[:nocc, None])[tro, tco]
    orb_vv = (me[None, nocc:] + me[nocc:, None])[trv, tcv]
    orb = cp.concatenate((orb_oo, orb_vv)) - 2.0 * pprpa.mu
    orb[:oo_dim] *= -1.0
    mv += orb[None, :] * T
    return cp.asnumpy(mv)


if __name__ == "__main__":
    # GPU mv_prod vs CPU mv_prod on a small cell, both s and t channels
    from pyscf.pbc import gto, dft as cdft
    from lib_pprpa.pprpa_davidson import ppRPA_Davidson, _pprpa_contraction
    a0 = 3.370137329
    cell = gto.M(atom=[["C", [0., 0., 0.]], ["C", [a0/2, a0/2, a0/2]]],
                 a=np.array([[0, a0, a0], [a0, 0, a0], [a0, a0, 0]]),
                 unit="bohr", basis="gth-dzv", pseudo="gth-pade", verbose=0)
    cell.mesh = [20, 20, 20]; cell.build()
    mf = cdft.RKS(cell, xc="pbe"); mf.exxdiv = None; mf.conv_tol = 1e-10; mf.kernel()
    nocc = cell.nelectron // 2; nmo = cell.nao; nvir = nmo - nocc
    eri = mf.with_df.get_mo_eri(mf.mo_coeff, compact=False)
    eri = eri.reshape(nmo, nmo, nmo, nmo).transpose(0, 2, 1, 3)
    vvvv = np.ascontiguousarray(eri[nocc:, nocc:, nocc:, nocc:])
    oovv = np.ascontiguousarray(eri[:nocc, :nocc, nocc:, nocc:])
    oooo = np.ascontiguousarray(eri[:nocc, :nocc, :nocc, :nocc])

    for multi in ("s", "t"):
        # CPU reference object
        cpu = ppRPA_Davidson(nocc, mf.mo_energy, Lpq=None, channel="hh",
                             nroot=2, residue_thresh=1e-10, trial="identity")
        cpu.mu = 0.0; cpu.use_eri(vvvv, oovv, oooo); cpu.multi = multi
        cpu.check_parameter()
        # GPU object (same dims), attach GPU contraction
        gpu = ppRPA_Davidson(nocc, mf.mo_energy, Lpq=None, channel="hh",
                             nroot=2, residue_thresh=1e-10, trial="identity")
        gpu.mu = 0.0; gpu.multi = multi; gpu.check_parameter()
        attach_gpu_eri_contraction(gpu, vvvv, oovv, oooo)
        rng = np.random.default_rng(0)
        tv = rng.standard_normal((7, cpu.full_dim))
        mv_cpu = _pprpa_contraction(cpu, tv)
        mv_gpu = gpu.contraction(tv)
        print(f"multi={multi}  full_dim={cpu.full_dim}  "
              f"max|gpu-cpu mv_prod| = {np.abs(mv_gpu - mv_cpu).max():.3e}")
