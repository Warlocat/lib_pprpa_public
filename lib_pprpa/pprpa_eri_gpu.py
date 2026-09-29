"""GPU (cupy) batched ERI contraction for the pp-RPA Davidson use_eri path, from
the packed pair Gram matrices of ``gpu_ao2mo``.

Replaces the CPU per-trial-vector loop in lib_pprpa.pprpa_davidson._pprpa_contraction
(the `_use_eri` branch).  With Zs = z^T + s z (s = +1 singlet, -1 triplet) the
(anti)symmetrised product of the CPU routine is the exchange-like contraction

    K_vv[a,b] = sum_cd (ac|bd) Zs_vv[c,d] + sum_ij (ia|jb) Zs_oo[i,j]
    K_oo[i,j] = sum_kl (ik|jl) Zs_oo[k,l] + sum_ab (ia|jb) Zs_vv[a,b]

for all trial vectors at once.  Each stored element (pq|rs), Q <= P, stands for
its permutation images.  For a chunk of pair rows p the kernel gathers the
stored elements once as G[p, r, q, s] = w (pq|rs), with (r, s) run over both
orders for the compact blocks and w halving the elements whose images coincide
(Q = P, and q = p); two GEMMs then give K4[p, r] += sum_qs G Zs[q, s] and
K4[q, r] += sum_ps G Zs[p, s], and K = K4 + s K4^T supplies the (rs|pq) images.
The gather is bound by device bandwidth and the GEMMs cost the same flops as
the physicist form.

The blocks stay on the device when they fit (mode='resident'), are uploaded in
row chunks from pinned host memory once per MVP otherwise (mode='streamed'), or
are divided over several ``devices`` (mode='split', partial K4 summed on the
host).  attach_gpu_eri_contraction(pprpa, vvvv, oovv, oooo) accepts NumPy or
CuPy packed blocks, sets up the use_eri dims, and overrides pprpa.contraction.
"""
import math
import numpy as np
import cupy as cp
from pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import pin_memory

from lib_pprpa.gpu_mem import eri_bytes, fits_resident, free_bytes, is_pinned, pack_offset
from lib_pprpa.gpu_multi import default_devices, each

__all__ = ['attach_gpu_eri_contraction', 'release_gpu_eri', 'pack_eri']

_INV_SQRT2 = 1.0 / math.sqrt(2.0)
# Share of the free device memory the unfolded row chunk (and its upload) may occupy.
_CHUNK_MEM_FRAC = 0.3

# G[p', r, q, s] = w * E[P(p, q), Q(r, s)] for the pair rows p = pa + p' whose
# packed rows start at ``base``; 0 where the element is not stored (Q > P, or
# q > p of a compact block).  w = 1/2 per coincidence of images: Q == P, and
# q == p of a compact block (the (qp|rs) GEMM then repeats the (pq|rs) one).
_UNFOLD = cp.ElementwiseKernel(
    'raw float64 E, int64 base, int64 pa, int64 nR, int64 nQ, int64 nS, int32 compact',
    'float64 out',
    '''
    long long t = i;
    const long long s = t % nS; t /= nS;
    const long long q = t % nQ; t /= nQ;
    const long long r = t % nR; t /= nR;
    const long long p = pa + t;
    long long P, Q;
    double w = 1.0;
    bool stored = true;
    if (compact) {
        stored = q <= p;
        P = p * (p + 1) / 2 + q;
        const long long hi = r > s ? r : s;
        const long long lo = r > s ? s : r;
        Q = hi * (hi + 1) / 2 + lo;
        if (q == p) w *= 0.5;
    } else {
        P = p * nQ + q;
        Q = r * nS + s;
    }
    stored = stored && Q <= P;
    if (Q == P) w *= 0.5;
    out = stored ? w * E[P * (P + 1) / 2 + Q - base] : 0.0;
    ''',
    'pprpa_unfold_packed')


def attach_gpu_eri_contraction(pprpa, vvvv, oovv, oooo, mode="auto", rows=None, devices=None,
                               verbose=None):
    """Keep the packed active-space ERI on GPU (or stream it) and route Davidson MVP through cupy.

    Args:
        vvvv, oovv, oooo : the packed Gram triangles of ``gpu_ao2mo_blocks`` (NumPy
            or CuPy); ``pack_eri`` converts physicist tensors.
    Kwargs:
        mode : 'auto', 'resident', 'split' or 'streamed'.  'auto' keeps the
            blocks on the device when they are already cupy arrays or fit
            gpu_mem.RESIDENT_VRAM_FRAC of its memory, splits them over
            ``devices`` when their combined memory holds them, and streams them
            otherwise.  Streamed host blocks that are not already pinned are
            copied into pinned memory once.
        rows : pair rows of a block unfolded per chunk; default from the free
            device memory.
        devices : CUDA device ids for the split mode (default: the current device).
    """
    log = logger.new_logger(pprpa, logger.INFO if verbose is None else verbose)
    nocc, nvir = pprpa.nocc, pprpa.nvir
    blocks = _check_packed(vvvv, oovv, oooo, nocc, nvir)
    devices = default_devices(devices)
    if mode == "auto":
        on_device = all(isinstance(x, cp.ndarray) for x in blocks.values())
        if on_device or fits_resident(nocc, nvir):
            mode = "resident"
        elif len(devices) > 1 and fits_resident(
                nocc, nvir, total_bytes=len(devices) * _min_total_bytes(devices)):
            mode = "split"
        else:
            mode = "streamed"
    if mode not in ("resident", "split", "streamed"):
        raise ValueError(f"unknown mode {mode!r}")
    pprpa._use_eri = True
    split = None
    if mode == "resident":
        dev = {k: cp.asarray(v) for k, v in blocks.items()}
        pieces = _pieces(dev, nocc, nvir, rows, streamed=False)
    elif mode == "streamed":
        host = {k: _pinned_host(v) for k, v in blocks.items()}
        pieces = _pieces(host, nocc, nvir, rows, streamed=True)
    else:
        split = _split_upload(blocks, nocc, nvir, rows, devices)
        pieces = None
    pprpa._gpu_eri_pieces = pieces
    pprpa._gpu_eri_split = split
    pprpa._gpu_eri_mode = mode
    pprpa._gpu_mo_energy = cp.asarray(pprpa.mo_energy)
    # a tiny host array so kernel()'s `data_type = pprpa.vvvv.dtype` still works
    pprpa.vvvv = np.empty(0, dtype=np.float64)
    pprpa.oovv = None
    pprpa.oooo = None
    pprpa.contraction = lambda tri_vec: _gpu_eri_contraction(pprpa, tri_vec)
    chunks = pieces if pieces is not None else sum(split[1], [])
    log.info("GPU ERI contraction: mode=%s chunks=%d devices=%s packed eri=%.2f GB nocc=%d nvir=%d",
             mode, len(chunks), devices, eri_bytes(nocc, nvir) / 1e9, nocc, nvir)
    return pprpa


def pack_eri(vvvv, oovv, oooo):
    """Packed Gram triangles (the format of ``gpu_ao2mo_blocks``) from physicist
    blocks vvvv[a,b,c,d] = <ab|cd>, oovv[i,j,a,b] = <ij|ab>, oooo (NumPy or CuPy;
    meant for small blocks, e.g. tests and CPU-built integrals)."""
    def pack(block, compact):
        xp = cp.get_array_module(block)
        if compact:
            n = block.shape[0]
            E = block.transpose(0, 2, 1, 3).reshape(n * n, n * n)       # (ac|bd) at [(a,c),(b,d)]
            p, q = np.tril_indices(n)
            idx = xp.asarray(p * n + q)                                  # compact pairs p >= q
            E = E[idx][:, idx]
        else:
            no, nv = block.shape[0], block.shape[2]
            E = block.transpose(0, 2, 1, 3).reshape(no * nv, no * nv)   # (ia|jb) at [(i,a),(j,b)]
        r, c = (xp.asarray(x) for x in np.tril_indices(E.shape[0]))
        return xp.ascontiguousarray(E[r, c])
    return pack(vvvv, True), pack(oovv, False), pack(oooo, True)


def release_gpu_eri(pprpa):
    """Drop the ERI blocks held by attach_gpu_eri_contraction and free the cupy pools.

    The solver keeps its eigenvalues and eigenvectors; its contraction is no
    longer usable.
    """
    split = pprpa.__dict__.pop("_gpu_eri_split", None)
    for attr in ("_gpu_eri_pieces", "_gpu_eri_mode", "_gpu_mo_energy", "contraction"):
        pprpa.__dict__.pop(attr, None)
    if split is not None:
        devices, parts = split
        parts.clear()
        each(devices, lambda rank: cp.get_default_memory_pool().free_all_blocks())
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _check_packed(vvvv, oovv, oooo, nocc, nvir):
    blocks = {"vvvv": (vvvv, pack_offset(pack_offset(nvir))),
              "oooo": (oooo, pack_offset(pack_offset(nocc))),
              "oovv": (oovv, pack_offset(nocc * nvir))}
    for name, (block, n) in blocks.items():
        if block.ndim != 1 or block.shape[0] != n:
            raise ValueError(f"{name}: expected a packed block of {n} elements, got shape "
                             f"{block.shape} (pack_eri converts physicist tensors)")
    return {name: block for name, (block, n) in blocks.items()}


def _min_total_bytes(devices):
    def total(rank):
        return cp.cuda.runtime.memGetInfo()[1]
    return min(each(devices, total))


def _pinned_host(block):
    block = block.get() if isinstance(block, cp.ndarray) else np.asarray(block)
    return block if is_pinned(block) else pin_memory(block)


def _row_range(name, nocc, nvir, pa, pb):
    """Pair rows [P0, P1) of block rows [pa, pb) and the packed offset of P0."""
    if name == "oovv":
        P0, P1 = pa * nvir, pb * nvir
    else:
        P0, P1 = pack_offset(pa), pack_offset(pb)
    return P0, P1, pack_offset(P0)


def _pieces(blocks, nocc, nvir, rows, streamed, budget=None):
    """[(name, pa, pb, strip)]: row chunks of every block, ``strip`` the packed
    rows [pa, pb) (a view of the block).  Chunks are sized so that the unfolded
    G (and the upload, when streamed) stay within ``budget`` bytes."""
    if budget is None:
        budget = int(_CHUNK_MEM_FRAC * free_bytes())
    out = []
    for name, block in blocks.items():
        if name == "oovv":
            nrow = nocc
            per_row = 8 * nocc * nvir * nvir + (8 * nvir * nocc * nvir if streamed else 0)
        else:
            nrow = nvir if name == "vvvv" else nocc
            per_row = 8 * nrow**3 + (8 * nrow * pack_offset(nrow) if streamed else 0)
        r = rows if rows is not None else max(1, budget // max(1, per_row))
        r = max(1, min(nrow, int(r)))
        for pa in range(0, nrow, r):
            pb = min(nrow, pa + r)
            P0, P1, base = _row_range(name, nocc, nvir, pa, pb)
            out.append((name, pa, pb, block[base:pack_offset(P1)]))
    return out


def _split_upload(blocks, nocc, nvir, rows, devices):
    """Every device holds a share of the row chunks of each block (by bytes)."""
    host = {k: (v.get() if isinstance(v, cp.ndarray) else np.asarray(v)) for k, v in blocks.items()}
    n = len(devices)
    budget = int(_CHUNK_MEM_FRAC * _min_total_bytes(devices))
    pieces = _pieces(host, nocc, nvir, rows, streamed=False, budget=budget)
    parts = [[] for _ in devices]
    load = [0] * n
    for piece in pieces:                  # largest chunks first onto the lightest device
        rank = min(range(n), key=load.__getitem__)
        parts[rank].append(piece)
        load[rank] += piece[3].nbytes

    def upload(rank):
        return [(name, pa, pb, cp.asarray(strip)) for name, pa, pb, strip in parts[rank]]
    return devices, each(devices, upload)


def _chunk(K4vv, K4oo, name, strip, pa, pb, Zs_vv, Zs_oo, nocc, nvir):
    """Add the stored elements of rows [pa, pb) of one block (``strip``: their
    packed rows, on the device) to K4vv / K4oo."""
    ntri = Zs_vv.shape[0]
    np_ = pb - pa
    base = _row_range(name, nocc, nvir, pa, pb)[2]
    if name == "oovv":
        G = cp.empty((np_, nocc, nvir, nvir))            # G[i', j, a, b] = w (ia|jb)
        _UNFOLD(strip, base, pa, nocc, nvir, nvir, 0, G)
        Gm = G.reshape(np_ * nocc, nvir * nvir)
        A = Gm @ Zs_vv.reshape(ntri, -1).T                # K_oo[i, j] += sum_ab (ia|jb) Zs_vv[a, b]
        K4oo[:, pa:pb, :] += A.reshape(np_, nocc, ntri).transpose(2, 0, 1)
        C = Gm.T @ Zs_oo[:, pa:pb, :].reshape(ntri, -1).T  # K_vv[a, b] += sum_ij (ia|jb) Zs_oo[i, j]
        K4vv += C.reshape(nvir, nvir, ntri).transpose(2, 0, 1)
        return
    n, K4, Zs = (nvir, K4vv, Zs_vv) if name == "vvvv" else (nocc, K4oo, Zs_oo)
    nq = pb                                               # rows p hold q <= p < pb
    G = cp.empty((np_, n, nq, n))                         # G[p', r, q, s] = w (pq|rs)
    _UNFOLD(strip, base, pa, n, nq, n, 1, G)
    A = G.reshape(np_ * n, nq * n) @ Zs[:, :nq, :].reshape(ntri, -1).T
    K4[:, pa:pb, :] += A.reshape(np_, n, ntri).transpose(2, 0, 1)
    B = cp.zeros((n * nq, ntri))                          # K[q, r] += sum_ps (pq|rs) Zs[p, s]
    for p in range(np_):
        B += G[p].reshape(n * nq, n) @ Zs[:, pa + p, :].T
    K4[:, :nq, :] += B.reshape(n, nq, ntri).transpose(2, 1, 0)


def _accumulate(pieces, K4vv, K4oo, Zs_vv, Zs_oo, nocc, nvir):
    for name, pa, pb, strip in pieces:
        _chunk(K4vv, K4oo, name, cp.asarray(strip), pa, pb, Zs_vv, Zs_oo, nocc, nvir)


def _split_products(split, Zs_vv, Zs_oo, nocc, nvir):
    """K4 partials from each device's chunks, summed on the host."""
    devices, parts = split
    zv, zo = Zs_vv.get(), Zs_oo.get()
    ntri = zv.shape[0]

    def partial(rank):
        K4vv = cp.zeros((ntri, nvir, nvir))
        K4oo = cp.zeros((ntri, nocc, nocc))
        _accumulate(parts[rank], K4vv, K4oo, cp.asarray(zv), cp.asarray(zo), nocc, nvir)
        return K4vv.get(), K4oo.get()
    out = each(devices, partial)
    return cp.asarray(sum(o[0] for o in out)), cp.asarray(sum(o[1] for o in out))


def _gpu_eri_contraction(pprpa, tri_vec):
    nocc, nvir = pprpa.nocc, pprpa.nvir
    oo_dim = pprpa.oo_dim
    sign = 1.0 if pprpa.multi == "s" else -1.0
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
    # the CPU routine contracts z^T and then (anti)symmetrises the product
    Zs_oo = z_oo.transpose(0, 2, 1) + sign * z_oo
    Zs_vv = z_vv.transpose(0, 2, 1) + sign * z_vv

    if pprpa._gpu_eri_split is not None:
        K4vv, K4oo = _split_products(pprpa._gpu_eri_split, Zs_vv, Zs_oo, nocc, nvir)
    else:
        K4vv = cp.zeros((ntri, nvir, nvir))
        K4oo = cp.zeros((ntri, nocc, nocc))
        _accumulate(pprpa._gpu_eri_pieces, K4vv, K4oo, Zs_vv, Zs_oo, nocc, nvir)
    prod_vv = K4vv + sign * K4vv.transpose(0, 2, 1)
    prod_oo = K4oo + sign * K4oo.transpose(0, 2, 1)

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
