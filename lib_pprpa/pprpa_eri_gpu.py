"""GPU (cupy) Davidson products for the pp-RPA use_eri path, replacing the per-vector CPU
loop of pprpa_davidson._pprpa_contraction, from the active-space ERIs stored as the
pair-space operator of one multiplicity (built from the Gram rows of ``gpu_ao2mo``).

The trial vectors of ppRPA_Davidson live on the pairs p >= q (singlet) or p > q
(triplet) of the occupied and of the virtual orbitals, with 1/sqrt(2) on p = q.  In
that basis the ERI part of the product is one symmetric matrix,

    M[(pq), (rs)] = f_pq f_rs (<pq|rs> + s <pq|sr>),  <pq|rs> = (pr|qs),
    f_pp = 1/sqrt(2), s = +1 (singlet) / -1 (triplet),

with blocks M_oo, M_vv and B = M[oo, vv].  M_oo and M_vv are kept as row strips of
their lower triangles (the diagonal block of a strip in full and halved), B as row
strips, so a product is two GEMMs per strip, Y[R] += S X[:R1] and Y[:R1] += S^T X[R]
(for B: Y_oo[R] += S X_vv and Y_vv += S^T X_oo[R]), with no gather and no zeros.
Both terms of a row whose first orbital is p sit in Gram rows of first orbital
max(p, r) with r <= p, so a strip over the first orbitals [pa, pb) is converted on
the device from the Gram rows of those orbitals alone.  ``gpu_ao2mo_pair_operator``
does that to every strip of the FFT ao2mo as it is built, so the Gram blocks are
never stored and only M (about 2/3 of them; B is half of oovv) reaches the host.
``attach_gpu_eri_contraction`` takes that operator, or Gram blocks, which it converts
at the first product for the multiplicity the solver then has.
"""
import numpy as np
import cupy as cp
from cupy import cublas
from pyscf.lib import logger

from lib_pprpa.gpu_ao2mo import gpu_ao2mo_strips
from lib_pprpa.gpu_mem import empty_pinned, free_bytes, pack_offset, plan_budget
from lib_pprpa.gpu_multi import default_devices, each

__all__ = ['PairOperator', 'attach_gpu_eri_contraction', 'gpu_ao2mo_pair_operator',
           'pack_eri', 'pair_operator_bytes', 'release_gpu_eri']

# Pair rows per strip.  The diagonal block of a strip is stored in full, which adds
# rows / npair to the storage of a block; the GEMMs are bandwidth-bound at any width.
_STRIP_ROWS = 1024
# (ntri, full_dim) float64 arrays one product holds besides the strips: T, X, Y and
# the two of Y^T + orb T.
_PRODUCT_ARRAYS = 5

_PREAMBLE = r'''
__device__ long long tri(long long a, long long b) {
    return a > b ? a * (a + 1) / 2 + b : b * (b + 1) / 2 + a;
}
__device__ double fpair(long long p, long long q) {
    return p == q ? 0.70710678118654752440 : 1.0;
}
'''

# Rows [R0, R0 + n) and columns [0, ncol) of M_oo / M_vv from the compact Gram rows
# starting at packed offset ``base``: <pq|rs> = (pr|qs) at tri(tri(p, r), tri(q, s)).
# The columns from R0 on form the diagonal block, which both GEMMs use: halved.
_SYM_STRIP = cp.ElementwiseKernel(
    'raw float64 E, raw int32 pp, raw int32 pq, int64 base, int64 R0, int64 ncol, float64 sgn',
    'float64 out',
    '''
    const long long P = R0 + i / ncol, Q = i % ncol;
    const long long p = pp[P], q = pq[P], r = pp[Q], s = pq[Q];
    const double v = E[tri(tri(p, r), tri(q, s)) - base]
                     + sgn * E[tri(tri(p, s), tri(q, r)) - base];
    out = (Q >= R0 ? 0.5 : 1.0) * fpair(p, q) * fpair(r, s) * v;
    ''',
    'pprpa_pair_operator_sym', preamble=_PREAMBLE)

# Rows [R0, R0 + n) of B from the oovv Gram rows starting at ``base``: <ij|ab> =
# (ia|jb), Gram pair index i nv + a.
_B_STRIP = cp.ElementwiseKernel(
    'raw float64 E, raw int32 op, raw int32 oq, raw int32 vp, raw int32 vq, int64 base, '
    'int64 R0, int64 ncol, int64 nv, float64 sgn',
    'float64 out',
    '''
    const long long P = R0 + i / ncol, Q = i % ncol;
    const long long ii = op[P], j = oq[P], a = vp[Q], b = vq[Q];
    const double v = E[tri(ii * nv + a, j * nv + b) - base]
                     + sgn * E[tri(ii * nv + b, j * nv + a) - base];
    out = fpair(ii, j) * fpair(a, b) * v;
    ''',
    'pprpa_pair_operator_b', preamble=_PREAMBLE)


def _pair_offset(p, multi):
    """Index of the first pair (p, .) in the trial-vector order (np.tril_indices)."""
    return pack_offset(p) if multi == "s" else pack_offset(p - 1)


def pair_operator_bytes(nocc, nvir, multi):
    """Bytes of M (lower triangles of M_oo, M_vv and all of B), before the diagonal blocks."""
    noo, nvv = _pair_offset(nocc, multi), _pair_offset(nvir, multi)
    return 8 * (pack_offset(noo) + pack_offset(nvv) + noo * nvv)


class PairOperator:
    """The pair-space operator of one multiplicity as pinned host strips
    [(kind, R0, R1, S)], kind 'oo', 'vv' or 'ov' (B); see the module docstring."""

    def __init__(self, nocc, nvir, multi, strips):
        self.nocc, self.nvir, self.multi, self.strips = int(nocc), int(nvir), multi, strips

    @property
    def nbytes(self):
        return sum(strip[3].nbytes for strip in self.strips)


def gpu_ao2mo_pair_operator(cell, cocc, cvir, mesh, multi, rows=None, pair_blk=None,
                            devices=None, verbose=None):
    """The pair-space operator of multiplicity ``multi`` ('s' or 't') straight from the
    FFT ao2mo (``gpu_ao2mo.gpu_ao2mo_strips``): each Gram strip is converted on its
    device and only the operator strips reach the host.

    Kwargs:
        rows : first orbitals per operator strip; default up to _STRIP_ROWS pairs.
        pair_blk, devices : as for ``gpu_ao2mo.gpu_ao2mo_blocks``.
    Returns:
        A PairOperator in pinned host memory, for ``attach_gpu_eri_contraction``.
    """
    if multi not in ("s", "t"):
        raise ValueError(f"multi must be 's' or 't', got {multi!r}")
    nocc, nvir = np.asarray(cocc).shape[1], np.asarray(cvir).shape[1]
    sink = _OperatorSink(nocc, nvir, multi, rows)
    gpu_ao2mo_strips(cell, cocc, cvir, mesh, sink, pair_blk, devices, verbose)
    return PairOperator(nocc, nvir, multi, sink.strips)


def attach_gpu_eri_contraction(pprpa, *eri, mode="auto", rows=None, devices=None, verbose=None):
    """Route the use_eri Davidson products through the pair-space operator on the GPU.

    Args:
        eri : a PairOperator (``gpu_ao2mo_pair_operator``), or the packed Gram
            triangles vvvv, oovv, oooo of ``gpu_ao2mo_blocks`` (NumPy or CuPy;
            ``pack_eri`` converts physicist tensors), which are converted
            at the first product for the solver's multiplicity and then dropped.
    Kwargs:
        mode : 'auto', 'resident', 'split' or 'streamed'.  'auto' keeps the strips on
            the device when they fit its free memory beside one more strip (the
            Gram slice being converted) and the product arrays of the largest trial
            block (max_vec + nroot vectors), divides them over ``devices`` when an
            equal share plus one strip fits each, and streams them otherwise.
        rows : first orbitals per strip when converting Gram blocks; default up to
            _STRIP_ROWS pairs.
        devices : CUDA device ids for the split mode (default: the current device).
    """
    log = logger.new_logger(pprpa, logger.INFO if verbose is None else verbose)
    nocc, nvir = pprpa.nocc, pprpa.nvir
    if len(eri) == 1 and isinstance(eri[0], PairOperator):
        pre, blocks, multi, need = eri[0], None, eri[0].multi, eri[0].nbytes
        if (pre.nocc, pre.nvir) != (nocc, nvir):
            raise ValueError(f"operator for nocc={pre.nocc} nvir={pre.nvir}, solver has "
                             f"nocc={nocc} nvir={nvir}")
    elif len(eri) == 3:
        pre, blocks, multi = None, _check_packed(*eri, nocc, nvir), None
        need = pair_operator_bytes(nocc, nvir, "s")
    else:
        raise TypeError("expected a PairOperator or the vvvv, oovv, oooo Gram blocks")
    devices = default_devices(devices)
    if mode == "auto":
        if pre is not None:
            strip = max(S.nbytes for *_, S in pre.strips)
        else:
            strip = max(s[-1] for s in _strips(nocc, nvir, "s", rows))
        temps = 8 * _PRODUCT_ARRAYS * (pprpa.max_vec + pprpa.nroot) * (
            _pair_offset(nocc, "s") + _pair_offset(nvir, "s"))
        free = each(devices, lambda rank: free_bytes())
        nshare = max(devices.count(d) for d in devices)
        if need + strip + temps <= plan_budget(free[0]):
            mode = "resident"
        elif (len(devices) > 1 and
              need / len(devices) + strip + temps <= plan_budget(min(free) // nshare)):
            mode = "split"
        else:
            mode = "streamed"
    if mode not in ("resident", "split", "streamed"):
        raise ValueError(f"unknown mode {mode!r}")
    pprpa._use_eri = True
    pprpa._gpu_eri = dict(blocks=blocks, prebuilt=pre, mode=mode, rows=rows, devices=devices,
                          multi=multi, strips=None, log=log)
    # a tiny host array so kernel()'s `data_type = pprpa.vvvv.dtype` still works
    pprpa.vvvv = np.empty(0, dtype=np.float64)
    pprpa.oovv = None
    pprpa.oooo = None
    pprpa.contraction = lambda tri_vec: _contraction(pprpa, tri_vec)
    pprpa.eri_nbytes = lambda: _eri_nbytes(pprpa)
    log.info("GPU pair operator: mode=%s devices=%s operator=%.2f GB%s nocc=%d nvir=%d",
             mode, devices, need / 1e9, f" ({multi})" if pre else " (singlet, from Gram)",
             nocc, nvir)
    return pprpa


def release_gpu_eri(pprpa):
    """Drop the operator and the Gram blocks held by attach_gpu_eri_contraction and free
    the cupy pools; the solver keeps its eigenvalues and eigenvectors."""
    op = pprpa.__dict__.pop("_gpu_eri", None)
    pprpa.__dict__.pop("contraction", None)
    pprpa.__dict__.pop("eri_nbytes", None)
    if op is not None:
        devices = op["devices"]
        op.clear()
        each(devices, lambda rank: cp.get_default_memory_pool().free_all_blocks())
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


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


def _check_packed(vvvv, oovv, oooo, nocc, nvir):
    blocks = {"vvvv": (vvvv, pack_offset(pack_offset(nvir))),
              "oooo": (oooo, pack_offset(pack_offset(nocc))),
              "oovv": (oovv, pack_offset(nocc * nvir))}
    for name, (block, n) in blocks.items():
        if block.ndim != 1 or block.shape[0] != n:
            raise ValueError(f"{name}: expected a packed block of {n} elements, got shape "
                             f"{block.shape} (pack_eri converts physicist tensors)")
    return {name: block for name, (block, n) in blocks.items()}


def _row_strips(n, multi, rows, pa=0, pb=None):
    """[(pa, pb, R0, R1)]: whole first orbitals within [pa, pb) of an n-orbital pair
    space and their pair rows, at most ``rows`` orbitals or _STRIP_ROWS pairs each (at
    least one orbital); empty ranges (orbital 0 of a triplet) are skipped."""
    pb = n if pb is None else pb
    out = []
    while pa < pb:
        R0 = _pair_offset(pa, multi)
        if rows is not None:
            pc = min(pb, pa + max(1, int(rows)))
        else:
            pc = pa + 1
            while pc < pb and _pair_offset(pc + 1, multi) - R0 <= _STRIP_ROWS:
                pc += 1
        R1 = _pair_offset(pc, multi)
        if R1 > R0:
            out.append((pa, pc, R0, R1))
        pa = pc
    return out


def _strips(nocc, nvir, multi, rows):
    """[(kind, pa, pb, R0, R1, nbytes)] of the whole operator; B runs over occupied rows."""
    out = []
    for kind, n in (("oo", nocc), ("vv", nvir), ("ov", nocc)):
        for pa, pb, R0, R1 in _row_strips(n, multi, rows):
            ncol = _pair_offset(nvir, multi) if kind == "ov" else R1
            out.append((kind, pa, pb, R0, R1, 8 * (R1 - R0) * ncol))
    return out


def _pairs(n, multi):
    p, q = np.tril_indices(n, 0 if multi == "s" else -1)
    return cp.asarray(p, dtype=np.int32), cp.asarray(q, dtype=np.int32)


def _convert_strip(kind, R0, R1, E, base, pairs, nvir, sgn):
    """Operator rows [R0, R1) of ``kind`` on the device from the Gram rows ``E`` (packed
    offset ``base``) that hold their first orbitals."""
    if kind == "ov":
        ncol = pairs["vv"][0].size
        S = cp.empty((R1 - R0, ncol))
        _B_STRIP(E, *pairs["oo"], *pairs["vv"], base, R0, ncol, nvir, sgn, S)
    else:
        S = cp.empty((R1 - R0, R1))
        _SYM_STRIP(E, *pairs[kind], base, R0, R1, sgn, S)
    return S


class _OperatorSink:
    """Converts each Gram strip of ``gpu_ao2mo_strips`` into the operator strips over
    the same first orbitals and keeps those in pinned host memory."""
    _KIND = {"vvvv": "vv", "oooo": "oo", "oovv": "ov"}

    def __init__(self, nocc, nvir, multi, rows):
        self.nocc, self.nvir, self.multi, self.rows = nocc, nvir, multi, rows
        self.strips = []
        self._pairs = {}

    def _ncol(self, kind, R1):
        return _pair_offset(self.nvir, self.multi) if kind == "ov" else R1

    def device_bytes(self, name):
        """One converted strip (at least one orbital of pairs) and the pair index arrays."""
        kind = self._KIND[name]
        n = self.nvir if kind == "vv" else self.nocc
        npair = _pair_offset(self.nocc, self.multi) + _pair_offset(self.nvir, self.multi)
        return 8 * max(_STRIP_ROWS, n) * self._ncol(kind, _pair_offset(n, self.multi)) + 8 * npair

    def prepare(self, name, tasks):
        kind = self._KIND[name]
        n = self.nvir if kind == "vv" else self.nocc
        sgn = 1.0 if self.multi == "s" else -1.0
        plan = {}
        for pa, pb in tasks:
            plan[pa, pb] = []
            for _, _, R0, R1 in _row_strips(n, self.multi, self.rows, pa, pb):
                host = empty_pinned((R1 - R0, self._ncol(kind, R1)))
                self.strips.append((kind, R0, R1, host))
                plan[pa, pb].append((R0, R1, host))

        def emit(pa, pb, buf, base):
            pairs = self._device_pairs()
            for R0, R1, host in plan[pa, pb]:
                _convert_strip(kind, R0, R1, buf, base, pairs, self.nvir, sgn).get(out=host)
        return emit

    def _device_pairs(self):
        dev = cp.cuda.Device().id
        if dev not in self._pairs:
            self._pairs[dev] = {"oo": _pairs(self.nocc, self.multi),
                                "vv": _pairs(self.nvir, self.multi)}
        return self._pairs[dev]


def _convert(blocks, plan, nocc, nvir, multi, host):
    """Build the strips of ``plan`` on the current device from uploaded Gram slices;
    pinned host copies if ``host``."""
    sgn = 1.0 if multi == "s" else -1.0
    pairs = {"oo": _pairs(nocc, multi), "vv": _pairs(nvir, multi)}
    out = []
    for kind, pa, pb, R0, R1, _ in plan:
        if kind == "ov":
            g0, g1 = pack_offset(pa * nvir), pack_offset(pb * nvir)
            E = cp.asarray(blocks["oovv"][g0:g1])
        else:
            g0, g1 = pack_offset(pack_offset(pa)), pack_offset(pack_offset(pb))
            E = cp.asarray(blocks["vvvv" if kind == "vv" else "oooo"][g0:g1])
        S = _convert_strip(kind, R0, R1, E, g0, pairs, nvir, sgn)
        E = None
        if host:
            h = empty_pinned(S.shape)
            S.get(out=h)
            S = h
        out.append((kind, R0, R1, S))
    return out


def _assign(sizes, n):
    """Indices of ``sizes`` per rank: largest first onto the least-loaded rank."""
    parts, load = [[] for _ in range(n)], [0] * n
    for i in sorted(range(len(sizes)), key=sizes.__getitem__, reverse=True):
        rank = min(range(n), key=load.__getitem__)
        parts[rank].append(i)
        load[rank] += sizes[i]
    return parts


def _build(op, nocc, nvir):
    mode, devices, multi, pre = op["mode"], op["devices"], op["multi"], op["prebuilt"]
    t0 = op["log"].init_timer()
    if pre is not None:
        items = pre.strips
        if mode == "streamed":
            strips = items
        elif mode == "resident":
            strips = [(k, R0, R1, cp.asarray(S)) for k, R0, R1, S in items]
        else:
            parts = _assign([S.nbytes for *_, S in items], len(devices))
            strips = each(devices, lambda rank: [(k, R0, R1, cp.asarray(S)) for k, R0, R1, S
                                                 in (items[i] for i in parts[rank])])
    else:
        blocks = op["blocks"]
        plan = _strips(nocc, nvir, multi, op["rows"])
        if mode == "split":
            parts = _assign([s[-1] for s in plan], len(devices))
            strips = each(devices, lambda rank: _convert(
                blocks, [plan[i] for i in parts[rank]], nocc, nvir, multi, False))
        else:
            strips = _convert(blocks, plan, nocc, nvir, multi, mode == "streamed")
    op.update(strips=strips, blocks=None, prebuilt=None)
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()   # dropped host blocks, if unreferenced
    flat = sum(strips, []) if mode == "split" else strips
    op["log"].timer("pair operator placement" if pre is not None else "pair operator build", *t0)
    op["log"].info("GPU pair operator (%s, %s): %d strips, %.2f GB", multi, mode, len(flat),
                   sum(strip[3].nbytes for strip in flat) / 1e9)


def _eri_nbytes(pprpa):
    """Bytes of the operator strips for the solver's multiplicity, plus the Gram blocks
    they are converted from while those are held (until the first product)."""
    op = pprpa._gpu_eri
    if op["strips"] is not None:
        flat = sum(op["strips"], []) if op["mode"] == "split" else op["strips"]
        return sum(S.nbytes for *_, S in flat)
    if op["prebuilt"] is not None:
        return op["prebuilt"].nbytes
    plan = _strips(pprpa.nocc, pprpa.nvir, pprpa.multi, op["rows"])
    return sum(s[-1] for s in plan) + sum(b.nbytes for b in op["blocks"].values())


def _apply(strips, X, Y, noo):
    """Y += M X over ``strips`` (device or pinned host arrays)."""
    Xo, Xv, Yo, Yv = X[:noo], X[noo:], Y[:noo], Y[noo:]
    for kind, R0, R1, S in strips:
        S = cp.asarray(S)
        if kind == "ov":
            cublas.gemm("N", "N", S, Xv, out=Yo[R0:R1], beta=1.0)
            cublas.gemm("T", "N", S, Xo[R0:R1], out=Yv, beta=1.0)
        else:
            Xs, Ys = (Xo, Yo) if kind == "oo" else (Xv, Yv)
            cublas.gemm("N", "N", S, Xs[:R1], out=Ys[R0:R1], beta=1.0)
            cublas.gemm("T", "N", S, Xs[R0:R1], out=Ys[:R1], beta=1.0)


def _contraction(pprpa, tri_vec):
    op = pprpa._gpu_eri
    nocc, nvir, multi = pprpa.nocc, pprpa.nvir, pprpa.multi
    if op["multi"] is None:
        op["multi"] = multi
    if op["multi"] != multi:
        raise ValueError(f"pair operator is for multi={op['multi']!r}, the solver has "
                         f"{multi!r}; attach the ERIs again")
    if op["strips"] is None:
        _build(op, nocc, nvir)
    noo = _pair_offset(nocc, multi)
    T = cp.asarray(tri_vec)
    X = cp.ascontiguousarray(T.T)                 # (full_dim, ntri)
    if op["mode"] == "split":
        xh = X.get()

        def partial(rank):
            Xd = cp.asarray(xh)
            Yd = cp.zeros_like(Xd)
            _apply(op["strips"][rank], Xd, Yd, noo)
            return Yd.get()
        Y = cp.asarray(sum(each(op["devices"], partial)))
    else:
        Y = cp.zeros_like(X)
        _apply(op["strips"], X, Y, noo)

    # orbital-energy diagonal term: (e_p + e_q - 2 mu), hh block negated
    k = 0 if multi == "s" else -1
    me = cp.asarray(pprpa.mo_energy)
    eo, ev = me[:nocc], me[nocc:]
    orb = cp.concatenate(((eo[None, :] + eo[:, None])[cp.tril_indices(nocc, k)],
                          (ev[None, :] + ev[:, None])[cp.tril_indices(nvir, k)])) - 2.0 * pprpa.mu
    orb[:noo] *= -1.0
    return cp.asnumpy(Y.T + orb[None, :] * T)
