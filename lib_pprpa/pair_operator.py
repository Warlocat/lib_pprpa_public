"""The ERI part of the pp-RPA matrix as the pair-space operator of one multiplicity (NumPy).

The trial vectors of ppRPA_Davidson live on the pairs p >= q (singlet) or p > q
(triplet) of the occupied and of the virtual orbitals, with 1/sqrt(2) on p = q.  In
that basis the ERI part of the product is one symmetric matrix,

    M[(pq), (rs)] = f_pq f_rs (<pq|rs> + s <pq|sr>),  <pq|rs> = (pr|qs),
    f_pp = 1/sqrt(2), s = +1 (singlet) / -1 (triplet),

with blocks M_oo, M_vv and B = M[oo, vv], kept as row strips of the lower triangles of
M_oo and M_vv (the diagonal block of a strip in full and halved) and of B.  It is built
from the packed Gram blocks of ``pack_eri``, ``gram_from_lpq`` or
``gpu_ao2mo.gpu_ao2mo_blocks``.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pyscf import lib

__all__ = ['STRIP_ROWS', 'PairOperator', 'apply_pair_operator', 'check_gram',
           'gram_from_lpq', 'operator_strips', 'pack_eri', 'pack_offset', 'pair_offset',
           'pair_operator', 'pair_operator_bytes', 'row_strips']

# Pair rows per strip.  The diagonal block of a strip is stored in full, which adds
# rows / npair to the storage of a block; the GEMMs are bandwidth-bound at any width.
STRIP_ROWS = 1024
# Operator rows per build task: the index arrays of a task are a few times this many
# rows of the operator, so the threads together hold tens of MB beyond M.
_BUILD_ROWS = 64
# Gram rows per GEMM in gram_from_lpq: a block this size is the only transient beyond
# the packed output.
_GRAM_BLOCK_BYTES = 1 << 30


def pack_offset(n):
    '''Offset of row ``n`` of a packed lower triangle, n (n + 1) / 2: the number
    of pairs p >= q of n orbitals, and the length of an n-row triangle.'''
    n = int(n)
    return n * (n + 1) // 2


def pair_offset(p, multi):
    """Index of the first pair (p, .) in the trial-vector order (np.tril_indices)."""
    return pack_offset(p) if multi == "s" else pack_offset(p - 1)


def pair_operator_bytes(nocc, nvir, multi):
    """Bytes of M (lower triangles of M_oo, M_vv and all of B), before the diagonal blocks."""
    noo, nvv = pair_offset(nocc, multi), pair_offset(nvir, multi)
    return 8 * (pack_offset(noo) + pack_offset(nvv) + noo * nvv)


class PairOperator:
    """The pair-space operator of one multiplicity as row strips [(kind, R0, R1, S)],
    kind 'oo', 'vv' or 'ov' (B); see the module docstring."""

    def __init__(self, nocc, nvir, multi, strips):
        self.nocc, self.nvir, self.multi, self.strips = int(nocc), int(nvir), multi, strips

    @property
    def nbytes(self):
        return sum(strip[3].nbytes for strip in self.strips)


def row_strips(n, multi, rows, pa=0, pb=None):
    """[(pa, pb, R0, R1)]: whole first orbitals within [pa, pb) of an n-orbital pair
    space and their pair rows, at most ``rows`` orbitals or STRIP_ROWS pairs each (at
    least one orbital); empty ranges (orbital 0 of a triplet) are skipped."""
    pb = n if pb is None else pb
    out = []
    while pa < pb:
        R0 = pair_offset(pa, multi)
        if rows is not None:
            pc = min(pb, pa + max(1, int(rows)))
        else:
            pc = pa + 1
            while pc < pb and pair_offset(pc + 1, multi) - R0 <= STRIP_ROWS:
                pc += 1
        R1 = pair_offset(pc, multi)
        if R1 > R0:
            out.append((pa, pc, R0, R1))
        pa = pc
    return out


def operator_strips(nocc, nvir, multi, rows=None):
    """[(kind, pa, pb, R0, R1, nbytes)] of the whole operator; B runs over occupied rows."""
    out = []
    for kind, n in (("oo", nocc), ("vv", nvir), ("ov", nocc)):
        for pa, pb, R0, R1 in row_strips(n, multi, rows):
            ncol = pair_offset(nvir, multi) if kind == "ov" else R1
            out.append((kind, pa, pb, R0, R1, 8 * (R1 - R0) * ncol))
    return out


def _pack_rows(rows, P0, out):
    """Rows P0, P0 + 1, ... of the packed lower triangle ``out``: the leading P0 + k + 1
    columns of rows[k]."""
    off = pack_offset(P0)
    for k, row in enumerate(rows):
        n = P0 + k + 1
        out[off:off + n] = row[:n]
        off += n


def pack_eri(vvvv, oovv, oooo):
    """Packed Gram triangles (the format of ``gpu_ao2mo_blocks``) from physicist blocks
    vvvv[a,b,c,d] = <ab|cd>, oovv[i,j,a,b] = <ij|ab>, oooo: (pq|rs) for Q <= P at
    P (P + 1) / 2 + Q, P = (p, r) over the compact pairs p >= r of vvvv and oooo and
    over i nvir + a of oovv.  One first orbital per thread task, so no block is copied
    whole."""
    no, nv = oovv.shape[0], oovv.shape[2]
    out = {"vv": np.empty(pack_offset(pack_offset(nv))), "ov": np.empty(pack_offset(no * nv)),
           "oo": np.empty(pack_offset(pack_offset(no)))}
    tril = {"vv": np.tril_indices(nv), "oo": np.tril_indices(no)}

    def task(job):
        kind, a = job
        if kind == "ov":
            # rows (i, a): (ia|jb) = <ij|ab> over (j, b)
            rows = np.asarray(oovv[a]).transpose(1, 0, 2).reshape(nv, no * nv)
            _pack_rows(rows, a * nv, out[kind])
        else:
            # rows (a, c), c <= a: (ac|bd) = <ab|cd> over the compact (b, d)
            block = vvvv if kind == "vv" else oooo
            rows = np.asarray(block[a, :, :a + 1]).transpose(1, 0, 2)[:, tril[kind][0],
                                                                      tril[kind][1]]
            _pack_rows(rows, pack_offset(a), out[kind])

    jobs = [("vv", a) for a in reversed(range(nv))] + [("ov", i) for i in range(no)] + \
           [("oo", i) for i in range(no)]
    with ThreadPoolExecutor(lib.num_threads()) as ex:
        list(ex.map(task, jobs))
    return out["vv"], out["ov"], out["oo"]


def _gram(L):
    """Packed lower triangle of L^T L for L (naux, npair), _GRAM_BLOCK_BYTES of rows per
    GEMM, so the square matrix never exists."""
    n = L.shape[1]
    out = np.empty(pack_offset(n))
    blk = max(1, _GRAM_BLOCK_BYTES // (8 * n))
    for P0 in range(0, n, blk):
        P1 = min(n, P0 + blk)
        _pack_rows(L[:, P0:P1].T @ L[:, :P1], P0, out)
    return out


def gram_from_lpq(Lpq, nocc):
    """Packed Gram triangles vvvv, oovv, oooo (the format of ``pack_eri``) straight from
    real three-index factors Lpq[L, p, q] = Lpq[L, q, p] of the active orbitals,
    occupied first, (pq|rs) = sum_L Lpq[L, p, q] Lpq[L, r, s]; the four-index blocks
    are never formed."""
    Lpq = np.asarray(Lpq)
    naux, nmo = Lpq.shape[:2]
    o, v = slice(0, nocc), slice(nocc, nmo)
    Lvv = lib.pack_tril(np.ascontiguousarray(Lpq[:, v, v]))
    Lov = np.ascontiguousarray(Lpq[:, o, v]).reshape(naux, -1)
    Loo = lib.pack_tril(np.ascontiguousarray(Lpq[:, o, o]))
    return _gram(Lvv), _gram(Lov), _gram(Loo)


def check_gram(vvvv, oovv, oooo, nocc, nvir):
    """The packed Gram triangles as NumPy arrays, after checking their lengths."""
    blocks = {"vvvv": (vvvv, pack_offset(pack_offset(nvir))),
              "oovv": (oovv, pack_offset(nocc * nvir)),
              "oooo": (oooo, pack_offset(pack_offset(nocc)))}
    for name, (block, n) in blocks.items():
        if np.ndim(block) != 1 or np.shape(block)[0] != n:
            raise ValueError(f"{name}: expected a packed Gram block of {n} elements, got "
                             f"shape {np.shape(block)}")
    return tuple(np.asarray(block) for block, _ in blocks.values())


def _tri(a, b):
    hi, lo = np.maximum(a, b), np.minimum(a, b)
    return hi * (hi + 1) // 2 + lo


def _tri_tables(n):
    """T[p, r] = tri(p, r) for orbitals p, r < n, TT[x] = x (x + 1) / 2 over its values,
    and T2 = TT[T]."""
    orb = np.arange(n)
    T = _tri(orb[:, None], orb[None, :])
    x = np.arange(pack_offset(n))
    TT = x * (x + 1) // 2
    return T, TT[T], TT


def _fpair(p, q):
    return np.where(p == q, np.sqrt(0.5), 1.0)


def _sym_rows(out, G, T, T2, TT, pp, pq, R0, r0, r1, sgn):
    """Rows [r0, r1) of the strip of M_oo / M_vv that starts at row R0, over its columns
    [0, R1), from the compact Gram triangle G, <pq|rs> = (pr|qs) at tri(tri(p, r),
    tri(q, s)); the columns from R0 on form the diagonal block, which both GEMMs of a
    product use: halved."""
    p, q = pp[r0:r1, None], pq[r0:r1, None]
    r, s = pp[None, :out.shape[1]], pq[None, :out.shape[1]]
    # rows and columns are pairs p >= q and r >= s, so tri(p, r) >= tri(q, s)
    np.take(G, T2[p, r] + T[q, s], out=out, mode="clip")
    a, b = T[p, s], T[q, r]
    hi = np.maximum(a, b)
    b += a
    b -= hi
    term = np.take(G, TT[hi] + b, mode="clip")
    term *= sgn
    out += term
    out *= _fpair(p, q) * _fpair(r, s)
    out[:, R0:] *= 0.5


def _b_rows(out, G, TT, io, jo, av, bv, r0, r1, nvir, sgn):
    """Rows [r0, r1) of B from the oovv Gram triangle G, <ij|ab> = (ia|jb), Gram pair
    index i nvir + a."""
    i, j = io[r0:r1, None], jo[r0:r1, None]
    ivir, jvir = i * nvir, j * nvir
    # i >= j and a >= b, so i nvir + a >= j nvir + b; for i > j also i nvir + b > j nvir + a
    np.take(G, TT[ivir + av] + (jvir + bv), out=out, mode="clip")
    idx = TT[ivir + bv] + (jvir + av)
    diag = io[r0:r1] == jo[r0:r1]
    if diag.any():
        # i = j: <ii|ba> = <ii|ab>, the same Gram element as the first term
        idx[diag] = TT[ivir[diag] + av] + (jvir[diag] + bv)
    term = np.take(G, idx, mode="clip")
    term *= sgn
    out += term
    out *= _fpair(i, j) * _fpair(av, bv)


def pair_operator(vvvv, oovv, oooo, nocc, nvir, multi, rows=None):
    """The pair-space operator of multiplicity ``multi`` ('s' or 't') from the packed
    Gram blocks of ``pack_eri``, as NumPy strips (``rows`` as for ``row_strips``), built
    _BUILD_ROWS rows per thread task."""
    if multi not in ("s", "t"):
        raise ValueError(f"multi must be 's' or 't', got {multi!r}")
    sgn = 1.0 if multi == "s" else -1.0
    k = 0 if multi == "s" else -1
    pairs = {"oo": np.tril_indices(nocc, k), "vv": np.tril_indices(nvir, k)}
    gram = {"oo": np.asarray(oooo), "vv": np.asarray(vvvv), "ov": np.asarray(oovv)}
    tables = {"oo": _tri_tables(nocc), "vv": _tri_tables(nvir)}
    x = np.arange(nocc * nvir)
    tt_ov = x * (x + 1) // 2
    strips, tasks = [], []
    for kind, _, _, R0, R1, nbytes in operator_strips(nocc, nvir, multi, rows):
        S = np.empty((R1 - R0, nbytes // (8 * (R1 - R0))))
        strips.append((kind, R0, R1, S))
        tasks += [(kind, R0, r0, min(R1, r0 + _BUILD_ROWS), S)
                  for r0 in range(R0, R1, _BUILD_ROWS)]

    def task(job):
        kind, R0, r0, r1, S = job
        out = S[r0 - R0:r1 - R0]
        if kind == "ov":
            _b_rows(out, gram[kind], tt_ov, *pairs["oo"], *pairs["vv"], r0, r1, nvir, sgn)
        else:
            _sym_rows(out, gram[kind], *tables[kind], *pairs[kind], R0, r0, r1, sgn)

    with ThreadPoolExecutor(lib.num_threads()) as ex:
        list(ex.map(task, tasks))
    return PairOperator(nocc, nvir, multi, strips)


def apply_pair_operator(op, X):
    """M X for trial vectors X (full_dim, ntri) in the pair basis of ``op.multi``."""
    noo = pair_offset(op.nocc, op.multi)
    Y = np.zeros_like(X)
    Xo, Xv, Yo, Yv = X[:noo], X[noo:], Y[:noo], Y[noo:]
    for kind, R0, R1, S in op.strips:
        if kind == "ov":
            Yo[R0:R1] += S @ Xv
            Yv += S.T @ Xo[R0:R1]
        else:
            Xs, Ys = (Xo, Yo) if kind == "oo" else (Xv, Yv)
            Ys[R0:R1] += S @ Xs[:R1]
            Ys[:R1] += S.T @ Xs[R0:R1]
    return Y
