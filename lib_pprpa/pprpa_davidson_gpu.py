"""GPU AO-direct contraction for the pbc Gamma-point pp-RPA Davidson solver.

cupy port of the AO-direct branch of ``pprpa_davidson._pprpa_contraction``.  The
heavy exchange build runs on a gpu4pyscf pbc KRKS via ``get_k`` (validated to
match CPU FFTDF to ~1e-12 for symmetric/antisymmetric trial densities); the
small Davidson subspace bookkeeping stays on CPU, so the contraction returns
numpy and the stock driver (kernel/subspace_diag/expand_space) is unchanged.

Usage (AO-direct pp-RPA object from pprpaobj(mf, ..., mo_eri=False)):
    from gpu4pyscf.pbc import dft as gdft
    kmf = gdft.KRKS(cell, kpts=[[0,0,0]], xc=mf.xc); kmf.exxdiv = None
    kmf.mo_coeff=[cp.asarray(mf.mo_coeff)]; kmf.mo_energy=[...]; kmf.mo_occ=[...]
    attach_gpu_contraction(pprpa, kmf)
    pprpa.kernel("t")
"""
import types
import numpy as np
import cupy as cp

_INV_SQRT2 = 1.0 / np.sqrt(2.0)


def _contraction_gpu(pprpa, tri_vec):
    nocc, nvir = pprpa.nocc, pprpa.nvir
    scf = pprpa._scf
    # gpu4pyscf KRKS is not a subclass of pyscf KSCF; duck-type a k-point SCF.
    assert getattr(scf, "kpts", None) is not None, \
        "GPU pp-RPA contraction requires a (gpu) k-point SCF as _scf"
    assert len(scf.kpts) == 1 and abs(np.asarray(scf.kpts)).max() < 1e-9, \
        "Only Gamma-point KSCF is supported in GPU AO-direct contraction."
    cell = scf.cell
    kpts = np.zeros((1, 3))
    nel = cell.nelectron // 2
    mc = cp.asarray(scf.mo_coeff[0])[:, nel - nocc: nel + nvir]
    mo_o, mo_v = mc[:, :nocc], mc[:, nocc:]
    mo_energy = cp.asarray(pprpa.mo_energy)

    # Singlet amplitudes are symmetric (include the diagonal: tril offset 0),
    # triplet antisymmetric (strictly-lower: offset -1).  These index the packing
    # of each trial vector into the oo/vv blocks (matches the CPU contraction).
    is_s = 1 if pprpa.multi == "s" else 0
    tro, tco = cp.tril_indices(nocc, is_s - 1)
    trv, tcv = cp.tril_indices(nvir, is_s - 1)

    ntri = tri_vec.shape[0]
    tv = cp.asarray(tri_vec)
    nao = cell.nao

    # Unpack each trial vector into full oo/vv MO matrices, transform to an AO
    # density (dms), contract with exchange (get_k), then transform back and
    # repack -- a cupy port of pprpa_davidson._pprpa_contraction (AO-direct).
    # The 1/sqrt(2) on the diagonal is the pp-RPA normalization convention.
    # trial-vector AO densities (vectorised over the whole subspace block)
    z_oo = cp.zeros((ntri, nocc, nocc))
    z_vv = cp.zeros((ntri, nvir, nvir))
    if nocc > 0:
        z_oo[:, tro, tco] = tv[:, :pprpa.oo_dim]
        z_oo[:, cp.arange(nocc), cp.arange(nocc)] *= _INV_SQRT2
    if nvir > 0:
        z_vv[:, trv, tcv] = tv[:, pprpa.oo_dim:]
        z_vv[:, cp.arange(nvir), cp.arange(nvir)] *= _INV_SQRT2
    dms = cp.zeros((ntri, nao, nao))
    if nvir > 0:
        dms += cp.einsum('pa,nba,qb->npq', mo_v, z_vv, mo_v)
    if nocc > 0:
        dms += cp.einsum('pi,nji,qj->npq', mo_o, z_oo, mo_o)

    # exchange on GPU via the single-kpt fft_jk.get_k (reshapes (nset,nao,nao);
    # the KRKS/FFTDF get_jk wrappers misread a 4D dm as KUHF spin=2).  The batch
    # is memory-bound (each density's exchange intermediate ~ nao*ngrid): batch
    # the whole subspace at small nao (~5x speedup), chunk down toward per-dm at
    # large nao so it fits GPU memory.  Matches CPU FFTDF get_k to ~1e-12.
    from gpu4pyscf.pbc.df import fft_jk
    ngrid = int(np.prod(cell.mesh))
    chunk = max(1, int(3.0e9 / (nao * ngrid * 16)))
    K = cp.empty((ntri, nao, nao))
    for s in range(0, ntri, chunk):
        Kc = fft_jk.get_k(pprpa._gpu_fftdf, dms[s:s+chunk], hermi=0,
                          kpt=np.zeros(3), exxdiv=None)
        K[s:s+chunk] = cp.asarray(Kc).reshape(-1, nao, nao)
        cp.get_default_memory_pool().free_all_blocks()

    # Back-transform K to MO oo/vv blocks, (anti)symmetrize per multiplicity,
    # rotate upper->lower triangle, and repack into the trial-product vector.
    sign = 1.0 if pprpa.multi == "s" else -1.0
    mv = cp.zeros((ntri, pprpa.full_dim))
    if nocc > 0:
        p_oo = cp.einsum('pi,npq,qj->nij', mo_o, K, mo_o)
        p_oo = p_oo + sign * p_oo.transpose(0, 2, 1)
        p_oo = cp.ascontiguousarray(p_oo.transpose(0, 2, 1))   # upper -> lower
        p_oo[:, cp.arange(nocc), cp.arange(nocc)] *= _INV_SQRT2
        mv[:, :pprpa.oo_dim] = p_oo[:, tro, tco]
    if nvir > 0:
        p_vv = cp.einsum('pa,npq,qb->nab', mo_v, K, mo_v)
        p_vv = p_vv + sign * p_vv.transpose(0, 2, 1)
        p_vv = cp.ascontiguousarray(p_vv.transpose(0, 2, 1))
        p_vv[:, cp.arange(nvir), cp.arange(nvir)] *= _INV_SQRT2
        mv[:, pprpa.oo_dim:] = p_vv[:, trv, tcv]

    # Orbital-energy diagonal: (e_p + e_q - 2*mu); the oo (hole-hole) block
    # carries a -1 sign relative to the vv (particle-particle) block.
    orb_oo = (mo_energy[None, :nocc] + mo_energy[:nocc, None])[tro, tco]
    orb_vv = (mo_energy[None, nocc:] + mo_energy[nocc:, None])[trv, tcv]
    orb = cp.concatenate((orb_oo, orb_vv)) - 2.0 * pprpa.mu
    orb[:pprpa.oo_dim] *= -1.0
    mv += orb * tv
    # return numpy: the (small) Davidson subspace bookkeeping stays on CPU
    return cp.asnumpy(mv)


def attach_gpu_contraction(pprpa, gpu_scf):
    """Switch an AO-direct pp-RPA object to the GPU contraction with a gpu KSCF."""
    from gpu4pyscf.pbc.df.fft import FFTDF
    assert pprpa._ao_direct, "GPU contraction requires _ao_direct=True (mo_eri=False)"
    pprpa._scf = gpu_scf
    pprpa.mo_coeff = None
    # dedicated FFTDF for batched get_k over the trial-vector subspace
    pprpa._gpu_fftdf = FFTDF(gpu_scf.cell, np.zeros((1, 3)))
    pprpa.contraction = types.MethodType(_contraction_gpu, pprpa)
    return pprpa
