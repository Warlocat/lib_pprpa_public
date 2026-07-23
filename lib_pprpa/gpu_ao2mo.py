"""GPU (cupy) Gamma-point FFT ao2mo for pp-RPA: builds the active-space
vvvv / oovv / oooo blocks directly on the GPU (no CPU ao2mo, no GDF).

Same FFT MO-integral transform as pyscf.pbc.df.fft_ao2mo, in cupy:
  rho_pq(r) = mo_p(r) mo_q(r)  ->  fft  ->  eri_chem[(pq),(rs)] = sum_G rho_pq(G)* wG rho_rs(G)
then permuted to the physicist convention pp-RPA expects (matches pprpaobj):
  vvvv[a,b,c,d] = <ab|cd>,  oovv[i,j,a,b] = <ij|ab>,  oooo[i,j,k,l] = <ij|kl>.

Memory: the codensities (n_pair, ngrid) are built ON THE FLY in pair_blk-sized
chunks for BOTH the left (FFT'd) and right (contracted) index, so the full
(n_pair, ngrid) array is never materialized.  Peak ~ 4 * pair_blk * ngrid * 8 B,
independent of the active-space size.  pair_blk defaults to an automatic value
from the free GPU memory (pass an int to override).

Validated element-wise vs CPU pprpaobj(mo_eri=True) (run this file as __main__).
"""
import numpy as np
import cupy as cp
from gpu4pyscf.pbc import tools as gtools
from gpu4pyscf.pbc.dft import numint as gnumint

try:
    from gpu4pyscf.lib.cupy_helper import get_avail_mem
except Exception:                                        # pragma: no cover
    def get_avail_mem():
        free, _ = cp.cuda.runtime.memGetInfo()
        return free


def _mo_on_grid(cell, mo, mesh):
    """MO values on the uniform grid: (nmo, ngrid) cupy real (Gamma)."""
    coords = cell.gen_uniform_grids(mesh)
    ao = gnumint.eval_ao_kpts(cell, coords, kpts=np.zeros((1, 3)), deriv=0)[0]
    ao = cp.asarray(ao)                       # (ngrid, nao), real at Gamma
    return cp.asarray(mo).T @ ao.T            # (nmo, ngrid)


def _auto_pair_blk(ng, npair):
    """Choose pair_blk as LARGE as free GPU memory allows: the codensity double
    loop costs (npair/pair_blk)**2 iterations, so big blocks are strongly preferred
    (small blocks are pathologically slow).  Peak per block is dominated by the FFT
    (complex, 16 B) plus its cuFFT workspace (~2x) and a couple real buffers, ~64
    B/element; budget ~45% of free memory, floor 512 for stability."""
    avail = float(get_avail_mem())
    blk = int(0.45 * avail / (ng * 64.0))
    blk = max(512, blk)
    return min(blk, npair) if npair else blk


def _block_lowmem(moA, moB, wcoulG, mesh, pair_blk):
    """eri_chem[(ab),(cd)] = sum_r v_ab(r) rho_cd(r), v = ifft(wcoulG*fft(rho)),
    with rho_ab = moA[a]*moB[b].  Codensities built on the fly in pair_blk chunks
    for both indices (never the full (npair, ngrid) array).  Matches
    pyscf.pbc.df.fft_ao2mo._contract_compact (same FFT norm)."""
    nA, ng = moA.shape[0], moA.shape[1]
    nB = moB.shape[0]
    npair = nA * nB
    if pair_blk is None:
        pair_blk = _auto_pair_blk(ng, npair)
    idx = cp.arange(npair)
    ai = idx // nB                               # flat pair p -> (a, b), row-major
    bi = idx % nB
    out = cp.empty((npair, npair))
    for p0 in range(0, npair, pair_blk):
        p1 = min(p0 + pair_blk, npair)
        rhoL = moA[ai[p0:p1]] * moB[bi[p0:p1]]                    # (blk, ng)
        vR = gtools.ifft(gtools.fft(rhoL, mesh) * wcoulG, mesh).real
        rhoL = None
        for q0 in range(0, npair, pair_blk):
            q1 = min(q0 + pair_blk, npair)
            rhoR = moA[ai[q0:q1]] * moB[bi[q0:q1]]                # (blk, ng)
            out[p0:p1, q0:q1] = vR.dot(rhoR.T)
            rhoR = None
        vR = None
    return out


def gpu_ao2mo_blocks(cell, cocc, cvir, mesh, pair_blk=None, return_gpu=False):
    """Return vvvv, oovv, oooo (physicist, real).  cupy if return_gpu else numpy.

    pair_blk=None (default) auto-selects the chunk size from free GPU memory.
    Pass an int to force a specific block size (smaller = less memory, more FFTs).
    """
    moO = _mo_on_grid(cell, cocc, mesh)
    moV = _mo_on_grid(cell, cvir, mesh)
    no, nv, ng = moO.shape[0], moV.shape[0], moO.shape[1]
    coulG = cp.asarray(gtools.get_coulG(cell, mesh=mesh))
    wcoulG = coulG * (cell.vol / ng)

    # vvvv: chem (ac|bd) -> <ab|cd> via transpose(0,2,1,3)
    vvvv = _block_lowmem(moV, moV, wcoulG, mesh, pair_blk).reshape(nv, nv, nv, nv)
    vvvv = cp.ascontiguousarray(vvvv.transpose(0, 2, 1, 3))
    # oooo: chem (ik|jl) -> <ij|kl>
    oooo = _block_lowmem(moO, moO, wcoulG, mesh, pair_blk).reshape(no, no, no, no)
    oooo = cp.ascontiguousarray(oooo.transpose(0, 2, 1, 3))
    # oovv: chem (ia|jb) -> <ij|ab>; pairs (i,a) on both sides
    ovov = _block_lowmem(moO, moV, wcoulG, mesh, pair_blk).reshape(no, nv, no, nv)
    oovv = cp.ascontiguousarray(ovov.transpose(0, 2, 1, 3))     # (i,j,a,b)
    if return_gpu:
        return vvvv, oovv, oooo
    return cp.asnumpy(vvvv), cp.asnumpy(oovv), cp.asnumpy(oooo)


if __name__ == "__main__":
    # validate vs CPU pprpaobj(mo_eri=True) on a small cell
    from pyscf.pbc import gto, dft as cdft
    a0 = 3.370137329
    cell = gto.M(atom=[["C", [0., 0., 0.]], ["C", [a0/2, a0/2, a0/2]]],
                 a=np.array([[0, a0, a0], [a0, 0, a0], [a0, a0, 0]]),
                 unit="bohr", basis="gth-szv", pseudo="gth-pade", verbose=0)
    cell.mesh = [20, 20, 20]; cell.build()
    mf = cdft.RKS(cell, xc="pbe"); mf.exxdiv = None; mf.conv_tol = 1e-10; mf.kernel()
    nocc = cell.nelectron // 2; nvir = cell.nao - nocc; nmo = cell.nao
    cocc = mf.mo_coeff[:, :nocc]; cvir = mf.mo_coeff[:, nocc:]
    eri = mf.with_df.get_mo_eri(mf.mo_coeff, compact=False)
    eri = eri.reshape(nmo, nmo, nmo, nmo).transpose(0, 2, 1, 3)   # -> <pq|rs>
    vvvv_c = eri[nocc:, nocc:, nocc:, nocc:]
    oovv_c = eri[:nocc, :nocc, nocc:, nocc:]
    oooo_c = eri[:nocc, :nocc, :nocc, :nocc]
    # test both auto and a forced small pair_blk
    for pb in (None, 3):
        vvvv_g, oovv_g, oooo_g = gpu_ao2mo_blocks(cell, cocc, cvir, cell.mesh, pair_blk=pb)
        print(f"pair_blk={pb}:")
        for name, g, c in [("vvvv", vvvv_g, vvvv_c), ("oovv", oovv_g, oovv_c), ("oooo", oooo_g, oooo_c)]:
            print(f"  {name}: shape {g.shape}  max|gpu-cpu| = {np.abs(g - np.asarray(c)).max():.3e}")
