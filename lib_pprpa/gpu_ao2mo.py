"""GPU (cupy) Gamma-point FFT ao2mo for pp-RPA: builds the active-space
vvvv / oovv / oooo blocks directly on the GPU (no CPU ao2mo, no GDF).

Same FFT MO-integral transform as pyscf.pbc.df.fft_ao2mo, in cupy:
  rho_pq(r) = mo_p(r) mo_q(r)  ->  fft  ->  eri_chem[(pq),(rs)] = sum_G rho_pq(G)* wG rho_rs(G)
then permuted to the physicist convention pp-RPA expects (matches pprpaobj):
  vvvv[a,b,c,d] = <ab|cd>,  oovv[i,j,a,b] = <ij|ab>,  oooo[i,j,k,l] = <ij|kl>.

Validated element-wise vs CPU pprpaobj(mo_eri=True) (run this file as __main__).
"""
import numpy as np
import cupy as cp
from gpu4pyscf.pbc import tools as gtools
from gpu4pyscf.pbc.dft import numint as gnumint


def _mo_on_grid(cell, mo, mesh):
    """MO values on the uniform grid: (nmo, ngrid) cupy real (Gamma)."""
    coords = cell.gen_uniform_grids(mesh)
    ao = gnumint.eval_ao_kpts(cell, coords, kpts=np.zeros((1, 3)), deriv=0)[0]
    ao = cp.asarray(ao)                       # (ngrid, nao), real at Gamma
    return cp.asarray(mo).T @ ao.T            # (nmo, ngrid)


def _codensities(moA, moB):
    """Real codensities mo_a(r)*mo_b(r): (nA*nB, ngrid)."""
    nA, ng = moA.shape[0], moA.shape[1]
    nB = moB.shape[0]
    return (moA[:, None, :] * moB[None, :, :]).reshape(nA * nB, ng)


def _block(rho, wcoulG, mesh, pair_blk=4000):
    """eri_chem[(ab),(cd)] = sum_r v_ab(r) rho_cd(r), v = ifft(wcoulG*fft(rho)).
    This matches pyscf.pbc.df.fft_ao2mo._contract_compact (same FFT norm)."""
    npair = rho.shape[0]
    out = cp.empty((npair, npair))
    for p0 in range(0, npair, pair_blk):
        p1 = min(p0 + pair_blk, npair)
        vR = gtools.ifft(gtools.fft(rho[p0:p1], mesh) * wcoulG, mesh).real
        out[p0:p1] = vR.dot(rho.T)
    return out


def gpu_ao2mo_blocks(cell, cocc, cvir, mesh, pair_blk=2000, return_gpu=False):
    """Return vvvv, oovv, oooo (physicist, real).  cupy if return_gpu else numpy."""
    moO = _mo_on_grid(cell, cocc, mesh)
    moV = _mo_on_grid(cell, cvir, mesh)
    no, nv, ng = moO.shape[0], moV.shape[0], moO.shape[1]
    coulG = cp.asarray(gtools.get_coulG(cell, mesh=mesh))
    wcoulG = coulG * (cell.vol / ng)

    # vvvv: chem (ac|bd) -> <ab|cd> via transpose(0,2,1,3)
    rho = _codensities(moV, moV)
    vvvv = _block(rho, wcoulG, mesh, pair_blk).reshape(nv, nv, nv, nv)
    vvvv = cp.ascontiguousarray(vvvv.transpose(0, 2, 1, 3)); rho = None
    # oooo: chem (ik|jl) -> <ij|kl>
    rho = _codensities(moO, moO)
    oooo = _block(rho, wcoulG, mesh, pair_blk).reshape(no, no, no, no)
    oooo = cp.ascontiguousarray(oooo.transpose(0, 2, 1, 3)); rho = None
    # oovv: chem (ia|jb) -> <ij|ab>; pairs (i,a) on both sides
    rho = _codensities(moO, moV)
    ovov = _block(rho, wcoulG, mesh, pair_blk).reshape(no, nv, no, nv); rho = None
    oovv = cp.ascontiguousarray(ovov.transpose(0, 2, 1, 3))     # (i,j,a,b)
    if return_gpu:
        return vvvv, oovv, oooo
    return cp.asnumpy(vvvv), cp.asnumpy(oovv), cp.asnumpy(oooo)


if __name__ == "__main__":
    # validate vs CPU pprpaobj(mo_eri=True) on a small cell
    from pyscf.pbc import gto, dft as cdft
    from lib_pprpa.grad.ase_utils import pprpaobj
    a0 = 3.370137329
    cell = gto.M(atom=[["C", [0., 0., 0.]], ["C", [a0/2, a0/2, a0/2]]],
                 a=np.array([[0, a0, a0], [a0, 0, a0], [a0, a0, 0]]),
                 unit="bohr", basis="gth-szv", pseudo="gth-pade", verbose=0)
    cell.mesh = [20, 20, 20]; cell.build()
    mf = cdft.RKS(cell, xc="pbe"); mf.exxdiv = None; mf.conv_tol = 1e-10; mf.kernel()
    nocc = cell.nelectron // 2; nvir = cell.nao - nocc; nmo = cell.nao
    cocc = mf.mo_coeff[:, :nocc]; cvir = mf.mo_coeff[:, nocc:]
    # CPU reference blocks: replicate pprpaobj's mo_eri path exactly
    eri = mf.with_df.get_mo_eri(mf.mo_coeff, compact=False)
    eri = eri.reshape(nmo, nmo, nmo, nmo).transpose(0, 2, 1, 3)   # -> <pq|rs>
    vvvv_c = eri[nocc:, nocc:, nocc:, nocc:]
    oovv_c = eri[:nocc, :nocc, nocc:, nocc:]
    oooo_c = eri[:nocc, :nocc, :nocc, :nocc]
    vvvv_g, oovv_g, oooo_g = gpu_ao2mo_blocks(cell, cocc, cvir, cell.mesh)
    for name, g, c in [("vvvv", vvvv_g, vvvv_c), ("oovv", oovv_g, oovv_c), ("oooo", oooo_g, oooo_c)]:
        print(f"{name}: shape {g.shape} vs {np.asarray(c).shape}  "
              f"max|gpu-cpu| = {np.abs(g - np.asarray(c)).max():.3e}")
