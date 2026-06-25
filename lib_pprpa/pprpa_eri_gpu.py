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

attach_gpu_eri_contraction(pprpa, vvvv, oovv, oooo) accepts numpy OR cupy blocks,
sets up the use_eri dims, and overrides pprpa.contraction.

Validate:  python pprpa_eri_gpu.py   (compares GPU mv_prod vs CPU mv_prod to ~1e-12)
"""
import math
import numpy as np
import cupy as cp

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def attach_gpu_eri_contraction(pprpa, vvvv, oovv, oooo):
    """Keep the active-space ERI on GPU and route Davidson MVP through cupy."""
    pprpa._use_eri = True
    nvir = pprpa.nvir
    nocc = pprpa.nocc
    # reshape to matmul form once: (nv^2,nv^2),(no^2,no^2),(no^2,nv^2)
    pprpa._gpu_vvvv = cp.ascontiguousarray(cp.asarray(vvvv).reshape(nvir * nvir, nvir * nvir))
    pprpa._gpu_oooo = cp.ascontiguousarray(cp.asarray(oooo).reshape(nocc * nocc, nocc * nocc))
    pprpa._gpu_oovv = cp.ascontiguousarray(cp.asarray(oovv).reshape(nocc * nocc, nvir * nvir))
    pprpa._gpu_mo_energy = cp.asarray(pprpa.mo_energy)
    # a tiny host array so kernel()'s `data_type = pprpa.vvvv.dtype` still works
    pprpa.vvvv = np.empty(0, dtype=np.float64)
    pprpa.oovv = None
    pprpa.oooo = None
    pprpa.contraction = lambda tri_vec: _gpu_eri_contraction(pprpa, tri_vec)
    return pprpa


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

    prod_vv = zvvT @ pprpa._gpu_vvvv.T            # vvvv . z_vv
    prod_oo = zooT @ pprpa._gpu_oooo.T            # oooo . z_oo
    prod_vv = prod_vv + zooT @ pprpa._gpu_oovv    # + oovv^T . z_oo
    prod_oo = prod_oo + zvvT @ pprpa._gpu_oovv.T  # + oovv   . z_vv

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
