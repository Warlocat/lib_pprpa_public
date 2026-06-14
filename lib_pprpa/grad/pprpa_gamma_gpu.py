"""GPU (gpu4pyscf) accelerated pp-RPA Gamma-point nuclear gradient.

Drop-in GPU counterpart of ``lib_pprpa.grad.pprpa_gamma``.  The relaxed 1-RDM /
energy-weighted density is built by the CPU routine ``make_rdm1_relaxed_rhf_pprpa``
(small MO-space block algebra) but driven by a GPU response (``make_gpu_vresp``),
so the CPHF/Z-vector solve runs on the GPU; the gradient *assembly* (hcore,
overlap, J/K, Vxc, fxc, pseudo-potential) also runs on the GPU.

Term -> gpu4pyscf primitive map (D=KS ref dm, P=relaxed corr dm, T=D+P,
W=energy-weighted dm, X=pp-RPA 2-RDM amplitude density):
  hcore   : krhf_g.hcore_generator, einsum('kxij,kji->x', h1ao, T)
  ovlp    : krhf_g.contract_h1e_dm(s1, W)            (note +=, W carries -dme0)
  J ref   : polarization Q(T)-Q(P) via jk_energy_per_atom(j=1,sr=lr=hyb), FFTDF
  hyb K   : Q(T)-Q(P) via jk_energy_per_atom(j=0,sr=lr=hyb), AFTDF   (only if hyb)
  pairing : ek(0.5(X-X^T)) - ek(0.5(X+X^T)) via jk_energy_per_atom(sr=lr=2), AFTDF
  Vxc     : 2*einsum(v1ao[1:,atom], T) from grad_utils_gpu_pbc._contract_xc_kernel
  fxc     : 1*einsum(f1vo[1:,atom], D)  (same routine; factor 1 = CPU's 0.5*..*2)
  PP nl   : krhf_g.vppnl_nuc_grad(T)

Key conventions / gpu4pyscf quirks (see also grad_utils_gpu_pbc):
* J derivatives use FFTDF (``get_j_e1`` exists, fast); K derivatives use AFTDF
  because FFTDF ``get_k_e1`` is NotImplemented.  AFT matches FFT to ~1e-8.
* pp-RPA pairing exchange: ``get_ek_ip1`` assumes a symmetric density -> it gives
  +Q_K for the antisymmetric part of X (triplet) and -Q_K for the symmetric part
  (singlet); splitting X handles both.  sr/lr=2 because that routine carries the
  SCF's 0.5 exchange factor while pp-RPA pairing is the full bare exchange.

Requirements / scope
--------------------
* Gamma point, RKS (or RHF) reference only; LDA/GGA functionals.
* For large nao the AFT/FFT exchange kernels need the memory-adaptive ``blksize``
  patch in ``pbc/df/{fft_jk,aft_jk}.py`` (else OOM at fine grids).
* Validated: vs CPU 2e-10 (C2 diamond, up to gth-tzv2p, singlet+triplet);
  vs finite difference 2.9e-7 (C2, off-grid).  Runs on the 63-atom NV cell.
"""
import numpy as np
import cupy as cp

from pyscf import lib
from lib_pprpa.grad.pprpa import make_rdm1_relaxed_rhf_pprpa
from lib_pprpa.grad import pprpa_gamma as _cpu
from lib_pprpa.grad.grad_utils import get_xy_full
from lib_pprpa.grad.grad_utils_gpu_pbc import _contract_xc_kernel as _cxk_gpu
from lib_pprpa.grad.grad_utils_gpu_pbc import nr_rks_fxc as _nr_rks_fxc_gpu


def _aftdf(cell, kpts):
    """AFTDF set up for energy-gradient J/K (FFTDF lacks get_k_e1)."""
    from gpu4pyscf.pbc.df.aft import AFTDF
    adf = AFTDF(cell, kpts)
    adf.mesh = cell.mesh
    return adf


def make_gpu_vresp(cell, mf):
    """GPU CPHF/orbital-hessian response (singlet=None, hermi=1) for Gamma KS.

    vresp(dm) = J[dm] + fxc(x)dm  (-0.5*hyb*K[dm] for hybrids).  Accepts and
    returns numpy so it is a drop-in for the CPU ``vresp`` consumed by
    make_rdm1_relaxed_rhf_pprpa; the grid response runs on the GPU.
    """
    import cupy as cp
    from gpu4pyscf.pbc import dft as gdft
    kmf = gdft.KRKS(cell, kpts=np.zeros((1, 3)), xc=mf.xc)
    kmf.exxdiv = None
    ni = kmf._numint
    grids = kmf.grids
    if grids.coords is None:
        grids.build()
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc)
    kpts = np.zeros((1, 3))
    dm0 = cp.asarray(mf.make_rdm1())

    def vresp(dm):
        dmg = cp.asarray(dm)
        v = _nr_rks_fxc_gpu(ni, cell, grids, mf.xc, dm0, dmg, kpts=kpts, hermi=1)
        v = v + kmf.get_j(cell, dmg[None], hermi=1, kpts=kpts)[0]
        if abs(hyb) > 1e-12:
            v = v - 0.5 * hyb * kmf.get_k(cell, dmg[None], hermi=1, kpts=kpts)[0]
        return cp.asnumpy(v)
    return vresp


def grad_elec(pprpa_grad, xy, mult, atmlst=None):
    mf = pprpa_grad.mf
    pprpa = pprpa_grad.base
    cell = mf.mol
    is_ks = hasattr(mf, "xc")
    kpts = mf.kpts
    if atmlst is None:
        atmlst = range(cell.natm)
    assert mult in ("t", "s"), "invalid mult %r" % mult
    assert len(kpts) == 1 and abs(np.asarray(kpts)).max() < 1e-9, \
        "GPU pp-RPA gradient is Gamma-point only"

    from gpu4pyscf.pbc import dft as gdft, scf as gscf
    from gpu4pyscf.pbc.grad import krhf as krhf_g

    nocc_all = cell.nelectron // 2
    nocc, nvir = pprpa.nocc, pprpa.nvir
    nfo = nocc_all - nocc
    mo = mf.mo_coeff

    # --- relaxed density / energy-weighted density (CPU, validated) ----------
    kmf_cpu = _cpu.rhf_to_krhf(mf)
    kg_cpu = kmf_cpu.nuc_grad_method()
    if is_ks:
        vresp = make_gpu_vresp(cell, mf)   # GPU grid response for the CPHF solve
    else:
        vresp = None
    P_mo, W_mo = make_rdm1_relaxed_rhf_pprpa(
        pprpa, mf, xy=xy, mult=mult, cphf_max_cycle=pprpa_grad.cphf_max_cycle,
        cphf_conv_tol=pprpa_grad.cphf_conv_tol, vresp=vresp)
    W = mo @ W_mo @ mo.T \
        - kg_cpu.make_rdm1e(kmf_cpu.mo_energy, kmf_cpu.mo_coeff, kmf_cpu.mo_occ)[0]
    P = mo @ P_mo @ mo.T
    pprpa_grad.rdm1e = P
    D = kmf_cpu.make_rdm1()[0]
    T = D + P
    occ_y, vir_x = get_xy_full(xy, pprpa.oo_dim, mult)
    cocc = mo[:, nfo:nfo+nocc]
    cvir = mo[:, nfo+nocc:nfo+nocc+nvir]
    X = cvir @ vir_x @ cvir.T + cocc @ occ_y @ cocc.T

    # --- GPU assembly --------------------------------------------------------
    if is_ks:
        kmf = gdft.KRKS(cell, kpts=kpts, xc=mf.xc)
    else:
        kmf = gscf.KRHF(cell, kpts=kpts)
    kmf.exxdiv = mf.exxdiv
    kmf.mo_coeff = [cp.asarray(mo)]
    kmf.mo_energy = [cp.asarray(mf.mo_energy)]
    kmf.mo_occ = [cp.asarray(mf.mo_occ)]
    # Main object keeps the default FFTDF: J derivatives, hcore, overlap and the
    # XC skeletons are all far cheaper through FFT.  Exchange derivatives (pp-RPA
    # pairing, and the hybrid reference K) have no FFT get_k_e1, so a separate
    # AFTDF object is used only for those (built lazily below).
    gg = kmf.nuc_grad_method()

    ni = kmf._numint if is_ks else None
    if is_ks:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc)
    else:
        omega, alpha, hyb = 0.0, 1.0, 1.0
    exxdiv = mf.exxdiv
    need_k = (not is_ks) or abs(hyb) > 1e-12

    def _kmf_aft():
        kf = (gdft.KRKS(cell, kpts=kpts, xc=mf.xc) if is_ks
              else gscf.KRHF(cell, kpts=kpts))
        kf.exxdiv = mf.exxdiv
        kf.mo_coeff = kmf.mo_coeff; kf.mo_energy = kmf.mo_energy
        kf.mo_occ = kmf.mo_occ
        kf.with_df = _aftdf(cell, kpts); kf.rsjk = None
        return kf

    Dg, Pg, Tg, Wg, Xg = (cp.asarray(a) for a in (D, P, T, W, X))
    aoslices = cell.aoslice_by_atom()
    natm = cell.natm
    de = cp.zeros((natm, 3))

    # hcore (kinetic + local PP) contracted with the total density
    hcore_deriv = krhf_g.hcore_generator(gg, cell, kpts)
    for ia in range(natm):
        de[ia] += cp.einsum('kxij,kji->x', hcore_deriv(ia), Tg[None]).real

    # J reference via the polarization identity Q(D+P)-Q(P), batched (FFTDF).
    eJ = krhf_g.jk_energy_per_atom(
        kmf, cp.stack([Tg, Pg])[:, None], kpts, j_factor=1.0,
        sr_factor=0.0, lr_factor=0.0, omega=0.0, exxdiv=None)
    de += cp.asarray(eJ[0] - eJ[1])

    kaft = _kmf_aft() if need_k else None
    if need_k and abs(hyb) > 1e-12:
        # hybrid reference exchange K (AFTDF) via the same polarization identity
        def dvk(dm):
            return cp.asarray(krhf_g.jk_energy_per_atom(
                kaft, dm[None], kpts, j_factor=0.0, sr_factor=hyb, lr_factor=hyb,
                omega=omega, exxdiv=exxdiv))
        de += dvk(Tg) - dvk(Pg)

    # pp-RPA pairing exchange Tr[K[X]^x X] (AFTDF).  get_ek_ip1 assumes a
    # symmetric density: it returns +Q_K for the antisymmetric part of X and
    # -Q_K for the symmetric part (the sym/antisym exchange cross term is zero),
    # so X is split.  A pure singlet/triplet has only one component, so the
    # vanishing one is skipped.  sr/lr=2: jk_energy_per_atom's K carries the SCF
    # 0.5 exchange factor while pp-RPA pairing is the full bare exchange.
    def _pair(dm):
        return cp.asarray(krhf_g.jk_energy_per_atom(
            kaft, dm[None], kpts, j_factor=0.0, sr_factor=2.0, lr_factor=2.0,
            omega=omega, exxdiv=None))
    Xa = (Xg - Xg.T) * 0.5
    Xs = (Xg + Xg.T) * 0.5
    if kaft is None:
        kaft = _kmf_aft()
    if float(cp.abs(Xa).max()) > 1e-10:
        de += _pair(Xa)
    if float(cp.abs(Xs).max()) > 1e-10:
        de -= _pair(Xs)

    # Vxc skeleton (contract with T) + fxc.P skeleton (contract with D)
    if is_ks:
        f1vo, v1ao = _cxk_gpu(kmf, mf.xc, Pg, dm0=Dg, with_vxc=True)
        for ia in range(natm):
            p0, p1 = aoslices[ia, 2:]
            de[ia] += cp.einsum('xij,ij->x', v1ao[1:, p0:p1], Tg[p0:p1]).real * 2
            de[ia] += cp.einsum('xij,ij->x', f1vo[1:, p0:p1], Dg[p0:p1]).real

    de = de.get()

    # overlap (energy-weighted) and nonlocal pseudo-potential
    s1 = gg.get_ovlp(cell, kpts)
    de += krhf_g.contract_h1e_dm(cell, s1, Wg[None], hermi=1)
    de += krhf_g.vppnl_nuc_grad(cell, T[None], kpts=kpts)

    return de[list(atmlst)] if not isinstance(atmlst, range) else de


class Gradients(_cpu.Gradients):
    """GPU pp-RPA Gamma-point gradient (see module docstring)."""

    def grad_elec(self, xy, mult, atmlst):
        return grad_elec(self, xy, mult, atmlst)


Grad = Gradients
