"""Consistent GPU GDF ppRPA gradient for three-dimensional Gamma cells.

This backend keeps one integral approximation throughout the calculation:

* the CPU PySCF reference uses :class:`pyscf.pbc.df.GDF`;
* :func:`lib_pprpa.pbc_gdf.make_pprpa_gdf` builds the ppRPA Davidson operator
  from that object's three-index factors; and
* every Coulomb/exchange derivative, including the ppRPA pair-density term and
  the CPHF response, uses GPU4PySCF's periodic GDF implementation.

It deliberately does not fall back to FFTDF or AFTDF.  Range-separated
functionals are rejected until their long-range GDF derivative is available.
The initial scope is real restricted Gamma-point HF and LDA/GGA/global-hybrid
KS references in three-dimensional cells.
"""

from __future__ import annotations

import copy
import inspect

import cupy as cp
import numpy as np

from lib_pprpa.grad import pprpa_gamma as _cpu
from lib_pprpa.grad.grad_utils import get_xy_full
from lib_pprpa.grad.grad_utils_gpu_pbc import _contract_xc_kernel as _cxk_gpu
from lib_pprpa.grad.pprpa import make_rdm1_relaxed_rhf_pprpa
from lib_pprpa.grad.pprpa_gamma_gpu import make_gpu_vresp
from lib_pprpa.pbc_gdf import validate_gdf_context, validate_gdf_mf


def _xc_coefficients(mf, gpu_mf):
    """Return ``(omega, alpha, hybrid)`` and enforce the supported XC scope."""
    if not hasattr(mf, "xc"):
        return 0.0, 1.0, 1.0
    xc_type = gpu_mf._numint._xc_type(mf.xc)
    if xc_type not in ("LDA", "GGA"):
        raise NotImplementedError(
            "GDF ppRPA gradients currently support LDA and GGA functionals only")
    omega, alpha, hybrid = gpu_mf._numint.rsh_and_hybrid_coeff(mf.xc)
    if abs(omega) > 1e-12:
        raise NotImplementedError(
            "range-separated GDF ppRPA gradients are not implemented; "
            "using AFT for the long-range term would mix integral backends")
    return float(omega), float(alpha), float(hybrid)


def make_gpu_gdf_mf(mf):
    """Mirror a converged CPU Gamma GDF reference on GPU without changing it.

    The already-built CPU auxiliary basis (after any exponent pruning) is
    copied exactly.  The GPU metric threshold and reciprocal mesh are then set
    to the CPU values.
    """
    cpu_df = validate_gdf_mf(mf)
    # Building Lpq normally initializes auxcell already.  Keep this helper safe
    # for direct callers too.
    if cpu_df.auxcell is None:
        cpu_df.build(j_only=False)

    from gpu4pyscf.pbc import dft as gdft, scf as gscf
    from gpu4pyscf.pbc.df.df import GDF as GPU_GDF

    cell = mf.cell
    kpts = np.asarray(mf.kpts).reshape(-1, 3)
    if hasattr(mf, "xc"):
        gpu_mf = gdft.KRKS(cell, kpts=kpts, xc=mf.xc)
        # Periodic CPU and GPU RKS both use UniformGrids by default.  Preserve
        # an explicitly selected mesh because it defines the XC energy/response.
        if getattr(getattr(mf, "grids", None), "mesh", None) is not None:
            gpu_mf.grids.mesh = np.asarray(mf.grids.mesh, dtype=int)
    else:
        gpu_mf = gscf.KRHF(cell, kpts=kpts)

    gpu_df = GPU_GDF(cell, kpts)
    # Mirror the basis that actually generated the authoritative CPU CDERI,
    # not merely a possibly mutated user-facing basis name.
    gpu_df.auxbasis = copy.deepcopy(cpu_df.auxcell.basis)
    gpu_df.exp_to_discard = None
    gpu_df.linear_dep_threshold = cpu_df.linear_dep_threshold
    mesh = cpu_df.mesh if cpu_df.mesh is not None else cell.mesh
    gpu_df.mesh = np.asarray(mesh, dtype=int)
    gpu_df.is_gamma_point = True
    gpu_df.build(j_only=False)

    gpu_mf.with_df = gpu_df
    gpu_mf.rsjk = None
    gpu_mf.exxdiv = mf.exxdiv
    gpu_mf.max_memory = mf.max_memory
    gpu_mf.verbose = mf.verbose
    gpu_mf.mo_coeff = cp.asarray(mf.mo_coeff)[None]
    gpu_mf.mo_energy = cp.asarray(mf.mo_energy)[None]
    gpu_mf.mo_occ = cp.asarray(mf.mo_occ)[None]
    gpu_mf.e_tot = mf.e_tot
    gpu_mf.converged = mf.converged
    return gpu_mf


class _GDFDerivativeEngine:
    """One reusable GPU4PySCF three-center derivative engine."""

    def __init__(self, gpu_mf):
        from gpu4pyscf.pbc.df.grad.krhf import _jk_energy_per_atom
        from gpu4pyscf.pbc.df.int3c2e import SRInt3c2eOpt
        from gpu4pyscf.pbc.df.rsdf_builder import _guess_omega
        from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh

        required = {
            "hermi", "j_factor", "k_factor", "exxdiv", "omega",
            "linear_dep_threshold",
        }
        try:
            parameters = set(inspect.signature(_jk_energy_per_atom).parameters)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "cannot inspect GPU4PySCF's periodic GDF derivative API") from exc
        if not callable(_jk_energy_per_atom) or not required.issubset(parameters):
            missing = ", ".join(sorted(required - parameters))
            raise RuntimeError(
                "the installed GPU4PySCF does not provide the GDF derivative "
                "interface required by ppRPA (including hermi=2); missing "
                f"parameters: {missing or 'callable implementation'}. Install "
                "the exact compatible GPU4PySCF revision documented in the "
                "GDF validation report.")

        self.gpu_mf = gpu_mf
        self.kpts = np.asarray(gpu_mf.kpts).reshape(-1, 3)
        with_df = gpu_mf.with_df
        cell = gpu_mf.cell
        kmesh = kpts_to_kmesh(
            cell, self.kpts, rcut=cell.rcut + 10, bound_by_supmol=False)
        self.opt = SRInt3c2eOpt(
            cell, with_df.auxcell, _guess_omega(cell), kmesh).build()
        self.linear_dep_threshold = with_df.linear_dep_threshold
        self._jk_energy_per_atom = _jk_energy_per_atom

    def jk(self, dm, *, hermi, j_factor, k_factor, exxdiv=None):
        """Derivative of ``j_factor*E_J - k_factor/2*E_K`` per atom."""
        if hermi not in (1, 2):
            raise ValueError("the periodic GDF derivative supports hermi=1 or 2")
        if hermi == 2 and abs(j_factor) > 1e-15:
            raise ValueError("an antisymmetric density has no Coulomb contribution")
        dm = cp.asarray(dm)
        nao = self.gpu_mf.cell.nao
        if dm.shape != (nao, nao):
            raise ValueError(f"GDF derivative density must be ({nao}, {nao})")
        return np.asarray(self._jk_energy_per_atom(
            self.opt, dm[None], self.kpts, hermi=hermi,
            j_factor=j_factor, k_factor=k_factor, exxdiv=exxdiv,
            omega=0.0, linear_dep_threshold=self.linear_dep_threshold))


def grad_elec(pprpa_grad, xy, mult, atmlst=None):
    """Evaluate the electronic GDF ppRPA gradient."""
    mf = pprpa_grad.mf
    pprpa = pprpa_grad.base
    validate_gdf_context(pprpa, mf)

    cell = mf.cell
    if atmlst is None:
        atmlst = range(cell.natm)
    if mult not in ("s", "t"):
        raise ValueError(f"invalid multiplicity {mult!r}")

    from gpu4pyscf.pbc.grad import krhf as krhf_g

    is_ks = hasattr(mf, "xc")
    nocc_all = cell.nelectron // 2
    nocc, nvir = pprpa.nocc, pprpa.nvir
    nfo = nocc_all - nocc
    mo = np.asarray(mf.mo_coeff)

    gpu_mf = make_gpu_gdf_mf(mf)
    _, _, hybrid = _xc_coefficients(mf, gpu_mf)
    gpu_grad = gpu_mf.nuc_grad_method()
    gdf_deriv = _GDFDerivativeEngine(gpu_mf)

    kmf_cpu = _cpu.rhf_to_krhf(mf)
    kg_cpu = kmf_cpu.nuc_grad_method()
    vresp = make_gpu_vresp(cell, mf, gpu_mf=gpu_mf) if is_ks else None
    p_mo, w_mo = make_rdm1_relaxed_rhf_pprpa(
        pprpa, mf, xy=xy, mult=mult,
        cphf_max_cycle=pprpa_grad.cphf_max_cycle,
        cphf_conv_tol=pprpa_grad.cphf_conv_tol, vresp=vresp)

    w = mo @ w_mo @ mo.T - kg_cpu.make_rdm1e(
        kmf_cpu.mo_energy, kmf_cpu.mo_coeff, kmf_cpu.mo_occ)[0]
    p = mo @ p_mo @ mo.T
    pprpa_grad.rdm1e = p
    d = kmf_cpu.make_rdm1()[0]
    total = d + p

    occ_y, vir_x = get_xy_full(
        xy, pprpa.oo_dim, mult, nocc=nocc, nvir=nvir)
    cocc = mo[:, nfo:nfo + nocc]
    cvir = mo[:, nfo + nocc:nfo + nocc + nvir]
    pair = cvir @ vir_x @ cvir.T + cocc @ occ_y @ cocc.T

    dg, pg, tg, wg, xg = (cp.asarray(a) for a in (d, p, total, w, pair))
    natm = cell.natm
    de = np.zeros((natm, 3))

    # One-electron skeleton (kinetic + local pseudopotential).
    hcore_deriv = krhf_g.hcore_generator(gpu_grad, cell, gpu_mf.kpts)
    for ia in range(natm):
        de[ia] += cp.asnumpy(cp.einsum(
            "kxij,kji->x", hcore_deriv(ia), tg[None]).real)

    # All reference-density Coulomb/exchange terms use the same GDF metric as
    # the ppRPA operator.  Q(D+P)-Q(P) isolates Q(D)+the D/P cross derivative.
    de += gdf_deriv.jk(
        tg, hermi=1, j_factor=1.0, k_factor=hybrid,
        exxdiv=mf.exxdiv)
    de -= gdf_deriv.jk(
        pg, hermi=1, j_factor=1.0, k_factor=hybrid,
        exxdiv=mf.exxdiv)

    # Pair-density exchange.  GPU4PySCF's GDF derivative is the derivative of
    # -(k_factor/4) Tr[D K(D)].  k_factor=2 therefore has the ppRPA magnitude;
    # antisymmetric/triplet and symmetric/singlet sectors enter with opposite
    # signs under the ppRPA metric convention.
    xa = (xg - xg.T) * 0.5
    xs = (xg + xg.T) * 0.5
    if float(cp.max(cp.abs(xa))) > 1e-10:
        de += gdf_deriv.jk(
            xa, hermi=2, j_factor=0.0, k_factor=2.0, exxdiv=None)
    if float(cp.max(cp.abs(xs))) > 1e-10:
        de -= gdf_deriv.jk(
            xs, hermi=1, j_factor=0.0, k_factor=2.0, exxdiv=None)

    # XC potential skeleton and its response to the relaxed density.  This
    # helper contains no Coulomb/exchange term; those are entirely GDF above.
    if is_ks:
        f1vo, v1ao = _cxk_gpu(
            gpu_mf, mf.xc, pg, dm0=dg, with_vxc=True)
        aoslices = cell.aoslice_by_atom()
        for ia in range(natm):
            p0, p1 = aoslices[ia, 2:]
            de[ia] += cp.asnumpy(cp.einsum(
                "xij,ij->x", v1ao[1:, p0:p1], tg[p0:p1]).real * 2)
            de[ia] += cp.asnumpy(cp.einsum(
                "xij,ij->x", f1vo[1:, p0:p1], dg[p0:p1]).real)

    # Pulay overlap and nonlocal pseudopotential terms.
    s1 = gpu_grad.get_ovlp(cell, gpu_mf.kpts)
    de += krhf_g.contract_h1e_dm(cell, s1, wg[None], hermi=1)
    de += krhf_g.vppnl_nuc_grad(cell, total[None], kpts=gpu_mf.kpts)

    return de[list(atmlst)]


class Gradients(_cpu.Gradients):
    """Consistent periodic GDF ppRPA gradient driver."""

    def __init__(self, pprpa, mf, mult="t", state=0):
        validate_gdf_context(pprpa, mf)
        self.mf = mf
        self.base = pprpa
        self.mol = mf.cell
        self.cell = mf.cell
        self.state = state
        self.verbose = self.mol.verbose
        self.mult = mult
        self.rdm1e = None
        self.atmlst = None
        self.de = None

    def grad_elec(self, xy, mult, atmlst):
        return grad_elec(self, xy, mult, atmlst)


Grad = Gradients


__all__ = ["Grad", "Gradients", "grad_elec", "make_gpu_gdf_mf"]
