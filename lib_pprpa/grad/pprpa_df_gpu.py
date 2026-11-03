"""Nuclear gradient of the molecular DF pp-RPA on the GPU (gpu4pyscf DF RHF/RKS reference).

The expression of ``grad.pprpa.grad_elec`` (without XC grid response, as there), assembled per
atom as in ``grad.pprpa_gamma_gdf_gpu``: hcore and overlap terms from gpu4pyscf's
``_hcore_energy``, Coulomb/exchange with auxiliary-basis response from the DF
``jk_energy_per_atom``, and the XC potential and kernel derivatives from gpu4pyscf's TDRKS
``_contract_xc_kernel``.  The relaxed density is ``make_rdm1_relaxed_rhf_pprpa`` with the GPU
CPHF response and pair exchange.  Pairs with ``mol_df_gpu.make_pprpa_df_gpu``.
Scope: HF, LDA/GGA and global hybrids; range-separated functionals are rejected.
"""
import cupy as cp
import numpy as np
from gpu4pyscf.grad import tdrks as tdrks_grad
from gpu4pyscf.grad.rhf import contract_h1e_dm

from lib_pprpa.grad import pprpa as _cpu
from lib_pprpa.grad.grad_utils import get_xy_full
from lib_pprpa.grad.pprpa import make_rdm1_relaxed_rhf_pprpa

__all__ = ['Grad', 'Gradients', 'grad_elec', 'make_gpu_vresp']


def _host(x):
    return x.get() if isinstance(x, cp.ndarray) else np.asarray(x)


def make_gpu_vresp(mf):
    """CPHF response (``singlet=None, hermi=1``) of a gpu4pyscf SCF: CuPy in -> CuPy out,
    NumPy in -> NumPy out."""
    native = mf.gen_response(singlet=None, hermi=1)

    def vresp(dm):
        v = native(cp.asarray(dm))
        return v if isinstance(dm, cp.ndarray) else v.get()
    vresp.accepts_device_arrays = True
    return vresp


def grad_elec(pprpa_grad, xy, mult, atmlst=None):
    mf = pprpa_grad.mf
    pprpa = pprpa_grad.base
    mol = mf.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    if mult not in ('s', 't'):
        raise ValueError(f'invalid multiplicity {mult!r}')
    is_ks = hasattr(mf, 'xc')
    hyb = 1.0
    if is_ks:
        omega, _, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        if omega != 0:
            raise NotImplementedError('range-separated functionals')

    def pair_get_k(dms, hermi):
        return mf.get_k(mol, cp.asarray(dms), hermi=hermi).get()

    # The relaxed-density algebra is on (nmo, nmo) host matrices; it reads only the orbitals.
    cphf_out = {}
    p_mo, w_mo = make_rdm1_relaxed_rhf_pprpa(
        pprpa, mf.to_cpu(), xy=xy, mult=mult, cphf_max_cycle=pprpa_grad.cphf_max_cycle,
        cphf_conv_tol=pprpa_grad.cphf_conv_tol, vresp=make_gpu_vresp(mf),
        pair_get_k=pair_get_k, cphf_x0=pprpa_grad.cphf_x0, cphf_x0_out=cphf_out)
    pprpa_grad.cphf_x0 = cphf_out.get('cphf_x0_ao')

    gpu_grad = mf.nuc_grad_method()
    mo = cp.asarray(mf.mo_coeff)
    d = mf.make_rdm1()
    p = mo @ cp.asarray(p_mo) @ mo.T
    w = mo @ cp.asarray(w_mo) @ mo.T - gpu_grad.make_rdm1e(
        cp.asarray(mf.mo_energy), mo, cp.asarray(mf.mo_occ))
    pprpa_grad.rdm1e = p.get()
    total = d + p

    nocc, nvir = pprpa.nocc, pprpa.nvir
    nfo = mol.nelectron // 2 - nocc
    occ_y, vir_x = get_xy_full(xy, pprpa.oo_dim, mult, nocc=nocc, nvir=nvir)
    cocc = mo[:, nfo:nfo + nocc]
    cvir = mo[:, nfo + nocc:nfo + nocc + nvir]
    pair = cvir @ cp.asarray(vir_x) @ cvir.T + cocc @ cp.asarray(occ_y) @ cocc.T

    # _hcore_energy subtracts the overlap derivative contracted with its energy-weighted
    # density; the pp-RPA one is -w.
    de = _host(gpu_grad._hcore_energy(total, -w))
    # Q(D+P) - Q(P) is the reference term plus the D/P cross term.
    de += _host(gpu_grad.jk_energy_per_atom(total, j_factor=1, k_factor=hyb, hermi=1))
    de -= _host(gpu_grad.jk_energy_per_atom(p, j_factor=1, k_factor=hyb, hermi=1))
    # Pair-density exchange: k_factor=2 is the pp-RPA magnitude; the antisymmetric
    # (triplet) and symmetric (singlet) parts enter with opposite signs.
    xa = (pair - pair.T) * .5
    xs = (pair + pair.T) * .5
    if float(abs(xa).max()) > 1e-10:
        de += _host(gpu_grad.jk_energy_per_atom(xa, j_factor=0, k_factor=2, hermi=2))
    if float(abs(xs).max()) > 1e-10:
        de -= _host(gpu_grad.jk_energy_per_atom(xs, j_factor=0, k_factor=2, hermi=1))

    if is_ks:
        # v1ao: derivative of the reference XC potential; f1vo: of the kernel applied to P.
        f1vo, _, v1ao, _ = tdrks_grad._contract_xc_kernel(
            mf.TDA().Gradients(), mf.xc, p, None, with_vxc=True, with_kxc=False,
            singlet=True)
        de += contract_h1e_dm(mol, v1ao[1:], total, hermi=1)
        de += contract_h1e_dm(mol, f1vo[1:] * .5, d, hermi=1)
    return de[list(atmlst)]


class Gradients(_cpu.Gradients):
    """pp-RPA gradient for a ``make_pprpa_df_gpu`` solver and its gpu4pyscf DF reference."""

    _keys = _cpu.Gradients._keys | {'cphf_x0'}

    def __init__(self, pprpa, mf, mult='t', state=0):
        if getattr(mf, 'with_df', None) is None:
            raise TypeError('the GPU molecular pp-RPA gradient needs a density-fitted reference')
        self.mf = mf
        self.base = pprpa
        self.mol = mf.mol
        self.state = state
        self.verbose = self.mol.verbose
        self.mult = mult
        # AO-basis CPHF solution of the previous evaluation, the next one's initial guess
        self.cphf_x0 = None
        self.rdm1e = None
        self.atmlst = None
        self.de = None

    def grad_elec(self, xy, mult, atmlst):
        return grad_elec(self, xy, mult, atmlst)


Grad = Gradients
