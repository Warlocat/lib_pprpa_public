"""Consistent Gaussian-density-fitted ppRPA setup for Gamma-point cells.

The ordinary molecular ppRPA path obtains three-index MO integrals ``Lpq``
from the density-fitted mean-field object and feeds them to the Davidson
operator.  This module provides the same contract for periodic Gamma-point
calculations and records enough provenance to prevent a later analytical
gradient from silently using a different integral backend.

The corresponding nuclear gradient is implemented in
``lib_pprpa.grad.pprpa_gamma_gdf_gpu``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

from lib_pprpa.pprpa_davidson import ppRPA_Davidson
from lib_pprpa.pyscf_util import get_pyscf_input_sc


@dataclass(frozen=True)
class PBCGDFContext:
    """Provenance for a periodic GDF ppRPA operator."""

    mf_id: int
    df_id: int
    signature: tuple[Any, ...]
    nocc_act: int
    nvir_act: int
    operator_signature: tuple[str, ...]


def _gamma_kpts(mf):
    kpts = getattr(mf, "kpts", None)
    if kpts is None:
        kpts = getattr(mf, "kpt", np.zeros(3))
    return np.asarray(kpts).reshape(-1, 3)


def _array_digest(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def _basis_signature(with_df):
    auxcell = getattr(with_df, "auxcell", None)
    # Record both the user-facing specification and the basis actually built.
    # If ``auxbasis`` is mutated after a GDF build, the existing CPU auxcell
    # and a freshly mirrored GPU GDF would otherwise silently diverge.
    return (
        repr(with_df.auxbasis),
        repr(auxcell.basis) if auxcell is not None else None,
    )


def _operator_signature(pprpa):
    """Hash the Davidson inputs that must survive until differentiation."""
    if pprpa.Lpq is not None:
        loo = pprpa.Lpq[:, :pprpa.nocc, :pprpa.nocc]
        lvv = pprpa.Lpq[:, pprpa.nocc:, pprpa.nocc:]
    else:
        loo, lvv = pprpa.Loo, pprpa.Lvv
    if loo is None or lvv is None:
        raise ValueError("the ppRPA solver does not contain GDF Loo/Lvv factors")
    return (
        str(pprpa.channel),
        _array_digest(pprpa.mo_energy),
        _array_digest(loo),
        _array_digest(lvv),
    )


def gdf_signature(mf):
    """Return the settings that define the periodic GDF ppRPA Hamiltonian."""
    with_df = validate_gdf_mf(mf)
    mesh = getattr(with_df, "mesh", None)
    if mesh is None:
        mesh = mf.cell.mesh
    grids = getattr(mf, "grids", None)
    xc_mesh = getattr(grids, "mesh", None)
    if xc_mesh is not None:
        xc_mesh = tuple(np.asarray(xc_mesh, dtype=int))
    return (
        _basis_signature(with_df),
        getattr(with_df, "exp_to_discard", None),
        getattr(with_df, "linear_dep_threshold", None),
        tuple(np.asarray(mesh, dtype=int)),
        bool(getattr(with_df, "_j_only", False)),
        tuple(_gamma_kpts(mf).ravel()),
        repr(mf.cell.basis),
        repr(mf.cell.pseudo),
        _array_digest(mf.cell.atom_coords()),
        _array_digest(mf.cell.lattice_vectors()),
        float(mf.cell.precision),
        getattr(mf, "xc", None),
        getattr(mf, "exxdiv", None),
        xc_mesh,
        getattr(grids, "level", None),
        _array_digest(mf.mo_coeff),
        _array_digest(mf.mo_energy),
        _array_digest(mf.mo_occ),
    )


def validate_gdf_mf(mf):
    """Validate the restricted, real, Gamma-point CPU GDF reference."""
    from pyscf.pbc.df.df import GDF
    from pyscf.pbc.scf.hf import SCF

    if not isinstance(mf, SCF):
        raise TypeError("periodic GDF ppRPA requires a PySCF PBC SCF object")
    kpts = _gamma_kpts(mf)
    if len(kpts) != 1 or not np.allclose(kpts[0], 0.0, atol=1e-10):
        raise NotImplementedError("periodic GDF ppRPA currently supports Gamma point only")
    if mf.cell.dimension != 3:
        raise NotImplementedError("periodic GDF ppRPA currently supports 3D cells only")
    if np.asarray(mf.mo_coeff).ndim != 2 or np.iscomplexobj(mf.mo_coeff):
        raise NotImplementedError("periodic GDF ppRPA currently requires real restricted orbitals")
    with_df = getattr(mf, "with_df", None)
    if not isinstance(with_df, GDF):
        raise TypeError(
            "periodic GDF ppRPA requires mf.with_df to be pyscf.pbc.df.GDF")
    if bool(getattr(with_df, "_j_only", False)):
        raise ValueError(
            "periodic GDF ppRPA requires a full (J and K) GDF build; "
            "mf.with_df._j_only is true")
    df_kpts = np.asarray(getattr(with_df, "kpts", kpts)).reshape(-1, 3)
    if df_kpts.shape != kpts.shape or not np.allclose(
            df_kpts, kpts, atol=1e-10):
        raise ValueError("mf.with_df.kpts does not match the SCF Gamma point")
    if not getattr(mf, "converged", False):
        raise RuntimeError("the GDF mean-field reference is not converged")
    return with_df


def attach_gdf_context(pprpa, mf):
    """Attach an explicitly constructed ``Lpq`` solver to its GDF reference.

    Prefer :func:`make_pprpa_gdf`, which constructs both objects together.
    This lower-level function is provided for callers that need to customize
    the Davidson object before running it.
    """
    validate_gdf_mf(mf)
    if pprpa._use_eri or pprpa._ao_direct:
        raise ValueError("a GDF ppRPA solver must use Lpq, not ERI/AO-direct contractions")
    if pprpa.Lpq is None and (pprpa.Loo is None or pprpa.Lvv is None):
        raise ValueError("the ppRPA solver does not contain GDF Lpq factors")
    pprpa._scf = mf
    pprpa._pbc_gdf_context = PBCGDFContext(
        mf_id=id(mf),
        df_id=id(mf.with_df),
        signature=gdf_signature(mf),
        nocc_act=pprpa.nocc,
        nvir_act=pprpa.nvir,
        operator_signature=_operator_signature(pprpa),
    )
    return pprpa


def validate_gdf_context(pprpa, mf):
    """Reject mixed GDF/FFT/AFT energy-gradient configurations."""
    validate_gdf_mf(mf)
    context = getattr(pprpa, "_pbc_gdf_context", None)
    if not isinstance(context, PBCGDFContext):
        raise RuntimeError(
            "missing periodic GDF provenance; construct the solver with "
            "make_pprpa_gdf (or explicitly call attach_gdf_context)")
    if context.mf_id != id(mf) or context.df_id != id(mf.with_df):
        raise RuntimeError("the ppRPA GDF operator and gradient reference are different objects")
    if context.signature != gdf_signature(mf):
        raise RuntimeError("the GDF settings or mean-field orbitals changed after Lpq was built")
    if context.nocc_act != pprpa.nocc or context.nvir_act != pprpa.nvir:
        raise RuntimeError("the recorded GDF active space no longer matches the ppRPA solver")
    if context.operator_signature != _operator_signature(pprpa):
        raise RuntimeError("the recorded GDF Davidson operator changed after construction")
    if pprpa._use_eri or pprpa._ao_direct:
        raise RuntimeError("the ppRPA operator is not using GDF Lpq contractions")
    return context


def get_pyscf_input_sc_gdf(
        mf, nocc_act=None, nvir_act=None, dump_file=None, with_dip=False):
    """Build the Gamma-point GDF ``Lpq`` factors used by the ppRPA operator."""
    validate_gdf_mf(mf)
    return get_pyscf_input_sc(
        mf, nocc_act=nocc_act, nvir_act=nvir_act, dump_file=dump_file,
        cholesky=False, with_dip=with_dip)


def make_pprpa_gdf(
        mf, channel="pp", nocc_act=None, nvir_act=None, dump_file=None,
        mu=None, **davidson_kwargs):
    """Construct a Davidson ppRPA solver from the reference's periodic GDF.

    The returned object is tagged with the exact mean-field/GDF provenance.
    Run ``pprpa.kernel('s')`` or ``pprpa.kernel('t')`` as usual, then evaluate
    the matching gradient with ``pprpa_gamma_gdf_gpu.Gradients``.
    """
    nocc, mo_energy, lpq = get_pyscf_input_sc_gdf(
        mf, nocc_act=nocc_act, nvir_act=nvir_act, dump_file=dump_file)
    pprpa = ppRPA_Davidson(
        nocc, mo_energy, lpq, channel=channel, **davidson_kwargs)
    if mu is not None:
        pprpa.mu = mu
    return attach_gdf_context(pprpa, mf)


__all__ = [
    "PBCGDFContext",
    "attach_gdf_context",
    "gdf_signature",
    "get_pyscf_input_sc_gdf",
    "make_pprpa_gdf",
    "validate_gdf_context",
    "validate_gdf_mf",
]
