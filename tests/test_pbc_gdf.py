"""CPU-side contract tests for the periodic GDF ppRPA entry point."""

import numpy as np
import pytest


def test_triplet_single_orbital_block_keeps_dimension():
    from lib_pprpa.grad.grad_utils import get_xy_full

    # A one-orbital antisymmetric block has no packed elements.  Supplying the
    # known active-space dimensions must still reconstruct its 1x1 zero block.
    xy = np.arange(6, dtype=float)
    occ, vir = get_xy_full(xy, 0, "t", nocc=1, nvir=4)
    assert occ.shape == (1, 1)
    assert vir.shape == (4, 4)
    assert occ[0, 0] == 0.0


@pytest.fixture(scope="module")
def gdf_reference():
    from pyscf.pbc import df, dft, gto

    cell = gto.Cell()
    cell.atom = [("H", (0.0, 0.0, 0.0)), ("H", (0.7, 0.2, 0.1))]
    cell.a = np.eye(3) * 4.0
    cell.unit = "Bohr"
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pbe"
    cell.precision = 1e-8
    cell.verbose = 0
    cell.build()

    mf = dft.RKS(cell, xc="pbe")
    mf.exxdiv = None
    mf.with_df = df.GDF(cell)
    mf.with_df.auxbasis = "weigend"
    mf.conv_tol = 1e-10
    mf.kernel()
    assert mf.converged
    return mf


def test_periodic_lpq_reproduces_gdf_mo_integrals(gdf_reference):
    from lib_pprpa.pbc_gdf import get_pyscf_input_sc_gdf

    mf = gdf_reference
    nocc, _, lpq = get_pyscf_input_sc_gdf(mf)
    eri_lpq = np.einsum("Lpq,Lrs->pqrs", lpq, lpq)
    eri_gdf = mf.with_df.ao2mo(
        mf.mo_coeff, compact=False).reshape(eri_lpq.shape)

    assert nocc == mf.cell.nelectron // 2
    np.testing.assert_allclose(eri_lpq, eri_gdf, rtol=0.0, atol=1e-12)


def test_gdf_solver_provenance_survives_davidson(gdf_reference):
    from lib_pprpa.pbc_gdf import make_pprpa_gdf, validate_gdf_context

    mf = gdf_reference
    solver = make_pprpa_gdf(
        mf, nroot=1, residue_thresh=1e-9, trial="identity")
    validate_gdf_context(solver, mf)
    solver.kernel("s")

    # Davidson releases the full Lpq tensor, but the split GDF factors and
    # immutable reference provenance remain valid for the gradient.
    assert solver.Lpq is None
    assert solver.Loo is not None and solver.Lvv is not None
    validate_gdf_context(solver, mf)


def test_gdf_provenance_rejects_changed_reference(gdf_reference):
    from lib_pprpa.pbc_gdf import make_pprpa_gdf, validate_gdf_context

    mf = gdf_reference
    solver = make_pprpa_gdf(mf, nroot=1)
    old_xc = mf.xc
    try:
        mf.xc = "b3lyp"
        with pytest.raises(RuntimeError, match="settings or mean-field orbitals changed"):
            validate_gdf_context(solver, mf)
    finally:
        mf.xc = old_xc

    old_mesh = np.asarray(mf.grids.mesh).copy()
    try:
        mf.grids.mesh = old_mesh + 2
        with pytest.raises(RuntimeError, match="settings or mean-field orbitals changed"):
            validate_gdf_context(solver, mf)
    finally:
        mf.grids.mesh = old_mesh

    old_energy = solver.mo_energy.copy()
    try:
        solver.mo_energy[0] += 1e-6
        with pytest.raises(RuntimeError, match="Davidson operator changed"):
            validate_gdf_context(solver, mf)
    finally:
        solver.mo_energy[:] = old_energy

    old_channel = solver.channel
    try:
        solver.channel = "hh"
        with pytest.raises(RuntimeError, match="Davidson operator changed"):
            validate_gdf_context(solver, mf)
    finally:
        solver.channel = old_channel

    # Accessing auxcell after a basis-label change rebuilds PySCF's cached
    # auxiliary Cell, so keep this destructive provenance check last.
    old_auxbasis = mf.with_df.auxbasis
    try:
        mf.with_df.auxbasis = "def2-universal-jkfit"
        with pytest.raises(RuntimeError, match="settings or mean-field orbitals changed"):
            validate_gdf_context(solver, mf)
    finally:
        mf.with_df.auxbasis = old_auxbasis


def test_gdf_rejects_j_only_operator(gdf_reference):
    from lib_pprpa.pbc_gdf import validate_gdf_mf

    mf = gdf_reference
    old_j_only = mf.with_df._j_only
    try:
        mf.with_df._j_only = True
        with pytest.raises(ValueError, match=r"full \(J and K\) GDF build"):
            validate_gdf_mf(mf)
    finally:
        mf.with_df._j_only = old_j_only
