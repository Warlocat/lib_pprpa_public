"""GPU contract test for mirroring the authoritative CPU periodic GDF."""

import numpy as np
import pytest


def _gpu4pyscf_available():
    try:
        import cupy as cp
        import gpu4pyscf  # noqa: F401
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _gpu4pyscf_available(), reason="GPU4PySCF CUDA device is unavailable")


def test_gpu_gdf_mirror_reproduces_cpu_jk():
    import cupy as cp
    from pyscf.pbc import df, dft, gto

    from lib_pprpa.grad.pprpa_gamma_gdf_gpu import make_gpu_gdf_mf

    gamma = np.zeros((1, 3))
    cell = gto.Cell()
    cell.atom = [("H", (0.03, -0.02, 0.01)), ("H", (1.5, 0.2, -0.1))]
    cell.a = np.diag([7.0, 7.2, 7.4])
    cell.unit = "Bohr"
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pbe"
    cell.ke_cutoff = 25.0
    cell.precision = 1e-7
    cell.verbose = 0
    cell.build()

    mf = dft.RKS(cell, xc="pbe")
    mf.exxdiv = None
    mf.grids.mesh = cell.mesh
    mf.with_df = df.GDF(cell, kpts=gamma)
    mf.with_df.auxbasis = {
        "H": [[0, [3.0, 1.0]], [0, [0.8, 1.0]], [1, [1.0, 1.0]]]
    }
    mf.with_df.linear_dep_threshold = 1e-10
    mf.with_df.build(j_only=False)

    overlap = mf.get_ovlp()
    overlap_e, overlap_u = np.linalg.eigh(overlap)
    mf.mo_coeff = overlap_u / np.sqrt(overlap_e)[None, :]
    mf.mo_energy = np.arange(cell.nao, dtype=float)
    mf.mo_occ = np.zeros(cell.nao)
    mf.mo_occ[:cell.nelectron // 2] = 2.0
    mf.e_tot = 0.0
    mf.converged = True

    rng = np.random.default_rng(14917)
    dm = rng.standard_normal((cell.nao, cell.nao))
    dm = (dm + dm.T) * 0.05
    cpu_j, cpu_k = mf.with_df.get_jk(
        dm, hermi=1, kpts=gamma, exxdiv=None)

    gpu_mf = make_gpu_gdf_mf(mf)
    gpu_j, gpu_k = gpu_mf.with_df.get_jk(
        cp.asarray(dm), hermi=1, kpts=gamma, exxdiv=None)

    np.testing.assert_allclose(cp.asnumpy(gpu_j), cpu_j, rtol=0.0, atol=2e-10)
    np.testing.assert_allclose(cp.asnumpy(gpu_k), cpu_k, rtol=0.0, atol=2e-10)
    assert gpu_mf.with_df.get_naoaux() == mf.with_df.get_naoaux()
    assert repr(gpu_mf.with_df.auxcell.basis) == repr(mf.with_df.auxcell.basis)
