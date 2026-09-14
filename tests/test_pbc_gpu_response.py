"""Focused regression test for the native GPU PBC CPHF response adapter."""

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


@pytest.mark.parametrize("xc", ("pbe", "b3lyp"))
def test_native_gamma_response_adapter(xc):
    import cupy as cp
    from gpu4pyscf.pbc import dft as gdft
    from pyscf.pbc import dft as cdft
    from pyscf.pbc import gto

    from lib_pprpa.grad.pprpa_gamma_gpu import make_gpu_vresp

    gamma = np.zeros((1, 3))
    cell = gto.Cell()
    cell.atom = [("C", (0.0, 0.0, 0.0)), ("C", (0.9, 0.9, 0.92))]
    cell.a = np.array([[0.0, 1.8, 1.8],
                       [1.8, 0.0, 1.8],
                       [1.8, 1.8, 0.0]])
    cell.unit = "Angstrom"
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pbe"
    cell.ke_cutoff = 30.0
    cell.precision = 1e-9
    cell.verbose = 0
    cell.build()

    template = gdft.KRKS(cell, kpts=gamma, xc=xc)
    template.exxdiv = None
    overlap = cp.asarray(template.get_ovlp(cell, gamma))[0].real
    overlap_e, overlap_u = cp.linalg.eigh(overlap)
    coeff = overlap_u / cp.sqrt(overlap_e)[None, :]
    nocc = cell.nelectron // 2
    mo_occ = cp.zeros(cell.nao)
    mo_occ[:nocc] = 2.0
    mo_energy = cp.arange(cell.nao, dtype=cp.float64)

    mf = cdft.RKS(cell, xc=xc)
    mf.exxdiv = None
    mf.mo_coeff = cp.asnumpy(coeff)
    mf.mo_energy = cp.asnumpy(mo_energy)
    mf.mo_occ = cp.asnumpy(mo_occ)
    mf.converged = True

    rng = np.random.default_rng(9173)
    dm1 = rng.standard_normal((cell.nao, cell.nao))
    dm1 = dm1 + dm1.T
    dm1 *= 0.02 / abs(dm1).max()

    adapter = make_gpu_vresp(cell, mf)
    actual = adapter(dm1)
    repeat = adapter(dm1)

    direct = gdft.KRKS(cell, kpts=gamma, xc=xc)
    direct.exxdiv = None
    direct.mo_coeff = coeff[None]
    direct.mo_energy = mo_energy[None]
    direct.mo_occ = mo_occ[None]
    direct.grids.build()
    native = direct.gen_response(singlet=None, hermi=1)
    expected = cp.asnumpy(native(cp.asarray(dm1).reshape(
        1, 1, cell.nao, cell.nao))[0, 0])

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(repeat, actual)
    with pytest.raises(ValueError, match="Gamma CPHF response expects"):
        adapter(dm1[:-1, :-1])
