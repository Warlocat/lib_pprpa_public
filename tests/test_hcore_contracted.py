"""Density-contracted hcore derivative against the per-atom hcore_generator loop."""

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

KPTS = np.zeros((1, 3))


def _cell(pseudo):
    from pyscf.pbc import gto
    a0 = 3.5668
    cell = gto.Cell()
    cell.atom = [["C", [0.0, 0.0, 0.0]], ["C", [a0 / 4 + 0.05, a0 / 4, a0 / 4]]]
    cell.a = np.array([[0, a0 / 2, a0 / 2], [a0 / 2, 0, a0 / 2], [a0 / 2, a0 / 2, 0]])
    if pseudo:
        cell.basis = "gth-szv"
        cell.pseudo = "gth-pade"
    else:
        cell.basis = "sto-3g"
    cell.mesh = [15, 15, 15]
    cell.precision = 1e-12
    cell.verbose = 0
    cell.build()
    return cell


def _density(nao):
    dm = np.random.default_rng(7).standard_normal((nao, nao))
    return dm + dm.T


def _reference_cpu(cell, dm):
    from pyscf.pbc import scf
    hcore_deriv = scf.KRHF(cell, KPTS).nuc_grad_method().hcore_generator(cell, KPTS)
    return np.array([np.einsum('xkij,kji->x', hcore_deriv(ia), dm[None]).real
                     for ia in range(cell.natm)])


def _reference_gpu(cell, dm):
    import cupy as cp
    from gpu4pyscf.pbc import scf
    hcore_deriv = scf.KRHF(cell, KPTS).nuc_grad_method().hcore_generator(cell, KPTS)
    dmg = cp.asarray(dm)[None]
    return np.array([cp.einsum('kxij,kji->x', hcore_deriv(ia), dmg).real.get()
                     for ia in range(cell.natm)])


@pytest.mark.parametrize("pseudo", [True, False])
@pytest.mark.parametrize("free", [None, 96_000])
def test_matches_per_atom_generator(pseudo, free, monkeypatch):
    """Whole mesh in one block and ~100-point blocks; pyscf's CPU generator is pseudo-only."""
    import lib_pprpa.grad.hcore_contracted as hc
    if free is not None:
        monkeypatch.setattr(hc, "free_bytes", lambda: free)
    cell = _cell(pseudo)
    dm = _density(cell.nao)
    de = hc.hcore_deriv_contracted(cell, KPTS, dm)
    ref = _reference_cpu(cell, dm) if pseudo else _reference_gpu(cell, dm)
    assert de.shape == (cell.natm, 3)
    np.testing.assert_allclose(de, ref, rtol=0, atol=1e-9)
