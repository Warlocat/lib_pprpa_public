"""GPU pp-RPA gradient when the SCF drops a linearly dependent AO combination (nmo < nao):
the relaxed density counts the virtual orbitals from the MOs, and the GPU gradient matches
the CPU driver."""

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


@pytest.mark.parametrize("mult", ("s", "t"))
def test_gradient_with_fewer_mos_than_aos(mult):
    from pyscf.pbc import dft, gto
    from pyscf.scf import addons
    from lib_pprpa.gpu_ao2mo import gpu_ao2mo_blocks
    from lib_pprpa.grad import pprpa_gamma as cpu
    from lib_pprpa.grad import pprpa_gamma_gpu as gpu
    from lib_pprpa.pprpa_davidson import ppRPA_Davidson
    from lib_pprpa.pprpa_eri_gpu import attach_gpu_eri_contraction
    a0 = 3.5668
    cell = gto.M(atom=[["C", [0.0, 0.0, 0.0]], ["C", [a0 / 4 + 0.05, a0 / 4, a0 / 4]]],
                 a=np.array([[0, a0 / 2, a0 / 2], [a0 / 2, 0, a0 / 2], [a0 / 2, a0 / 2, 0]]),
                 basis="gth-szv", pseudo="gth-pade", charge=2, ke_cutoff=100.0, verbose=0)
    # canonical orthogonalization that drops exactly the smallest overlap eigenvector
    w = np.linalg.eigvalsh(np.asarray(cell.pbc_intor("int1e_ovlp")))
    thresh = np.sqrt(w[0] * w[1])
    mf = addons.remove_linear_dep_(dft.RKS(cell, xc="pbe"), threshold=thresh, lindep=thresh)
    mf.exxdiv = None
    mf.conv_tol = 1e-10
    mf.kernel()
    nocc = cell.nelectron // 2
    assert len(mf.mo_energy) == cell.nao - 1
    eri = gpu_ao2mo_blocks(cell, mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:], cell.mesh)
    mp = ppRPA_Davidson(nocc, mf.mo_energy, Lpq=None, channel="pp", nroot=2,
                        residue_thresh=1e-10, trial="identity")
    mp.mu = 0.0
    attach_gpu_eri_contraction(mp, *eri, mode="resident")
    mp.kernel(mult)
    xy = np.array((mp.xy_s if mult == "s" else mp.xy_t)[0], copy=True)
    de = []
    for module in (cpu, gpu):
        g = module.Gradients(mp, mf, mult, 0)
        g.cphf_conv_tol = 1e-10
        g.cphf_max_cycle = 100
        de.append(g.grad_elec(xy, mult, range(cell.natm)))
    assert np.abs(de[1] - de[0]).max() < 1e-7 * np.abs(de[0]).max()
