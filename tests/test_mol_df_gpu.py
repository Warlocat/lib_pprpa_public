"""GPU molecular DF pp-RPA (mol_df_gpu, grad.pprpa_df_gpu) against the CPU DF path."""
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

ATOM = "O 0.02 -0.01 0.03; H 0 -.757 .587; H 0.05 .757 .587"


def _scf(xc):
    from gpu4pyscf import dft, scf
    from pyscf import gto

    mol = gto.M(atom=ATOM, basis="6-31g*", charge=2, verbose=0)
    mf = (scf.RHF(mol) if xc == "hf" else dft.RKS(mol, xc=xc)).density_fit(
        auxbasis="def2-universal-jkfit")
    if xc != "hf":
        mf.grids.level = 4
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    # the CPU reference shares the GPU orbitals, grids and auxiliary basis
    return mf, mf.to_cpu()


def _cpu_solver(mf_cpu, mult, **kw):
    from lib_pprpa.pprpa_davidson import ppRPA_Davidson
    from lib_pprpa.pyscf_util import get_pyscf_input_mol

    nocc, mo_energy, lpq = get_pyscf_input_mol(mf_cpu, **kw)
    solver = ppRPA_Davidson(nocc, mo_energy, lpq, channel="pp", nroot=3,
                            residue_thresh=1e-10, max_iter=200, trial="identity")
    solver.mu = 0.0
    solver.kernel(mult)
    return solver


@pytest.mark.parametrize("act", [{}, {"nocc_act": 3, "nvir_act": 10}])
def test_gram_blocks_match_cpu_factors(act):
    import cupy as cp
    from lib_pprpa.mol_df_gpu import df_gram_blocks, get_pyscf_input_mol_gpu
    from lib_pprpa.pprpa_eri_gpu import pack_eri
    from lib_pprpa.pyscf_util import get_pyscf_input_mol

    mf, mf_cpu = _scf("pbe")
    nocc, mo_energy, lpq = get_pyscf_input_mol_gpu(mf, **act)
    nocc_ref, mo_energy_ref, lpq_ref = get_pyscf_input_mol(mf_cpu, **act)
    assert nocc == nocc_ref
    assert np.allclose(mo_energy, mo_energy_ref, atol=1e-12)
    eri = np.einsum("Lpr,Lqs->pqrs", lpq_ref, lpq_ref)     # <pq|rs> = (pr|qs)
    o, v = slice(0, nocc), slice(nocc, None)
    ref = pack_eri(eri[v, v, v, v], eri[o, o, v, v], eri[o, o, o, o])
    for got, want in zip(df_gram_blocks(lpq, nocc), ref):
        assert abs(cp.asnumpy(got) - want).max() < 1e-10


@pytest.mark.parametrize("xc", ["hf", "pbe", "b3lyp"])
@pytest.mark.parametrize("mult", ["s", "t"])
def test_energy_and_gradient_match_cpu(xc, mult):
    # explicit class: other modules (grad.pprpa_gamma) rebind ppRPA_Davidson.Gradients
    from lib_pprpa.grad.pprpa import Gradients as CpuGradients
    from lib_pprpa.grad.pprpa_df_gpu import Gradients
    from lib_pprpa.mol_df_gpu import make_pprpa_df_gpu

    mf, mf_cpu = _scf(xc)
    ref = _cpu_solver(mf_cpu, mult)
    solver = make_pprpa_df_gpu(mf, channel="pp", nroot=3, residue_thresh=1e-10,
                               max_iter=200, verbose=0)
    solver.mu = 0.0
    solver.kernel(mult)
    exci = solver.exci_s if mult == "s" else solver.exci_t
    exci_ref = ref.exci_s if mult == "s" else ref.exci_t
    assert abs(exci - exci_ref).max() < 1e-9

    for state in range(2):
        g_ref = CpuGradients(ref, mf_cpu, mult, state)
        g_ref.verbose = 0
        g_ref.cphf_conv_tol = 1e-10
        g = Gradients(solver, mf, mult, state)
        g.verbose = 0
        g.cphf_conv_tol = 1e-10
        assert abs(g.kernel() - g_ref.kernel()).max() < 1e-7
