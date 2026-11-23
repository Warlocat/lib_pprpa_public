"""The fused Gamma response (J inside the fxc pass) against gpu4pyscf's KRKS.gen_response:
the full matrix and the vir-occ block, LDA and GGA; make_gpu_vresp's routing."""

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


def _diamond(xc):
    from pyscf.pbc import dft, gto
    a0 = 3.370137329
    cell = gto.M(atom=[["C", [0., 0., 0.]], ["C", [a0 / 2, a0 / 2, a0 / 2]]],
                 a=np.array([[0, a0, a0], [a0, 0, a0], [a0, a0, 0]]),
                 unit="bohr", basis="gth-dzv", pseudo="gth-pade", mesh=[20] * 3, verbose=0)
    mf = dft.RKS(cell, xc=xc)
    mf.exxdiv = None
    mf.conv_tol = 1e-10
    mf.kernel()
    return cell, mf


def _trial(mf):
    occ = mf.mo_occ
    o, v = mf.mo_coeff[:, occ > 0], mf.mo_coeff[:, occ == 0]
    t = np.random.default_rng(0).standard_normal((v.shape[1], o.shape[1]))
    dm = v @ t @ o.T
    return o, v, t, dm + dm.T


@pytest.mark.parametrize("xc", ("pbe", "lda,vwn"))
def test_fused_response_matches_gen_response(xc):
    import cupy as cp
    from gpu4pyscf.pbc import dft as gdft
    from lib_pprpa.gpu_response import fused_response
    cell, mf = _diamond(xc)
    o, v, t, dm = _trial(mf)
    kmf = gdft.KRKS(cell, kpts=np.zeros((1, 3)), xc=xc)
    kmf.exxdiv = None
    kmf.mo_coeff = cp.asarray(mf.mo_coeff)[None]
    kmf.mo_energy = cp.asarray(mf.mo_energy)[None]
    kmf.mo_occ = cp.asarray(mf.mo_occ)[None]
    kmf.grids.build()
    nao = cell.nao
    ref = cp.asnumpy(kmf.gen_response(singlet=None, hermi=1)(
        cp.asarray(dm).reshape(1, 1, nao, nao))[0, 0])
    respond = fused_response(cell, kmf)
    got = cp.asnumpy(respond(cp.asarray(dm)))
    assert np.abs(got - ref).max() < 1e-10 * np.abs(ref).max()
    got_ket = cp.asnumpy(respond(cp.asarray(dm), ket=cp.asarray(o)))
    assert np.abs(got_ket - ref @ o).max() < 1e-10 * np.abs(ref @ o).max()


def test_make_gpu_vresp_routes():
    """LDA/GGA get the fused response, a hybrid gen_response; both serve ``ket``."""
    from gpu4pyscf.pbc import dft as gdft
    from lib_pprpa.grad.pprpa_gamma_gpu import _fused_applies, make_gpu_vresp
    cell, mf = _diamond("pbe")
    o, v, t, dm = _trial(mf)
    for xc, fused in (("pbe", True), ("pbe0", False)):
        mf.xc = xc
        assert _fused_applies(gdft.KRKS(cell, kpts=np.zeros((1, 3)), xc=xc)) == fused
        vresp = make_gpu_vresp(cell, mf, emulate=False)
        ref = vresp(dm) @ o
        np.testing.assert_allclose(vresp(dm, ket=o), ref, rtol=0, atol=1e-10 * np.abs(ref).max())
