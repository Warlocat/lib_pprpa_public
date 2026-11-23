'''Gamma-point KS response (J + fxc)[D] on the uniform grid in two AO passes.

gpu4pyscf's ``KRKS.gen_response`` builds fxc[D] (density and potential passes) and
J[D] (FFTDF ``get_j``: two more) apart: three passes, four nao^2 x ngrid products.
Here one density pass serves both (rho1[0] is J's density), J's potential joins the
fxc weights, and one potential pass builds the matrix: two passes, two products.  With
a ``ket`` (the occupied orbitals of a CPHF iteration) it returns V @ ket directly from
Z = ao0^T (aow ket) + aow^T (ao0 ket), 2 nk / nao of a nao^2 product.  LDA/GGA without
exact exchange, real symmetric D (``hermi=1``), FFTDF J; matches ``gen_response`` up to
the summation order.
'''

import numpy as np
import cupy as cp
from gpu4pyscf.pbc import tools as gtools
from gpu4pyscf.pbc.dft.numint import _scale_ao

__all__ = ['fused_response']

_KPTS0 = np.zeros((1, 3))


def fused_response(cell, kmf):
    '''``respond(dm, ket=None)`` for the Gamma KRKS ``kmf`` (KNumInt on its uniform grids,
    LDA/GGA): (J + fxc)[dm] (nao, nao) for a real symmetric CuPy dm, or that matrix
    times ``ket`` (nao, nk), CuPy.
    '''
    ni, grids, xc = kmf._numint, kmf.grids, kmf.xc
    if grids.coords is None:
        grids.build()
    xctype = ni._xc_type(xc)
    deriv = 0 if xctype == 'LDA' else 1
    ncomp = 1 + 3 * deriv
    mesh = np.asarray(cell.mesh)
    ngrid = int(np.prod(mesh))
    weight = cell.vol / ngrid
    fxc = ni.cache_xc_kernel1(cell, grids, xc, kmf.make_rdm1(), 0, _KPTS0, is_rhf=True)[2]
    order = cp.asarray(grids.argsort(tile=8))              # block_loop(sort_grids=True)
    coulG = gtools.get_coulG(cell, mesh=mesh)

    def blocks():
        p0 = 0
        for ao_ks, w, _ in ni.block_loop(cell, grids, deriv, _KPTS0, sort_grids=True):
            p1 = p0 + w.size
            yield p0, p1, ao_ks
            p0 = p1

    def potential_weights(dm):
        rho = cp.empty((ncomp, ngrid))
        for p0, p1, ao_ks in blocks():
            rho[:, p0:p1] = ni.eval_rho(cell, ao_ks, dm[None], xctype=xctype,
                                        hermi=1).real.reshape(ncomp, -1)
        rho_g = cp.empty(ngrid)
        rho_g[order] = rho[0]
        vj = gtools.ifft(coulG * gtools.fft(rho_g[None], mesh), mesh).real[0]
        wv = cp.einsum('xg,xyg->yg', rho, fxc) * weight
        wv[0] += vj[order] * weight
        wv[0] *= .5                    # the matrix is Z + Z^T, as _vxc_mat with hermi=1
        return wv

    def scaled(ao, wv):
        return _scale_ao(ao, wv[0]) if xctype == 'LDA' else _scale_ao(ao[:4], wv)

    def respond(dm, ket=None):
        wv = potential_weights(dm)
        nk = cell.nao if ket is None else ket.shape[1]
        Z = cp.zeros((cell.nao, nk))
        for p0, p1, ao_ks in blocks():
            ao = ao_ks[0]
            ao0 = ao if xctype == 'LDA' else ao[0]
            aow = scaled(ao, wv[:, p0:p1])
            if ket is None:
                Z += ao0.T @ aow
            else:
                Z += ao0.T @ (aow @ ket)
                Z += aow.T @ (ao0 @ ket)
        return Z + Z.T if ket is None else Z

    return respond
