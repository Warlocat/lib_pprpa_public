"""Molecular density-fitted pp-RPA on the GPU from a gpu4pyscf DF RHF/RKS reference.

``get_pyscf_input_mol_gpu`` matches ``pyscf_util.get_pyscf_input_mol_r`` (same active-space
selection, the factors of ``mf.with_df``) with the MO transform on the device.
``df_gram_blocks`` forms the packed Gram triangles of ``gpu_ao2mo_blocks``,
E[(pr),(qs)] = (pr|qs) = sum_L L_pr L_qs, from those factors, and ``make_pprpa_df_gpu`` runs
the Davidson products on them through ``pprpa_eri_gpu``.  The Gram blocks are formed whole on
one device (small molecules).  The gradient is ``lib_pprpa.grad.pprpa_df_gpu``.
"""
import cupy as cp
from gpu4pyscf.lib.cupy_helper import contract

from lib_pprpa.pprpa_davidson import ppRPA_Davidson
from lib_pprpa.pprpa_eri_gpu import attach_gpu_eri_contraction

__all__ = ['df_gram_blocks', 'get_pyscf_input_mol_gpu', 'make_pprpa_df_gpu']


def get_pyscf_input_mol_gpu(mf, nocc_act=None, nvir_act=None):
    """Active-space pp-RPA input of a gpu4pyscf density-fitted RHF/RKS.

    Returns:
        nocc_act, mo_energy_act (NumPy) and Lpq (naux, nmo_act, nmo_act), CuPy.
    """
    with_df = getattr(mf, 'with_df', None)
    if with_df is None:
        raise TypeError('the GPU molecular pp-RPA needs a density-fitted reference '
                        '(mf.density_fit())')
    if with_df._cderi is None:
        with_df.build()
    if len(with_df._cderi) != 1:
        raise NotImplementedError('the DF factors must sit on one device')
    nmo = len(mf.mo_energy)
    nocc = mf.mol.nelectron // 2
    nocc_act = nocc if nocc_act is None else min(nocc, nocc_act)
    nvir_act = nmo - nocc if nvir_act is None else min(nmo - nocc, nvir_act)
    act = slice(nocc - nocc_act, nocc + nvir_act)
    mo_energy = cp.asnumpy(cp.asarray(mf.mo_energy)[act])
    mo = cp.asarray(mf.mo_coeff)[:, act]
    nmo_act = nocc_act + nvir_act
    lpq = cp.empty((with_df.naux, nmo_act, nmo_act))
    p0 = 0
    for cderi, _ in with_df.loop():
        p1 = p0 + cderi.shape[0]
        lpq[p0:p1] = contract('Lnq,np->Lpq', contract('Lmn,mq->Lnq', cderi, mo), mo)
        p0 = p1
    return nocc_act, mo_energy, lpq


def _packed_tril(E):
    r, c = cp.tril_indices(E.shape[0])
    return E[r, c]


def _compact_gram(L):
    """Packed (pr|qs) over the compact pairs p >= q of one symmetric factor block."""
    p, q = cp.tril_indices(L.shape[1])
    Lc = L[:, p, q]
    return _packed_tril(Lc.T @ Lc)


def df_gram_blocks(lpq, nocc):
    """Packed Gram triangles vvvv, oovv, oooo (``gpu_ao2mo_blocks`` layout) from DF factors.

    Args:
        lpq : (naux, nmo, nmo) MO factors, NumPy or CuPy.
        nocc : occupied orbitals of ``lpq`` (the first ``nocc``).
    Returns:
        vvvv, oovv, oooo as 1-D CuPy arrays.
    """
    lpq = cp.asarray(lpq)
    naux = lpq.shape[0]
    lov = lpq[:, :nocc, nocc:].reshape(naux, -1)    # pair (i, a) at i * nvir + a
    return (_compact_gram(lpq[:, nocc:, nocc:]), _packed_tril(lov.T @ lov),
            _compact_gram(lpq[:, :nocc, :nocc]))


def make_pprpa_df_gpu(mf, channel='pp', nocc_act=None, nvir_act=None, verbose=None,
                      **davidson_kwargs):
    """A ppRPA_Davidson whose products run on the GPU from the DF factors of ``mf``.

    Kwargs:
        davidson_kwargs : passed to ppRPA_Davidson.  Only identity trial vectors are
            available: the subspace guess reads the factors on the host.
    """
    if davidson_kwargs.get('trial', 'identity') != 'identity':
        raise ValueError('make_pprpa_df_gpu supports trial="identity" only')
    nocc, mo_energy, lpq = get_pyscf_input_mol_gpu(mf, nocc_act, nvir_act)
    pprpa = ppRPA_Davidson(nocc, mo_energy, None, channel=channel, **davidson_kwargs)
    blocks = df_gram_blocks(lpq, nocc)
    del lpq
    return attach_gpu_eri_contraction(pprpa, *blocks,
                                      verbose=mf.verbose if verbose is None else verbose)
