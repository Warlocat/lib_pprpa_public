import h5py
import math
import numpy
import scipy

import pyscf
import pyscf.dft
from pyscf.lib import unpack_tril
from pyscf import df
from pyscf.ao2mo import _ao2mo

from lib_pprpa.analyze import get_pprpa_nto, get_pprpa_dm
from lib_pprpa.pprpa_util import start_clock, stop_clock, get_nocc_nvir_frac


# get input from PySCF
def get_pyscf_input_mol(
        mf, auxbasis=None, nocc_act=None, nvir_act=None, dump_file=None,
        sort_mo=False, cholesky=False, with_dip=False):
    """
    Get input from PySCF for a molecular calculation.
    Will be saved to a file if `dump_file` is specified.
    mo_dipole will also be saved if `with_dip` is True. Only implemented for RHF/RKS.
    """

    if isinstance(mf, pyscf.scf.uhf.UHF) or isinstance(mf, pyscf.dft.uks.UKS):
        return get_pyscf_input_mol_u(
            mf, auxbasis=auxbasis, nocc_act=nocc_act, nvir_act=nvir_act,
            dump_file=dump_file, sort_mo=sort_mo)

    if isinstance(mf, pyscf.scf.rohf.ROHF) or\
            isinstance(mf, pyscf.dft.roks.ROKS):
        mf_new = mf.copy()
        mf_new.mo_coeff = [mf.mo_coeff, mf.mo_coeff]
        mf_new.mo_energy = [mf.mo_energy, mf.mo_energy]
        return get_pyscf_input_mol_u(
            mf_new, auxbasis=auxbasis, nocc_act=nocc_act, nvir_act=nvir_act,
            dump_file=dump_file, sort_mo=sort_mo)

    if isinstance(mf, pyscf.scf.rhf.RHF) or isinstance(mf, pyscf.dft.rks.RKS):
        return get_pyscf_input_mol_r(
            mf, auxbasis=auxbasis, nocc_act=nocc_act, nvir_act=nvir_act,
            dump_file=dump_file, sort_mo=sort_mo, cholesky=cholesky, with_dip=with_dip)

    if isinstance(mf, pyscf.scf.ghf.GHF) or isinstance(mf, pyscf.dft.gks.GKS):
        return get_pyscf_input_mol_g(
            mf, auxbasis=auxbasis, nocc_act=nocc_act, nvir_act=nvir_act,
            dump_file=dump_file, sort_mo=sort_mo)


def get_pyscf_input_mol_r(
        mf, auxbasis=None, nocc_act=None, nvir_act=None, dump_file=None,
        sort_mo=False, cholesky=False, with_dip=False):
    """Get ppRPA input from a PySCF molecular SCF calculation.

    Args:
        mf (pyscf.scf.RHF/pyscf.dft.RKS): molecular mean-field object.
        auxbasis (string, optional): auxiliary basis set. Defaults to None.
        nocc_act (int, optional): number of active occupied orbitals. Defaults to None.
        nvir_act (int, optional): number of active virtual orbitals. Defaults to None.
        dump_file (str, optional): file name to dump matrix for lib_pprpa. Defaults to None.
        cholesky (bool, optional): use Cholesky decomposition. Defaults to False.
        with_dip (bool, optional): save dipole moment. Defaults to False.

    Returns:
        nocc_act (int): number of occupied orbitals in the active space.
        mo_energy_act (double array): orbital energy in the active space.
        Lpq (double ndarray): three-center density-fitting matrix in active MO space.
    """
    start_clock("getting input for molecule ppRPA from PySCF")

    nmo = len(mf.mo_energy)
    nocc = mf.mol.nelectron // 2
    nvir = nmo - nocc
    mo_energy = numpy.array(mf.mo_energy)
    mo_coeff = numpy.array(mf.mo_coeff)

    if sort_mo is True:
        occ_index = numpy.where(mf.mo_occ > 0.5)[0]
        vir_index = numpy.where(mf.mo_occ < 0.5)[0]
        print("sorting molecular orbitals")
        print("occ index = ", occ_index)
        print("vir index = ", vir_index)
        if occ_index[-1] < vir_index[0]:
            print("warning: no sorting is performed!")
        mo_energy_occ = mo_energy[occ_index]
        mo_energy_vir = mo_energy[vir_index]
        mo_energy = numpy.concatenate((mo_energy_occ, mo_energy_vir))
        mo_coeff_occ = mo_coeff[:, occ_index]
        mo_coeff_vir = mo_coeff[:, vir_index]
        mo_coeff = numpy.concatenate((mo_coeff_occ, mo_coeff_vir), axis=1)

    nocc_act = nocc if nocc_act is None else min(nocc, nocc_act)
    nvir_act = nvir if nvir_act is None else min(nvir, nvir_act)
    nmo_act = nocc_act + nvir_act
    mo_energy_act = numpy.array(mo_energy[(nocc - nocc_act) : (nocc + nvir_act)])

    if with_dip:
        mo_dip = calculate_dipole_moment_MO(mf, nocc_act=nocc_act, nvir_act=nvir_act)
    else:
        mo_dip = None

    if cholesky is False:
        if getattr(mf, 'with_df', None):
            pass
        else:
            mf.with_df = df.DF(mf.mol)
            if auxbasis is not None:
                mf.with_df.auxbasis = auxbasis
            else:
                try:
                    mf.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
                except RuntimeError:
                    mf.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=False)
            mf._keys.update(['with_df'])

        naux = mf.with_df.get_naoaux()
        ijslice = (nocc-nocc_act, nocc+nvir_act, nocc-nocc_act, nocc+nvir_act)
        try:
            Lpq = None
            Lpq = _ao2mo.nr_e2(mf.with_df._cderi, mo_coeff, ijslice, aosym='s2', out=Lpq)
            Lpq = Lpq.reshape(naux, nmo_act, nmo_act)
        except AttributeError:
            # Chunking if filename passed as _cderi.
            Lpq = []
            for LpqR in mf.with_df.loop():
                tmp = None
                tmp = _ao2mo.nr_e2(LpqR, mo_coeff, ijslice, aosym='s2', out=tmp)
                Lpq.append(tmp)
            Lpq = numpy.vstack(Lpq).reshape(naux, nmo_act, nmo_act)
            Lpq = numpy.ascontiguousarray(Lpq)
    else:
        mo_coeff_act = numpy.array(mo_coeff[:, (nocc - nocc_act) : (nocc + nvir_act)])
        if hasattr(mf, 'with_df'):
            full_eri = mf.with_df.ao2mo(mo_coeff_act, compact=True)
        else:
            # TODO: use non density-fitting ERI
            mf.with_df = df.DF(mf.mol)
            if auxbasis is not None:
                mf.with_df.auxbasis = auxbasis
            else:
                try:
                    mf.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
                except RuntimeError:
                    mf.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=False)
            mf._keys.update(['with_df'])
            full_eri = mf.with_df.ao2mo(mo_coeff_act, compact=True)

        cd = Cholesky(full_eri, err_tol=1e-5, aosym='s4')
        Lpq = cd.kernel()
        Lpq = Lpq.reshape(-1, nmo_act, nmo_act)
        naux = Lpq.shape[0]

    if dump_file is not None:
        _write_dump_file(dump_file, nocc_act, mo_energy_act, Lpq, mo_dip=mo_dip)

    print("\nget input for lib_pprpa from PySCF (molecule)")
    print("nmo = %-d, nocc= %-d, nvir = %-d" % (nmo, nocc, nvir))
    print("nmo_act = %-d, nocc_act= %-d, nvir_act = %-d" % (nmo_act, nocc_act, nvir_act))
    print("naux = %-d" % naux)
    print("dump h5py file = %-s" % dump_file)

    stop_clock("getting input for molecule ppRPA from PySCF")

    if with_dip:
        return nocc_act, mo_energy_act, Lpq, mo_dip
    else:
        return nocc_act, mo_energy_act, Lpq


def get_pyscf_input_mol_u(
        mf, auxbasis=None, nocc_act=None, nvir_act=None, dump_file=None,
        sort_mo=False):
    """Get ppRPA input from a PySCF unrestricted calculation.

    Args:
        mf (pyscf.scf.UHF/pyscf.dft.UKS): molecular mean-field object.

    Kwargs:
        auxbasis (str): name of the auxiliary basis set. Default to None.
        nocc_act (tuple of int): number of active occupied orbitals,
            (Nalpha, Nbeta). Default to None.
        nvir_act (tuple of int): number of active virtual orbitals,
            (Nalpha, Nbeta). Default to None.
        dump_file (str): name of the file to dump matrices for lib_pprpa.
            Default to None.

    Returns:
        nocc_act (tuple of int): number of occupied orbitals
            in the active space.
        mo_energy_act (list of double ndarray): orbital energies in the active space.
        Lpq (list of double ndarray): three-center density fitting matrix
            in the active MO space.
    """
    from pyscf import df
    from pyscf.ao2mo import _ao2mo

    start_clock("getting input for unrestricted ppRPA from PySCF")

    nmo = (len(mf.mo_energy[0]), len(mf.mo_energy[1]))
    nocc = mf.nelec
    nvir = (nmo[0] - nocc[0], nmo[1] - nocc[1])
    mo_energy = numpy.array(mf.mo_energy)
    mo_coeff = numpy.array(mf.mo_coeff)
    if sort_mo is True:
        for s in range(2):
            occ_index = numpy.where(mf.mo_occ[s] > 0.5)[0]
            vir_index = numpy.where(mf.mo_occ[s] < 0.5)[0]
            spin = "alpha" if s == 0 else "beta"
            print("sorting %s spin molecular orbitals" % spin)
            print("occ index = ", occ_index)
            print("vir index = ", vir_index)
            if occ_index[-1] < vir_index[0]:
                print("warning: no sorting is performed!")
            mo_energy_occ = mo_energy[s][occ_index]
            mo_energy_vir = mo_energy[s][vir_index]
            mo_energy[s] = numpy.concatenate((mo_energy_occ, mo_energy_vir))
            mo_coeff_occ = mo_coeff[s][:, occ_index]
            mo_coeff_vir = mo_coeff[s][:, vir_index]
            mo_coeff[s] = numpy.concatenate((mo_coeff_occ, mo_coeff_vir), axis=1)

    if nocc_act is None:
        nocc_act = nocc
    else:
        if isinstance(nocc_act, (int, numpy.int64)):
            nocc_act = [nocc_act, nocc_act]
        nocc_act = (min(nocc[0], nocc_act[0]), min(nocc[1], nocc_act[1]))
    if nvir_act is None:
        nvir_act = nvir
    else:
        if isinstance(nvir_act, (int, numpy.int64)):
            nvir_act = [nvir_act, nvir_act]
        nvir_act = (min(nvir[0], nvir_act[0]), min(nvir[1], nvir_act[1]))
    nmo_act = (nocc_act[0] + nvir_act[0], nocc_act[1] + nvir_act[1])
    mo_energy_act = [
        mo_energy[0, (nocc[0]-nocc_act[0]):(nocc[0]+nvir_act[0])],
        mo_energy[1, (nocc[1]-nocc_act[1]):(nocc[1]+nvir_act[1])]]

    if getattr(mf, 'with_df', None):
        pass
    else:
        mf.with_df = df.DF(mf.mol)
        if auxbasis is not None:
            mf.with_df.auxbasis = auxbasis
        else:
            try:
                mf.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
            except:
                mf.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=False)
        mf._keys.update(['with_df'])

    naux = mf.with_df.get_naoaux()
    ijslice = [
        (
            nocc[0]-nocc_act[0], nocc[0]+nvir_act[0],
            nocc[0]-nocc_act[0], nocc[0]+nvir_act[0]),
        (
            nocc[1]-nocc_act[1], nocc[1]+nvir_act[1],
            nocc[1]-nocc_act[1], nocc[1]+nvir_act[1]),
    ]
    Lpq_a = None
    Lpq_b = None
    Lpq_a = _ao2mo.nr_e2(mf.with_df._cderi, mo_coeff[0], ijslice[0],
                         aosym='s2', out=Lpq_a)
    Lpq_a = Lpq_a.reshape(naux, nmo_act[0], nmo_act[0])
    Lpq_b = _ao2mo.nr_e2(mf.with_df._cderi, mo_coeff[1], ijslice[1],
                         aosym='s2', out=Lpq_b)
    Lpq_b = Lpq_b.reshape(naux, nmo_act[1], nmo_act[1])
    Lpq = [Lpq_a, Lpq_b]

    if dump_file is not None:
        _write_dump_file(dump_file, nocc_act, mo_energy_act, Lpq)

    print("\nget input for lib_pprpa from PySCF (molecule)")
    print("nmo = %-d (%-d alpha, %-d beta)" %
          (nmo[0] + nmo[1], nmo[0], nmo[1]), end=', ')
    print("nocc = %-d (%-d alpha, %-d beta)" %
          (nocc[0] + nocc[1], nocc[0], nocc[1]), end=', ')
    print("nvir = %-d (%-d alpha, %-d beta)" %
          (nvir[0] + nvir[1], nvir[0], nvir[1]))
    print("nmo_act = %-d (%-d alpha, %-d beta)" %
          (nmo_act[0] + nmo_act[1], nmo_act[0], nmo_act[1]), end=', ')
    print("nocc_act = %-d (%-d alpha, %-d beta)" %
          (nocc_act[0] + nocc_act[1], nocc_act[0], nocc_act[1]), end=', ')
    print("nvir_act = %-d (%-d alpha, %-d beta)" %
          (nvir_act[0] + nvir_act[1], nvir_act[0], nvir_act[1]))
    print("naux = %-d" % naux)
    print("dump h5py file = %-s" % dump_file)

    stop_clock("getting input for unrestricted ppRPA from PySCF")

    return nocc_act, mo_energy_act, Lpq


class Cholesky:
    def __init__(self, eri, err_tol=1e-6, aosym='s1'):
        """Constructor for incore pivoted cholesky

        Parameters
        ----------
        eri : ERI matrix
            s1 or s4 symmetry. For s1, either 4d or 2d is accepted.
        err_tol : float, optional
            threshold to stop pivoted cholesky.
            The default is 1e-5.
        aosym : str, optional
            eri symmetry, by default 's1'
        """
        self.eri = eri
        self.err_tol = err_tol
        self.aosym = aosym
        if aosym == 's1':
            if eri.ndim == 4:
                self.norb = eri.shape[0]
            elif eri.ndim == 2:
                self.norb = int(numpy.rint(numpy.sqrt(eri.shape[0])))
            self.eri = eri.reshape(self.norb**2, -1)
        elif aosym == 's4':
            npair = self.eri.shape[0]
            norb = math.floor(math.sqrt(1 + 8 * npair) / 2)
            self.norb = norb
        else:
            raise NotImplementedError

    def kernel(self, overwrite_input=False):
        """Fast pivoted cholesky using LAPACK

        Parameters
        ----------
        overwrite_input : bool, optional
            overwrites the input ERI for efficiency, by default False

        Returns
        -------
        numpy.ndarray
            (naux, norb, norb) array containing CDERI
        """

        erimat = self.eri

        pivoted_cholesky = scipy.linalg.lapack.get_lapack_funcs('pstrf', (erimat,))
        # documentation for ?pstrf:
        # https://www.netlib.org/lapack/explore-html/de/de0/group__pstrf.html
        copy = scipy.linalg.get_blas_funcs('copy', (erimat,))
        cderi_perm, piv, rank, info = pivoted_cholesky(erimat.T, tol=self.err_tol, lower=0, overwrite_a=overwrite_input)
        cderi = numpy.zeros((rank, erimat.shape[0]), dtype=erimat.dtype)
        invpiv = numpy.argsort(piv)
        cderi_ravel = cderi.reshape(-1)

        # Now we must permute the columns of the cholesky factor
        # In addition, the lower triangle is filled with garbage,
        # so we shouldn't copy that.
        # Only the first `rank` columns are needed.
        for j in range(erimat.shape[0]):
            i = invpiv[j]
            # the invocation of BLAS ?copy is equivalent to
            # Lpq[:min(rank, i+1), j] = Lpq_perm[:min(rank, i+1), i]
            copy(cderi_perm[: min(rank, i + 1), i], cderi_ravel, offy=j, incy=cderi.shape[1], n=min(rank, i + 1))
        cderi_perm = None
        if self.aosym == 's1':
            return cderi
        else:  # s4
            return unpack_tril(cderi)


def nr_e2_cross(
        eri, mo_coeff1, mo_coeff2, nocc, nocc_act, nvir_act,
        aosym='s1', mosym='s1', out=None, ao_loc=None):
    """A wrapper for pyscf.ao2mo.nr_e2 to compute density fitting
        MO integrals in active space with two sets of MO coefficients
        as left and right coeff.
        Lpq = C1.T Lmn C2

    Args:
        eri (double ndarray): density fitting AO integrals.
        mo_coeff1 (double ndarray): MO coefficients for the left index.
        mo_coeff2 (double ndarray): MO coefficients for the right index.
        nocc (int): number of occupied orbitals.
        nocc_act (int): number of active occupied orbitals.
        nvir_act (int): number of active virtual orbitals.

    Kwargs:
        aosym (str): symmetry of AO integrals. Default to 's1'.
        mosym (str): symmetry of MO integrals. Default to 's1'.
        out (double ndarray): output array. Default to None.
        ao_loc: see pyscf.ao2mo._ao2mo

    Returns:
        Lpq (double ndarray): density fitting MO integrals.
    """
    from pyscf.ao2mo import _ao2mo
    mo_coeff = numpy.asarray(numpy.hstack((mo_coeff1, mo_coeff2)), order='F')
    offset_mo1 = mo_coeff1.shape[1]
    ijslice = (nocc-nocc_act, nocc+nvir_act,
               offset_mo1+nocc-nocc_act, offset_mo1+nocc+nvir_act)
    return _ao2mo.nr_e2(eri, mo_coeff, ijslice, aosym, mosym, out, ao_loc)


def get_Lmo_ghf(mf, naux=None, mo_coeff=None, nocc_act=None, nvir_act=None):
    """Get density-fitting MO integrals Lpq from a PySCF GHF calculation.

    Args:
        mf (RHF/RKS/GHF/GKS/JHF/JKS object): molecular mean-field object.
            It is designed to be compatible with all of above mf classes.
            For RHF, it will first generate GHF-type parameters from RHF object
            and do the subsequent procedure as GHF.

    Kwargs:
        naux (int): number of auxiliary basis functions. Default to None.
        mo_coeff (double/complex ndarray): MO coefficients. Default to None.
        nocc_act (int): number of active occupied orbitals. Default to None.
        nvir_act (int): number of active virtual orbitals. Default to None.

    Returns:
        Lpq (double/complex ndarray): density fitting MO integrals
            in active space (naux, nmo_act, nmo_act).
    """
    from pyscf import scf, dft
    from pyscf.ao2mo import _ao2mo

    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if naux is None:
        naux = mf.with_df.get_naoaux()

    nao = mf.mol.nao_nr()
    nocc = mf.mol.nelec[0] + mf.mol.nelec[1]
    nmo = len(mf.mo_energy)
    if isinstance(mf, scf.hf.RHF) or isinstance(mf, dft.rks.RKS):
        nmo *= 2
    nvir = nmo - nocc
    nocc_act = nocc if nocc_act is None else min(nocc, nocc_act)
    nvir_act = nvir if nvir_act is None else min(nvir, nvir_act)
    nmo_act = nocc_act + nvir_act

    if isinstance(mf, scf.ghf.GHF) or isinstance(mf, dft.gks.GKS):
        mo_coeff_a = mf.mo_coeff[:nao, :]
        mo_coeff_b = mf.mo_coeff[nao:, :]
    elif isinstance(mf, scf.hf.RHF) or isinstance(mf, dft.rks.RKS):
        mo_coeff_a = numpy.zeros((nao, nmo))
        mo_coeff_b = numpy.zeros((nao, nmo))
        for i in range(nmo // 2):
            mo_coeff_a[:, 2 * i] = mf.mo_coeff[:, i]
            mo_coeff_b[:, 2 * i + 1] = mf.mo_coeff[:, i]
    else:
        try:
            from socutils.scf import spinor_hf
            from socutils.dft import dft
        except ImportError:
            raise ImportError('socutils is required for JHF/JKS.')
        if isinstance(mf, spinor_hf.SpinorSCF) or isinstance(mf, dft.SpinorDFT):
            c2 = numpy.vstack(mf.mol.sph2spinor_coeff())
            mo_sph = numpy.dot(c2, mf.mo_coeff)
            mo_coeff_a = mo_sph[:nao, :]
            mo_coeff_b = mo_sph[nao:, :]
        else:
            raise ValueError('mf should be GHF/GKS/JKS/JKS.')

    if mo_coeff_a.dtype == numpy.double:  # if mo_coeff is real
        ijslice = (nocc - nocc_act, nocc + nvir_act, nocc - nocc_act, nocc + nvir_act)
        Lpq = _ao2mo.nr_e2(mf.with_df._cderi, mo_coeff_a, ijslice, aosym='s2')
        Lpq += _ao2mo.nr_e2(mf.with_df._cderi, mo_coeff_b, ijslice, aosym='s2')
    elif mo_coeff_a.dtype == numpy.complex128:  # if mo_coeff is complex
        Lpq = numpy.zeros((naux, nmo_act * nmo_act), dtype=numpy.complex128)
        Lpq += nr_e2_cross(mf.with_df._cderi, mo_coeff_a.real, mo_coeff_a.real, nocc, nocc_act, nvir_act, aosym='s2')
        Lpq += nr_e2_cross(mf.with_df._cderi, mo_coeff_b.real, mo_coeff_b.real, nocc, nocc_act, nvir_act, aosym='s2')
        Lpq += nr_e2_cross(mf.with_df._cderi, mo_coeff_a.imag, mo_coeff_a.imag, nocc, nocc_act, nvir_act, aosym='s2')
        Lpq += nr_e2_cross(mf.with_df._cderi, mo_coeff_b.imag, mo_coeff_b.imag, nocc, nocc_act, nvir_act, aosym='s2')
        Lpq += 1.0j * nr_e2_cross(
            mf.with_df._cderi, mo_coeff_a.real, mo_coeff_a.imag, nocc, nocc_act, nvir_act, aosym='s2'
        )
        Lpq += 1.0j * nr_e2_cross(
            mf.with_df._cderi, mo_coeff_b.real, mo_coeff_b.imag, nocc, nocc_act, nvir_act, aosym='s2'
        )
        Lpq -= 1.0j * nr_e2_cross(
            mf.with_df._cderi, mo_coeff_a.imag, mo_coeff_a.real, nocc, nocc_act, nvir_act, aosym='s2'
        )
        Lpq -= 1.0j * nr_e2_cross(
            mf.with_df._cderi, mo_coeff_b.imag, mo_coeff_b.real, nocc, nocc_act, nvir_act, aosym='s2'
        )
    else:
        raise ValueError('mo_coeff should be either float64 or complex128.')

    return Lpq.reshape(naux, nmo_act, nmo_act)


def get_pyscf_input_mol_g(
        mf, auxbasis=None, nocc_act=None, nvir_act=None, dump_file=None,
        sort_mo=False):
    """Get ppRPA input from a PySCF generalized or spinor-based calculation.

    Args:
        mf (RHF/RKS/GHF/GKS/JHF/JKS object): molecular mean-field object.
        It is designed to be compatible with all of above mf classes.
        For RHF, it will first generate GHF-type parameters from RHF object
        and do the subsequent procedure as GHF.

    Kwargs:
        auxbasis (str): name of the auxiliary basis set. Default to None.
        nocc_act (int): number of active occupied orbitals. Default to None.
        nvir_act (int): number of active virtual orbitals. Default to None.
        dump_file (str): name of the file to dump matrices for lib_pprpa.
            Default to None.

    Returns:
        nocc_act (int): number of occupied orbitals in the active space.
        mo_energy_act (double ndarray): orbital energies in the active space.
        Lpq (double ndarray): three-center density fitting matrix
            in the active MO space.
    """

    start_clock('getting input for generalized ppRPA from PySCF')

    from pyscf.scf.hf import RHF
    from pyscf.dft.rks import RKS

    if isinstance(mf, RHF) or isinstance(mf, RKS):
        nmo = len(mf.mo_energy) * 2
        nao = mf.mo_coeff.shape[0]
        mo_energy_ab = numpy.concatenate((mf.mo_energy, mf.mo_energy))
        mo_energy = numpy.zeros_like(mo_energy_ab)
        mo_occ = numpy.zeros(nmo, dtype=mf.mo_occ.dtype)
        mo_coeff = numpy.zeros((nao*2, nmo), dtype=mf.mo_coeff.dtype)
        # Put alpha and beta parameters in the GHF order
        for i in range(len(mf.mo_energy)):
            mo_energy[2 * i] = mo_energy_ab[i]
            mo_energy[2 * i + 1] = mo_energy_ab[i]
            mo_occ[2 * i] = mf.mo_occ[i]/2
            mo_occ[2 * i + 1] = mf.mo_occ[i]/2
            mo_coeff[:nao, 2 * i] = mf.mo_coeff[:, i]
            mo_coeff[nao:, 2 * i + 1] = mf.mo_coeff[:, i]
    else:
        nmo = len(mf.mo_energy)
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        mo_coeff = mf.mo_coeff
    nocc = mf.mol.nelec[0] + mf.mol.nelec[1]
    nvir = nmo - nocc
    if sort_mo is True:
        occ_index = numpy.where(mo_occ > 0.5)[0]
        vir_index = numpy.where(mo_occ < 0.5)[0]
        print("sorting molecular orbitals")
        print("occ index = ", occ_index)
        print("vir index = ", vir_index)
        if occ_index[-1] < vir_index[0]:
            print("warning: no sorting is performed!")
        mo_energy_occ = mo_energy[occ_index]
        mo_energy_vir = mo_energy[vir_index]
        mo_energy = numpy.concatenate((mo_energy_occ, mo_energy_vir))
        mo_coeff_occ = mo_coeff[:, occ_index]
        mo_coeff_vir = mo_coeff[:, vir_index]
        mo_coeff = numpy.concatenate((mo_coeff_occ, mo_coeff_vir), axis=1)

    nocc_act = nocc if nocc_act is None else min(nocc, nocc_act)
    nvir_act = nvir if nvir_act is None else min(nvir, nvir_act)
    nmo_act = nocc_act + nvir_act
    mo_energy_act = mo_energy[(nocc - nocc_act) : (nocc + nvir_act)]

    from pyscf import df

    if getattr(mf, 'with_df', None):
        pass
    else:
        mf.with_df = df.DF(mf.mol)
        if auxbasis is not None:
            mf.with_df.auxbasis = auxbasis
        else:
            try:
                mf.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
            except:
                mf.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=False)
        mf._keys.update(['with_df'])

    naux = mf.with_df.get_naoaux()
    Lpq = get_Lmo_ghf(mf, naux=naux, mo_coeff=mo_coeff, nocc_act=nocc_act, nvir_act=nvir_act)

    if dump_file is not None:
        _write_dump_file(dump_file, nocc_act, mo_energy_act, Lpq)

    print('\nget input for lib_pprpa from PySCF (molecule)')
    print('nmo = %-d, nocc= %-d, nvir = %-d' % (nmo, nocc, nvir))
    print('nmo_act = %-d, nocc_act= %-d, nvir_act = %-d' % (nmo_act, nocc_act, nvir_act))
    print('naux = %-d' % naux)
    print('dump h5py file = %-s' % dump_file)

    stop_clock('getting input for generalized ppRPA from PySCF')

    return nocc_act, mo_energy_act, Lpq


def get_pyscf_input_mol_frac(
        mf, auxbasis=None, nocc_act=None, nvir_act=None, dump_file=None,
        sort_mo=True):
    """Get ppRPA input from a PySCF unrestricted calculation.

    Args:
        mf (pyscf.scf.UHF/pyscf.dft.UKS): molecular mean-field object.

    Kwargs:
        auxbasis (str): name of the auxiliary basis set. Default to None.
        nocc_act (tuple of int): number of active occupied orbitals,
            (Nalpha, Nbeta). Default to None.
        nvir_act (tuple of int): number of active virtual orbitals,
            (Nalpha, Nbeta). Default to None.
        dump_file (str): name of the file to dump matrices for lib_pprpa.
            Default to None.

    Returns:
        nocc_act (tuple of int): number of occupied orbitals
            in the active space.
        mo_energy_act (list of double ndarray): orbital energies in the active space.
        Lpq (list of double ndarray): three-center density fitting matrix
            in the active MO space.
    """
    from pyscf import df
    from pyscf.ao2mo import _ao2mo

    start_clock("getting input for ppRPA with fractional occupation from PySCF")

    nmo = (len(mf.mo_energy[0]), len(mf.mo_energy[1]))
    if sort_mo is True:
        nocc, nvir, frac_nocc = get_nocc_nvir_frac(
            mf.mo_occ, sort_mo=True, mo_energy=mf.mo_energy,
            mo_coeff=mf.mo_coeff)
    else:
        nocc, nvir, frac_nocc = get_nocc_nvir_frac(mf.mo_occ)
    mo_energy = numpy.array(mf.mo_energy)
    mo_coeff = numpy.array(mf.mo_coeff)

    if nocc_act is None:
        nocc_act = nocc
    else:
        if isinstance(nocc_act, (int, numpy.int64)):
            nocc_act = [nocc_act, nocc_act]
        nocc_act = (min(nocc[0], nocc_act[0]), min(nocc[1], nocc_act[1]))
    if nvir_act is None:
        nvir_act = nvir
    else:
        if isinstance(nvir_act, (int, numpy.int64)):
            nvir_act = [nvir_act, nvir_act]
        nvir_act = (min(nvir[0], nvir_act[0]), min(nvir[1], nvir_act[1]))
    nmo_act = (
        nocc_act[0] + nvir_act[0] - frac_nocc[0],
        nocc_act[1] + nvir_act[1] - frac_nocc[1])
    mo_energy_act = [
        mo_energy[0, (nocc[0]-nocc_act[0]):(nocc[0]+nvir_act[0])],
        mo_energy[1, (nocc[1]-nocc_act[1]):(nocc[1]+nvir_act[1])]]
    mo_occ= [
        mf.mo_occ[0, (nocc[0]-nocc_act[0]):(nocc[0]+nvir_act[0])],
        mf.mo_occ[1, (nocc[1]-nocc_act[1]):(nocc[1]+nvir_act[1])]]

    if getattr(mf, 'with_df', None):
        pass
    else:
        mf.with_df = df.DF(mf.mol)
        if auxbasis is not None:
            mf.with_df.auxbasis = auxbasis
        else:
            try:
                mf.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
            except:
                mf.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=False)
        mf._keys.update(['with_df'])

    naux = mf.with_df.get_naoaux()
    ijslice = [
        (
            nocc[0]-nocc_act[0], nocc[0]+nvir_act[0]-frac_nocc[0],
            nocc[0]-nocc_act[0], nocc[0]+nvir_act[0]-frac_nocc[0]),
        (
            nocc[1]-nocc_act[1], nocc[1]+nvir_act[1]-frac_nocc[1],
            nocc[1]-nocc_act[1], nocc[1]+nvir_act[1]-frac_nocc[1]),
    ]
    Lpq_a = None
    Lpq_b = None
    Lpq_a = _ao2mo.nr_e2(mf.with_df._cderi, mo_coeff[0], ijslice[0],
                         aosym='s2', out=Lpq_a)
    Lpq_a = Lpq_a.reshape(naux, nmo_act[0], nmo_act[0])
    Lpq_b = _ao2mo.nr_e2(mf.with_df._cderi, mo_coeff[1], ijslice[1],
                         aosym='s2', out=Lpq_b)
    Lpq_b = Lpq_b.reshape(naux, nmo_act[1], nmo_act[1])
    Lpq = [Lpq_a, Lpq_b]

    if dump_file is not None:
        _write_dump_file(dump_file, nocc_act, mo_energy_act, Lpq)

    print("\nget input for lib_pprpa from PySCF (molecule)")
    print("nmo = %-d (%-d alpha, %-d beta)" %
          (nmo[0] + nmo[1], nmo[0], nmo[1]), end=', ')
    print("nocc = %-d (%-d alpha, %-d beta)" %
          (nocc[0] + nocc[1], nocc[0], nocc[1]), end=', ')
    print("nvir = %-d (%-d alpha, %-d beta)" %
          (nvir[0] + nvir[1], nvir[0], nvir[1]))
    print("nmo_act = %-d (%-d alpha, %-d beta)" %
          (nmo_act[0] + nmo_act[1], nmo_act[0], nmo_act[1]), end=', ')
    print("nocc_act = %-d (%-d alpha, %-d beta)" %
          (nocc_act[0] + nocc_act[1], nocc_act[0], nocc_act[1]), end=', ')
    print("nvir_act = %-d (%-d alpha, %-d beta)" %
          (nvir_act[0] + nvir_act[1], nvir_act[0], nvir_act[1]))
    print("naux = %-d" % naux)
    print("dump h5py file = %-s" % dump_file)

    stop_clock("getting input for ppRPA with fractional occupation from PySCF")

    return mo_occ, mo_energy_act, Lpq


def get_pyscf_input_sc(
        kmf, nocc_act=None, nvir_act=None, dump_file=None, cholesky=False, with_dip=False):
    """Get ppRPA input from a PySCF supercell SCF calculation.

    Args:
        mf (pyscf.pbc.scf.RHF/pyscf.pbc.dft.RKS): supercell mean-field object.
        nocc_act (int, optional): number of active occupied orbitals. Defaults to None.
        nvir_act (int, optional): number of active virtual orbitals. Defaults to None.
        dump_file (str, optional): file name to dump matrix for lib_pprpa. Defaults to None.
        cholesky (bool, optional): use Cholesky decomposition. Defaults to False.
        with_dip (bool, optional): Calculate dipole moment in MO space. Defaults to False.

    Returns:
        nocc_act (int): number of occupied orbitals in the active space.
        mo_energy_act (double array): orbital energy in the active space.
        Lpq (double ndarray): three-center density-fitting matrix in active MO space.
        mo_dip (numpy.ndarray, upon request): Dipole moment in MO space.
            Only returned if with_dip is True.
    """
    from pyscf import lib
    from pyscf.ao2mo import _ao2mo
    from pyscf.pbc.df.fft_ao2mo import _format_kpts

    start_clock("getting input for supercell ppRPA from PySCF")

    nmo = len(kmf.mo_energy)
    nocc = int(numpy.sum(kmf.mo_occ) / 2)
    nvir = nmo - nocc
    mo_energy = numpy.array(kmf.mo_energy)
    mo_coeff = numpy.asarray(kmf.mo_coeff)
    nao = mo_coeff.shape[0]
    naux = kmf.with_df.get_naoaux()
    kpts = kmf.with_df.kpts
    max_memory = max(
        4000, kmf.max_memory-lib.current_memory()[0]-nao**2*naux*8/1e6)

    nocc_act = nocc if nocc_act is None else min(nocc, nocc_act)
    nvir_act = nvir if nvir_act is None else min(nvir, nvir_act)
    nmo_act = nocc_act + nvir_act
    mo_energy_act = numpy.array(mo_energy[(nocc - nocc_act) : (nocc + nvir_act)])

    mo = numpy.asarray(mo_coeff, order='F')
    ijslice = (nocc-nocc_act, nocc+nvir_act, nocc-nocc_act, nocc+nvir_act)

    kptijkl = _format_kpts(kpts)

    if with_dip:
        mo_dip = calculate_dipole_moment_MO(kmf, nocc_act=nocc_act, nvir_act=nvir_act)
    else:
        mo_dip = None

    if cholesky is False:
        Lpq = []
        for LpqR, _, _ in kmf.with_df.sr_loop(
                kptijkl[: 2], max_memory=0.8 * max_memory, compact=False):
            tmp = None
            tmp = _ao2mo.nr_e2(
                LpqR.reshape(-1, nao, nao), mo, ijslice, aosym='s1', mosym='s1',
                out=tmp)
            Lpq.append(tmp)
        Lpq = numpy.vstack(Lpq).reshape(naux, nmo_act, nmo_act)
        Lpq = numpy.ascontiguousarray(Lpq)
    else:
        mo_coeff_act = numpy.array(mo_coeff[:, (nocc - nocc_act) : (nocc + nvir_act)])
        full_eri = kmf.with_df.ao2mo(mo_coeff_act, compact=True)
        cd = Cholesky(full_eri, err_tol=1e-5, aosym='s4')
        Lpq = cd.kernel()
        Lpq = Lpq.reshape(-1, nmo_act, nmo_act)
        naux = Lpq.shape[0]

    if dump_file is not None:
        _write_dump_file(dump_file, nocc_act, mo_energy_act, Lpq, mo_dip=mo_dip)

    print("\nget input for lib_pprpa from PySCF (supercell)")
    print("nmo = %-d, nocc= %-d, nvir = %-d" % (nmo, nocc, nvir))
    print("nmo_act = %-d, nocc_act= %-d, nvir_act = %-d" %
          (nmo_act, nocc_act, nvir_act))
    print("naux = %-d" % naux)
    print("dump h5py file = %-s" % dump_file)

    stop_clock("getting input for supercell ppRPA from PySCF")

    if with_dip:
        return nocc_act, mo_energy_act, Lpq, mo_dip
    else:
        return nocc_act, mo_energy_act, Lpq


def get_pyscf_input_sc_g(kmf, nocc_act=None, nvir_act=None, dump_file=None):
    """Get ppRPA input from a PySCF supercell SCF calculation.

    Args:
        mf (pyscf.pbc.scf.GHF/pyscf.pbc.dft.GKS): supercell mean-field object.
        nocc_act (int, optional): number of active occupied orbitals. Defaults to None.
        nvir_act (int, optional): number of active virtual orbitals. Defaults to None.
        dump_file (str, optional): file name to dump matrix for lib_pprpa. Defaults to None.

    Returns:
        nocc_act (int): number of occupied orbitals in the active space.
        mo_energy_act (double array): orbital energy in the active space.
        Lpq (double/complex ndarray): three-center density-fitting matrix in active MO space.
    """
    from pyscf import lib
    from pyscf.ao2mo import _ao2mo
    from pyscf.pbc.df.fft_ao2mo import _format_kpts

    start_clock("getting input for supercell ppRPA from PySCF")

    nmo = len(kmf.mo_energy)
    nocc = int(numpy.sum(kmf.mo_occ))
    nvir = nmo - nocc
    mo_energy = numpy.array(kmf.mo_energy)
    mo_coeff = numpy.asarray(kmf.mo_coeff)
    nao = mo_coeff.shape[0]//2
    naux = kmf.with_df.get_naoaux()
    kpts = kmf.with_df.kpts
    max_memory = max(
        4000, kmf.max_memory-lib.current_memory()[0]-nao**2*naux*8/1e6)

    nocc_act = nocc if nocc_act is None else min(nocc, nocc_act)
    nvir_act = nvir if nvir_act is None else min(nvir, nvir_act)
    nmo_act = nocc_act + nvir_act
    mo_energy_act = mo_energy[(nocc-nocc_act):(nocc+nvir_act)]

    from pyscf.pbc import scf, dft
    if isinstance(kmf, scf.ghf.GHF) or isinstance(kmf, dft.gks.GKS):
        mo_coeff_a = kmf.mo_coeff[:nao, :]
        mo_coeff_b = kmf.mo_coeff[nao:, :]
    else:
        raise ValueError("kmf should be GHF/JHF/JKS/JKS.")

    ijslice = (nocc-nocc_act, nocc+nvir_act, nocc-nocc_act, nocc+nvir_act)

    kptijkl = _format_kpts(kpts)
    Lpq = []
    for LpqR, _, _ in kmf.with_df.sr_loop(
            kptijkl[: 2], max_memory=0.8 * max_memory, compact=False):
        LpqR = LpqR.reshape(-1, nao, nao)
        if mo_coeff_a.dtype == numpy.double:
            tmp = _ao2mo.nr_e2(LpqR, mo_coeff_a, ijslice, aosym='s1', mosym='s1')
            tmp += _ao2mo.nr_e2(LpqR, mo_coeff_b, ijslice, aosym='s1', mosym='s1')
        elif mo_coeff_a.dtype == numpy.complex128:
            tmp = numpy.zeros((LpqR.shape[0], nmo_act*nmo_act), dtype=numpy.complex128)
            tmp += nr_e2_cross(LpqR, mo_coeff_a.real, mo_coeff_a.real, nocc, nocc_act, nvir_act, aosym='s1', mosym='s1')
            tmp += nr_e2_cross(LpqR, mo_coeff_b.real, mo_coeff_b.real, nocc, nocc_act, nvir_act, aosym='s1', mosym='s1')
            tmp += nr_e2_cross(LpqR, mo_coeff_a.imag, mo_coeff_a.imag, nocc, nocc_act, nvir_act, aosym='s1', mosym='s1')
            tmp += nr_e2_cross(LpqR, mo_coeff_b.imag, mo_coeff_b.imag, nocc, nocc_act, nvir_act, aosym='s1', mosym='s1')
            tmp += 1.0j * nr_e2_cross(LpqR, mo_coeff_a.real, mo_coeff_a.imag, nocc, nocc_act, nvir_act, aosym='s1', mosym='s1')
            tmp += 1.0j * nr_e2_cross(LpqR, mo_coeff_b.real, mo_coeff_b.imag, nocc, nocc_act, nvir_act, aosym='s1', mosym='s1')
            tmp -= 1.0j * nr_e2_cross(LpqR, mo_coeff_a.imag, mo_coeff_a.real, nocc, nocc_act, nvir_act, aosym='s1', mosym='s1')
            tmp -= 1.0j * nr_e2_cross(LpqR, mo_coeff_b.imag, mo_coeff_b.real, nocc, nocc_act, nvir_act, aosym='s1', mosym='s1')
        Lpq.append(tmp)
    Lpq = numpy.vstack(Lpq).reshape(naux, nmo_act, nmo_act)

    if dump_file is not None:
        f = h5py.File(name="%s.h5" % dump_file, mode="w")
        f["nocc"] = numpy.asarray(nocc_act)
        f["mo_energy"] = numpy.asarray(mo_energy_act)
        f["Lpq"] = numpy.asarray(Lpq)
        f.close()

    print("\nget input for lib_pprpa from PySCF (supercell)")
    print("nmo = %-d, nocc= %-d, nvir = %-d" % (nmo, nocc, nvir))
    print("nmo_act = %-d, nocc_act= %-d, nvir_act = %-d" %
          (nmo_act, nocc_act, nvir_act))
    print("naux = %-d" % naux)
    print("dump h5py file = %-s" % dump_file)

    stop_clock("getting input for supercell ppRPA from PySCF")

    return nocc_act, mo_energy_act, Lpq

def calculate_dipole_moment_MO(mf, nocc_act=None, nvir_act=None):
    """Calculate the dipole moment in MO space
    Args:
        mf (PySCF mean-field object): PySCF mean-field object (Can be for molecule or PBC).
    Kwargs:
        nocc_act (int): active space size for occupied orbitals. Leave None for full space calculation
        nvir_act (int): active space size for virtual orbitals. Leave None for full space calculation

    Returns:
        mo_dip (numpy.ndarray): shape (nmo, 3)
    """
    nmo = len(mf.mo_energy)
    nocc = int(numpy.sum(mf.mo_occ) / 2)
    nvir = nmo - nocc
    nocc_act = nocc if nocc_act is None else min(nocc, nocc_act)
    nvir_act = nvir if nvir_act is None else min(nvir, nvir_act)
    mo_coeff = numpy.asarray(mf.mo_coeff)
    print("Computing dipole moment in MO space")
    from pyscf.tdscf.rhf import _charge_center
    # Same formulation as in the TDDFT module
    with mf.mol.with_common_orig(_charge_center(mf.mol)):
        ao_dip = mf.mol.intor_symmetric("int1e_r", comp=3)
    mo_dip = mo_coeff.T @ ao_dip @ mo_coeff
    mo_dip = mo_dip[:, nocc - nocc_act : nocc + nvir_act, nocc - nocc_act : nocc + nvir_act]
    return mo_dip

# natural transition orbital
def get_pprpa_nto_pyscf(mf, multi, state, xy, nocc, nvir):
    """Get natural transition orbital coefficient and weight with PySCF.

    Args:
        mf (PySCF mean-field object): PySCF mean-field object.
        multi (char): multiplicity.
        state (int): index of the desired state.
        xy (double ndarray): ppRPA eigenvector.
        nocc (int or int array): number of (active) occupied orbitals.
        nvir (int or int array): number of (active) virtual orbitals.

    Returns:
        Natural transition orbital coefficient and weight, see get_pprpa_nto().
    """
    nocc_full = mf.mol.nelectron // 2
    mo_coeff = numpy.asarray(mf.mo_coeff)
    return get_pprpa_nto(multi, state, xy, nocc, nvir, mo_coeff, nocc_full)


def get_pprpa_dm_pyscf(mf, multi, state, xy, nocc, nvir):
    """Get density matrix with PySCF.

    Args:
        mf (PySCF mean-field object): PySCF mean-field object.
        multi (char): multiplicity.
        state (int): index of the desired state.
        xy (double ndarray): ppRPA eigenvector.
        nocc (int or int array): number of (active) occupied orbitals.
        nvir (int or int array): number of (active) virtual orbitals.

    Returns:
        density matrix, see get_pprpa_nto().
    """
    nocc_full = mf.mol.nelectron // 2
    mo_coeff = numpy.asarray(mf.mo_coeff)
    return get_pprpa_dm(
        multi, state, xy, nocc, nvir, mo_coeff, nocc_full, full_return=False)


def create_frac_scf_object(mf, frac_spin, frac_orb, frac_occ):
    """Create a mean-field object with hacked get_occ function to allow
    SCF calculations with fractional occupation numbers.

    Args:
        mf (PySCF mean-field object): unrestricted PySCF mean-field object.
        frac_spin (int list): spin channel(s) for fractionally occupied
            orbitals. Options: [0] or [1] or [0, 1].
        frac_orb (int list): orbital indices with fractional occupation
            numbers.
        frac_occ (double list): list of fractional occupation numbers.

    Return:
        frac_mf: hacked mean-field object with modified get_occ function.
    """
    import pyscf

    frac_mf = mf
    frac_mf.frac_spin_added = frac_spin
    frac_mf.frac_orb_added = frac_orb
    frac_mf.frac_occ_added = frac_occ

    def get_occ(mf, mo_energy=None, mo_coeff=None):
        "hacked get_occ function for PySCF"
        if mo_energy is None: mo_energy = mf.mo_energy
        if mo_coeff is None: mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ

        for ispin in mf.frac_spin_added:
            if len(mf.frac_spin_added) == 2:
                for idx, iorb in enumerate(mf.frac_orb_added):
                    mo_occ[ispin][iorb] = mf.frac_occ_added[ispin][idx]
            else:
                for idx, iorb in enumerate(mf.frac_orb_added):
                    mo_occ[ispin][iorb] = mf.frac_occ_added[idx]
        return mo_occ

    if isinstance(mf, (pyscf.scf.uhf.UHF, pyscf.dft.uks.UKS)):
        pyscf.scf.uhf.UHF.get_occ = get_occ

    return frac_mf


def get_pyscf_input_mol_eri_r(
        mf, nocc_act=None, nvir_act=None, dump_file=None,
        sort_mo=False, incore=True, return_raw=False):
    """Get ppRPA use eri input from a PySCF molecular SCF calculation.

    Args:
        mf (pyscf.scf.RHF/pyscf.dft.RKS): molecular mean-field object.
        nocc_act (int, optional): number of active occupied orbitals. Defaults to None.
        nvir_act (int, optional): number of active virtual orbitals. Defaults to None.
        dump_file (str, optional): file name to dump matrix for lib_pprpa. Defaults to None.
        return_raw (bool, optional): whether to return raw four-center integrals.

    Returns:
        nocc_act (int): number of occupied orbitals in the active space.
        mo_energy_act (double array): orbital energy in the active space.
        vvvv (double ndarray): four-center two-electron integrals
        oooo (double ndarray): four-center two-electron integrals
        oovv (double ndarray): four-center two-electron integrals
    """
    from pyscf import df
    from pyscf.ao2mo import _ao2mo

    start_clock("getting input for molecule ppRPA from PySCF")

    nmo = len(mf.mo_energy)
    nocc = mf.mol.nelectron // 2
    nvir = nmo - nocc
    mo_energy = numpy.array(mf.mo_energy)
    mo_coeff = numpy.array(mf.mo_coeff)

    if sort_mo is True:
        occ_index = numpy.where(mf.mo_occ > 0.5)[0]
        vir_index = numpy.where(mf.mo_occ < 0.5)[0]
        print("sorting molecular orbitals")
        print("occ index = ", occ_index)
        print("vir index = ", vir_index)
        if occ_index[-1] < vir_index[0]:
            print("warning: no sorting is performed!")
        mo_energy_occ = mo_energy[occ_index]
        mo_energy_vir = mo_energy[vir_index]
        mo_energy = numpy.concatenate((mo_energy_occ, mo_energy_vir))
        mo_coeff_occ = mo_coeff[:, occ_index]
        mo_coeff_vir = mo_coeff[:, vir_index]
        mo_coeff = numpy.concatenate((mo_coeff_occ, mo_coeff_vir), axis=1)

    nocc_act = nocc if nocc_act is None else min(nocc, nocc_act)
    nvir_act = nvir if nvir_act is None else min(nvir, nvir_act)
    nmo_act = nocc_act + nvir_act
    mo_energy_act = mo_energy[(nocc-nocc_act):(nocc+nvir_act)]

    from pyscf import ao2mo
    if incore:
        eri_raw = ao2mo.incore.full(mf._eri, mo_coeff[:,(nocc-nocc_act):(nocc+nvir_act)])
        if dump_file is not None:
            f = h5py.File(name="%s.h5" % dump_file, mode="w")
            f["nocc"] = numpy.asarray(nocc_act)
            f["mo_energy"] = numpy.asarray(mo_energy_act)
            f["eri_raw"] = numpy.asarray(eri_raw)
            f.close()
        # (pq|rs)
        eri_raw = ao2mo.restore(1, eri_raw, nmo_act)
        # <pq|rs>
        eri_raw = eri_raw.transpose(0, 2, 1, 3)
        oooo = eri_raw[:nocc_act, :nocc_act, :nocc_act, :nocc_act]
        vvvv = eri_raw[nocc_act:, nocc_act:, nocc_act:, nocc_act:]
        oovv = eri_raw[:nocc_act, :nocc_act, nocc_act:, nocc_act:]
    else:
        raise NotImplementedError("only incore=True is implemented.")
    

    print("\nget input for lib_pprpa from PySCF (molecule)")
    print("nmo = %-d, nocc= %-d, nvir = %-d" % (nmo, nocc, nvir))
    print("nmo_act = %-d, nocc_act= %-d, nvir_act = %-d" %
          (nmo_act, nocc_act, nvir_act))
    print("dump h5py file = %-s" % dump_file)

    stop_clock("getting input for molecule ppRPA from PySCF")

    if not return_raw:
        return nocc_act, mo_energy_act, vvvv, oooo, oovv
    else:
        return nocc_act, mo_energy_act, eri_raw


def _write_dump_file(filename, nocc, mo_energy, Lpq, mo_dip=None):
    """Write ppRPA inputs to a file.
    see get_pyscf_input_mol_r for details about the inputs.
    """
    filename += ".h5" if not filename.endswith(".h5") else ""
    with h5py.File(name=filename, mode="w") as f:
        f["nocc"] = numpy.asarray(nocc)
        f["mo_energy"] = numpy.asarray(mo_energy)
        f["Lpq"] = numpy.asarray(Lpq)
        if mo_dip is not None:
            f["mo_dipole"] = numpy.asarray(mo_dip)
    print("Saved h5py file = %-s" % filename)

def read_dump_file(filename: str, with_dip: bool = False):
    """Read ppRPA inputs from a file.
    Args:
        filename (str): name of the file to dump matrices for lib_pprpa.
        with_dip (bool, optional): whether to read dipole integrals.
            Defaults to False.
    returns:
        nocc (numpy.ndarray): number of occupied orbitals in the active space.
        mo_energy (numpy.ndarray): orbital energies in the active space.
        Lpq (numpy.ndarray): three-center density fitting matrix
            in the active MO space.
        mo_dip (numpy.ndarray, optional): Dipole integrals in MO space.
            Only returned if with_dip is True.
    """
    filename += ".h5" if not filename.endswith(".h5") else ""
    with h5py.File(name=filename, mode="r") as f:
        nocc = numpy.asarray(f["nocc"])
        mo_energy = numpy.asarray(f["mo_energy"])
        Lpq = numpy.asarray(f["Lpq"])
        if "mo_dipole" in f.keys() and with_dip:
            mo_dip = numpy.asarray(f["mo_dipole"])
            return nocc, mo_energy, Lpq, mo_dip
        elif with_dip:
            raise KeyError("The requested dipole integrals were not in the file.")
        return nocc, mo_energy, Lpq
