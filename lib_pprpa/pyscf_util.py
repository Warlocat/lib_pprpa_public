import h5py
import numpy

from lib_pprpa.analyze import get_pprpa_nto, get_pprpa_dm
from lib_pprpa.pprpa_util import start_clock, stop_clock, get_nocc_nvir_frac


# get input from PySCF
def get_pyscf_input_mol(
        mf, auxbasis=None, nocc_act=None, nvir_act=None, dump_file=None,
        sort_mo=False):
    import pyscf

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
            dump_file=dump_file, sort_mo=sort_mo)


def get_pyscf_input_mol_r(
        mf, auxbasis=None, nocc_act=None, nvir_act=None, dump_file=None,
        sort_mo=False):
    """Get ppRPA input from a PySCF molecular SCF calculation.

    Args:
        mf (pyscf.scf.RHF/pyscf.dft.RKS): molecular mean-field object.
        auxbasis (string, optional): auxiliary basis set. Defaults to None.
        nocc_act (int, optional): number of active occupied orbitals. Defaults to None.
        nvir_act (int, optional): number of active virtual orbitals. Defaults to None.
        dump_file (str, optional): file name to dump matrix for lib_pprpa. Defaults to None.

    Returns:
        nocc_act (int): number of occupied orbitals in the active space.
        mo_energy_act (double array): orbital energy in the active space.
        Lpq (double ndarray): three-center density-fitting matrix in active MO space.
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
    ijslice = (nocc-nocc_act, nocc+nvir_act, nocc-nocc_act, nocc+nvir_act)
    Lpq = None
    Lpq = _ao2mo.nr_e2(mf.with_df._cderi, mo_coeff, ijslice, aosym='s2', out=Lpq)
    Lpq = Lpq.reshape(naux, nmo_act, nmo_act)

    if dump_file is not None:
        f = h5py.File(name="%s.h5" % dump_file, mode="w")
        f["nocc"] = numpy.asarray(nocc_act)
        f["mo_energy"] = numpy.asarray(mo_energy_act)
        f["Lpq"] = numpy.asarray(Lpq)
        f.close()

    print("\nget input for lib_pprpa from PySCF (molecule)")
    print("nmo = %-d, nocc= %-d, nvir = %-d" % (nmo, nocc, nvir))
    print("nmo_act = %-d, nocc_act= %-d, nvir_act = %-d" %
          (nmo_act, nocc_act, nvir_act))
    print("naux = %-d" % naux)
    print("dump h5py file = %-s" % dump_file)

    stop_clock("getting input for molecule ppRPA from PySCF")

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
        f = h5py.File(name="%s.h5" % dump_file, mode="w")
        f["nocc"] = numpy.asarray(nocc_act)
        f["mo_energy"] = numpy.asarray(mo_energy_act)
        f["Lpq"] = numpy.asarray(Lpq)
        f.close()

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
        f = h5py.File(name="%s.h5" % dump_file, mode="w")
        f["nocc"] = numpy.asarray(nocc_act)
        f["mo_energy"] = numpy.asarray(mo_energy_act)
        f["Lpq"] = numpy.asarray(Lpq)
        f.close()

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


def get_pyscf_input_sc(kmf, nocc_act=None, nvir_act=None, dump_file=None):
    """Get ppRPA input from a PySCF supercell SCF calculation.

    Args:
        mf (pyscf.pbc.scf.RHF/pyscf.pbc.dft.RKS): supercell mean-field object.
        nocc_act (int, optional): number of active occupied orbitals. Defaults to None.
        nvir_act (int, optional): number of active virtual orbitals. Defaults to None.
        dump_file (str, optional): file name to dump matrix for lib_pprpa. Defaults to None.

    Returns:
        nocc_act (int): number of occupied orbitals in the active space.
        mo_energy_act (double array): orbital energy in the active space.
        Lpq (double ndarray): three-center density-fitting matrix in active MO space.
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
    mo_energy_act = mo_energy[(nocc-nocc_act):(nocc+nvir_act)]

    mo = numpy.asarray(mo_coeff, order='F')
    ijslice = (nocc-nocc_act, nocc+nvir_act, nocc-nocc_act, nocc+nvir_act)

    kptijkl = _format_kpts(kpts)
    Lpq = []
    for LpqR, _, _ in kmf.with_df.sr_loop(
            kptijkl[: 2], max_memory=0.8 * max_memory, compact=False):
        tmp = None
        tmp = _ao2mo.nr_e2(
            LpqR.reshape(-1, nao, nao), mo, ijslice, aosym='s1', mosym='s1',
            out=tmp)
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

    if isinstance(mf, pyscf.scf.uhf.UHF):
        pyscf.scf.uhf.UHF.get_occ = get_occ

    return frac_mf


def get_fxc_r(mf):
    import pyscf
    dm0 = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    ni = mf._numint
    make_rho = ni._gen_rho_evaluator(mf.mol, dm0, hermi=1, with_lapl=False)[0]
    xctype = ni._xc_type(mf.xc)
    mem_now = pyscf.lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory * 0.8 - mem_now)

    mo_coeff = mf.mo_coeff
    nao = mo_coeff.shape[0]
    nmo = mo_coeff.shape[1]

    fxc_mat = numpy.zeros((nmo, nmo, nmo, nmo))
    if xctype == 'LDA':
        ao_deriv = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mf.mol, mf.grids, nao, ao_deriv, max_memory):
            rho = make_rho(0, ao, mask, xctype)
            fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
            wfxc = fxc[0, 0] * weight

            mo_value = numpy.einsum('rm,mp->rp', ao, mo_coeff, optimize=True)
            rho_value = numpy.einsum(
                'rp,rq->rpq', mo_value, mo_value, optimize=True)
            w_rho = numpy.einsum('rpq,r->rpq', rho_value, wfxc, optimize=True)
            w = numpy.einsum(
                'rpq,rmn->pqmn', rho_value, w_rho, optimize=True) * 2
            fxc_mat += w.transpose(0, 3, 1, 2)

    elif xctype == 'GGA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mf.mol, mf.grids, nao, ao_deriv, max_memory):
            rho = make_rho(0, ao, mask, xctype)
            fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
            wfxc = fxc * weight
            mo_value = numpy.einsum('xrm,mp->xrp', ao, mo_coeff, optimize=True)
            rho_value = numpy.einsum(
                'xrp,rq->xrpq', mo_value, mo_value[0], optimize=True)
            rho_value[1:4] += numpy.einsum(
                'rp,xrq->xrpq', mo_value[0], mo_value[1:4], optimize=True)
            w_rho = numpy.einsum(
                'xyr,xrpq->yrpq', wfxc, rho_value, optimize=True)
            w = numpy.einsum('xrpq,xrmn->pqmn', w_rho, rho_value, optimize=True)
            fxc_mat += w.transpose(0, 3, 1, 2)

    elif xctype == 'MGGA':
        raise NotImplementedError

    return fxc_mat
