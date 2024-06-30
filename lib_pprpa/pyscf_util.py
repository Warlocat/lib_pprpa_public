import h5py
import numpy

from lib_pprpa.nto import get_pprpa_nto, get_pprpa_dm
from lib_pprpa.pprpa_util import start_clock, stop_clock


# get input from PySCF
def get_pyscf_input_mol(
        mf, auxbasis=None, nocc_act=None, nvir_act=None, dump_file=None):
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
    mo = numpy.asarray(mf.mo_coeff, order='F')
    ijslice = (nocc-nocc_act, nocc+nvir_act, nocc-nocc_act, nocc+nvir_act)
    Lpq = None
    Lpq = _ao2mo.nr_e2(mf.with_df._cderi, mo, ijslice, aosym='s2', out=Lpq)
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
        mf, auxbasis=None, nocc_act=None, nvir_act=None, dump_file=None):
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
    mo_energy = numpy.asarray(mf.mo_energy)
    
    if nocc_act is None:
        nocc_act = nocc
    else:
        nocc_act = (min(nocc[0], nocc_act[0]), min(nocc[1], nocc_act[1]))
    if nvir_act is None:
        nvir_act = nvir
    else:
        nvir_act = (min(nvir[0], nvir_act[0]), min(nvir[1], nvir_act[1]))
    nmo_act = (nocc_act[0] + nvir_act[0], nocc_act[1] + nvir_act[1])
    mo_energy_act =[
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
    mo = mf.mo_coeff
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
    Lpq_a = _ao2mo.nr_e2(mf.with_df._cderi, mo[0], ijslice[0], 
                         aosym='s2', out=Lpq_a)
    Lpq_a = Lpq_a.reshape(naux, nmo_act[0], nmo_act[0])
    Lpq_b = _ao2mo.nr_e2(mf.with_df._cderi, mo[1], ijslice[1], 
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
