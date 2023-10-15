import numpy


def get_pyscf_input_mol(mf, auxbasis=None, nocc_act=None, nvir_act=None):
    """Get ppRPA input from a PySCF molecular SCF calculation.

    Args:
        mf (pyscf.scf.RHF/pyscf.dft.RKS): molecular mean-field object.
        auxbasis (string, optional): auxiliary basis set. Defaults to None.
        nocc_act (int, optional): number of active occupied orbitals. Defaults to None.
        nvir_act (int, optional): number of active virtual orbitals. Defaults to None.

    Returns:
        nocc_act (int): number of occupied orbitals in the active space.
        mo_energy_act (double array): orbital energy in the active space.
        Lpq (double ndarray): three-center density-fitting matrix in active MO space.
    """
    from pyscf import df
    from pyscf.ao2mo import _ao2mo

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

    print("\nget input for lib_pprpa from PySCF (molecule)")
    print("nmo = %-d, nocc= %-d, nvir = %-d" % (nmo, nocc, nvir))
    print("nmo_act = %-d, nocc_act= %-d, nvir_act = %-d" % (nmo_act, nocc_act, nvir_act))
    print("naux = %-d" % naux)

    return nocc_act, mo_energy_act, Lpq


def get_pyscf_input_sc(kmf, nocc_act=None, nvir_act=None):
    """Get ppRPA input from a PySCF supercell SCF calculation.

    Args:
        mf (pyscf.pbc.scf.RHF/pyscf.pbc.dft.RKS): supercell mean-field object.
        nocc_act (int, optional): number of active occupied orbitals. Defaults to None.
        nvir_act (int, optional): number of active virtual orbitals. Defaults to None.

    Returns:
        nocc_act (int): number of occupied orbitals in the active space.
        mo_energy_act (double array): orbital energy in the active space.
        Lpq (double ndarray): three-center density-fitting matrix in active MO space.
    """
    from pyscf import lib
    from pyscf.ao2mo import _ao2mo
    from pyscf.pbc.df.fft_ao2mo import _format_kpts

    nmo = len(kmf.mo_energy)
    nocc = int(numpy.sum(kmf.mo_occ) / 2)
    nvir = nmo - nocc
    mo_energy = numpy.array(kmf.mo_energy)
    mo_coeff = numpy.asarray(kmf.mo_coeff)
    nao = mo_coeff.shape[0]
    naux = kmf.with_df.get_naoaux()
    kpts = kmf.with_df.kpts
    max_memory = max(4000, kmf.max_memory-lib.current_memory()[0]-nao**2*naux*8/1e6)

    nocc_act = nocc if nocc_act is None else min(nocc, nocc_act)
    nvir_act = nvir if nvir_act is None else min(nvir, nvir_act)
    nmo_act = nocc_act + nvir_act
    mo_energy_act = mo_energy[(nocc-nocc_act):(nocc+nvir_act)]

    mo = numpy.asarray(mo_coeff, order='F')
    ijslice = (nocc-nocc_act, nocc+nvir_act, nocc-nocc_act, nocc+nvir_act)

    kptijkl = _format_kpts(kpts)
    Lpq = []
    for LpqR, _, _ in kmf.with_df.sr_loop(kptijkl[:2], max_memory=0.8*max_memory, compact=False):
        tmp = None
        tmp = _ao2mo.nr_e2(LpqR.reshape(-1, nao, nao), mo, ijslice, aosym='s1', mosym='s1', out=tmp)
        Lpq.append(tmp)
    Lpq = numpy.vstack(Lpq).reshape(naux, nmo_act, nmo_act)

    print("\nget input for lib_pprpa from PySCF (supercell)")
    print("nmo = %-d, nocc= %-d, nvir = %-d" % (nmo, nocc, nvir))
    print("nmo_act = %-d, nocc_act= %-d, nvir_act = %-d" % (nmo_act, nocc_act, nvir_act))
    print("naux = %-d" % naux)

    return nocc_act, mo_energy_act, Lpq
