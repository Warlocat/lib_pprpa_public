import numpy as np
import os, gc, h5py
from functools import reduce
from socutils.prop.gfac.spinor_hf import int_gfac_2c
from lib_pprpa.grad.grad_utils import make_rdm1_relaxed, make_tdm1
from pyscf.data.nist import LIGHT_SPEED
from pyscf.lib import einsum
# try:
#     from pyscf import tblis_einsum
#     tblis = True
# except ImportError:
#     tblis = False
# if tblis:
#     einsum = tblis_einsum.contract
#     print("Using tblis for einsum.")

def pseudo_spin_state(xy, oo_dim, gfac_int_mo, gfac_mf = None, two_pseudo_spin = 2):
    """
    Returns the pseudo spin eigenstates.
    """
    if xy.shape[0] != two_pseudo_spin + 1:
        print("Warning: The pseudo spin dimension is not consistent with the input.")
        print("         This could happen for pprpa_direct objects or for wrong input.")
        print("         Use the first 2S+1 solutions instead.")
        xy = xy[:two_pseudo_spin + 1]
    nroots = two_pseudo_spin + 1
    nact = gfac_int_mo.shape[1]

    if gfac_mf is None:
        gfac_mf = np.zeros(3)

    # construct the g-factor matrix in cartesian basis
    gfac = np.zeros((3, nroots, nroots), dtype=xy.dtype)
    for i in range(nroots):
        gfac[:, i, i] = gfac_mf
        den = make_rdm1(xy[i], oo_dim)
        assert len(den) == nact
        for j in range(i+1):
            tdm, diag = make_tdm1(xy[i], xy[j], oo_dim)
            gfac[:, i, j] = einsum("ij,kji->k",tdm,gfac_int_mo)
            gfac[:, j, i] = gfac[:, i, j].conj()
        print(gfac[:, i, i].real)
            
    # construct the G matrix
    st = two_pseudo_spin / 2.0
    # J. Chem. Theory Comput. 2019, 15, 1560    S.I. Eqn. (12)-(14)
    Gmat = np.einsum("imn,jnm->ij", gfac, gfac) * 3.0 / st / (st + 1.0) / (2.0 * st + 1.0)

    print("g-tensor matrix:")
    print(Gmat.real)

    # diagonalize the G matrix
    e_G, c_G = np.linalg.eigh(Gmat)
    print("g-tensor eigenvalues:")
    print(np.sqrt(e_G))
    print("magnetic axis")
    print(c_G)

    # g-factor matrix in the pseudo spin basis
    gfac = np.einsum("imn,ij->jmn", gfac, c_G)
    # choose "z" as the pseudo spin quantization axis and construct the pseudo spin states
    e_mu, c_mu = np.linalg.eigh(gfac[2, :, :])
    print(r"pseudo spin eigenvalues: (G_e \tilde{S}_z)")
    print(e_mu)

    # phase factor
    for i in range(nroots-1):
        g_fac_x = reduce(np.dot, (c_mu[:, i].conj(), gfac[0, :, :], c_mu[:, i+1]))
        phase = np.sqrt(g_fac_x.real**2 + g_fac_x.imag**2) / g_fac_x
        c_mu[:, i+1] *= phase
    print("pseudo spin eigenvectors:")
    print(c_mu)

    return c_mu

def spin_spin_pt_incore(xy, oo_dim, mo_coeff, mol, den_hf = None):
    assert oo_dim == 0 or xy[0].shape[0] == oo_dim, "The pt Hamiltonian is only for TDA pp-RPA."
    nao = mol.nao_nr()
    nroots = len(xy)

    pp_ints = mol.intor("int2e_ip1ip2", comp=9).reshape(3,3,nao,nao,nao,nao)
    pp_ints = pp_ints + pp_ints.transpose(0,1,3,2,5,4) + pp_ints.transpose(0,1,2,3,5,4) + pp_ints.transpose(0,1,3,2,4,5)
    trace = pp_ints[0,0] + pp_ints[1,1] + pp_ints[2,2]
    pp_ints[0,0] -= trace / 3 * 2
    pp_ints[1,1] -= trace / 3 * 2
    pp_ints[2,2] -= trace / 3 * 2
    pp_ints = pp_ints / 4.0 / LIGHT_SPEED**2

    # fc_ints = mol.intor("int4c1e", comp=1).reshape(nao,nao,nao,nao) # int4c1e not recognized by PySCF
    ss_ints = np.zeros((2*nao,2*nao,2*nao,2*nao), dtype=np.complex128)
    ss_ints[:nao,:nao,:nao,:nao] = pp_ints[2,2]
    ss_ints[nao:,nao:,nao:,nao:] = ss_ints[:nao,:nao,:nao,:nao]
    ss_ints[:nao,:nao,nao:,nao:] = -ss_ints[:nao,:nao,:nao,:nao]
    ss_ints[nao:,nao:,:nao,:nao] = -ss_ints[:nao,:nao,:nao,:nao]
    ss_ints[:nao,nao:,:nao,:nao] = (pp_ints[0,2] - 1.0j*pp_ints[1,2])
    ss_ints[nao:,:nao,:nao,:nao] = ss_ints[:nao,nao:,:nao,:nao].conj()
    ss_ints[:nao,nao:,nao:,nao:] = -ss_ints[:nao,nao:,:nao,:nao]
    ss_ints[nao:,:nao,nao:,nao:] = -ss_ints[:nao,nao:,:nao,:nao].conj()
    ss_ints[:nao,:nao,:nao,nao:] = (pp_ints[2,0] - 1.0j*pp_ints[2,1])
    ss_ints[:nao,:nao,nao:,:nao] = ss_ints[:nao,:nao,:nao,nao:].conj()
    ss_ints[nao:,nao:,:nao,nao:] = -ss_ints[:nao,:nao,:nao,nao:]
    ss_ints[nao:,nao:,nao:,:nao] = -ss_ints[:nao,:nao,:nao,nao:].conj()
    ss_ints[:nao,nao:,:nao,nao:] = (pp_ints[0,0] - pp_ints[1,1] - 1j*pp_ints[0,1] - 1j*pp_ints[1,0])
    ss_ints[nao:,:nao,nao:,:nao] = ss_ints[:nao,nao:,:nao,nao:].conj()
    ss_ints[:nao,nao:,nao:,:nao] = (pp_ints[0,0] + pp_ints[1,1] + 1j*pp_ints[0,1] - 1j*pp_ints[1,0])
    ss_ints[nao:,:nao,:nao,nao:] = ss_ints[:nao,nao:,nao:,:nao].conj()

    if den_hf is not None:
        ss_mf = np.einsum("msnr,sm->nr", ss_ints, den_hf) - np.einsum("mrns,sm->nr", ss_ints, den_hf)

    from lib_pprpa.gradient import get_xy_full
    ss_mat = np.zeros((nroots, nroots), dtype=np.complex128)
    for i in range(nroots):
        for j in range(i+1):
            y_i, x_i = get_xy_full(xy[i], oo_dim)
            y_j, x_j = get_xy_full(xy[j], oo_dim)    
            if oo_dim == 0:    
                x_i_ao = reduce(np.dot, (mo_coeff, x_i, mo_coeff.T))
                x_j_ao = reduce(np.dot, (mo_coeff, x_j, mo_coeff.T))
            else:
                x_i_ao = reduce(np.dot, (mo_coeff, y_i.conj(), mo_coeff.T))
                x_j_ao = reduce(np.dot, (mo_coeff, y_j.conj(), mo_coeff.T))
            tmp = einsum("ijkl,ik->jl", ss_ints, x_i_ao.conj())
            ss_mat[i, j] = 0.5*einsum("jl,jl", tmp, x_j_ao)
            if den_hf is not None:
                tdm_ao = np.einsum("nb,rd,ab,ad->nr", mo_coeff.conj(), mo_coeff, x_i.conj(), x_j, optimize=True)
                ss_mat[i, j] += np.einsum("nr,nr->", tdm_ao, ss_mf)
            ss_mat[j, i] = ss_mat[i, j].conj()
    if oo_dim != 0:
        ss_mat = -ss_mat.conj()
    return ss_mat

def spin_spin_pt_direct(xy, oo_dim, mo_coeff, mol):
    assert oo_dim == 0 or xy[0].shape[0] == oo_dim, "The pt Hamiltonian is only for TDA pp-RPA."
    from lib_pprpa.gradient import get_xy_full
    nao = mol.nao_nr()
    nroots = len(xy)

    fac = 1.0 / 8.0 / LIGHT_SPEED**2
    ss_mat = np.zeros((nroots, nroots), dtype=np.complex128)

    from pyscf.scf import jk
    total_tasks = nroots*(nroots+1)//2
    nfinished = 0
    for j in range(nroots):
        print("Calculating spin-spin interaction matrix: %d/%d" % (nfinished, total_tasks), flush=True, end="\r")
        y_j, x_j = get_xy_full(xy[j], oo_dim)
        if oo_dim == 0:
                x_j = reduce(np.dot, (mo_coeff, x_j, mo_coeff.T))
        else:
                x_j = reduce(np.dot, (mo_coeff, y_j.conj(), mo_coeff.T))
        x_j_aa = x_j[:nao,:nao]
        x_j_bb = x_j[nao:,nao:]
        x_j_ab = x_j[:nao,nao:] + x_j[nao:,:nao]
        faa_0, faa_1, faa_2, faa_3, fbb_0, fbb_1, fbb_2, fbb_3, fab_0, fab_1, fab_2, fab_3 \
        = jk.get_jk(mol, 
            [x_j_aa, x_j_aa, x_j_aa, x_j_aa, x_j_bb, x_j_bb, x_j_bb, x_j_bb, x_j_ab, x_j_ab, x_j_ab, x_j_ab],
            scripts=['msnr,sr->mn', 'smnr,sr->mn', 'msrn,sr->mn', 'smrn,sr->mn', 
                     'msnr,sr->mn', 'smnr,sr->mn', 'msrn,sr->mn', 'smrn,sr->mn', 
                     'msnr,sr->mn', 'smnr,sr->mn', 'msrn,sr->mn', 'smrn,sr->mn'],
            intor='cint2e_ip1ip2_sph')
        faa = faa_0 + faa_1 + faa_2 + faa_3
        fbb = fbb_0 + fbb_1 + fbb_2 + fbb_3
        fab = fab_0 + fab_1 + fab_2 + fab_3
        faa = faa.reshape(3,3,nao,nao)
        fbb = fbb.reshape(3,3,nao,nao)
        fab = fab.reshape(3,3,nao,nao)
        trace_aa = faa[0,0] + faa[1,1] + faa[2,2]
        trace_bb = fbb[0,0] + fbb[1,1] + fbb[2,2]
        trace_ab = fab[0,0] + fab[1,1] + fab[2,2]
        faa[0,0] -= trace_aa / 3
        faa[1,1] -= trace_aa / 3
        faa[2,2] -= trace_aa / 3
        fbb[0,0] -= trace_bb / 3
        fbb[1,1] -= trace_bb / 3
        fbb[2,2] -= trace_bb / 3
        fab[0,0] -= trace_ab / 3
        fab[1,1] -= trace_ab / 3
        fab[2,2] -= trace_ab / 3

        for i in range(j+1):
                y_i, x_i = get_xy_full(xy[i], oo_dim)
                if oo_dim == 0:
                    x_i = reduce(np.dot, (mo_coeff, x_i, mo_coeff.T))
                else:
                    x_i = reduce(np.dot, (mo_coeff, y_i.conj(), mo_coeff.T))
                x_i_aa = x_i[:nao,:nao]
                x_i_bb = x_i[nao:,nao:]
                x_i_ab = x_i[:nao,nao:] + x_i[nao:,:nao]
                ss_mat[i, j] = einsum("mn,mn->", faa[2,2], x_i_aa.conj())
                ss_mat[i, j]+= einsum("mn,mn->", fbb[2,2], x_i_bb.conj())
                ss_mat[i, j]-= einsum("mn,mn->", fab[2,2], x_i_ab.conj())
                ss_mat[i, j]+= einsum("mn,mn->", faa[0,2] + 1.0j*faa[1,2], x_i_ab.conj())
                ss_mat[i, j]-= einsum("mn,mn->", fab[0,2] + 1.0j*fab[1,2], x_i_bb.conj())
                ss_mat[i, j]+= einsum("mn,mn->", fab[0,2] - 1.0j*fab[1,2], x_i_aa.conj())
                ss_mat[i, j]-= einsum("mn,mn->", fbb[0,2] - 1.0j*fbb[1,2], x_i_ab.conj())
                ss_mat[i, j]+= einsum("mn,mn->", fbb[0,0] - fbb[1,1] - 2j*fbb[0,1], x_i_aa.conj())
                ss_mat[i, j]+= einsum("mn,mn->", faa[0,0] - faa[1,1] + 2j*faa[0,1], x_i_bb.conj())
                ss_mat[i, j] *= fac
                ss_mat[j, i] = ss_mat[i, j].conj()

                nfinished += 1
    print()
    if oo_dim != 0:
        ss_mat = -ss_mat.conj()
    return ss_mat

def spin_spin_pt_ri(xy, oo_dim, mo_coeff, mol, auxbasis = None, filename = None, ao_integrals = None):
    assert oo_dim == 0 or xy[0].shape[0] == oo_dim, "The pt Hamiltonian is only for TDA pp-RPA."
    from lib_pprpa.gradient import get_xy_full
    nao = mol.nao_nr()
    nroots = len(xy)

    fac = 1.0 / 8.0 / LIGHT_SPEED**2
    ss_mat = np.zeros((nroots, nroots), dtype=np.complex128)

    if ao_integrals is None:
        ao_integrals = True if mo_coeff.shape[0]//2 < mo_coeff.shape[1] else False
    if ao_integrals:
        print("Calculating spin-spin interaction matrix in AO basis.")
    else:
        print("Calculating spin-spin interaction matrix in MO basis.")

    # Calculate or read RI integrals in given basis.
    if filename is not None and os.path.exists(filename):
        if ao_integrals:
            with h5py.File(filename, "r") as f:
                int3c = f["int3c"][()]
                int3c_ipip = f["int3c_ipip"][()]
            naux = int3c.shape[-1]
            assert int3c.shape[0] == mo_coeff.shape[0]//2, "The saved RI integrals have wrong dimension."
        else:
            with h5py.File(filename, "r") as f:
                int3c_ip = f["int3c_ip"][()]
                int3c_ip2 = f["int3c_ip2"][()]
            naux = int3c_ip.shape[-1]
            assert int3c_ip.shape[1] == mo_coeff.shape[1], "The saved RI integrals have wrong dimension."
    else:
        from pyscf import df
        auxmol = df.make_auxmol(mol, auxbasis=auxbasis)
        naux = auxmol.nao_nr()
        int3c = df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1').transpose(2,0,1).reshape(naux,nao*nao)
        int2c = auxmol.intor('int2c2e', aosym='s1', comp=1)
        int3c = np.linalg.solve(int2c.T, int3c).T.reshape(nao,nao,naux) # int3c = int3c2e * int2c2e^-1 in AO basis
        int2c_ipip = auxmol.intor('int2c2e_ip1ip2', aosym='s1', comp=9).reshape(3,3,naux,naux)
        trace = int2c_ipip[0,0] + int2c_ipip[1,1] + int2c_ipip[2,2]
        int2c_ipip[0,0] -= trace / 3
        int2c_ipip[1,1] -= trace / 3
        int2c_ipip[2,2] -= trace / 3
        if ao_integrals:
            int3c_ipip = einsum("ijp,xypq->xyijq", int3c, int2c_ipip)
            del int2c_ipip, int2c, trace
            gc.collect()
            if filename is not None:
                with h5py.File(filename, "w") as f:
                    f.create_dataset("int3c", data=int3c, shape=int3c.shape)
                    f.create_dataset("int3c_ipip", data=int3c_ipip, shape=int3c_ipip.shape)
        else:
            nmo = mo_coeff.shape[1]
            ca = mo_coeff[:nao,:]
            cb = mo_coeff[nao:,:]
            int3c_ip = np.zeros((3,nmo,nmo,naux), dtype=np.complex128)
            tmp = einsum("mi,mnp->inp", ca.conj(), int3c)
            int3c_ip[0] = einsum("nj,inp->ijp", cb, tmp)
            int3c_ip[1] = einsum("nj,inp->ijp", cb, tmp)*(-1.0j)
            int3c_ip[2] = einsum("nj,inp->ijp", ca, tmp)
            tmp = einsum("mi,mnp->inp", cb.conj(), int3c)
            int3c_ip[0] += einsum("nj,inp->ijp", ca, tmp)
            int3c_ip[1] += einsum("nj,inp->ijp", ca, tmp)*1.0j
            int3c_ip[2] -= einsum("nj,inp->ijp", cb, tmp)
            del int3c, ca, cb, tmp
            gc.collect()
            int3c_ip2 = einsum("xijp,xypq->yijq", int3c_ip, int2c_ipip)
            del int2c_ipip, int2c, trace
            gc.collect()
            if filename is not None:
                with h5py.File(filename, "w") as f:
                    f.create_dataset("int3c_ip", data=int3c_ip, shape=int3c_ip.shape)
                    f.create_dataset("int3c_ip2", data=int3c_ip2, shape=int3c_ip2.shape)
            
    total_tasks = nroots*(nroots+1)//2
    nfinished = 0
    if ao_integrals:
        for j in range(nroots):
            y_j, x_j = get_xy_full(xy[j], oo_dim)
            if oo_dim == 0:
                x_j = reduce(np.dot, (mo_coeff, x_j, mo_coeff.T.conj()))
            else:
                x_j = reduce(np.dot, (mo_coeff, y_j.conj(), mo_coeff.T.conj()))
            x_j_aa = x_j[:nao,:nao]
            x_j_bb = x_j[nao:,nao:]
            x_j_ab = x_j[:nao,nao:] + x_j[nao:,:nao]
            faa = einsum("nrp,sr->nsp", int3c, x_j_aa)
            fbb = einsum("nrp,sr->nsp", int3c, x_j_bb)
            fab = einsum("nrp,sr->nsp", int3c, x_j_ab)
            for i in range(j+1):
                print("Calculating spin-spin interaction matrix: %d/%d" % (nfinished, total_tasks), flush=True, end="\r")
                y_i, x_i = get_xy_full(xy[i], oo_dim)
                if oo_dim == 0:
                    x_i = reduce(np.dot, (mo_coeff, x_i, mo_coeff.T.conj()))
                else:
                    x_i = reduce(np.dot, (mo_coeff, y_i.conj(), mo_coeff.T.conj()))
                x_i_aa = x_i[:nao,:nao]
                x_i_bb = x_i[nao:,nao:]
                x_i_ab = x_i[:nao,nao:] + x_i[nao:,:nao]
                tmp = np.zeros_like(x_i_aa, dtype=np.complex128)
                tmp += einsum("msp,nsp->mn", int3c_ipip[2,2], faa)
                tmp += einsum("msp,nsp->mn", int3c_ipip[0,2] - 1.0j*int3c_ipip[1,2], fab)
                tmp += einsum("msp,nsp->mn", int3c_ipip[0,0] - int3c_ipip[1,1] - 2j*int3c_ipip[0,1], fbb)
                ss_mat[i, j] = einsum("mn,mn->", tmp, x_i_aa.conj())
                tmp = np.zeros_like(x_i_bb, dtype=np.complex128)
                tmp += einsum("msp,nsp->mn", int3c_ipip[2,2], fbb)
                tmp -= einsum("msp,nsp->mn", int3c_ipip[0,2] + 1.0j*int3c_ipip[1,2], fab)
                tmp += einsum("msp,nsp->mn", int3c_ipip[0,0] - int3c_ipip[1,1] + 2j*int3c_ipip[0,1], faa)
                ss_mat[i, j] += einsum("mn,mn->", tmp, x_i_bb.conj())
                tmp = np.zeros_like(x_i_ab, dtype=np.complex128)
                tmp -= einsum("msp,nsp->mn", int3c_ipip[2,2], fab)
                tmp += einsum("msp,nsp->mn", int3c_ipip[0,2] + 1.0j*int3c_ipip[1,2], faa)
                tmp -= einsum("msp,nsp->mn", int3c_ipip[0,2] - 1.0j*int3c_ipip[1,2], fbb)
                ss_mat[i, j] += einsum("mn,mn->", tmp, x_i_ab.conj())
                ss_mat[i, j] *= fac
                ss_mat[j, i] = ss_mat[i, j].conj()

                nfinished += 1
    else:
        for i in range(nroots):
            y_i, x_i = get_xy_full(xy[i], oo_dim)
            if oo_dim != 0:
                x_i = y_i.conj()
            tmp = einsum("ik,yklq->yilq", x_i.conj(), int3c_ip)
            for j in range(i+1):
                print("Calculating spin-spin interaction matrix: %d/%d" % (nfinished, total_tasks), flush=True, end="\r")
                y_j, x_j = get_xy_full(xy[j], oo_dim)
                if oo_dim != 0:
                    x_j = y_j.conj()
                tmp2 = einsum("yilq,jl->yijq", tmp, x_j)
                ss_mat[i, j] = einsum("yijq,yijq->", tmp2, int3c_ip2) * fac
                ss_mat[j, i] = ss_mat[i, j].conj()
                nfinished += 1
    print()
    if oo_dim != 0:
        ss_mat = -ss_mat.conj()
    return ss_mat


def zfs_tensor_ss(xy, oo_dim, mo_coeff, mol, ri = True, auxbasis = None):
    from lib_pprpa.gradient import get_xy_full
    nao = mol.nao_nr()
    if ri or auxbasis != None:
        from pyscf import df
        auxmol = df.make_auxmol(mol, auxbasis=auxbasis)
        naux = auxmol.nao_nr()
        int3c = df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1').transpose(2,0,1).reshape(naux,nao*nao)
        int2c = auxmol.intor('int2c2e', aosym='s1', comp=1)
        int3c = np.linalg.solve(int2c.T, int3c).T.reshape(nao,nao,naux) # int3c = int3c2e * int2c2e^-1 in AO basis
        int2c_ipip = auxmol.intor('int2c2e_ip1ip2', aosym='s1', comp=9).reshape(3,3,naux,naux)
        trace = int2c_ipip[0,0] + int2c_ipip[1,1] + int2c_ipip[2,2]
        int2c_ipip[0,0] -= trace / 3
        int2c_ipip[1,1] -= trace / 3
        int2c_ipip[2,2] -= trace / 3
        del int2c, trace
        gc.collect()
        nmo = mo_coeff.shape[1]
        ca = mo_coeff[:nao,:]
        cb = mo_coeff[nao:,:]
        int3c_ip = np.zeros((3,nmo,nmo,naux), dtype=np.complex128)
        tmp = einsum("mi,mnp->inp", ca.conj(), int3c)
        int3c_ip[0] = einsum("nj,inp->ijp", ca, tmp)
        int3c_ip[1] = einsum("nj,inp->ijp", ca, tmp)
        int3c_ip[2] = einsum("nj,inp->ijp", ca, tmp)
        tmp = einsum("mi,mnp->inp", cb.conj(), int3c)
        int3c_ip[0] += einsum("nj,inp->ijp", cb, tmp)
        int3c_ip[1] += einsum("nj,inp->ijp", cb, tmp)
        int3c_ip[2] += einsum("nj,inp->ijp", cb, tmp)
        del int3c, ca, cb, tmp
        gc.collect()

        nroots = len(xy)
        D_ss = []
        for j in range(nroots):
            y_j, x_j = get_xy_full(xy[j], oo_dim)
            if oo_dim != 0:
                x_j = y_j.conj()
            tmp1 = einsum("qs,xpqP->xpsP", x_j, int3c_ip)
            tmp2 = einsum("pr,yrsQ->ypsQ", x_j.conj(), int3c_ip)
            tmp = einsum("xpsP,ypsQ->xyPQ", tmp1, tmp2)
            D_ss_tmp = np.zeros((3,3), dtype=np.complex128)
            for x in range(3):
                for y in range(3):
                    D_ss_tmp[x,y] = einsum("PQ,PQ->", tmp[x,y], int2c_ipip[x,y])
            D_ss.append(D_ss_tmp / 2.0 / LIGHT_SPEED**2)
        return np.array(D_ss)
    else:
        raise NotImplementedError("Direct calculation of the ZFS tensor is not implemented yet.")

def sz_eigenstates(xy, oo_dim, mo_coeff, mol, exci, spinor = False):
    """
    Returns the eigenstates of the spin-z Hamiltonian.
    """
    nroots = len(xy)
    nao = mo_coeff.shape[0]//2
    ovlp = mol.intor("int1e_ovlp")
    sz_ao = np.zeros((2*nao,2*nao), dtype=ovlp.dtype)
    sz_ao[:nao,:nao] = 0.5*ovlp
    sz_ao[nao:,nao:] = -0.5*ovlp
    if spinor:
        from socutils.scf.spinor_hf import sph2spinor
        sz_ao = sph2spinor(mol, sz_ao)
    sz_mo = reduce(np.dot, (mo_coeff.T.conj(), sz_ao, mo_coeff))
    sz_mat = np.zeros((nroots, nroots), dtype=xy.dtype)
    for i in range(nroots):
        for j in range(i+1):
            tdm, diag = make_tdm1(xy[i], xy[j], oo_dim)
            sz_mat[i, j] = einsum("ij,ji->", sz_mo, tdm)
            sz_mat[j, i] = sz_mat[i, j].conj()
    e, c = np.linalg.eigh(sz_mat)
    print(e)
    return np.dot(c.T, xy)  

def zfs_tensor_ss_spin_density_mf(den, mol, pseudo_spin = 1, ri = True, auxbasis = None):
    from lib_pprpa.gradient import make_rdm1
    nao = mol.nao_nr()
    if ri or auxbasis != None:
        from pyscf import df
        auxmol = df.make_auxmol(mol, auxbasis=auxbasis)
        naux = auxmol.nao_nr()
        int3c = df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1').transpose(2,0,1).reshape(naux,nao*nao)
        int2c = auxmol.intor('int2c2e', aosym='s1', comp=1)
        int3c = np.linalg.solve(int2c.T, int3c).T.reshape(nao,nao,naux) # int3c = int3c2e * int2c2e^-1 in AO basis
        int2c_ipip = auxmol.intor('int2c2e_ip1ip2', aosym='s1', comp=9).reshape(3,3,naux,naux)
        trace = int2c_ipip[0,0] + int2c_ipip[1,1] + int2c_ipip[2,2]
        int2c_ipip[0,0] -= trace / 3
        int2c_ipip[1,1] -= trace / 3
        int2c_ipip[2,2] -= trace / 3
        del int2c, trace
        gc.collect()
        
        fac = 1.0 / 8.0 / LIGHT_SPEED**2 / pseudo_spin / (pseudo_spin - 0.5)
        D_ss = []
        if den.ndim == 3:
            assert den[0].shape[0] == nao
            den_spin = den[0] - den[1]
        elif den.ndim == 2:
            assert den.shape[0] == nao*2
            den_spin = den[:nao,:nao] - den[nao:,nao:]
        tmp1 = einsum("mn,nmP->P", den_spin, int3c)
        tmp2 = einsum("P,xyPQ->xyQ", tmp1, int2c_ipip)
        tmp3 = einsum("xyQ,Q->xy", tmp2, tmp1)
        tmp1 = einsum("tm,mnP->tnP", den_spin, int3c)
        tmp2 = einsum("tnP,ntQ->PQ", tmp1, tmp1)
        tmp3 -= einsum("PQ,xyPQ->xy", tmp2, int2c_ipip)
        D_ss.append(tmp3 * fac)
        return np.array(D_ss)
    else:
        raise NotImplementedError("Direct calculation of the ZFS tensor is not implemented yet.")
    


def zfs_tensor_ss_spin_density(xy, oo_dim, mo_coeff, mol, pseudo_spin = 1, ri = True, auxbasis = None):
    from lib_pprpa.grad.grad_utils import get_xy_full
    nao = mol.nao_nr()
    if ri or auxbasis != None:
        from pyscf import df
        auxmol = df.make_auxmol(mol, auxbasis=auxbasis)
        naux = auxmol.nao_nr()
        int3c = df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1').transpose(2,0,1).reshape(naux,nao*nao)
        int2c = auxmol.intor('int2c2e', aosym='s1', comp=1)
        int3c = np.linalg.solve(int2c.T, int3c).T.reshape(nao,nao,naux) # int3c = int3c2e * int2c2e^-1 in AO basis
        int2c_ipip = auxmol.intor('int2c2e_ip1ip2', aosym='s1', comp=9).reshape(3,3,naux,naux)
        trace = int2c_ipip[0,0] + int2c_ipip[1,1] + int2c_ipip[2,2]
        int2c_ipip[0,0] -= trace / 3
        int2c_ipip[1,1] -= trace / 3
        int2c_ipip[2,2] -= trace / 3
        del int2c, trace
        gc.collect()
        nmo = mo_coeff.shape[1]
        ca = mo_coeff[:nao,:]
        cb = mo_coeff[nao:,:]
        int3c_ip = np.zeros((3,nmo,nmo,naux), dtype=np.complex128)
        tmp = einsum("mi,mnp->inp", ca.conj(), int3c)
        int3c_ip[0] = einsum("nj,inp->ijp", cb, tmp)
        int3c_ip[1] = einsum("nj,inp->ijp", cb, tmp)*(-1.0j)
        int3c_ip[2] = einsum("nj,inp->ijp", ca, tmp)
        tmp = einsum("mi,mnp->inp", cb.conj(), int3c)
        int3c_ip[0] += einsum("nj,inp->ijp", ca, tmp)
        int3c_ip[1] += einsum("nj,inp->ijp", ca, tmp)*1.0j
        int3c_ip[2] -= einsum("nj,inp->ijp", cb, tmp)
        del int3c, ca, cb, tmp
        gc.collect()
        fac = 1.0 / 8.0 / LIGHT_SPEED**2 / pseudo_spin / (pseudo_spin - 0.5)
        nroots = len(xy)
        D_ss = []
        for j in range(nroots):
            y_j, x_j = get_xy_full(xy[j], oo_dim)
            if oo_dim != 0:
                x_j = y_j.conj()
            tmp1 = einsum("qs,xpqP->xpsP", x_j, int3c_ip)
            tmp2 = einsum("pr,xrsQ->xpsQ", x_j.conj(), int3c_ip)
            tmp = 2.0*einsum("psP,psQ->PQ", tmp1[2], tmp2[2]) \
                - einsum("psP,psQ->PQ", tmp1[1], tmp2[1]) \
                - einsum("psP,psQ->PQ", tmp1[0], tmp2[0])
            D_ss.append(einsum("PQ,xyPQ->xy", tmp, int2c_ipip) * fac)
        return np.array(D_ss)
    else:
        raise NotImplementedError("Direct calculation of the ZFS tensor is not implemented yet.")
    

