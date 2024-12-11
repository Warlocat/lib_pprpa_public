import numpy as np
from lib_pprpa.pprpa_util import start_clock, stop_clock

from lib_pprpa import pprpa_davidson


def _pprpa_contraction(pprpa, tri_vec):
    """GppRPA contraction.

    Args:
        pprpa (ppRPA_Davidson): ppRPA_Davidson object.
        tri_vec (double/complex ndarray): trial vector.

    Returns:
        mv_prod (double/complex ndarray): product between ppRPA matrix and trial vectors.
    """
    nocc, nvir, nmo = pprpa.nocc, pprpa.nvir, pprpa.nmo
    naux = pprpa.naux
    mo_energy = pprpa.mo_energy
    Lpq = pprpa.Lpq
    Lpi = pprpa.Lpi
    Lpa = pprpa.Lpa

    ntri = tri_vec.shape[0]
    mv_prod = np.zeros(shape=[ntri, pprpa.full_dim], dtype=tri_vec.dtype)

    tri_row_o, tri_col_o = np.tril_indices(nocc, -1)
    tri_row_v, tri_col_v = np.tril_indices(nvir, -1)

    for ivec in range(ntri):
        z_oo = np.zeros(shape=[nocc, nocc], dtype=tri_vec.dtype)
        z_oo[tri_row_o, tri_col_o] = tri_vec[ivec][:pprpa.oo_dim]
        z_oo[np.diag_indices(nocc)] *= 1.0 / np.sqrt(2)
        z_oo = np.ascontiguousarray(z_oo.T)
        z_vv = np.zeros(shape=[nvir, nvir], dtype=tri_vec.dtype)
        z_vv[tri_row_v, tri_col_v] = tri_vec[ivec][pprpa.oo_dim:]
        z_vv[np.diag_indices(nvir)] *= 1.0 / np.sqrt(2)
        z_vv = np.ascontiguousarray(z_vv.T)

        # Lpqz_{L,pr} = \sum_s Lpq_{L,ps} z_{rs}
        Lpq_z = np.zeros(shape=[naux*nmo, nmo], dtype=tri_vec.dtype)
        if pprpa._use_Lov:
            Lpq_z[:, :nocc] = np.matmul(Lpi.reshape(naux * nmo, nocc), z_oo)
            Lpq_z[:, nocc:] = np.matmul(Lpa.reshape(naux * nmo, nvir), z_vv)
        else:
            Lpq_z[:, :nocc] = np.matmul(
                Lpq[:, :, :nocc].reshape(naux * nmo, nocc), z_oo)
            Lpq_z[:, nocc:] = np.matmul(
                Lpq[:, :, nocc:].reshape(naux * nmo, nvir), z_vv)
        Lpq_z = Lpq_z.reshape(naux, nmo, nmo)

        # MV_{pq} = \sum_{Lr} Lpq_{L,pr} Lpqz_{L,qr} - Lpq_{L,qr} Lpqz_{L,pr}
        # -MV_{qp}* = - \sum_{Lr} Lpq_{L,rp} Lpqz_{L,qr}^* + Lpq_{L,rq} Lpqz_{L,pr}^*
        Lpq_z = Lpq_z.transpose(1, 0, 2).conj()
        Lpq_z = Lpq_z.reshape(nmo, naux * nmo)
        if pprpa._use_Lov:
            prod_oo = np.matmul(Lpq_z[:nocc], Lpi.reshape(naux * nmo, nocc))
            prod_oo -= prod_oo.T
            prod_vv = np.matmul(Lpq_z[nocc:], Lpa.reshape(naux * nmo, nvir))
            prod_vv -= prod_vv.T
        else:
            prod_oo = np.matmul(Lpq_z[:nocc], Lpq[:, :, :nocc].reshape(naux * nmo, nocc))
            prod_oo -= prod_oo.T
            prod_vv = np.matmul(Lpq_z[nocc:], Lpq[:, :, nocc:].reshape(naux * nmo, nvir))
            prod_vv -= prod_vv.T
        prod_oo[np.diag_indices(nocc)] *= 1.0 / np.sqrt(2)
        prod_vv[np.diag_indices(nvir)] *= 1.0 / np.sqrt(2)
        mv_prod[ivec][: pprpa.oo_dim] =\
            -prod_oo[tri_row_o, tri_col_o].conj()
        mv_prod[ivec][pprpa.oo_dim:] = \
            -prod_vv[tri_row_v, tri_col_v].conj()

    orb_sum_oo = np.asarray(
        mo_energy[None, : nocc] + mo_energy[: nocc, None]) - 2.0 * pprpa.mu
    orb_sum_oo = orb_sum_oo[tri_row_o, tri_col_o]
    orb_sum_vv = np.asarray(
        mo_energy[None, nocc:] + mo_energy[nocc:, None]) - 2.0 * pprpa.mu
    orb_sum_vv = orb_sum_vv[tri_row_v, tri_col_v]
    for ivec in range(ntri):
        oz_oo = -orb_sum_oo * tri_vec[ivec][:pprpa.oo_dim]
        mv_prod[ivec][:pprpa.oo_dim] += oz_oo
        oz_vv = orb_sum_vv * tri_vec[ivec][pprpa.oo_dim:]
        mv_prod[ivec][pprpa.oo_dim:] += oz_vv

    return mv_prod


# analysis functions
def _pprpa_print_eigenvector(
        nocc, nvir, thresh, channel, exci0, exci, xy):
    """Print dominant components of an eigenvector.

    Args:
        multi (string): multiplicity.
        nocc (int): number of occupied orbitals.
        nvir (int): number of virtual orbitals.
        thresh (double): threshold to print a pair.
        channel (string): "pp" for particle-particle or "hh" for hole-hole.
        exci0 (double): lowest eigenvalue.
        exci (double array): ppRPA eigenvalue.
        xy (double/complex ndarray): ppRPA eigenvector.
    """
    oo_dim = int((nocc - 1) * nocc / 2)
    print("\n     print GppRPA excitations:\n")

    tri_row_o, tri_col_o = np.tril_indices(nocc, -1)
    tri_row_v, tri_col_v = np.tril_indices(nvir, -1)

    nroot = len(exci)
    au2ev = 27.211386
    from lib_pprpa.gpprpa_direct import complex_matrix_norm
    from lib_pprpa.pprpa_davidson import pprpa_print_a_pair
    if channel == "pp":
        for iroot in range(nroot):
            print("#%-d excitation:  exci= %-12.6f  eV   2e=  %-12.6f  eV" %
                  (iroot + 1,
                   (exci[iroot] - exci0) * au2ev, exci[iroot] * au2ev))
            if nocc > 0:
                full = np.zeros(shape=[nocc, nocc], dtype=xy.dtype)
                full[tri_row_o, tri_col_o] = xy[iroot][:oo_dim]
                full = complex_matrix_norm(full)
                pairs = np.argwhere(full > thresh)
                for i, j in pairs:
                    pprpa_print_a_pair(
                        is_pp=False, p=i, q=j, percentage=full[i, j])

            full = np.zeros(shape=[nvir, nvir], dtype=xy.dtype)
            full[tri_row_v, tri_col_v] = xy[iroot][oo_dim:]
            full = complex_matrix_norm(full)
            pairs = np.argwhere(full > thresh)
            for a, b in pairs:
                pprpa_print_a_pair(
                    is_pp=True, p=a+nocc, q=b+nocc, percentage=full[a, b])
            print("")
    else:
        for iroot in range(nroot):
            print("#%-d de-excitation:  exci= %-12.6f  eV   2e=  %-12.6f  eV" %
                  (iroot + 1,
                   (exci[iroot] - exci0) * au2ev, exci[iroot] * au2ev))
            full = np.zeros(shape=[nocc, nocc], dtype=xy.dtype)
            full[tri_row_o, tri_col_o] = xy[iroot][:oo_dim]
            full = complex_matrix_norm(full)
            pairs = np.argwhere(full > thresh)
            for i, j in pairs:
                pprpa_print_a_pair(is_pp=False, p=i, q=j, percentage=full[i, j])

            if nvir > 0:
                full = np.zeros(shape=[nvir, nvir], dtype=xy.dtype)
                full[tri_row_v, tri_col_v] = xy[iroot][oo_dim:]
                full = complex_matrix_norm(full)
                pairs = np.argwhere(full > thresh)
                for a, b in pairs:
                    pprpa_print_a_pair(
                        is_pp=True, p=a+nocc, q=b+nocc, percentage=full[a, b])
            print("")

    return


def _analyze_pprpa_davidson(
        exci, xy, nocc, nvir, print_thresh=0.1, channel="pp"):
    print("\nanalyze ppRPA results.")
    _pprpa_print_eigenvector(
        nocc=nocc, nvir=nvir, thresh=print_thresh,
        channel=channel, exci0=exci[0], exci=exci, xy=xy)
    return


class GppRPA_Davidson(pprpa_davidson.ppRPA_Davidson):
    def __init__(
            self, nocc, mo_energy, Lpq, channel="pp", nroot=5, max_vec=200,
            max_iter=100, trial="identity", residue_thresh=1.0e-7,
            print_thresh=0.1):
        super().__init__(
            nocc, mo_energy, Lpq, channel, nroot, max_vec, max_iter, trial,
            residue_thresh, print_thresh)
        # This is not really for triplet, just for implementation purpose.
        # GHF-ppRPA shares the same equation with ppRPA triplet.
        self.multi = "t"
        return
    
    def check_memory(self):
        # intermediate in contraction; mv_prod, tri_vec, xy
        mem = (
            self.naux * self.nmo * self.nmo + 3 * self.max_vec * self.full_dim)\
                * 8 / 1.0e6
        if self.Lpq.dtype == np.complex128:
            mem *= 2
        if mem < 1000:
            print("GppRPA needs at least %d MB memory." % mem)
        else:
            print("GppRPA needs at least %.1f GB memory." % (mem / 1.0e3))
        return

    def kernel(self):
        self.check_parameter()

        # TODO: directly take Lpi and Lpa in the future
        if self._use_Lov is True and self.Lpq is not None:
            self.Lpi = np.ascontiguousarray(self.Lpq[:, :, :self.nocc])
            self.Lpa = np.ascontiguousarray(self.Lpq[:, :, self.nocc:])
            self.Lpq = None

        self.dump_flags()
        self.check_memory()
        start_clock("GppRPA Davidson:")
        pprpa_davidson.kernel(pprpa=self)
        stop_clock("GppRPA Davidson:")
        self.exci_t = self.exci.copy()
        self.xy_t = self.xy.copy()
        return

    def analyze(self):
        _analyze_pprpa_davidson(
            exci=self.exci_t,
            xy=self.xy_t, nocc=self.nocc, nvir=self.nvir,
            print_thresh=self.print_thresh, channel=self.channel)
        return
    
    def contraction(self, tri_vec):
        return _pprpa_contraction(self, tri_vec)