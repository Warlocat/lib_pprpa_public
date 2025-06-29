import numpy
import scipy
import scipy.linalg

from lib_pprpa.pprpa_direct import ppRPA_direct
from lib_pprpa.pprpa_util import inner_product, get_chemical_potential, start_clock, stop_clock

def diagonalize_pprpa_singlet(nocc, mo_energy, Lpq, fxc, mu=None):
    """Diagonalize singlet ppRPA matrix.

    Args:
        nocc (int): number of occupied orbitals.
        mo_energy (double array): orbital energy.
        Lpq (double ndarray): three-center density-fitting matrix in MO space.
        fxc (double ndarray): KS-DFT kernel matrix in MO space.
        mu (double, optional): chemical potential. Defaults to None.

    Returns:
        exci (double array): ppRPA eigenvalue.
        xy (double ndarray): ppRPA eigenvector.
        ec (double): singlet correlation energy.
    """
    nmo = len(mo_energy)
    nvir = nmo - nocc
    if mu is None:
        mu = get_chemical_potential(nocc=nocc, mo_energy=mo_energy)

    oo_dim = int((nocc + 1) * nocc / 2)

    # low triangular index (including diagonal)
    tri_row_o, tri_col_o = numpy.tril_indices(nocc)
    tri_row_v, tri_col_v = numpy.tril_indices(nvir)

    # =======================> Construct W <============================
    # K_{pq,rs} = fxc_{pq,rs} + <ps|qr>
    pass


def diagonalize_pprpa_triplet():
    pass

class RppRPAwDirect(ppRPA_direct):
    def __init__(
            self, nocc, mo_energy, Lpq, fxc, hh_state=5, pp_state=5, nelec="n-2",
            print_thresh=0.1, mo_dip=None, osc_channel="pp"):
        super().__init__(
            nocc, mo_energy, Lpq, hh_state=hh_state, pp_state=pp_state, 
            nelec=nelec, print_thresh=print_thresh, mo_dip=mo_dip, 
            osc_channel=osc_channel)
        self.fxc = fxc

    def kernel(self, multi):
        self.multi = multi
        self.check_parameter()
        self.dump_flags()
        self.check_memory()
        start_clock("ppRPA-w direct: %s" % multi)
        if self.multi == "s":
            self.exci_s, self.xy_s, self.ec_s = diagonalize_pprpa_singlet(
                nocc=self.nocc, mo_energy=self.mo_energy, Lpq=self.Lpq,
                mu=self.mu)
        elif multi == "t":
            self.exci_t, self.xy_t, self.ec_t = diagonalize_pprpa_triplet(
                nocc=self.nocc, mo_energy=self.mo_energy, Lpq=self.Lpq,
                mu=self.mu)
        stop_clock("ppRPA-w direct: %s" % multi)
        return
