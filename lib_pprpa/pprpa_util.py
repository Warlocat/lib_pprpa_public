import numpy
import time

def ij2index(r, c, row, col):
    """Get index of a row and column in a square matrix in a lower triangular matrix.

    Args:
        r (int): row index in s square matrix.
        c (int): column index in s square matrix.
        row (int array): row index array of a lower triangular matrix.
        col (int array): column index array of a lower triangular matrix.

    Returns:
        i (int): index in the lower triangular matrix.
    """
    for i in range(len(row)):
        if r == row[i] and c == col[i]:
            return i

    raise ValueError("cannot find the index!")


def inner_product(u, v, oo_dim):
    """Calculate ppRPA inner product.
    product = <Y1,Y2> - <X1,X2>, where X is occ-occ block, Y is vir-vir block.

    Args:
        u (double array): first vector.
        v (double array): second vector
        oo_dim (int): occ-occ block dimension

    Returns:
        inp (double): inner product.
    """
    inp = -numpy.sum(u[:oo_dim] * v[:oo_dim]) + numpy.sum(u[oo_dim:] * v[oo_dim:])
    return inp


def get_chemical_potential(nocc, mo_energy):
    """Get chemical potential as the average between HOMO and LUMO.
    In the case there is no occupied or virtual orbital, return 0.

    Args:
        nocc (int/int array): number of occupied orbitals.
        mo_energy (double array/double ndarray): orbital energy.

    Raises:
        ValueError: unrecognized mo_energy dimension.

    Returns:
        mu (double): chemical potential.
    """
    if mo_energy.ndim == 1:
        nmo = len(mo_energy)
        mu = 0.0 if nocc == 0 or nocc == nmo else (mo_energy[nocc-1] + mo_energy[nocc]) * 0.5
    elif mo_energy.ndim == 2:
        nmo = mo_energy.shape[1]
        if (nocc[0] == nocc[1] == 0) or (nocc[0] == nocc[1] == nmo):
            mu = 0.0
        else:
            assert nocc[0] >= nocc[1]
            homo = mo_energy[0][nocc[0]-1] if nocc[1] == 0 else max(mo_energy[0][nocc[0]-1], mo_energy[1][nocc[1]-1])
            lumo = min(mo_energy[0][nocc[0]], mo_energy[1][nocc[1]])
            mu = (homo + lumo) * 0.5
    else:
        raise ValueError("unrecognized mo_energy shape: %s" % (mo_energy.shape, ))

    return mu


def get_pprpa_input_act(nocc, mo_energy, Lpq, nocc_act, nvir_act):
    """Get basic input in active space.

    Args:
        nocc (int/int array): number of occupied orbitals.
        mo_energy (double array/double ndarray): orbital energy.
        Lpq (double ndarray): three-center density-fitting matrix in MO space.
        nocc_act (int/int array): number of occupied orbitals.
        nvir_act (int/int array): number of virtual orbitals.

    Returns:
        nocc_act (int/int array): number of occupied orbitals in active space.
        mo_energy_act (double array/double ndarray): orbital energy in active space.
        Lpq_act (double ndarray): three-center density-fitting matrix in MO space in active space.
    """
    nmo = len(mo_energy)
    nvir = nmo - nocc
    nocc_act = nocc if nocc_act > nocc else nocc_act
    nvir_act = nvir if nvir_act > nvir else nvir_act
    mo_energy_act = mo_energy[(nocc-nocc_act):(nocc+nvir_act)]
    Lpq_act = Lpq[:, (nocc-nocc_act):(nocc+nvir_act), (nocc-nocc_act):(nocc+nvir_act)]
    return nocc_act, mo_energy_act, Lpq_act


def print_citation():
    __version__ = "0.1"

    __doc__ = \
    """
\nlib_pprpa   version %s
A package for particle-particle random phase approximation.

    Thanks for using the ppRPA library!
    Any papers that use lib_pprpa should cite the these two papers:
    [1] Jiachen Li, Jincheng Yu, Weitao Yang.
        In preparation
    [2] Helen van Aggelen, Yang Yang, and Weitao Yang.
        "Exchange-correlation energy from pairing matrix fluctuation and the particle-particle random-phase approximation"
        Phys. Rev. A 88, 030501(R)

    If you used the active-space ppRPA, please cite
    [3] Jiachen Li, Jincheng Yu, Zehua Chen, and Weitao Yang.
        "Linear Scaling Calculations of Excitation Energies with Active-Space Particle-Particle Random-Phase Approximation"
        J. Phys. Chem. A 2023, 127, 37, 7811-7822

    If you used the Davidson algorithm in ppRPA, please cite
    [4] Yang Yang, Degao Peng, Jianfeng Lu, and Weitao Yang.
        "Excitation energies from particle-particle random phase approximation: Davidson algorithm and benchmark studies"
        J. Chem. Phys. 141, 124104 (2014)

    Have a nice day!\n
    """ % (__version__)

    print(__doc__)
    return


# time counting global variables and functions
clock_names = []
clocks = []

def _s_to_hms(t):
    decimal = t - int(t)
    t = int(t)
    seconds = int(t % 60)
    t = (t - seconds) / 60
    minutes = int(t % 60)
    hours = int((t - minutes) / 60)
    hms = "%.2f s" % (seconds + decimal)
    if minutes != 0 or hours != 0:
        hms = ("%d m " % minutes) + hms
    if hours != 0:
        hms = ("%d h " % hours) + hms
    return hms


def start_clock(clock_name):
    assert isinstance(clock_name, str) and clock_name not in clock_names
    clock_names.append(clock_name)
    clocks.append((time.process_time(), time.perf_counter()))
    print("begin %-s." % clock_name, flush=True)


def stop_clock(clock_name):
    assert isinstance(clock_name, str) and clock_name in clock_names
    idx = clock_names.index(clock_name)
    clock_end = (time.process_time(), time.perf_counter())
    cpu_time = _s_to_hms(clock_end[0] - clocks[idx][0])
    wall_time = _s_to_hms(clock_end[1] - clocks[idx][1])
    del clock_names[idx]
    del clocks[idx]

    print("finish %-s." % clock_name)
    print('    CPU time for %s %s, wall time %s\n' % (clock_name, cpu_time, wall_time), flush=True)
