import numpy
import time
from typing import List, Tuple


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
    inp = -numpy.sum(u[:oo_dim] * v[:oo_dim]) \
        + numpy.sum(u[oo_dim:] * v[oo_dim:])
    return inp


def get_chemical_potential(nocc, mo_energy):
    """Get chemical potential as the average between HOMO and LUMO.
    In the case there is no occupied or virtual orbital, return 0.

    Args:
        nocc (int/int array): number of occupied orbitals.
        mo_energy (double array/double ndarray/list of double array):
            orbital energy. double array for restricted calculation,
            double ndarray/list of double array for unrestricted
            calculation.

    Raises:
        ValueError: unrecognized mo_energy dimension.

    Returns:
        mu (double): chemical potential.
    """
    nspin = None
    if isinstance(mo_energy, list):
        nspin = len(mo_energy)
    elif isinstance(mo_energy, numpy.ndarray):
        nspin = mo_energy.ndim

    if nspin == 1:
        nmo = len(mo_energy)
        if nocc == 0 or nocc == nmo:
            mu = 0.0
        else:
            mu = (mo_energy[nocc-1] + mo_energy[nocc]) * 0.5
    elif nspin == 2:
        nmo = (len(mo_energy[0]), len(mo_energy[1]))
        if (nocc[0] == nocc[1] == 0) or (nocc[0] == nmo[0] and nocc[1]==nmo[1]):
            mu = 0.0
        else:
            assert nocc[0] >= nocc[1]
            if nocc[1] == 0:
                homo = mo_energy[0][nocc[0]-1]
            else:
                homo = max(mo_energy[0][nocc[0]-1], mo_energy[1][nocc[1]-1])
            lumo = min(mo_energy[0][nocc[0]], mo_energy[1][nocc[1]])
            mu = (homo + lumo) * 0.5
    else:
        raise ValueError("unrecognized mo_energy.")

    return mu


def get_pprpa_input_act(nocc, mo_energy, Lpq, nocc_act, nvir_act, mo_dip=None):
    """Get basic input in active space.

    Args:
        nocc (int/int array): number of occupied orbitals.
        mo_energy (double array/double ndarray): orbital energy.
        Lpq (double ndarray): three-center density-fitting matrix in MO space.
        nocc_act (int/int array): number of occupied orbitals.
        nvir_act (int/int array): number of virtual orbitals.
        mo_dip (double ndarray, optional): dipole moment matrix in MO space. Defaults to None.

    Returns:
        nocc_act (int/int array): number of occupied orbitals in active space.
        mo_energy_act (double array/double ndarray): orbital energy in active space.
        Lpq_act (double ndarray): three-center density-fitting matrix in MO space in active space.

        Only if mo_dip is provided:
        mo_dip_act (double ndarray): dipole moment matrix in MO space in active space.
    """
    nmo = len(mo_energy)
    nvir = nmo - nocc
    nocc_act = nocc if nocc_act > nocc else nocc_act
    nvir_act = nvir if nvir_act > nvir else nvir_act
    mo_energy_act = mo_energy[(nocc-nocc_act):(nocc+nvir_act)]
    Lpq_act = Lpq[:, (nocc-nocc_act):(nocc+nvir_act),
                  (nocc-nocc_act):(nocc+nvir_act)]
    if mo_dip is not None:
        mo_dip_act = mo_dip[:, nocc - nocc_act : nocc + nvir_act, nocc - nocc_act : nocc + nvir_act]
        return nocc_act, mo_energy_act, Lpq_act, mo_dip_act
    else:
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


def GMRES_Pople(AMultX, ADiag, B, maxIter = 30, tol = 1e-14, printLevel = 0):
    """ Solve AX = B using Pople's algorithm.
        It is designed for solving Z-vector equations for orbital relaxation.
        Poples's algorithm is highly related to the GMRES/Krylov-space methods.
        But interestingly, Pople's paper is earlier than the GMRES algorithm.
        (https://doi.org/10.1002/qua.560160825)

        A = ADiag*[I - Ap], Ap = I - ADiagInv*A, Bp = ADiagInv*B
        Ap*X = X - ADiagInv*A*X
        [I - Ap] X = Bp
        X = span[Bp, Ap*Bp, Ap^2*Bp, ...]
          = span[Bp, ADiagInv*A*Bp, ADiagInv*A^2*Bp, ...]

        ADiag is just for preconditioning. It is not necessary to
        be the exact diagonal elements of A.
    Args:
        AMultX (function): A function to multiply A with X.
        ADiag (double/complex array): Diagonal elements of A.
        B (double/complex array): Right-hand side of the equation.
        maxIter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.
        printLevel (int): Print level.
    Returns:
        X (double/complex array): Solution of the equation.
    """
    size = B.shape[0]
    assert size == len(ADiag)
    maxIter = min(maxIter, size + 1)

    Bp = B.copy()/numpy.sqrt(numpy.dot(B.conj().T, B).real)
    for ii in range(size):
        Bp[ii] = B[ii]/ADiag[ii]

    X_list = [Bp]
    AX_list = []
    converged = False
    for ii in range(1, maxIter):
        AX = AMultX(X_list[ii-1])
        for jj in range(size):
            AX[jj] = AX[jj]/ADiag[jj]
        AX_list.append(AX)
        for jj in range(ii):
            Xnorm = numpy.dot(X_list[jj].conj().T, X_list[jj])
            AX = AX - numpy.dot(X_list[jj].conj().T, AX)/Xnorm * X_list[jj]
        AXnorm = numpy.sqrt(numpy.dot(AX.conj().T, AX).real)/size
        if printLevel >= 1:
            print("gmres iter %d, residual norm: %e" % (ii, AXnorm))
        if AXnorm < tol:
            converged = True
            break
        else:
            X_list.append(AX)
    if not converged:
        print("ERROR: GMRES (Pople) did not converge in %d iterations!" % maxIter)
        exit()

    nbasis = len(X_list)
    A_sub = numpy.zeros((nbasis, nbasis), dtype = B.dtype)
    B_sub = numpy.zeros((nbasis), dtype = B.dtype)
    for ii in range(nbasis):
        B_sub[ii] = numpy.dot(X_list[ii].conj().T, Bp)
        for jj in range(nbasis):
            A_sub[ii,jj] = numpy.dot(X_list[ii].conj().T, AX_list[jj])
    X_sub = numpy.linalg.solve(A_sub, B_sub)
    X = numpy.zeros((size), dtype = B.dtype)
    for ii in range(nbasis):
        X = X + X_sub[ii]*X_list[ii]
    if printLevel >= 1:
        print("gmres residual subspace: ", numpy.linalg.norm(A_sub.dot(X_sub) - B_sub))
        print("gmres residual: ", numpy.linalg.norm(AMultX(X) - B))
    return X

def GMRES_wrapper(AMultX, ADiag, B, maxIter = 30, tol = 1e-14):
    """ Solve AX = B using GMRES algorithm.
        It is a wrapper of the scipy.sparse.linalg.gmres function.
    Args:
        AMultX (function): A function to multiply A with X.
        B (double/complex array): Right-hand side of the equation.
        maxIter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.
        printLevel (int): Print level.
    Returns:
        X (double/complex array): Solution of the equation.
    """
    from scipy.sparse.linalg import gmres, LinearOperator
    size = B.shape[0]
    A = LinearOperator((size, size), matvec = AMultX, dtype=B.dtype)
    M = numpy.diag(1.0/ADiag)
    X, info = gmres(A, B, M = M, maxiter = maxIter, tol = tol)
    if info != 0:
        print("ERROR: GMRES did not converge!")
        exit()
    return X

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
    print('    CPU time for %s %s, wall time %s\n' %
          (clock_name, cpu_time, wall_time), flush=True)


def get_nocc_nvir_frac(mo_occ, thresh=1.0e-8, sort_mo=False, mo_energy=None,
        mo_coeff=None):
    """Get number of occupied, fractionally occupied and virtual orbitals.
    Fractionally occupied orbitals are considered as both occupied and virtual.
    Reference: doi.org/10.1063/1.4817183.

    Args:
        mo_occ (double ndarray): occupation numbers of orbitals.
        thresh (double, optional): threshold to determine fractional charge.
            Defaults to 1.0e-5.

    Returns:
        nocc (int array): number of occupied orbitals, including fully
            and fractionally occupied orbitals.
        nvir (int array): number of virtual orbitals, including completely
            unoccupied and fractionally occupied orbitals.
        frac_nocc (int array): number of fractionally occupied orbitals.
    """
    nspin = 2
    nocc = numpy.zeros(shape=[2], dtype=int)
    nvir = numpy.zeros_like(nocc)
    nmo = numpy.zeros_like(nocc)
    frac_nocc = numpy.zeros_like(nocc)

    for s in range(nspin):
        nmo[s] = len(mo_occ[s])
        nocc[s] = len(numpy.argwhere(mo_occ[s] > thresh))
        nvir[s] = len(numpy.argwhere(mo_occ[s] < (1.0 - thresh)))
        frac_nocc[s] = nocc[s] + nvir[s] - nmo[s]

    if sort_mo:
        for s in range(nspin):
            occ_index = numpy.where(mo_occ[s] > thresh)[0]
            vir_index = numpy.where(mo_occ[s] < (1.0 - thresh))[0]
            frac_index = numpy.asarray(
                numpy.intersect1d(occ_index, vir_index))
            int1_index = numpy.asarray(
                [x for x in occ_index if x not in frac_index])
            int0_index = numpy.asarray(
                [x for x in vir_index if x not in frac_index])

            spin = 'alpha' if s == 0 else 'beta'
            mo_energy_int1 = mo_energy[s][int1_index]
            mo_energy_frac = mo_energy[s][frac_index]
            mo_energy_int0 = mo_energy[s][int0_index]
            mo_energy[s] = numpy.concatenate(
                (mo_energy_int1, mo_energy_frac, mo_energy_int0))
            mo_coeff_int1 = mo_coeff[s][int1_index]
            mo_coeff_frac = mo_coeff[s][frac_index]
            mo_coeff_int0 = mo_coeff[s][int0_index]
            mo_coeff[s] = numpy.concatenate(
                (mo_coeff_int1, mo_coeff_frac, mo_coeff_int0))

    return nocc, nvir, frac_nocc

def generate_spectrum(
    vee: numpy.ndarray,
    tdm: numpy.ndarray,
    ipol: str = "XYZ",
    energyRange: List[float] = [0.0, 10.0, 0.01],
    sigma: float = 0.1,
    nspin: int = 1,
    save_to: bool | str = False
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Generate absorption spectrum from excitation energies and transition dipole moments.
    (Based on the BSEResult.plotSpectrum() method in the westpy package
    Args:
        vee (numpy.ndarray): Array of excitation energies in eV
        tdm (numpy.ndarray): Array of transition dipole moments, shape (n_transitions, 3)
        ipol (str): Polarization component ("XX", "YY", "ZZ", "XY", "XZ", "YX", "YZ", "ZX", "ZY", or "XYZ")
        energyRange (List[float]): Energy range as [min, max, step] in eV
        sigma (float): Broadening width in eV
        nspin (int): Number of spin channels (1 or 2)
        save_to (bool | str, optional): Filename to save spectrum data.
            A default filename will be picked if True. No data saved if False.
            Defaults to False.
    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Energy axis (x) and absorption spectrum (y)
    """

    #import scipy.constants as sp
    #ev2ry=sp.e / sp.m_e / sp.e**4 * (8.0 * sp.epsilon_0**2 * sp.h**2) # 0.0734985857 Ry / eV
    ev2ry = 0.0734985857 # Ry / eV units

    # Validate inputs
    valid_ipols = ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ", "XYZ"]
    assert ipol in valid_ipols, f"ipol must be one of {valid_ipols}"
    n_ipol = len(ipol)

    xmin, xmax, dx = energyRange
    assert xmax > xmin, "Maximum energy must be greater than minimum energy"
    assert dx > 0.0, "Energy step must be positive"
    assert sigma > 0.0, "Broadening width must be positive"
    assert vee.shape[0] == tdm.shape[0], "Number of excitation energies must match transition dipole moments"
    assert tdm.shape[1] == 3, "Transition dipole moments must have 3 components (x, y, z)"

    # Sort arrays
    vee = numpy.abs(vee)
    idxs = numpy.argsort(vee)
    vee = vee[idxs]
    tdm = tdm[idxs]

    # Convert energy range to Rydberg units and create energy axis
    sigma_ev = sigma * ev2ry
    n_step = int((xmax - xmin) / dx) + 1
    energyAxis = numpy.linspace(xmin, xmax, n_step, endpoint=True)
    chiAxis = numpy.zeros(n_step, dtype=numpy.complex128)

    # Convert excitation energies to Rydberg units
    vee_ry = vee * ev2ry

    # Spin degeneracy factor
    degspin = 2.0 / nspin

    # Calculate susceptibility for each energy point
    for ie, energy in enumerate(energyAxis):
        freq_ev = energy * ev2ry  # Convert to Rydberg units

        # Calculate chi tensor components
        chi = numpy.zeros((n_ipol, n_ipol), dtype=numpy.complex128)

        for ip in range(n_ipol):  # requested Cartesian components
            for ip2 in range(n_ipol):
                num = tdm[:, ip] * tdm[:, ip2]
                den = freq_ev - vee_ry - 1j * sigma_ev
                tmp = numpy.sum(num / den)
                chi[ip, ip2] = tmp * degspin

        # Extract the requested polarization component
        if ipol == "XX":
            chiAxis[ie] = chi[0, 0]
        elif ipol == "XY":
            chiAxis[ie] = chi[0, 1]
        elif ipol == "XZ":
            chiAxis[ie] = chi[0, 2]
        elif ipol == "YX":
            chiAxis[ie] = chi[1, 0]
        elif ipol == "YY":
            chiAxis[ie] = chi[1, 1]
        elif ipol == "YZ":
            chiAxis[ie] = chi[1, 2]
        elif ipol == "ZX":
            chiAxis[ie] = chi[2, 0]
        elif ipol == "ZY":
            chiAxis[ie] = chi[2, 1]
        elif ipol == "ZZ":
            chiAxis[ie] = chi[2, 2]
        elif ipol == "XYZ":
            # Isotropic average with energy weighting
            chiAxis[ie] = (chi[0, 0] + chi[1, 1] + chi[2, 2]) * energy / 3.0 / numpy.pi

    if isinstance(save_to, str):
        filename = save_to + ".npz" if ".npz" not in save_to else save_to
    elif save_to:
        filename = "save_to.npz"
    else:
        filename = None

    if filename is not None:
        message = f"Saving spectrum to {filename}"
        numpy.savez(filename, x=energyAxis, y=chiAxis.imag, vee=vee, tdm=tdm)
    else:
        message = "Not saving spectrum data"

    print(message)
    # Return energy axis and imaginary part of susceptibility (absorption)
    return energyAxis, chiAxis.imag
