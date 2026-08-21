"""Small-molecule checks for the isolated ddCOSMO ppRPA extension."""

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import dft, gto
from pyscf.solvent.grad import ddcosmo_grad

from lib_pprpa.grad.ddcosmo_kernels import reference_gradient
from lib_pprpa.grad.pprpa_ddcosmo import (
    Gradients,
    ddcosmo_density_response_gradient,
    ddcosmo_density_response_gradient_polarization,
)
from lib_pprpa.pprpa_davidson import ppRPA_Davidson
from lib_pprpa.pyscf_util import get_pyscf_input_mol
from lib_pprpa.solvent.ddcosmo import attach_ddcosmo, state_energy


def _water_mf(displacement=0.0, xc="pbe"):
    mol = gto.M(
        atom=[
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, 1.0, 1.0 + displacement)),
            ("H", (0.0, -1.0, 1.0)),
        ],
        basis="sto-3g",
        unit="Bohr",
        verbose=0,
    )
    mf = dft.RKS(mol, xc=xc).density_fit()
    mf = attach_ddcosmo(mf, eps=20.0)
    mf.grids.level = 5
    mf.conv_tol = 1.0e-12
    mf.kernel()
    assert mf.converged
    return mf


def _pprpa(mf, channel, mult):
    nocc, mo_energy, lpq = get_pyscf_input_mol(mf)
    pprpa = ppRPA_Davidson(
        nocc,
        mo_energy,
        lpq,
        channel=channel,
        nroot=1,
        trial="identity",
        residue_thresh=1.0e-10,
    )
    pprpa.mu = 0.0
    pprpa.kernel(mult)
    return pprpa


def test_attach_requires_density_fitting():
    mol = gto.M(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", unit="Bohr")
    with pytest.raises(TypeError, match="density fitting"):
        attach_ddcosmo(dft.RKS(mol, xc="pbe"))


def test_td_helper_matches_density_polarization_identity():
    mf = _water_mf()
    dm0 = mf.make_rdm1()
    rng = np.random.default_rng(9)
    dm1 = rng.standard_normal(dm0.shape)
    dm1 = (dm1 + dm1.T) * 2.0e-4

    grad_td = ddcosmo_density_response_gradient(mf.with_solvent, dm0, dm1)
    grad_pol = ddcosmo_density_response_gradient_polarization(
        mf.with_solvent, dm0, dm1
    )
    assert np.max(np.abs(grad_td - grad_pol)) < 2.0e-8


def test_parallel_reference_kernel_matches_pyscf():
    mf = _water_mf()
    dm0 = mf.make_rdm1()
    grad_parallel = reference_gradient(mf.with_solvent, dm0)
    grad_pyscf = ddcosmo_grad.kernel(mf.with_solvent, dm0)
    assert np.max(np.abs(grad_parallel - grad_pyscf)) < 2.0e-10


@pytest.mark.parametrize(
    "xc,channel,mult",
    [
        ("pbe", "pp", "s"),
        ("pbe", "hh", "t"),
        ("b3lyp", "pp", "s"),
        ("b3lyp", "hh", "s"),
    ],
)
def test_full_state_gradient_matches_finite_difference(xc, channel, mult):
    displacement = 1.0e-3

    mf = _water_mf(xc=xc)
    pprpa = _pprpa(mf, channel, mult)
    analytical = Gradients(pprpa, mf, mult=mult, state=0).kernel()[1, 2]

    mf_plus = _water_mf(displacement, xc)
    pprpa_plus = _pprpa(mf_plus, channel, mult)
    mf_minus = _water_mf(-displacement, xc)
    pprpa_minus = _pprpa(mf_minus, channel, mult)
    numerical = (
        state_energy(mf_plus, pprpa_plus, mult, 0)
        - state_energy(mf_minus, pprpa_minus, mult, 0)
    ) / (2.0 * displacement)

    assert abs(analytical - numerical) < 3.0e-6
