"""Small-molecule checks for the isolated PCM ppRPA extension."""

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import dft, gto

from lib_pprpa.grad.pprpa_pcm import Gradients, _pcm_grad, pcm_density_response_gradient
from lib_pprpa.pprpa_davidson import ppRPA_Davidson
from lib_pprpa.pyscf_util import get_pyscf_input_mol
from lib_pprpa.solvent.pcm import (
    attach_pcm,
    canonical_pcm_method,
    require_df_pcm,
    state_energy,
)


def _water_mf(displacement=0.0, xc="pbe", method="IEF-PCM"):
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
    mf = attach_pcm(mf, eps=20.0, method=method)
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


def test_attach_requires_density_fitting_and_canonicalizes_method():
    mol = gto.M(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", unit="Bohr")
    with pytest.raises(TypeError, match="density fitting"):
        attach_pcm(dft.RKS(mol, xc="pbe"))

    mf = attach_pcm(dft.RKS(mol, xc="pbe").density_fit(), method="iefpcm")
    assert mf.with_solvent.method == "IEF-PCM"
    assert mf.with_solvent.equilibrium_solvation
    with pytest.raises(ValueError, match="Unsupported PCM method"):
        canonical_pcm_method("not-pcm")


def test_require_rejects_nonequilibrium_and_frozen_pcm():
    mol = gto.M(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", unit="Bohr")
    mf = attach_pcm(dft.RKS(mol, xc="pbe").density_fit())

    mf.with_solvent.equilibrium_solvation = False
    with pytest.raises(ValueError, match="equilibrium PCM"):
        require_df_pcm(mf)

    mf.with_solvent.equilibrium_solvation = True
    mf.with_solvent.frozen = True
    with pytest.raises(ValueError, match="Frozen PCM"):
        require_df_pcm(mf)


def test_pcm_gradient_memory_cap_is_temporary():
    class DummyPCM:
        max_memory = 330000

        def grad(self, dm):
            return self.max_memory, dm

    solvent_obj = DummyPCM()
    used_memory, returned_dm = _pcm_grad(solvent_obj, "density")
    assert used_memory == 12000
    assert returned_dm == "density"
    assert solvent_obj.max_memory == 330000


def test_density_response_polarization_is_exact_and_restores_cache():
    mf = _water_mf()
    solvent_obj = mf.with_solvent
    dm0 = mf.make_rdm1()
    rng = np.random.default_rng(11)
    dm1 = rng.standard_normal(dm0.shape)
    dm1 = (dm1 + dm1.T) * 2.0e-4

    response = pcm_density_response_gradient(solvent_obj, dm0, dm1)
    assert np.max(np.abs(np.squeeze(solvent_obj._intermediates["dm"]) - dm0)) < 1e-12

    scale = 1.0e-3
    grad_plus = solvent_obj.grad(dm0 + scale * dm1)
    grad_minus = solvent_obj.grad(dm0 - scale * dm1)
    numerical = (grad_plus - grad_minus) / (2.0 * scale)
    solvent_obj.kernel(dm0)
    assert np.max(np.abs(response - numerical)) < 2.0e-8


@pytest.mark.parametrize(
    "xc,channel,mult,method",
    [
        ("pbe", "pp", "s", "C-PCM"),
        ("pbe", "hh", "t", "IEF-PCM"),
        ("b3lyp", "pp", "s", "COSMO"),
        ("b3lyp", "hh", "s", "SS(V)PE"),
    ],
)
def test_full_state_gradient_matches_finite_difference(xc, channel, mult, method):
    displacement = 1.0e-3

    mf = _water_mf(xc=xc, method=method)
    pprpa = _pprpa(mf, channel, mult)
    analytical = Gradients(pprpa, mf, mult=mult, state=0).kernel()[1, 2]

    mf_plus = _water_mf(displacement, xc, method)
    pprpa_plus = _pprpa(mf_plus, channel, mult)
    mf_minus = _water_mf(-displacement, xc, method)
    pprpa_minus = _pprpa(mf_minus, channel, mult)
    numerical = (
        state_energy(mf_plus, pprpa_plus, mult, 0)
        - state_energy(mf_minus, pprpa_minus, mult, 0)
    ) / (2.0 * displacement)

    assert abs(analytical - numerical) < 3.0e-6
