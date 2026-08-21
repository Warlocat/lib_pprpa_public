"""Analytical molecular ppRPA gradients on an equilibrium ddCOSMO reference.

Only the *solvated-reference* model is implemented here.  Equilibrium
ddCOSMO is included in the density-fitted SCF energy and in the CPHF/orbital
response.  The ppRPA matrix is not augmented with the ddCOSMO response kernel.

The solvent derivative follows PySCF's ddCOSMO TDDFT-gradient decomposition:
the ordinary ddCOSMO reference gradient is supplemented by the derivative of
the solvent interaction between the reference density and the relaxed ppRPA
one-particle density correction.  There is no TD transition-density-square
term because the solvent kernel is not present in the ppRPA energy equation.
"""

from __future__ import annotations

import numpy as np

from pyscf import lib
from pyscf.lib import logger

from lib_pprpa.grad import pprpa as pprpa_grad
from lib_pprpa.grad.ddcosmo_kernels import (
    prepare_gradient_intermediates,
    reference_gradient as ddcosmo_reference_gradient,
    response_gradient as ddcosmo_response_gradient,
)
from lib_pprpa.pprpa_util import start_clock, stop_clock
from lib_pprpa.solvent.ddcosmo import (
    require_df_ddcosmo,
    require_zero_mu,
)


def ddcosmo_density_response_gradient(
    solvent_obj, dm_reference, dm_response, intermediates=None
):
    """Derivative of the ddCOSMO reference/response-density cross term.

    This is the ppRPA analogue of the ``dm0 * dmz1doo`` part of PySCF's
    ddCOSMO TDDFT gradient. The solvated-reference model does not put the
    solvent response kernel in the ppRPA eigenvalue equation.
    """
    dm_reference = np.asarray(dm_reference)
    dm_response = np.asarray(dm_response)
    if dm_reference.ndim != 2 or dm_response.shape != dm_reference.shape:
        raise ValueError("dm_reference and dm_response must be equal-size AO matrices")

    return ddcosmo_response_gradient(
        solvent_obj, dm_reference, dm_response, intermediates=intermediates
    )


def ddcosmo_density_response_gradient_polarization(
    solvent_obj, dm_reference, dm_response
):
    """Evaluate the same cross derivative through a polarization identity.

    This slower public-API expression is useful as an implementation check for
    the private PySCF TDDFT helper used above.  It is not used by the production
    gradient path.
    """
    dm_reference = np.asarray(dm_reference)
    dm_response = np.asarray(dm_response)
    if dm_reference.ndim != 2 or dm_response.shape != dm_reference.shape:
        raise ValueError("dm_reference and dm_response must be equal-size AO matrices")
    grad_plus = solvent_obj.grad(dm_reference + dm_response)
    grad_minus = solvent_obj.grad(dm_reference - dm_response)
    return 0.5 * (grad_plus - grad_minus)


def grad_elec(pprpa_grad_obj, xy, mult, atmlst=None):
    """Electronic plus ddCOSMO contribution for one ppRPA state."""
    mf = pprpa_grad_obj.mf
    solvent_obj = require_df_ddcosmo(mf)

    # Force the same equilibrium response used by PySCF's TDDFT gradients,
    # including every mf.gen_response() call inside the ppRPA Z-vector code.
    with lib.temporary_env(solvent_obj, equilibrium_solvation=True):
        de_solute = pprpa_grad.grad_elec(pprpa_grad_obj, xy, mult, atmlst)

        dm_reference = mf.make_rdm1()
        dm_response = pprpa_grad_obj.rdm1e

        shared_clock = "Build shared ddCOSMO gradient intermediates"
        start_clock(shared_clock)
        intermediates = prepare_gradient_intermediates(solvent_obj)
        stop_clock(shared_clock)

        reference_clock = "Calculate reference ddCOSMO gradient"
        start_clock(reference_clock)
        de_reference_solvent = ddcosmo_reference_gradient(
            solvent_obj, dm_reference, intermediates=intermediates
        )
        stop_clock(reference_clock)

        response_clock = "Calculate ppRPA ddCOSMO response gradient"
        start_clock(response_clock)
        de_response_solvent = ddcosmo_density_response_gradient(
            solvent_obj, dm_reference, dm_response, intermediates=intermediates
        )
        stop_clock(response_clock)

    if atmlst is not None:
        atom_indices = np.asarray(list(atmlst), dtype=int)
        de_reference_solvent = de_reference_solvent[atom_indices]
        de_response_solvent = de_response_solvent[atom_indices]

    pprpa_grad_obj.de_solute = de_solute
    pprpa_grad_obj.de_solvent_reference = de_reference_solvent
    pprpa_grad_obj.de_solvent_response = de_response_solvent
    return de_solute + de_reference_solvent + de_response_solvent


class Gradients(pprpa_grad.Gradients):
    """ppRPA gradient for a density-fitted equilibrium-ddCOSMO reference."""

    _keys = pprpa_grad.Gradients._keys | {
        "de_solute",
        "de_solvent_reference",
        "de_solvent_response",
    }

    def __init__(self, pprpa, mf, mult="t", state=0):
        require_df_ddcosmo(mf)
        require_zero_mu(pprpa)
        super().__init__(pprpa, mf, mult=mult, state=state)
        self.de_solute = None
        self.de_solvent_reference = None
        self.de_solvent_response = None

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info("solvent model = equilibrium ddCOSMO (solvated reference)")
        log.info("ddCOSMO epsilon = %.12g", self.mf.with_solvent.eps)
        log.info("ppRPA solvent kernel = not included")
        return self

    def grad_elec(self, xy, mult, atmlst):
        return grad_elec(self, xy, mult, atmlst)


Grad = Gradients
