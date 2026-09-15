"""Analytical molecular ppRPA gradients on an equilibrium PCM reference.

Only the solvated-reference model is implemented. Equilibrium PCM is included
in the density-fitted SCF energy and in the CPHF/orbital response. The ppRPA
matrix is not augmented with a solvent response kernel.

The solvent derivative contains the ordinary PCM reference gradient and the
derivative of the PCM interaction between the reference density and the
relaxed ppRPA one-particle density correction. The latter is evaluated by an
exact polarization identity for the quadratic PCM energy functional.
"""

from __future__ import annotations

import numpy as np

from pyscf import lib
from pyscf.lib import logger

from lib_pprpa.grad import pprpa as pprpa_grad
from lib_pprpa.pprpa_util import start_clock, stop_clock
from lib_pprpa.solvent.ddcosmo import require_zero_mu
from lib_pprpa.solvent.pcm import require_df_pcm


# PySCF 2.12 sizes the three-center PCM derivative block directly from
# ``pcmobj.max_memory``.  On large-memory nodes this can request a single
# >2**31-element libcint buffer and segfault before NumPy can report an
# allocation error.  This cap only changes batching; it does not approximate
# or otherwise alter the PCM gradient.
_PCM_GRAD_MEMORY_CAP_MB = 12000


def _pcm_grad(solvent_obj, dm):
    max_memory = min(float(solvent_obj.max_memory), _PCM_GRAD_MEMORY_CAP_MB)
    with lib.temporary_env(solvent_obj, max_memory=max_memory):
        return solvent_obj.grad(dm)


def pcm_density_response_gradient(solvent_obj, dm_reference, dm_response):
    """Derivative of the PCM reference/response-density cross term.

    The PCM energy and its nuclear gradient are quadratic functions of the
    solute electrostatic potential. Therefore, the central polarization
    identity exactly extracts the term linear in ``dm_response``. Calls to
    :meth:`PCM.grad` update PySCF's density-dependent cache, so the reference
    density is restored before returning.
    """
    dm_reference = np.asarray(dm_reference)
    dm_response = np.asarray(dm_response)
    if dm_reference.ndim != 2 or dm_response.shape != dm_reference.shape:
        raise ValueError("dm_reference and dm_response must be equal-size AO matrices")

    try:
        grad_plus = _pcm_grad(solvent_obj, dm_reference + dm_response)
        grad_minus = _pcm_grad(solvent_obj, dm_reference - dm_response)
    finally:
        solvent_obj.kernel(dm_reference)
    return 0.5 * (grad_plus - grad_minus)


def grad_elec(pprpa_grad_obj, xy, mult, atmlst=None):
    """Electronic plus PCM contribution for one ppRPA state."""
    mf = pprpa_grad_obj.mf
    solvent_obj = require_df_pcm(mf)

    # Include equilibrium PCM in every mf.gen_response() call made while the
    # ppRPA relaxed density/Z-vector is constructed.
    with lib.temporary_env(solvent_obj, equilibrium_solvation=True):
        de_solute = pprpa_grad.grad_elec(pprpa_grad_obj, xy, mult, atmlst)

        dm_reference = mf.make_rdm1()
        dm_response = pprpa_grad_obj.rdm1e

        reference_clock = "Calculate reference PCM gradient"
        start_clock(reference_clock)
        de_reference_solvent = _pcm_grad(solvent_obj, dm_reference)
        stop_clock(reference_clock)

        response_clock = "Calculate ppRPA PCM response gradient"
        start_clock(response_clock)
        de_response_solvent = pcm_density_response_gradient(
            solvent_obj, dm_reference, dm_response
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
    """ppRPA gradient for a density-fitted equilibrium-PCM reference."""

    _keys = pprpa_grad.Gradients._keys | {
        "de_solute",
        "de_solvent_reference",
        "de_solvent_response",
    }

    def __init__(self, pprpa, mf, mult="t", state=0):
        require_df_pcm(mf)
        require_zero_mu(pprpa)
        super().__init__(pprpa, mf, mult=mult, state=state)
        self.de_solute = None
        self.de_solvent_reference = None
        self.de_solvent_response = None

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info(
            "solvent model = equilibrium %s (solvated reference)",
            self.mf.with_solvent.method,
        )
        log.info("PCM epsilon = %.12g", self.mf.with_solvent.eps)
        log.info("ppRPA solvent kernel = not included")
        return self

    def grad_elec(self, xy, mult, atmlst):
        return grad_elec(self, xy, mult, atmlst)


Grad = Gradients


__all__ = ["Grad", "Gradients", "pcm_density_response_gradient"]
