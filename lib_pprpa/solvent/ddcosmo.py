"""Equilibrium ddCOSMO utilities for molecular, density-fitted ppRPA.

This module deliberately does not alter the ppRPA eigenvalue equation.  The
SCF orbitals, orbital energies, reference energy, and orbital response are
computed in equilibrium ddCOSMO.  ppRPA is then evaluated with the existing
electron-electron interaction in that solvated reference.
"""

from __future__ import annotations

from pyscf import solvent
from pyscf.df.df_jk import _DFHF
from pyscf.solvent import ddcosmo


def attach_ddcosmo(mf, eps=78.3553, solvent_obj=None):
    """Attach equilibrium ddCOSMO to an already density-fitted RHF/RKS object.

    Density fitting must be applied before this function is called.  This
    ordering preserves both the density-fitting auxiliary-basis response and
    PySCF's solvent wrapper.

    Parameters
    ----------
    mf
        A molecular ``RHF(...).density_fit()`` or
        ``RKS(...).density_fit()`` object.
    eps : float
        Static dielectric constant.
    solvent_obj
        Optional preconfigured :class:`pyscf.solvent.ddcosmo.DDCOSMO` object.

    Returns
    -------
    pyscf.scf.hf.SCF
        The density-fitted SCF object wrapped by equilibrium ddCOSMO.
    """
    if not isinstance(mf, _DFHF):
        raise TypeError(
            "ddCOSMO-ppRPA requires density fitting before solvation: "
            "build RKS/RHF, call .density_fit(), then attach_ddcosmo()."
        )

    if getattr(mf, "with_solvent", None) is not None:
        raise ValueError(
            "The SCF object already has a solvent. Apply density fitting first "
            "and call attach_ddcosmo() exactly once."
        )

    if solvent_obj is None:
        solvent_obj = ddcosmo.DDCOSMO(mf.mol)
    elif not isinstance(solvent_obj, ddcosmo.DDCOSMO):
        raise TypeError("solvent_obj must be a PySCF DDCOSMO object")

    solvent_obj.eps = float(eps)
    solvent_obj.equilibrium_solvation = True
    solvated_mf = solvent.ddCOSMO(mf, solvent_obj=solvent_obj)
    require_df_ddcosmo(solvated_mf)
    return solvated_mf


def require_df_ddcosmo(mf):
    """Validate the model assumptions required by the solvent gradient."""
    if not isinstance(mf, _DFHF):
        raise TypeError(
            "The ddCOSMO ppRPA implementation requires a density-fitted "
            "molecular RHF/RKS reference."
        )

    solvent_obj = getattr(mf, "with_solvent", None)
    if not isinstance(solvent_obj, ddcosmo.DDCOSMO):
        raise TypeError("The mean-field object must be wrapped by PySCF ddCOSMO")
    if solvent_obj.frozen:
        raise ValueError("Frozen ddCOSMO is not supported for ppRPA gradients")
    if not solvent_obj.equilibrium_solvation:
        raise ValueError(
            "Analytical ppRPA geometry gradients require equilibrium ddCOSMO "
            "(with_solvent.equilibrium_solvation = True)."
        )
    return solvent_obj


def require_zero_mu(pprpa):
    """Require the fixed chemical-potential convention used by gradients."""
    if pprpa.mu is None or abs(pprpa.mu) > 1.0e-14:
        raise ValueError(
            "Solvated-reference ppRPA state energies and gradients use mu = 0.0. "
            "Set pprpa.mu = 0.0 before pprpa.kernel()."
        )


def state_energy(mf, pprpa, mult="s", state=0):
    """Return the solvated-reference ppRPA total state energy.

    The sign follows the convention already used by the molecular ppRPA
    gradient examples: addition energies are added to the SCF energy and
    removal energies are subtracted.
    """
    require_df_ddcosmo(mf)
    require_zero_mu(pprpa)
    if mult == "s":
        excitation = pprpa.exci_s[state]
    elif mult == "t":
        excitation = pprpa.exci_t[state]
    else:
        raise ValueError("mult must be 's' or 't'")

    sign = 1.0 if pprpa.channel == "pp" else -1.0
    return mf.e_tot + sign * excitation
