"""Equilibrium PCM utilities for molecular, density-fitted ppRPA.

This module implements the same solvated-reference model used by the ddCOSMO
extension: PCM modifies the SCF reference and its orbital response, while the
ppRPA eigenvalue equation retains the ordinary electron-electron interaction.
"""

from __future__ import annotations

from pyscf import solvent
from pyscf.df.df_jk import _DFHF
from pyscf.solvent import pcm

from lib_pprpa.solvent.ddcosmo import require_zero_mu


_PCM_METHOD_ALIASES = {
    "C-PCM": "C-PCM",
    "CPCM": "C-PCM",
    "IEF-PCM": "IEF-PCM",
    "IEFPCM": "IEF-PCM",
    "COSMO": "COSMO",
    "SS(V)PE": "SS(V)PE",
}


def canonical_pcm_method(method):
    """Return the PySCF spelling of a supported PCM-family method."""
    key = str(method).strip().upper()
    try:
        return _PCM_METHOD_ALIASES[key]
    except KeyError as err:
        supported = ", ".join(sorted(set(_PCM_METHOD_ALIASES.values())))
        raise ValueError(
            f"Unsupported PCM method {method!r}; choose one of {supported}."
        ) from err


def attach_pcm(mf, eps=78.3553, method="IEF-PCM", solvent_obj=None):
    """Attach equilibrium PCM to an already density-fitted RHF/RKS object.

    Density fitting must precede the solvent wrapper. This ordering preserves
    both the density-fitting auxiliary-basis response and PySCF's PCM gradient
    wrapper.

    Parameters
    ----------
    mf
        A molecular ``RHF(...).density_fit()`` or
        ``RKS(...).density_fit()`` object.
    eps : float
        Static dielectric constant.
    method : str
        One of ``C-PCM``, ``IEF-PCM``, ``COSMO``, or ``SS(V)PE``.
    solvent_obj
        Optional preconfigured :class:`pyscf.solvent.pcm.PCM` object.

    Returns
    -------
    pyscf.scf.hf.SCF
        The density-fitted SCF object wrapped by equilibrium PCM.
    """
    if not isinstance(mf, _DFHF):
        raise TypeError(
            "PCM-ppRPA requires density fitting before solvation: build "
            "RKS/RHF, call .density_fit(), then attach_pcm()."
        )

    if getattr(mf, "with_solvent", None) is not None:
        raise ValueError(
            "The SCF object already has a solvent. Apply density fitting first "
            "and call attach_pcm() exactly once."
        )

    if solvent_obj is None:
        solvent_obj = pcm.PCM(mf.mol)
    elif not isinstance(solvent_obj, pcm.PCM):
        raise TypeError("solvent_obj must be a PySCF PCM object")

    solvent_obj.method = canonical_pcm_method(method)
    solvent_obj.eps = float(eps)
    solvent_obj.equilibrium_solvation = True
    solvated_mf = solvent.PCM(mf, solvent_obj=solvent_obj)
    require_df_pcm(solvated_mf)
    return solvated_mf


def require_df_pcm(mf):
    """Validate the model assumptions required by the PCM ppRPA gradient."""
    if not isinstance(mf, _DFHF):
        raise TypeError(
            "The PCM ppRPA implementation requires a density-fitted molecular "
            "RHF/RKS reference."
        )

    solvent_obj = getattr(mf, "with_solvent", None)
    if not isinstance(solvent_obj, pcm.PCM):
        raise TypeError("The mean-field object must be wrapped by PySCF PCM")
    canonical_pcm_method(solvent_obj.method)
    if solvent_obj.frozen:
        raise ValueError("Frozen PCM is not supported for ppRPA gradients")
    if not solvent_obj.equilibrium_solvation:
        raise ValueError(
            "Analytical ppRPA geometry gradients require equilibrium PCM "
            "(with_solvent.equilibrium_solvation = True)."
        )
    return solvent_obj


def state_energy(mf, pprpa, mult="s", state=0):
    """Return the solvated-reference PCM ppRPA total state energy."""
    require_df_pcm(mf)
    require_zero_mu(pprpa)
    if mult == "s":
        excitation = pprpa.exci_s[state]
    elif mult == "t":
        excitation = pprpa.exci_t[state]
    else:
        raise ValueError("mult must be 's' or 't'")

    sign = 1.0 if pprpa.channel == "pp" else -1.0
    return mf.e_tot + sign * excitation


__all__ = [
    "attach_pcm",
    "canonical_pcm_method",
    "require_df_pcm",
    "state_energy",
]
