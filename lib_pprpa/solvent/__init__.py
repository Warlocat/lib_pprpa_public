"""Implicit-solvent helpers for molecular ppRPA calculations.

The initial implementation in this package is a *solvated-reference* model:
the SCF reference and its orbital response include equilibrium ddCOSMO, while
the ppRPA eigenvalue equation itself is unchanged.
"""

from lib_pprpa.solvent.ddcosmo import (
    attach_ddcosmo,
    require_df_ddcosmo,
    require_zero_mu,
    state_energy,
)

__all__ = [
    "attach_ddcosmo",
    "require_df_ddcosmo",
    "require_zero_mu",
    "state_energy",
]
