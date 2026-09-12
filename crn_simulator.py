"""
crn_simulator.py  (v2 -- physically-grounded CSTR mass balance)

Generic mass-action chemical reaction network (CRN) simulator, built to match
the ChemComp formulation (Johnson et al. 2026, npj Unconventional Computing)
and the reservoir-computing readout approach used there and in Baltussen
et al. 2024 (Nature, formose CRC) -- but with a real CSTR mass balance
instead of ChemComp's constant-flux input abstraction.

Why this matters: a real continuous-flow reactor (and the Pioreactor's pump
loop) can only add/remove mass in two physically valid ways:
  1. Dilution: fresh reagent flows in at concentration C_in, well-stirred
     mixture flows out at whatever concentration is currently in the vessel.
     Net effect on every species: d[X]/dt += D * (C_in_X - [X]), where
     D = total_flow_rate / vessel_volume = 1/residence_time. This term is
     SELF-LIMITING -- as [X] -> C_in it can't overshoot.
  2. A first-order side process removing one species specifically (gas
     venting, precipitation, downstream consumption outside the modeled
     network): d[X]/dt -= k_out * [X]. Still self-limiting.

A CONSTANT flux term (d[X]/dt += constant, independent of [X]) has no
physical CSTR analog -- it can drive concentrations unboundedly negative,
which is exactly what happened when we tried to reproduce ChemComp's
I_E = -10 literally. We replace that with option (2) above.

Core object: a Reaction is `stoich_react * [species] -> stoich_prod * [species]`
with forward rate k_f and optional reverse rate k_r (mass-action kinetics).
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Reaction:
    name: str
    reactants: dict[str, float]   # species -> stoichiometric coefficient
    products: dict[str, float]
    k_f: float
    k_r: float = 0.0              # 0.0 => irreversible

    def rate(self, conc: dict[str, float]) -> float:
        """Net forward-minus-reverse rate (mass action)."""
        fwd = self.k_f
        for sp, n in self.reactants.items():
            fwd *= conc[sp] ** n
        rev = 0.0
        if self.k_r:
            rev = self.k_r
            for sp, n in self.products.items():
                rev *= conc[sp] ** n
        return fwd - rev


@dataclass
class ChemicalReactionNetwork:
    species: list[str]
    reactions: list[Reaction]

    # --- physical CSTR mass balance ---
    # D: shared dilution rate (1/residence_time). 0.0 => closed batch vessel,
    # no flow at all (fine for a well-mixed round-bottom flask experiment).
    dilution_rate: float = 0.0
    # inflow_conc[sp]: the concentration of species `sp` in the feed stream.
    # Only species actually being dosed need an entry; everything else
    # defaults to 0 (i.e. dilution washes it out, doesn't replenish it).
    # Value can be a float (constant) or callable f(t) -> float (driven input).
    inflow_conc: dict = field(default_factory=dict)
    # extra_removal[sp]: first-order removal rate constant for species that
    # leave via a separate physical/chemical process (venting, precipitation,
    # consumption downstream of the modeled network). d[sp]/dt -= k * [sp].
    extra_removal: dict = field(default_factory=dict)

    def _inflow_at(self, sp: str, t: float) -> float:
        val = self.inflow_conc.get(sp, 0.0)
        return val(t) if callable(val) else val

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        conc = dict(zip(self.species, y))
        d = {sp: 0.0 for sp in self.species}
        for rxn in self.reactions:
            r = rxn.rate(conc)
            for sp, n in rxn.reactants.items():
                d[sp] -= n * r
            for sp, n in rxn.products.items():
                d[sp] += n * r
        for sp in self.species:
            if self.dilution_rate:
                d[sp] += self.dilution_rate * (self._inflow_at(sp, t) - conc[sp])
            if sp in self.extra_removal:
                d[sp] -= self.extra_removal[sp] * conc[sp]
        return np.array([d[sp] for sp in self.species])

    def simulate(self, y0: dict, t_span: tuple, n_points: int = 1000,
                 method: str = "LSODA", **kwargs):
        y0_arr = np.array([y0[sp] for sp in self.species])
        t_eval = np.linspace(*t_span, n_points)
        sol = solve_ivp(self.rhs, t_span, y0_arr, t_eval=t_eval,
                         method=method, **kwargs)
        return sol.t, dict(zip(self.species, sol.y))


# ----------------------------------------------------------------------------
# Reservoir readout: ridge regression of a target y(t) onto concentration
# traces, following ChemComp eq. (2)-(3): y(t) = sum_i alpha_i * c_i(t)
# ----------------------------------------------------------------------------

def fit_readout(C: np.ndarray, y: np.ndarray, alpha: float = 1e-3):
    """
    C: (n_timepoints, n_species) reservoir concentration traces (design matrix)
    y: (n_timepoints,) or (n_timepoints, n_targets) target signal(s)
    Returns trained sklearn Ridge model (includes intercept/bias term).
    """
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(C, y)
    return model


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))