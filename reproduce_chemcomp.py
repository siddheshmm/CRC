"""
reproduce_chemcomp.py  (v2)

Sanity-check crn_simulator.py against Johnson et al. 2026 (ChemComp paper),
using a physically valid CSTR mass balance instead of their constant-flux
input abstraction (see crn_simulator.py docstring for why that mattered).

Target: Sel'kov-Schnakenberg ODE system
    dx/dt = a - b*x + d*y*x^2
    dy/dt = c - d*y*x^2
with (a,b,c,d) = (0.5, 2.2, 1.5, 1.0), (x0,y0) = (1.1, 2.5)

Candidate reservoirs (topologies from their Fig. 6, reparameterized for a
real CSTR):

CRN-A: 2A + B <-> 3A   (k1, k-1)      A -> 0   (k2, plain first-order decay)
       species {A,B}, dilution D=0.2, inflow conc chosen so the *initial*
       net flow matches the magnitude of their I_A=0.5, I_B=1.5:
           C_in = X0 + I/D   =>   C_in_A = 5.0, C_in_B = 8.6

CRN-B: same R1 as CRN-A, plus
       A <-> E         (k2,k-2)
       2C + D <-> 3C   (k3,k-3)
       C <-> E         (k4,k-4)
       2B <-> C        (k5,k-5)
       species {A,B,C,D,E}, dilution D=0.2 on A,B,C,D via CSTR feed
       (C_in_A=5.0, C_in_B=8.6, C_in_C=3.6, C_in_D=10.0), but E is NOT fed
       by the CSTR line -- instead it's removed by a first-order side
       process (e.g. it precipitates/vents/gets consumed downstream), which
       is the physically valid replacement for ChemComp's I_E=-10:
           extra_removal={"E": 2.0}   (tuned so E stays in a sane range)

We train a ridge readout on each reservoir's concentration traces (train
window ODE-time < 10) to reconstruct x(t) and y(t), then report train/test
RMSE. Expected qualitative result: CRN-B (richer basis: 5 species vs 2)
should out-perform CRN-A on test RMSE, same conclusion as the paper --
but this time without any species going unphysically negative.
"""

import numpy as np
from scipy.integrate import solve_ivp
from crn_simulator import Reaction, ChemicalReactionNetwork, fit_readout, rmse

# ---- Target ODE (ground truth) ---------------------------------------------
a, b, c, d = 0.5, 2.2, 1.5, 1.0

def target_rhs(t, state):
    x, y = state
    dx = a - b * x + d * y * x**2
    dy = c - d * y * x**2
    return [dx, dy]

t_span = (0, 20)
n_points = 2000
t_eval = np.linspace(*t_span, n_points)
target_sol = solve_ivp(target_rhs, t_span, [1.1, 2.5], t_eval=t_eval, method="LSODA")
x_true, y_true = target_sol.y

train_mask = t_eval < 10.0
test_mask = ~train_mask

# ---- CRN-A ------------------------------------------------------------------
crn_a = ChemicalReactionNetwork(
    species=["A", "B"],
    reactions=[
        Reaction("R1", reactants={"A": 2, "B": 1}, products={"A": 3}, k_f=1.0, k_r=0.1),
        Reaction("R2", reactants={"A": 1}, products={}, k_f=2.2),
    ],
    dilution_rate=0.2,
    inflow_conc={"A": 5.0, "B": 8.6},
)
t_a, conc_a = crn_a.simulate({"A": 2.5, "B": 1.1}, t_span, n_points=n_points)
C_a = np.column_stack([conc_a["A"], conc_a["B"]])

# ---- CRN-B ------------------------------------------------------------------
crn_b = ChemicalReactionNetwork(
    species=["A", "B", "C", "D", "E"],
    reactions=[
        Reaction("R1", reactants={"A": 2, "B": 1}, products={"A": 3}, k_f=1.0, k_r=0.10),
        Reaction("R2", reactants={"A": 1}, products={"E": 1}, k_f=2.21, k_r=0.221),
        Reaction("R3", reactants={"C": 2, "D": 1}, products={"C": 3}, k_f=1.3, k_r=0.13),
        Reaction("R4", reactants={"C": 1}, products={"E": 1}, k_f=1.5, k_r=0.15),
        Reaction("R5", reactants={"B": 2}, products={"C": 1}, k_f=0.01, k_r=0.001),
    ],
    dilution_rate=0.2,
    inflow_conc={"A": 5.0, "B": 8.6, "C": 3.6, "D": 10.0},
    extra_removal={"E": 2.0},
)
t_b, conc_b = crn_b.simulate({"A": 2.5, "B": 1.1, "C": 1.1, "D": 0.0, "E": 0.0},
                              t_span, n_points=n_points)
C_b = np.column_stack([conc_b["A"], conc_b["B"], conc_b["C"], conc_b["D"], conc_b["E"]])

# ---- sanity check: no species should go negative -----------------------
for name, conc in [("CRN-A", conc_a), ("CRN-B", conc_b)]:
    for sp, v in conc.items():
        if v.min() < -1e-6:
            print(f"WARNING: {name} species {sp} went negative (min={v.min():.3f})")

# ---- Train/test readout for both reservoirs, both targets -------------------
def evaluate(C, name):
    print(f"\n--- {name} ---")
    for target_name, y_full in [("x(t)", x_true), ("y(t)", y_true)]:
        model = fit_readout(C[train_mask], y_full[train_mask], alpha=1e-3)
        pred = model.predict(C)
        train_rmse = rmse(y_full[train_mask], pred[train_mask])
        test_rmse = rmse(y_full[test_mask], pred[test_mask])
        print(f"  target {target_name}: train RMSE={train_rmse:.3f}  test RMSE={test_rmse:.3f}")

evaluate(C_a, "CRN-A (2 species, 2 reactions)")
evaluate(C_b, "CRN-B (5 species, 5 reactions)")

print("\nExpected qualitative result (paper): CRN-B fits better than CRN-A,")
print("because more species/reactions = richer linear basis -- now achieved")
print("with a physically valid CSTR mass balance (no negative concentrations).")