"""
Cycle Reservoir with Jumps (CRJ) - minimal-complexity ESN topology
matching the "cycle-based + chords" idea used in ChemReservoir
(Yirik et al. 2025), built on Rodan & Tino's original CRJ architecture.

Nodes = pseudo-molecules, cycle edges = the base reaction ring,
jump/chord edges = the extra long-range connections ChemReservoir adds
to improve memory capacity.

Evaluated on the standard Short-Term Memory (STM) capacity task:
can a linear readout reconstruct u(t-k) from the reservoir state x(t)?
"""

import numpy as np
from sklearn.linear_model import Ridge

rng = np.random.default_rng(0)


def build_crj_weights(n_nodes, r_c=0.5, r_j=0.3, jump_size=5, v_in=0.5):
    """Cycle + jump ('chord') reservoir weight matrix, plus input weights."""
    W = np.zeros((n_nodes, n_nodes))
    # base cycle: node i -> node i+1
    for i in range(n_nodes):
        W[(i + 1) % n_nodes, i] = r_c
    # jump/chord connections every `jump_size` nodes (bidirectional, like chords)
    for i in range(0, n_nodes, jump_size):
        j = (i + jump_size) % n_nodes
        W[j, i] = r_j
        W[i, j] = r_j
    # fixed-sign, fixed-magnitude input weights (CRJ convention)
    w_in = v_in * rng.choice([-1, 1], size=n_nodes)
    return W, w_in


def run_reservoir(W, w_in, u, leak=1.0):
    """Drive the reservoir with input sequence u, return state matrix X (T x n_nodes)."""
    n_nodes = W.shape[0]
    x = np.zeros(n_nodes)
    states = []
    for u_t in u:
        pre_activation = W @ x + w_in * u_t
        x = (1 - leak) * x + leak * np.tanh(pre_activation)
        states.append(x.copy())
    return np.array(states)


def memory_capacity(u, X, max_delay=30, washout=100, ridge_alpha=1e-6):
    """Short-term memory capacity: sum of squared correlations for each delay k."""
    T = len(u)
    mc_total, mc_per_delay = 0.0, []
    for k in range(1, max_delay + 1):
        X_train = X[washout:T - k]
        y_train = u[washout - k if washout - k >= 0 else 0: T - k]
        y_target = u[washout:T - k]  # u(t-k) aligned with X(t)
        # align properly: target at time t is u(t-k)
        y_target = np.array([u[t - k] for t in range(washout, T - k)])
        model = Ridge(alpha=ridge_alpha)
        model.fit(X_train, y_target)
        y_pred = model.predict(X_train)
        if np.std(y_target) > 0 and np.std(y_pred) > 0:
            corr = np.corrcoef(y_target, y_pred)[0, 1]
        else:
            corr = 0.0
        mc_k = corr ** 2
        mc_per_delay.append(mc_k)
        mc_total += mc_k
    return mc_total, mc_per_delay


if __name__ == "__main__":
    N_NODES = 100
    T_STEPS = 3000
    u = rng.uniform(-1, 1, size=T_STEPS)  # i.i.d. uniform input, standard MC benchmark

    configs = {
        "cycle_only (no chords)": dict(r_c=0.5, r_j=0.0, jump_size=N_NODES),
        "cycle + chords (jump=5)": dict(r_c=0.5, r_j=0.3, jump_size=5),
        "cycle + chords (jump=10)": dict(r_c=0.5, r_j=0.3, jump_size=10),
    }

    for name, cfg in configs.items():
        W, w_in = build_crj_weights(N_NODES, **cfg)
        X = run_reservoir(W, w_in, u)
        mc_total, _ = memory_capacity(u, X, max_delay=30, washout=100)
        print(f"{name:28s} -> total memory capacity: {mc_total:.3f} (max possible = 30)")