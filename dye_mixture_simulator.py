"""
dye_mixture_simulator.py

Simulates dye-in-water dosing experiments using REAL, fitted calibration
coefficients from dye_analysis.py (or hypothetical ones for a dye you
haven't tested yet), so you can explore dosing schedules and candidate dye
pairs in silico before spending wet-lab time on them.

Unlike the titration_reservoir CRN, dyes don't react chemically -- they just
dilute and physically mix. So this is deliberately simpler: no ODE solver,
just volume-based concentration tracking (same math as extract_pulse_readings
in dye_analysis.py) plus a first-order mixing/settling relaxation (matching
the ~10-20s settle time observed empirically) and the linear calibration
model signal = intercept + slope*conc, with optional realistic noise.

Reuses unmix_mixed_dye's core logic conceptually but works entirely from
simulated data -- no CSV needed. This lets you:
  1. Reproduce the real red/blue finding (high correlation -> unstable
     unmixing) as a sanity check that the simulator matches reality.
  2. Screen a HYPOTHETICAL dye (e.g. yellow, with assumed/measured slopes)
     against blue or red to see if it separates better, before buying it.
  3. Test whether a different dosing schedule or noise level changes the
     unmixing outcome, without running a new physical experiment each time.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

AS7341_BANDS = [415, 445, 480, 515, 555, 590, 630, 680]


@dataclass
class DyeCalibration:
    """A dye's fitted (or hypothetical) per-band response: signal = intercept + slope*conc"""
    name: str
    slopes: dict      # {band: slope}
    intercepts: dict  # {band: intercept}

    @classmethod
    def from_fit_calibration_results(cls, name: str, results: dict):
        """results: output of dye_analysis.fit_calibration() -- {band: (slope, intercept, r2)}"""
        return cls(name=name,
                    slopes={b: v[0] for b, v in results.items()},
                    intercepts={b: v[1] for b, v in results.items()})


# --- Real fitted values from your actual red_dye / blue_dye experiments,
# auto-detected step times (copy these in directly from your console output
# any time you refit, so the simulator stays in sync with real data) ---
RED_DYE_REAL = DyeCalibration(
    name="red",
    slopes={415: 0.00953, 445: 0.08293, 480: 0.10482, 515: 0.12032,
            555: 0.10792, 590: 0.07011, 630: 0.06479, 680: 0.04137},
    intercepts={415: 0.01886, 445: 0.15096, 480: 0.15049, 515: 0.17787,
                555: 0.21274, 590: 0.22202, 630: 0.22842, 680: 0.11650},
)
BLUE_DYE_REAL = DyeCalibration(
    name="blue",
    slopes={415: 0.01312, 445: 0.14208, 480: 0.18872, 515: 0.21094,
            555: 0.20532, 590: 0.12730, 630: 0.13792, 680: 0.09544},
    intercepts={415: 0.01605, 445: 0.12274, 480: 0.11159, 515: 0.13233,
                555: 0.15887, 590: 0.17542, 630: 0.16882, 680: 0.08636},
)

# HYPOTHETICAL yellow dye -- NOT measured, just a plausible guess based on
# yellow dyes typically peaking in the blue-absorbing region (~440-460nm)
# and having low absorbance in the red end. Replace with real fitted values
# the moment you have a yellow-dye calibration run.
YELLOW_DYE_HYPOTHETICAL = DyeCalibration(
    name="yellow (HYPOTHETICAL -- not measured)",
    slopes={415: 0.03, 445: 0.28, 480: 0.15, 515: 0.04,
            555: 0.01, 590: 0.005, 630: 0.002, 680: 0.001},
    intercepts={b: 0.05 for b in AS7341_BANDS},  # placeholder flat baseline
)


def check_separability(dye_a: DyeCalibration, dye_b: DyeCalibration):
    """Same diagnostic used in dye_analysis.unmix_mixed_dye -- run this
    BEFORE any dosing simulation to see if a pair is even worth simulating
    further."""
    slope_a = np.array([dye_a.slopes[b] for b in AS7341_BANDS])
    slope_b = np.array([dye_b.slopes[b] for b in AS7341_BANDS])
    corr = np.corrcoef(slope_a, slope_b)[0, 1]
    A = np.column_stack([slope_a, slope_b])
    cond = np.linalg.cond(A)
    print(f"[separability check] {dye_a.name} vs {dye_b.name}: "
          f"correlation={corr:.3f}  condition_number={cond:.1f}")
    if corr > 0.95 or cond > 20:
        print("  -> Likely to unmix badly, same failure mode as real red/blue. "
              "Not worth simulating a full dosing run until you pick a more distinct pair.")
    else:
        print("  -> Looks separable in principle -- worth simulating further, "
              "and worth actually testing on real hardware.")
    return corr, cond


def simulate_dosing(dyes: list[DyeCalibration], pulses: list[tuple],
                     start_volume_mL: float, noise_std: float = 0.002,
                     mixing_tau_s: float = 15.0, seed: int = 0):
    """
    pulses: list of (dye_name, volume_mL, time_s) tuples, time in seconds
    from experiment start (simpler than wall-clock for a pure simulation).
    noise_std: Gaussian measurement noise added per band, per reading --
    set this from your real sensor's baseline noise floor (check the
    baseline std you saw in auto_align_pulses' output) for a realistic test.
    mixing_tau_s: first-order relaxation time constant for stirring/mixing
    to complete after each dose (~15-20s matches what we saw empirically
    in the real timing-check plots).

    Returns: t (seconds), true_conc {dye_name: array}, signal (n_timepoints, 8)
    """
    rng = np.random.default_rng(seed)
    dye_lookup = {d.name.split(" ")[0]: d for d in dyes}  # tolerate "(HYPOTHETICAL...)" suffix
    pulses = sorted(pulses, key=lambda p: p[2])
    t_end = pulses[-1][2] + 60
    t = np.arange(0, t_end, 1.0)

    cumulative = {d.name.split(" ")[0]: np.zeros_like(t) for d in dyes}
    total_vol = np.full_like(t, start_volume_mL)

    for dye_name, vol, dose_time in pulses:
        idx = t >= dose_time
        total_vol[idx] += vol
        # instantaneous jump then first-order relaxation toward the new
        # cumulative amount, approximating mixing time rather than an
        # instant step (more realistic than assuming perfect instant mixing)
        target = np.zeros_like(t)
        target[idx] = vol
        relax = 1 - np.exp(-(t - dose_time) / mixing_tau_s)
        relax[~idx] = 0
        cumulative[dye_name] += target * relax

    # cumulative volume added is a running sum across pulses of the same dye;
    # fix the above per-pulse increments into a true cumulative trace
    for dye_name in cumulative:
        cumulative[dye_name] = np.cumsum(np.diff(cumulative[dye_name], prepend=0).clip(min=0))

    true_conc = {name: cumulative[name] / total_vol for name in cumulative}

    n = len(t)
    signal = np.zeros((n, len(AS7341_BANDS)))
    for i, b in enumerate(AS7341_BANDS):
        band_signal = np.zeros(n)
        # baseline: average the dyes' intercepts once (not summed -- same
        # fix as the real unmixing code, avoids double-counting the blank)
        band_signal += np.mean([dye_lookup[name].intercepts[b] for name in true_conc])
        for name, conc in true_conc.items():
            band_signal += dye_lookup[name].slopes[b] * conc
        signal[:, i] = band_signal + rng.normal(0, noise_std, size=n)

    return t, true_conc, signal


if __name__ == "__main__":
    print("=== Sanity check: does the simulator reproduce the real red/blue finding? ===")
    check_separability(RED_DYE_REAL, BLUE_DYE_REAL)

    print("\n=== Screening a hypothetical yellow dye against blue (before buying it) ===")
    check_separability(YELLOW_DYE_HYPOTHETICAL, BLUE_DYE_REAL)

    print("\n=== Screening hypothetical yellow against red ===")
    check_separability(YELLOW_DYE_HYPOTHETICAL, RED_DYE_REAL)

    print("\n=== Simulated dosing run: red+blue, same schedule shape as your pseudorandom experiment ===")
    pulses = [("red", 1.0, 0), ("blue", 1.0, 180), ("red", 1.0, 360), ("blue", 1.0, 540),
              ("red", 1.0, 720), ("blue", 1.0, 900), ("red", 1.0, 1080), ("blue", 1.0, 1260)]
    t, true_conc, signal = simulate_dosing([RED_DYE_REAL, BLUE_DYE_REAL], pulses, start_volume_mL=10.0)

    from scipy.optimize import nnls
    print(f"\n{'t(s)':>6} {'true_red':>10} {'true_blue':>10} {'nnls_red':>10} {'nnls_blue':>10}")
    for check_t in [60, 240, 420, 600, 780, 960, 1140, 1260]:
        idx = min(np.searchsorted(t, check_t), len(t) - 1)
        A = np.column_stack([[RED_DYE_REAL.slopes[b] for b in AS7341_BANDS],
                              [BLUE_DYE_REAL.slopes[b] for b in AS7341_BANDS]])
        baseline = np.mean([[RED_DYE_REAL.intercepts[b], BLUE_DYE_REAL.intercepts[b]] for b in AS7341_BANDS], axis=1)
        y = signal[idx] - baseline
        recovered, _ = nnls(A, y)
        print(f"{t[idx]:>6.0f} {true_conc['red'][idx]:>10.4f} {true_conc['blue'][idx]:>10.4f} "
              f"{recovered[0]:>10.4f} {recovered[1]:>10.4f}")