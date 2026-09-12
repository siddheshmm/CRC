"""
dye_analysis.py

Reusable analysis for food-dye-in-water AS7341 calibration experiments.
Handles both:
  1. SINGLE-DYE calibration runs: fits a per-band linear response curve
     (signal vs. relative dye concentration) and reports R^2, so you can
     see whether each band behaves Beer-Lambert-linearly, and over what
     dynamic range it saturates/bends.
  2. MIXED-DYE unmixing runs: once you have single-dye calibration curves
     for each dye, uses them to try to recover individual dye
     concentrations from a combined multi-band reading -- this is the
     actual test of whether AS7341's 8 bands can resolve simultaneous
     "species" (dyes now, indicators/reagents later).

Dosing was done manually (not pump-logged), so pulse schedules are entered
by hand in the EXPERIMENTS config below, from your dosing-log
screenshots. Everything else (loading, alignment, fitting, unmixing) is
generic and reusable across every future export with the same structure.

USAGE: edit EXPERIMENTS at the bottom for each new run, then just run this
file. It prints calibration tables + R^2, saves calibration-curve plots as
PNGs, and (once mixed-dye CSVs are available) prints unmixing recovery
tables.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dataclasses import dataclass, field
from scipy.optimize import nnls
import os

AS7341_BANDS = [415, 445, 480, 515, 555, 590, 630, 680]

# Output directory for saved calibration plots -- change this to wherever
# you want plots written on your machine. Created automatically if missing.
OUTPUT_DIR = "./dye_analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# Loading
# ============================================================================

def load_as7341_wide(path: str) -> pd.DataFrame:
    """Load an as7341_spectrum_readings CSV (long format: one row per
    band per scan) and pivot to wide format: one row per scan timestamp,
    one column per band."""
    df = pd.read_csv(path)
    df["timestamp_localtime"] = pd.to_datetime(df["timestamp_localtime"])
    wide = df.pivot_table(index="timestamp_localtime", columns="band",
                           values="reading", aggfunc="mean")
    wide = wide.sort_index()
    wide.columns = [int(c) for c in wide.columns]
    return wide


def load_od_wide(path: str) -> pd.DataFrame:
    """Load an od_readings CSV, pivot to wide format keyed by (angle,channel)."""
    df = pd.read_csv(path)
    df["timestamp_localtime"] = pd.to_datetime(df["timestamp_localtime"])
    df["col"] = "od_a" + df["angle"].astype(str) + "_ch" + df["channel"].astype(str)
    wide = df.pivot_table(index="timestamp_localtime", columns="col",
                           values="od_reading", aggfunc="mean")
    return wide.sort_index()


# ============================================================================
# Dosing schedule handling
# ============================================================================

@dataclass
class Pulse:
    dye: str
    volume_mL: float
    timestamp: pd.Timestamp


def build_pulse_schedule(date_str: str, pulses_raw: list[tuple[str, float, str]]) -> list[Pulse]:
    """
    pulses_raw: list of (dye_name, volume_mL, 'H:MM AM/PM') tuples, straight
    from your dosing-log screenshots.
    date_str: 'YYYY-MM-DD' for that experiment.
    """
    pulses = []
    for dye, vol, time_str in pulses_raw:
        ts = pd.to_datetime(f"{date_str} {time_str}")
        pulses.append(Pulse(dye=dye, volume_mL=vol, timestamp=ts))
    return sorted(pulses, key=lambda p: p.timestamp)


def cumulative_volume_trace(pulses: list[Pulse], dye: str | None = None) -> list[tuple[pd.Timestamp, float]]:
    """Returns [(timestamp, cumulative_volume)] step trace. If dye is given,
    only counts that dye's pulses (for per-dye concentration in a mixed run);
    if None, counts every pulse (for total vial volume)."""
    trace = []
    cum = 0.0
    for p in pulses:
        if dye is None or p.dye == dye:
            cum += p.volume_mL
        trace.append((p.timestamp, cum))
    return trace


def volume_at(trace: list[tuple[pd.Timestamp, float]], t: pd.Timestamp) -> float:
    """Step-function lookup: cumulative volume at time t (0 before first pulse)."""
    val = 0.0
    for ts, cum in trace:
        if ts <= t:
            val = cum
        else:
            break
    return val


# ============================================================================
# Pre/post-pulse signal extraction
# ============================================================================

def extract_pulse_readings(as7341_wide: pd.DataFrame, pulses: list[Pulse],
                            start_volume_mL: float, settle_seconds: float = 20,
                            settle_window_seconds: float = 15):
    """
    For each pulse, computes:
      - relative concentration immediately after that pulse (total dye
        volume so far / total vial volume so far)
      - the AS7341 band reading averaged over a settle window starting
        `settle_seconds` after the pulse (giving mixing time to complete)
    Also includes the t=0 (pre-dose) baseline as concentration=0.

    Returns a DataFrame: columns = ['time','conc'] + AS7341_BANDS
    """
    all_pulses_trace = cumulative_volume_trace(pulses, dye=None)
    # assume single dye if only one dye name present; else caller should
    # pass a specific dye filter via cumulative_volume_trace externally
    dye_names = sorted(set(p.dye for p in pulses))
    dye_traces = {d: cumulative_volume_trace(pulses, dye=d) for d in dye_names}

    rows = []
    # baseline (pre-dose)
    t0 = pulses[0].timestamp - pd.Timedelta(seconds=30)
    baseline_window = as7341_wide[(as7341_wide.index >= t0 - pd.Timedelta(seconds=settle_window_seconds)) &
                                   (as7341_wide.index <= t0)]
    if len(baseline_window) > 0:
        row = {"time": t0, "conc": 0.0, "dye_volume": 0.0, "total_volume": start_volume_mL}
        for b in AS7341_BANDS:
            row[b] = baseline_window[b].mean()
        rows.append(row)

    for p in pulses:
        window_start = p.timestamp + pd.Timedelta(seconds=settle_seconds)
        window_end = window_start + pd.Timedelta(seconds=settle_window_seconds)
        window = as7341_wide[(as7341_wide.index >= window_start) & (as7341_wide.index <= window_end)]
        if len(window) == 0:
            continue
        total_vol = start_volume_mL + volume_at(all_pulses_trace, window_start)
        dye_vol = volume_at(dye_traces[p.dye], window_start)
        conc = dye_vol / total_vol
        row = {"time": window_start, "conc": conc, "dye_volume": dye_vol,
               "total_volume": total_vol, "dye": p.dye}
        for b in AS7341_BANDS:
            row[b] = window[b].mean()
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# Automatic change-point detection (robust to imprecise manual dosing times)
# ============================================================================

def detect_step_time(as7341_wide: pd.DataFrame, nominal_time: pd.Timestamp,
                      band: int = 555, search_before_s: float = 15,
                      search_after_s: float = 90, baseline_window_s: float = 25,
                      threshold_sigma: float = 4.0):
    """
    Instead of trusting `nominal_time` (a manually-logged dosing time) as
    exact, this searches a window around it for where the signal ACTUALLY
    steps away from its pre-dose baseline.

    Baseline: mean/std of `band` in the `baseline_window_s` seconds just
    before (nominal_time - search_before_s).
    Search window: from (nominal_time - search_before_s) to
    (nominal_time + search_after_s).
    Returns (detected_time, detected, baseline_mean, baseline_std) --
    `detected` is False if nothing in the search window clears the
    threshold, meaning this pulse produced no resolvable step (too small,
    or masked by an already-saturated signal) -- that's a real finding,
    not a bug, and calibration should flag/exclude such points rather than
    silently using the nominal time anyway.
    """
    baseline_end = nominal_time - pd.Timedelta(seconds=search_before_s)
    baseline_start = baseline_end - pd.Timedelta(seconds=baseline_window_s)
    baseline = as7341_wide[(as7341_wide.index >= baseline_start) & (as7341_wide.index <= baseline_end)][band]
    if len(baseline) < 2:
        return nominal_time, False, None, None
    baseline_mean, baseline_std = baseline.mean(), baseline.std()
    baseline_std = max(baseline_std, 1e-6)  # avoid zero-std false triggers

    search_start = nominal_time - pd.Timedelta(seconds=search_before_s)
    search_end = nominal_time + pd.Timedelta(seconds=search_after_s)
    search = as7341_wide[(as7341_wide.index >= search_start) & (as7341_wide.index <= search_end)][band]

    threshold = threshold_sigma * baseline_std
    deviation = (search - baseline_mean).abs()
    crossed = deviation[deviation > threshold]
    if len(crossed) == 0:
        return nominal_time, False, baseline_mean, baseline_std
    detected_time = crossed.index[0]
    return detected_time, True, baseline_mean, baseline_std


def auto_align_pulses(as7341_wide: pd.DataFrame, pulses: list[Pulse],
                       band: int = 555, **kwargs):
    """
    Runs detect_step_time for every pulse (searching relative to each
    pulse's own logged time), returns a list of dicts with both the
    logged and detected time, plus a `detected` flag. Print a comparison
    table so you can see exactly how far off (and which pulses failed to
    register) before trusting any calibration built on top of it.
    """
    results = []
    for p in pulses:
        det_time, detected, bmean, bstd = detect_step_time(as7341_wide, p.timestamp, band=band, **kwargs)
        offset_s = (det_time - p.timestamp).total_seconds()
        results.append(dict(pulse=p, detected_time=det_time, detected=detected,
                             offset_seconds=offset_s, baseline_mean=bmean, baseline_std=bstd))
    print(f"\n--- Auto step-detection (band={band}nm) ---")
    print(f"{'dye':>6} {'logged_time':>20} {'detected_time':>20} {'offset(s)':>10} {'found?':>7}")
    for r in results:
        print(f"{r['pulse'].dye:>6} {str(r['pulse'].timestamp):>20} {str(r['detected_time']):>20} "
              f"{r['offset_seconds']:>10.1f} {str(r['detected']):>7}")
    n_found = sum(r["detected"] for r in results)
    print(f"{n_found}/{len(results)} pulses produced a detectable step at band {band}nm.")
    return results


def extract_pulse_readings_auto(as7341_wide: pd.DataFrame, pulses: list[Pulse],
                                 start_volume_mL: float, detection_band: int = 555,
                                 settle_after_detection_s: float = 10,
                                 settle_window_s: float = 15, **detect_kwargs):
    """
    Same output shape as extract_pulse_readings(), but uses the DETECTED
    step time (not the logged/nominal time) as the anchor for the settle
    window. Pulses with no detectable step are EXCLUDED from the returned
    calibration points and reported separately -- including them anyway
    (at their nominal time) would just re-introduce the same noise we're
    trying to remove.
    """
    alignment = auto_align_pulses(as7341_wide, pulses, band=detection_band, **detect_kwargs)
    all_pulses_trace = cumulative_volume_trace(pulses, dye=None)
    dye_names = sorted(set(p.dye for p in pulses))
    dye_traces = {d: cumulative_volume_trace(pulses, dye=d) for d in dye_names}

    rows, skipped = [], []
    t0 = pulses[0].timestamp - pd.Timedelta(seconds=45)
    baseline_window = as7341_wide[(as7341_wide.index >= t0 - pd.Timedelta(seconds=settle_window_s)) &
                                   (as7341_wide.index <= t0)]
    if len(baseline_window) > 0:
        row = {"time": t0, "conc": 0.0, "dye_volume": 0.0, "total_volume": start_volume_mL}
        for b in AS7341_BANDS:
            row[b] = baseline_window[b].mean()
        rows.append(row)

    for r in alignment:
        p = r["pulse"]
        if not r["detected"]:
            skipped.append(p)
            continue
        window_start = r["detected_time"] + pd.Timedelta(seconds=settle_after_detection_s)
        window_end = window_start + pd.Timedelta(seconds=settle_window_s)
        window = as7341_wide[(as7341_wide.index >= window_start) & (as7341_wide.index <= window_end)]
        if len(window) == 0:
            skipped.append(p)
            continue
        total_vol = start_volume_mL + volume_at(all_pulses_trace, window_start)
        dye_vol = volume_at(dye_traces[p.dye], window_start)
        conc = dye_vol / total_vol
        row = {"time": window_start, "conc": conc, "dye_volume": dye_vol,
               "total_volume": total_vol, "dye": p.dye}
        for b in AS7341_BANDS:
            row[b] = window[b].mean()
        rows.append(row)

    if skipped:
        print(f"\nExcluded {len(skipped)} pulse(s) with no detectable step from calibration: "
              f"{[p.timestamp.strftime('%H:%M') for p in skipped]}")

    return pd.DataFrame(rows)




def fit_calibration(pulse_readings: pd.DataFrame, baseline_row: pd.Series | None = None):
    """
    Fits signal = slope*conc + intercept per band via least squares.
    Returns dict: band -> (slope, intercept, r_squared)
    """
    results = {}
    conc = pulse_readings["conc"].values
    for b in AS7341_BANDS:
        signal = pulse_readings[b].values
        if len(conc) < 2:
            continue
        A = np.column_stack([conc, np.ones_like(conc)])
        coef, *_ = np.linalg.lstsq(A, signal, rcond=None)
        slope, intercept = coef
        pred = A @ coef
        ss_res = np.sum((signal - pred) ** 2)
        ss_tot = np.sum((signal - signal.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        results[b] = (slope, intercept, r2)
    return results


def print_calibration(results: dict, label: str):
    print(f"\n--- Calibration: {label} ---")
    print(f"{'band(nm)':>10} {'slope':>14} {'intercept':>12} {'R^2':>8}")
    for b in AS7341_BANDS:
        slope, intercept, r2 = results[b]
        flag = "" if r2 > 0.9 else "  <-- weak/nonlinear fit, check saturation or noise"
        print(f"{b:>10} {slope:>14.5f} {intercept:>12.5f} {r2:>8.3f}{flag}")


def plot_calibration(pulse_readings: pd.DataFrame, results: dict, label: str, out_path: str):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    conc = pulse_readings["conc"].values
    for i, b in enumerate(AS7341_BANDS):
        ax = axes[i // 4, i % 4]
        signal = pulse_readings[b].values
        slope, intercept, r2 = results[b]
        ax.scatter(conc, signal, color="tab:blue")
        xs = np.linspace(0, conc.max() * 1.05, 50)
        ax.plot(xs, slope * xs + intercept, color="tab:red", linestyle="--")
        ax.set_title(f"{b}nm  R2={r2:.3f}")
        ax.set_xlabel("relative conc (mL dye / mL total)")
        ax.set_ylabel("AS7341 reading")
    fig.suptitle(f"Calibration curves: {label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved calibration plot -> {out_path}")


def plot_raw_spectrum(as7341_wide: pd.DataFrame, label: str, out_path: str,
                       pulses: list[Pulse] | None = None):
    """
    Plots all 8 AS7341 bands over time on one figure, optionally with
    vertical dashed lines marking logged pulse times (useful for eyeballing
    whether steps actually line up with dosing, same check we did manually
    for red_dye's timing issue).
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(AS7341_BANDS)))
    for band, color in zip(AS7341_BANDS, colors):
        ax.plot(as7341_wide.index, as7341_wide[band], marker='.', markersize=3,
                linewidth=1, color=color, label=f"{band}nm")

    if pulses:
        for p in pulses:
            ax.axvline(p.timestamp, color="red", linestyle="--", alpha=0.4)

    ax.set_title(f"AS7341 raw spectrum over time: {label}")
    ax.set_xlabel("time")
    ax.set_ylabel("AS7341 reading")
    ax.legend(loc="upper left", ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved raw spectrum plot -> {out_path}")


# ============================================================================
# Mixed-dye linear unmixing
# ============================================================================

def unmix_mixed_dye(as7341_wide: pd.DataFrame, mixed_pulses: list[Pulse],
                     start_volume_mL: float, calibrations: dict[str, dict],
                     settle_seconds: float = 20, settle_window_seconds: float = 15,
                     baseline_window_before_s: float = 45):
    """
    calibrations: {dye_name: {band: (slope, intercept, r2)}} from
    fit_calibration() on each dye's single-dye run -- only the SLOPES are
    used here, not the intercepts (see note below).

    For each mixed-dose timepoint, solves the linear system
        (signal_b - baseline_b) = sum_dye(slope_dye_b * conc_dye)
    via BOTH plain least squares and non-negative least squares (NNLS),
    reporting both. Plain least squares has no way to know concentrations
    can't be negative, so if the underlying spectral responses of the two
    dyes are highly correlated (nearly the same shape, just different
    magnitude -- check this with the correlation/condition-number
    diagnostic below), small measurement noise gets amplified into
    unstable, sign-flipping, unphysical results. NNLS constrains the
    solution to be non-negative, which won't fix genuine spectral overlap
    but stops the solver from returning nonsense like -1.0 for a
    concentration.

    baseline_b is THIS experiment's own pre-dose blank reading (not a sum
    of the single-dye runs' fitted intercepts -- see comment history for
    why that was wrong).
    """
    dyes = list(calibrations.keys())
    all_trace = cumulative_volume_trace(mixed_pulses, dye=None)
    dye_traces = {d: cumulative_volume_trace(mixed_pulses, dye=d) for d in dyes}

    # Diagnostic: how separable ARE these dyes, spectrally?
    slope_matrix = np.array([[calibrations[d][b][0] for d in dyes] for b in AS7341_BANDS])
    if len(dyes) == 2:
        corr = np.corrcoef(slope_matrix[:, 0], slope_matrix[:, 1])[0, 1]
        cond = np.linalg.cond(slope_matrix)
        print(f"\n[unmixing diagnostic] slope correlation between {dyes[0]} and {dyes[1]} "
              f"across bands: {corr:.3f}  |  design matrix condition number: {cond:.1f}")
        if corr > 0.95 or cond > 20:
            print("  -> HIGH correlation / condition number: these two dyes' spectral shapes "
                  "are too similar for reliable linear unmixing. Expect unstable results below, "
                  "regardless of solver -- this is a dye-selection limitation, not a fixable bug.")

    # Real baseline: this experiment's own pre-dose reading, not the
    # single-dye runs' fitted intercepts. Falls back to the earliest
    # available readings in the file if logging started too close to (or
    # after) the intended lookback window -- better to use a slightly
    # later baseline than to crash outright.
    t0 = mixed_pulses[0].timestamp - pd.Timedelta(seconds=baseline_window_before_s)
    baseline_window = as7341_wide[(as7341_wide.index >= t0 - pd.Timedelta(seconds=settle_window_seconds)) &
                                   (as7341_wide.index <= t0)]
    if len(baseline_window) == 0:
        print(f"WARNING: no data in the intended baseline window "
              f"({baseline_window_before_s}s before first pulse) -- logging "
              f"likely started too close to dosing. Falling back to the "
              f"earliest available readings in this file as the baseline "
              f"instead. Treat this baseline as less reliable than a proper "
              f"pre-dose blank.")
        baseline_window = as7341_wide.iloc[:max(1, int(settle_window_seconds / 5))]
        if len(baseline_window) == 0:
            raise ValueError("No AS7341 data at all in this file -- check the CSV path/contents.")
        data_start = as7341_wide.index.min()
        if data_start > mixed_pulses[0].timestamp:
            print(f"WARNING: logging actually starts at {data_start}, which is AFTER "
                  f"the first pulse at {mixed_pulses[0].timestamp} -- there is no true "
                  f"pre-dose data in this file at all. This fallback baseline already "
                  f"includes dye and will bias every recovered concentration. Fix the "
                  f"pulse schedule's date/time, or start logging before dosing next time.")
    baseline = {b: baseline_window[b].mean() for b in AS7341_BANDS}

    rows = []
    for p in mixed_pulses:
        window_start = p.timestamp + pd.Timedelta(seconds=settle_seconds)
        window_end = window_start + pd.Timedelta(seconds=settle_window_seconds)
        window = as7341_wide[(as7341_wide.index >= window_start) & (as7341_wide.index <= window_end)]
        if len(window) == 0:
            continue
        total_vol = start_volume_mL + volume_at(all_trace, window_start)
        known_conc = {d: volume_at(dye_traces[d], window_start) / total_vol for d in dyes}

        A_rows, y_rows = [], []
        for b in AS7341_BANDS:
            signal_b = window[b].mean()
            A_rows.append([calibrations[d][b][0] for d in dyes])   # slopes only
            y_rows.append(signal_b - baseline[b])                  # real measured baseline
        A = np.array(A_rows)
        y = np.array(y_rows)

        recovered_lstsq, *_ = np.linalg.lstsq(A, y, rcond=None)
        recovered_nnls, _ = nnls(A, y)

        row = {"time": window_start, "after_pulse_of": p.dye}
        for i, d in enumerate(dyes):
            row[f"known_{d}"] = known_conc[d]
            row[f"lstsq_{d}"] = recovered_lstsq[i]
            row[f"nnls_{d}"] = recovered_nnls[i]
        rows.append(row)

    result_df = pd.DataFrame(rows)
    print("\n--- Mixed-dye unmixing recovery (lstsq = unconstrained, nnls = non-negative) ---")
    print(result_df.to_string(index=False))
    return result_df


# ============================================================================
# OD sanity check (dyes should NOT move OD much -- OD is scattering, not
# absorption; a flat OD trace while AS7341 changes confirms that distinction
# empirically for this setup)
# ============================================================================

def check_od_flatness(od_wide: pd.DataFrame, label: str):
    print(f"\n--- OD flatness check: {label} (dyes absorb, shouldn't scatter) ---")
    for col in od_wide.columns:
        v = od_wide[col].dropna()
        if len(v) < 2:
            continue
        pct_change = (v.max() - v.min()) / (abs(v.mean()) + 1e-9) * 100
        print(f"  {col}: range=[{v.min():.5f}, {v.max():.5f}]  pct_change={pct_change:.1f}%")


# ============================================================================
# EXPERIMENTS CONFIG -- edit this per new export
# ============================================================================

EXPERIMENTS = {
    "red_dye": dict(
        as7341_csv="D:/Chemical RC/data/red_dye/as7341_spectrum_readings-Pulse_experiments-all_units-20260911232952.csv",
        od_csv="D:/Chemical RC/data/red_dye/od_readings-Pulse_experiments-all_units-20260911232952.csv",
        start_volume_mL=10.0,
        date="2026-09-11",
        pulses_raw=[
            ("red", 0.25, "9:41 PM"),
            ("red", 0.50, "9:44 PM"),
            ("red", 1.00, "9:47 PM"),
            ("red", 1.25, "9:50 PM"),
            ("red", 1.25, "9:53 PM"),
            ("red", 1.50, "9:56 PM"),
        ],
    ),
    "blue_dye": dict(
        as7341_csv="D:/Chemical RC/data/blue_dye/as7341_spectrum_readings-Pulse_experiments-all_units-20260911232246.csv",
        od_csv="D:/Chemical RC/data/blue_dye/od_readings-Pulse_experiments-all_units-20260911232246.csv",
        start_volume_mL=10.0,
        date="2026-09-11",
        # NOTE: source screenshot labeled these "AM" but the run started at
        # 10:10 PM and pulses follow sequentially after -- corrected to PM.
        pulses_raw=[
            ("blue", 0.25, "10:13 PM"),
            ("blue", 0.50, "10:16 PM"),
            ("blue", 1.00, "10:19 PM"),
            ("blue", 1.25, "10:22 PM"),
            ("blue", 1.25, "10:25 PM"),
            ("blue", 1.50, "10:28 PM"),
        ],
    ),
    "mixed_dye": dict(
        as7341_csv="D:/Chemical RC/data/mixed_dye/as7341_spectrum_readings-Pulse_experiments-all_units-20260912015332.csv",
        od_csv="D:/Chemical RC/data/mixed_dye/od_readings-Pulse_experiments-all_units-20260912015332.csv",
        start_volume_mL=10.0,
        date="2026-09-11",
        pulses_raw=[
            ("blue", 1.0, "11:30 PM"),
            ("red", 1.0, "11:35 PM"),
        ],
    ),
    "mixed_dye_alt_1": dict(
        as7341_csv="D:/Chemical RC/data/mixed_dye_alt_1/as7341_spectrum_readings-Pulse_experiments-all_units-20260912015842.csv",
        od_csv="D:/Chemical RC/data/mixed_dye_alt_1/od_readings-Pulse_experiments-all_units-20260912015842.csv",
        start_volume_mL=10.0,
        date="2026-09-12",
        pulses_raw=[
            ("red", 2.0, "1:05 AM"),
            ("blue", 1.0, "1:10 AM"),
        ],
    ),
    "mixed_dye_alt_2": dict(
        as7341_csv="D:/Chemical RC/data/mixed_dye_alt_2/as7341_spectrum_readings-Pulse_experiments-all_units-20260912020332.csv",
        od_csv="D:/Chemical RC/data/mixed_dye_alt_2/od_readings-Pulse_experiments-all_units-20260912020332.csv",
        start_volume_mL=10.0,
        date="2026-09-12",
        pulses_raw=[
            ("blue", 2.0, "1:25 AM"),
            ("red", 1.0, "1:30 AM"),
        ],
    ),
    "mixed_dye_pseudorandom": dict(
        as7341_csv="D:/Chemical RC/data/mixed_dye_pseudorandom/as7341_spectrum_readings-Pulse_experiments-all_units-20260912020837.csv",
        od_csv="D:/Chemical RC/data/mixed_dye_pseudorandom/od_readings-Pulse_experiments-all_units-20260912020837.csv",
        start_volume_mL=10.0,
        date="2026-09-12",
        pulses_raw=[
            ("red", 1.0, "12:33 AM"), ("blue", 1.0, "12:36 AM"),
            ("red", 1.0, "12:39 AM"), ("blue", 1.0, "12:42 AM"),
            ("red", 1.0, "12:45 AM"), ("blue", 1.0, "12:48 AM"),
            ("red", 1.0, "12:51 AM"), ("blue", 1.0, "12:54 AM"),
        ],
    ),
}


if __name__ == "__main__":
    calibrations = {}

    # --- Step 0: raw spectrum plot for every experiment that has a CSV ---
    for name, cfg in EXPERIMENTS.items():
        if cfg.get("as7341_csv") is None:
            continue
        as7341 = load_as7341_wide(cfg["as7341_csv"])
        pulses = build_pulse_schedule(cfg["date"], cfg["pulses_raw"])
        plot_raw_spectrum(as7341, name, os.path.join(OUTPUT_DIR, f"{name}_raw_spectrum.png"), pulses=pulses)

    # --- Step 1: fit single-dye calibrations, using AUTO-DETECTED step times
    # instead of trusting the logged (manual) dosing times exactly ---
    for name in ["red_dye", "blue_dye"]:
        cfg = EXPERIMENTS[name]
        as7341 = load_as7341_wide(cfg["as7341_csv"])
        pulses = build_pulse_schedule(cfg["date"], cfg["pulses_raw"])

        readings_naive = extract_pulse_readings(as7341, pulses, cfg["start_volume_mL"])
        results_naive = fit_calibration(readings_naive)
        print_calibration(results_naive, f"{name} (naive: logged times)")

        readings_auto = extract_pulse_readings_auto(as7341, pulses, cfg["start_volume_mL"])
        results_auto = fit_calibration(readings_auto)
        print_calibration(results_auto, f"{name} (auto-detected step times)")
        plot_calibration(readings_auto, results_auto, f"{name}_auto",
                          os.path.join(OUTPUT_DIR, f"{name}_calibration_auto.png"))
        calibrations[cfg["pulses_raw"][0][0]] = results_auto  # keyed by dye name ("red"/"blue")

        od = load_od_wide(cfg["od_csv"])
        check_od_flatness(od, name)

    # --- Step 2: run mixed-dye unmixing for any experiment with CSVs filled in ---
    for name, cfg in EXPERIMENTS.items():
        if "mixed" not in name or cfg["as7341_csv"] is None:
            continue
        as7341 = load_as7341_wide(cfg["as7341_csv"])
        pulses = build_pulse_schedule(cfg["date"], cfg["pulses_raw"])
        unmix_mixed_dye(as7341, pulses, cfg["start_volume_mL"], calibrations)