"""
continuous_flow_analysis.py

Analyzes a continuous-flow (CSTR-style) experiment: constant water inflow
(add_media) + outflow (remove_waste) to hold vial volume near a setpoint,
with red dye pulses (add_alt_media) dosed at a few points in time. Answers
two questions directly:

  1. Does the AS7341 signal stay flat/return to baseline under pure water
     flow (no dye) -- i.e. is continuous flow alone optically silent?
  2. Does each dye pulse cause a detectable spectral shift, and does the
     signal wash back out over time as the continuous flow dilutes the
     dye away -- consistent with the REAL dilution rate computed from the
     logged pump volumes (not an assumed one)?

Unlike the earlier manual dosing runs, dosing_events.csv here is pump-
logged with exact timestamps -- no auto-detection/change-point workaround
needed, we can anchor directly on the real event times.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

AS7341_BANDS = [415, 445, 480, 515, 555, 590, 630, 680]
OUTPUT_DIR = "./data/14th septcontinuous_flow_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# Loading
# ============================================================================

def load_as7341_wide(path: str) -> pd.DataFrame:
    """
    Loads and pivots to wide format, one row per SCAN (not per exact
    timestamp). Each band within one scan is logged a few milliseconds
    apart, not simultaneously -- pivoting on exact timestamp leaves most
    rows with only one non-NaN band, which is fine for column-wise
    aggregates like .mean() (NaN-safe, used correctly throughout the
    calibration work) but breaks anything needing all 8 bands aligned in
    the same row (e.g. band-to-band or band-to-volume correlation).
    Rounding to the nearest 200ms bucket groups each scan's 8 near-
    simultaneous readings into one row instead.
    """
    df = pd.read_csv(path)
    df["timestamp_localtime"] = pd.to_datetime(df["timestamp_localtime"])
    df["scan_bucket"] = df["timestamp_localtime"].dt.round("200ms")
    wide = df.pivot_table(index="scan_bucket", columns="band",
                           values="reading", aggfunc="mean")
    wide = wide.sort_index()
    wide.columns = [int(c) for c in wide.columns]
    wide.index.name = "timestamp_localtime"
    return wide


def load_dosing_events(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp_localtime"] = pd.to_datetime(df["timestamp_localtime"])
    return df.sort_values("timestamp_localtime")


def compute_volume_trace(dosing_df: pd.DataFrame, start_volume_mL: float = 10.0) -> pd.Series:
    """Real cumulative vial volume over time from logged pump events --
    add_media and add_alt_media increase it, remove_waste decreases it.
    This is exact (pump-logged), not the volume-from-images estimate used
    in the earlier manual dosing runs."""
    df = dosing_df.copy()
    sign = df["event"].map({"add_media": 1, "add_alt_media": 1, "remove_waste": -1}).fillna(0)
    delta = sign * df["volume_change_ml"]
    volume = start_volume_mL + delta.cumsum()
    return pd.Series(volume.values, index=df["timestamp_localtime"].values)


def cluster_dye_doses(dosing_df: pd.DataFrame, gap_seconds: float = 5.0):
    """Groups add_alt_media sub-pulses that occur within `gap_seconds` of
    each other into a single logical dosing event (the pump often splits
    one real dose into several rapid sub-pulses)."""
    alt = dosing_df[dosing_df["event"] == "add_alt_media"].sort_values("timestamp_localtime")
    if len(alt) == 0:
        return []
    doses = []
    current_group = [alt.iloc[0]]
    for i in range(1, len(alt)):
        gap = (alt.iloc[i]["timestamp_localtime"] - alt.iloc[i - 1]["timestamp_localtime"]).total_seconds()
        if gap <= gap_seconds:
            current_group.append(alt.iloc[i])
        else:
            doses.append(current_group)
            current_group = [alt.iloc[i]]
    doses.append(current_group)

    result = []
    for group in doses:
        total_vol = sum(row["volume_change_ml"] for row in group)
        start_time = group[0]["timestamp_localtime"]
        end_time = group[-1]["timestamp_localtime"]
        result.append(dict(start_time=start_time, end_time=end_time, total_volume_mL=total_vol,
                            n_subpulses=len(group)))
    return result


# ============================================================================
# Baseline stability check (pure water flow, before any dye)
# ============================================================================

def check_baseline_stability(as7341_wide: pd.DataFrame, before_time: pd.Timestamp):
    """Checks how flat the AS7341 signal is under pure water flow (no dye
    yet) -- answers 'does water alone stay stable' before we even look at
    dye effects."""
    window = as7341_wide[as7341_wide.index < before_time]
    print(f"\n--- Baseline stability check (pure water flow, before first dye dose at {before_time}) ---")
    print(f"  ({len(window)} readings in this window)")
    for b in AS7341_BANDS:
        v = window[b]
        pct_range = (v.max() - v.min()) / (abs(v.mean()) + 1e-9) * 100
        print(f"  {b}nm: mean={v.mean():.4f}  range=[{v.min():.4f}, {v.max():.4f}]  pct_range={pct_range:.1f}%")
    return window


# ============================================================================
# Volume-confound correction
# ============================================================================
# CRITICAL: even with pure water and NO dye, AS7341 signal correlates with
# vial volume at -0.73 to -0.83 across every band (checked directly against
# this experiment's own pre-dose water-only window). Any "washout" analysis
# on raw signal is mostly tracking this volume effect, not dye dilution.
# Fix: fit a per-band linear model of signal ~ volume during the pure-water
# (pre-dose) phase, then subtract that model's prediction from the observed
# signal everywhere to get a volume-corrected residual that should isolate
# the dye-specific effect.

def fit_volume_baseline_model(as7341_wide: pd.DataFrame, volume_trace: pd.Series,
                               before_time: pd.Timestamp):
    """Fits signal = a + b*volume per band, using ONLY the pure-water
    (pre-dye) window, so the model captures the volume artifact without
    any dye contamination."""
    water_only = as7341_wide[as7341_wide.index < before_time].dropna()
    vol_df = volume_trace.reset_index()
    vol_df.columns = ["t", "volume"]
    vol_df["t"] = pd.to_datetime(vol_df["t"])
    merged = pd.merge_asof(water_only.reset_index().rename(columns={"timestamp_localtime": "t"}).sort_values("t"),
                            vol_df.sort_values("t"), on="t", direction="nearest")

    models = {}
    print(f"\n--- Volume-confound model (fit on pure-water window only, n={len(merged)}) ---")
    for b in AS7341_BANDS:
        A = np.column_stack([merged["volume"], np.ones(len(merged))])
        coef, *_ = np.linalg.lstsq(A, merged[b], rcond=None)
        slope, intercept = coef
        pred = A @ coef
        r2 = 1 - np.sum((merged[b] - pred) ** 2) / np.sum((merged[b] - merged[b].mean()) ** 2)
        models[b] = (slope, intercept, r2)
        print(f"  {b}nm: signal = {intercept:.4f} + {slope:.5f}*volume   R^2={r2:.3f}")
    return models


def compute_volume_corrected_residual(as7341_wide: pd.DataFrame, volume_trace: pd.Series,
                                       volume_models: dict) -> pd.DataFrame:
    """Subtracts the volume model's prediction from observed signal at
    every timepoint -- what's left is (hopefully) the dye-specific effect,
    with the volume artifact removed."""
    vol_df = volume_trace.reset_index()
    vol_df.columns = ["t", "volume"]
    vol_df["t"] = pd.to_datetime(vol_df["t"])
    merged = pd.merge_asof(as7341_wide.reset_index().rename(columns={"timestamp_localtime": "t"}).sort_values("t"),
                            vol_df.sort_values("t"), on="t", direction="nearest")
    residual = pd.DataFrame({"t": merged["t"]})
    for b in AS7341_BANDS:
        slope, intercept, _ = volume_models[b]
        predicted = intercept + slope * merged["volume"]
        residual[b] = merged[b] - predicted
    residual = residual.set_index("t")
    residual.index.name = "timestamp_localtime"
    return residual


# ============================================================================
# Per-dose spectral shift + washout timing
# ============================================================================

def analyze_dose_response(as7341_wide: pd.DataFrame, dose: dict, dilution_rate_per_s: float,
                           pre_window_s: float = 30, post_window_s: float = 240,
                           return_tolerance_sigma: float = 2.0):
    """
    For one dye dose:
      - pre-dose baseline (mean + std) per band, from `pre_window_s` before the dose
      - peak deviation from baseline per band, in the post_window_s after the dose
      - time to return within `return_tolerance_sigma` baseline std of the
        pre-dose baseline, per band (None if it never returns within the window)
      - compares empirical return time to the THEORETICAL washout time
        constant from the real dilution rate (1/D), so you can judge
        whether the observed washout matches CSTR physics
    """
    t0 = dose["start_time"]
    baseline_window = as7341_wide[(as7341_wide.index >= t0 - pd.Timedelta(seconds=pre_window_s)) &
                                   (as7341_wide.index < t0)]
    post_window = as7341_wide[(as7341_wide.index >= t0) &
                               (as7341_wide.index <= t0 + pd.Timedelta(seconds=post_window_s))]

    print(f"\n--- Dose at {t0} ({dose['total_volume_mL']:.3f} mL red dye) ---")
    theoretical_tau_s = 1 / dilution_rate_per_s
    print(f"  Theoretical washout time constant (1/D from real flow rate): {theoretical_tau_s:.1f}s "
          f"(~{theoretical_tau_s/60:.2f} min)")

    results = {}
    for b in AS7341_BANDS:
        base_mean = baseline_window[b].mean()
        base_std = baseline_window[b].std()
        post = post_window[b]
        deviation = post - base_mean
        peak_idx = deviation.abs().idxmax()
        peak_dev = deviation.loc[peak_idx]
        peak_time_s = (peak_idx - t0).total_seconds()

        tolerance = max(return_tolerance_sigma * base_std, 1e-6)
        after_peak = post[post.index >= peak_idx]
        returned = after_peak[(after_peak - base_mean).abs() <= tolerance]
        return_time_s = (returned.index[0] - t0).total_seconds() if len(returned) > 0 else None

        results[b] = dict(baseline=base_mean, peak_deviation=peak_dev, peak_time_s=peak_time_s,
                           return_time_s=return_time_s)
        return_str = f"{return_time_s:.1f}s" if return_time_s is not None else "did NOT return within window"
        print(f"  {b}nm: baseline={base_mean:.4f}  peak_dev={peak_dev:+.4f} at t+{peak_time_s:.1f}s  "
              f"returned to baseline at t+{return_str}")
    return results


def plot_dose_response(as7341_wide: pd.DataFrame, dose: dict, out_path: str,
                        pre_window_s: float = 60, post_window_s: float = 300):
    t0 = dose["start_time"]
    window = as7341_wide[(as7341_wide.index >= t0 - pd.Timedelta(seconds=pre_window_s)) &
                          (as7341_wide.index <= t0 + pd.Timedelta(seconds=post_window_s))]
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(AS7341_BANDS)))
    for band, color in zip(AS7341_BANDS, colors):
        ax.plot(window.index, window[band], marker='.', markersize=3, color=color, label=f"{band}nm")
    ax.axvline(t0, color="red", linestyle="--", label="dose")
    ax.set_title(f"Dose response: {dose['total_volume_mL']:.3f}mL red dye at {t0}")
    ax.set_xlabel("time")
    ax.set_ylabel("AS7341 reading")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved dose response plot -> {out_path}")


def plot_full_overview(as7341_wide: pd.DataFrame, doses: list, volume_trace: pd.Series, out_path: str):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(AS7341_BANDS)))
    for band, color in zip(AS7341_BANDS, colors):
        axes[0].plot(as7341_wide.index, as7341_wide[band], marker='.', markersize=2,
                     linewidth=0.8, color=color, label=f"{band}nm")
    for dose in doses:
        axes[0].axvline(dose["start_time"], color="red", linestyle="--", alpha=0.6)
    axes[0].set_title("Full experiment: AS7341 spectrum with dye dose markers")
    axes[0].set_ylabel("AS7341 reading")
    axes[0].legend(loc="upper left", ncol=4, fontsize=8)

    axes[1].plot(volume_trace.index, volume_trace.values, color="tab:blue")
    for dose in doses:
        axes[1].axvline(dose["start_time"], color="red", linestyle="--", alpha=0.6)
    axes[1].set_ylabel("vial volume (mL)")
    axes[1].set_xlabel("time")
    axes[1].set_title("Vial volume over time (from logged pump events)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved full overview plot -> {out_path}")


if __name__ == "__main__":
    AS7341_CSV = "./data/14th sept/as7341_spectrum_readings-Pulse_experiments-all_units-20260914234210.csv"
    DOSING_CSV = "./data/14th sept/dosing_events-Pulse_experiments-all_units-20260914234210.csv"
    START_VOLUME_ML = 10.0

    as7341 = load_as7341_wide(AS7341_CSV)
    dosing = load_dosing_events(DOSING_CSV)
    volume_trace = compute_volume_trace(dosing, START_VOLUME_ML)
    doses = cluster_dye_doses(dosing)

    print(f"Found {len(doses)} dye dosing event(s):")
    for d in doses:
        print(f"  {d['start_time']}  total={d['total_volume_mL']:.3f}mL  ({d['n_subpulses']} sub-pulses)")

    # Real dilution rate from logged flow, not assumed
    add_media = dosing[dosing["event"] == "add_media"]
    duration_s = (dosing["timestamp_localtime"].max() - dosing["timestamp_localtime"].min()).total_seconds()
    flow_rate_mL_per_s = add_media["volume_change_ml"].sum() / duration_s
    dilution_rate_per_s = flow_rate_mL_per_s / START_VOLUME_ML
    print(f"\nReal flow rate: {flow_rate_mL_per_s*60:.2f} mL/min  "
          f"-> dilution rate D={dilution_rate_per_s:.5f}/s  -> residence time={1/dilution_rate_per_s:.1f}s")

    # 1. Baseline stability check (pure water, before first dose)
    check_baseline_stability(as7341, doses[0]["start_time"])

    # 2. Volume-confound correction -- fit on pure-water window, apply everywhere
    volume_models = fit_volume_baseline_model(as7341, volume_trace, doses[0]["start_time"])
    residual = compute_volume_corrected_residual(as7341, volume_trace, volume_models)

    # 3. Per-dose spectral shift + washout timing -- on the CORRECTED residual,
    # not raw signal, so this reflects dye effect with the volume artifact removed
    print("\n\n========== ANALYSIS ON VOLUME-CORRECTED RESIDUAL (dye effect only) ==========")
    for dose in doses:
        analyze_dose_response(residual, dose, dilution_rate_per_s)

    # 4. Plots (raw, for reference)
    plot_full_overview(as7341, doses, volume_trace, os.path.join(OUTPUT_DIR, "full_overview.png"))
    for i, dose in enumerate(doses):
        plot_dose_response(as7341, dose, os.path.join(OUTPUT_DIR, f"dose_{i+1}_response_raw.png"))
        plot_dose_response(residual, dose, os.path.join(OUTPUT_DIR, f"dose_{i+1}_response_corrected.png"))
