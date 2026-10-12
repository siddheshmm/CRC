#!/usr/bin/env python3
"""
Dye calibration analysis — Protocol 3 (fixed-volume optical calibration)

What this script does:
  1. Reads the run-log spreadsheet (fill/record timestamps for each vial).
  2. Reads the AS7341 spectrometer export (long format: one row per band
     per scan) and pivots it to one row per scan, one column per band.
  3. For every logged vial, extracts the scans that fall inside its
     [record_start, record_end) window and reduces them to a per-band
     mean + SD + sample count.
  4. Builds a local (time-interpolated) blank for every sample from the
     interleaved blank vials, and computes apparent absorbance
     A = -log10(I_sample / I_local_blank) per band.
  5. Aggregates the 3 independent replicates per concentration.
  6. Plots:
       - average raw spectrum per concentration (all 8 bands)
       - blank stability over the run (drift check)
       - calibration curve (raw intensity AND apparent absorbance) per
         band, with replicate error bars, linear fit and saturating fit
         overlaid
  7. Fits linear vs. a saturating (exponential-approach) model per band
     on the 9 concentration-mean points, and compares them with
     leave-one-concentration-out cross-validation (not R², which is
     uninformative with 9 points).
  8. Prints a plain-language "is this calibration good to go" summary:
     dynamic range, replicate CV%, blank drift, and which model/band
     the data actually supports.

Usage:
    python analyze_calibration.py \
        --run-log dye_calibration_run_log.xlsx \
        --spectra as7341_spectrum_readings.csv \
        --outdir calibration_analysis_output

If you don't pass --run-log / --spectra, the script looks for files
matching those patterns in the current directory.
"""

import argparse
import glob
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=RuntimeWarning)

BANDS = [415, 445, 480, 515, 555, 590, 630, 680]
MIN_SAMPLES_PER_WINDOW = 5   # below this, flag the row as too sparse to trust


# --------------------------------------------------------------------------
# 1. Load the run log
# --------------------------------------------------------------------------

def _parse_logged_time(val, ref_date):
    """Run-log times are entered like '12.05 PM' or '1.00 PM' (dot instead
    of colon). Convert to a real timestamp on ref_date."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return pd.NaT
    if isinstance(val, pd.Timestamp):
        # openpyxl sometimes returns a datetime.time or datetime object
        return pd.Timestamp.combine(ref_date, val.time())
    s = str(val).strip()
    if not s:
        return pd.NaT
    s = re.sub(r"(\d+)\.(\d+)", r"\1:\2", s)  # "12.05 PM" -> "12:05 PM"
    try:
        t = pd.to_datetime(s, format="%I:%M %p").time()
    except ValueError:
        try:
            t = pd.to_datetime(s).time()
        except Exception:
            return pd.NaT
    return pd.Timestamp.combine(ref_date, t)


def load_run_log(path, ref_date):
    df = pd.read_excel(path, sheet_name="Run Log", header=0)
    df = df[df["order"].apply(lambda v: isinstance(v, (int, float)) and not pd.isna(v))]
    df = df[df["order"] != "EX"]
    df["order"] = df["order"].astype(int)

    for col in ("fill_time", "record_start", "record_end"):
        df[col] = df[col].apply(lambda v: _parse_logged_time(v, ref_date))

    df["fraction"] = pd.to_numeric(df["fraction"], errors="coerce")
    df["replicate"] = pd.to_numeric(df["replicate"], errors="coerce")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. Load and pivot the spectrometer export
# --------------------------------------------------------------------------

def load_spectra(path):
    df = pd.read_csv(path)
    df["t"] = pd.to_datetime(df["timestamp_localtime"])
    wide = df.pivot_table(index="t", columns="band", values="reading", aggfunc="mean")
    wide = wide.reindex(columns=BANDS)
    wide = wide.sort_index()
    return wide


# --------------------------------------------------------------------------
# 3. Extract per-window stats
# --------------------------------------------------------------------------

def window_stats(spectra, start, end):
    if pd.isna(start) or pd.isna(end):
        return None
    sub = spectra.loc[(spectra.index >= start) & (spectra.index < end)]
    n = len(sub)
    if n == 0:
        return {"n_samples": 0, **{f"{b}_mean": np.nan for b in BANDS}, **{f"{b}_sd": np.nan for b in BANDS}}
    out = {"n_samples": n}
    for b in BANDS:
        out[f"{b}_mean"] = sub[b].mean()
        out[f"{b}_sd"] = sub[b].std()
    return out


def build_measurements(run_log, spectra):
    rows = []
    for _, r in run_log.iterrows():
        stats = window_stats(spectra, r["record_start"], r["record_end"])
        row = {
            "order": r["order"],
            "type": r["type"],
            "fraction": r["fraction"],
            "replicate": r["replicate"],
            "record_start": r["record_start"],
            "record_end": r["record_end"],
        }
        row.update(stats)
        rows.append(row)
    meas = pd.DataFrame(rows)
    meas["window_midpoint"] = meas["record_start"] + (meas["record_end"] - meas["record_start"]) / 2
    return meas


# --------------------------------------------------------------------------
# 4. Local blank interpolation + apparent absorbance
# --------------------------------------------------------------------------

def add_local_blank_and_absorbance(meas):
    blanks = meas[meas["type"] == "blank"].sort_values("window_midpoint").reset_index(drop=True)
    if blanks.empty:
        raise RuntimeError("No blank rows found/matched — cannot compute local blanks.")

    blank_t = blanks["window_midpoint"].values.astype("datetime64[ns]").astype(np.int64)

    for b in BANDS:
        col = f"{b}_mean"
        blank_vals = blanks[col].values
        valid = ~np.isnan(blank_vals)
        if valid.sum() < 1:
            meas[f"{b}_blank_local"] = np.nan
            continue
        meas[f"{b}_blank_local"] = np.interp(
            meas["window_midpoint"].values.astype("datetime64[ns]").astype(np.int64),
            blank_t[valid], blank_vals[valid],
        )

    samples = meas[meas["type"] == "sample"].copy()
    for b in BANDS:
        I = samples[f"{b}_mean"]
        I_blank = samples[f"{b}_blank_local"]
        ratio = I / I_blank
        samples[f"{b}_ratio"] = ratio
        with np.errstate(invalid="ignore", divide="ignore"):
            samples[f"{b}_A"] = -np.log10(ratio)
    return meas, blanks, samples


# --------------------------------------------------------------------------
# 5. Replicate aggregation
# --------------------------------------------------------------------------

def aggregate_by_fraction(samples):
    usable = samples[samples["n_samples"] > 0]
    agg_rows = []
    for frac, g in usable.groupby("fraction"):
        row = {"fraction": frac, "n_replicates": len(g)}
        for b in BANDS:
            row[f"{b}_I_mean"] = g[f"{b}_mean"].mean()
            row[f"{b}_I_sd"] = g[f"{b}_mean"].std()
            row[f"{b}_A_mean"] = g[f"{b}_A"].mean()
            row[f"{b}_A_sd"] = g[f"{b}_A"].std()
            mean_I = row[f"{b}_I_mean"]
            row[f"{b}_I_cv_pct"] = (row[f"{b}_I_sd"] / mean_I * 100) if mean_I else np.nan
        agg_rows.append(row)
    out = pd.DataFrame(agg_rows).sort_values("fraction").reset_index(drop=True)
    short = out[out["n_replicates"] < 3]
    if len(short):
        print(f"\n[!] Fraction(s) with fewer than 3 usable replicates (missing data reduced n): "
              + ", ".join(f"{r.fraction:.2f} (n={int(r.n_replicates)})" for r in short.itertuples()))
    return out


# --------------------------------------------------------------------------
# 6. Models + leave-one-concentration-out CV
# --------------------------------------------------------------------------

def detector_diagnostics(spectra_long, spectra_wide):
    """Protocol clause: 'Record detector saturation/background settings; do not
    subtract an initial dark correction twice.'

    The Pioreactor export gives normalised readings, not raw ADC counts. This
    infers the underlying quantisation so we can report where each band sits
    relative to the ADC full scale, and check nothing is clipping or has been
    background-subtracted to a floor."""
    vals = spectra_long["reading"].values
    uniq = np.unique(vals)
    gaps = np.diff(np.sort(uniq))
    gaps = gaps[gaps > 0]
    q = np.min(gaps) if len(gaps) else np.nan          # quantisation step
    full_scale = (1.0 / q) if (q and not np.isnan(q)) else np.nan

    rows = []
    for b in BANDS:
        s = spectra_long[spectra_long["band"] == b]["reading"]
        counts_max = s.max() / q if q else np.nan
        counts_min = s.min() / q if q else np.nan
        rows.append({
            "band": b,
            "max_reading": s.max(),
            "min_reading": s.min(),
            "implied_counts_max": counts_max,
            "implied_counts_min": counts_min,
            "pct_of_full_scale_at_max": counts_max / full_scale * 100 if full_scale else np.nan,
            "n_at_ceiling": int((s >= s.max() * 0.999).sum()),
            "n_at_zero": int((s <= 0).sum()),
        })
    return pd.DataFrame(rows), q, full_scale


def carryover_test(meas):
    """Protocol clause: 'interleave water blanks to check carryover/drift'.
    Drift is handled by the blank-stability plot. This covers CARRYOVER:
    if rinsing is inadequate, a blank measured right after a high-concentration
    vial will read systematically different from one after a low-concentration
    vial. Tests blank reading against the concentration of the preceding sample."""
    m = meas.sort_values("order").reset_index(drop=True)
    rows = []
    for i, r in m.iterrows():
        if r["type"] != "blank":
            continue
        prev = m.iloc[:i]
        prev_samples = prev[prev["type"] == "sample"]
        if prev_samples.empty:
            continue
        prev_frac = prev_samples.iloc[-1]["fraction"]
        if r["n_samples"] == 0 or pd.isna(prev_frac):
            continue
        row = {"order": r["order"], "preceding_fraction": prev_frac}
        for b in BANDS:
            row[f"{b}"] = r[f"{b}_mean"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df) < 3:
        return df, pd.DataFrame()

    # CONFOUND CHECK: if the randomisation happened to place higher
    # concentrations later in the session, then "blank vs preceding
    # concentration" is indistinguishable from "blank vs time" (drift).
    order_conc_r = np.corrcoef(df["order"].values.astype(float),
                               df["preceding_fraction"].values.astype(float))[0, 1]

    results = []
    for b in BANDS:
        x = df["preceding_fraction"].values.astype(float)
        y = df[f"{b}"].values.astype(float)
        ok = ~np.isnan(y)
        if ok.sum() < 3 or np.ptp(x[ok]) == 0:
            continue
        slope, intercept = np.polyfit(x[ok], y[ok], 1)
        r = np.corrcoef(x[ok], y[ok])[0, 1]
        # partial correlation of blank vs preceding conc, controlling for order
        o = df["order"].values.astype(float)[ok]
        xx, yy, oo = x[ok], y[ok], o
        def resid(a, b_):
            s = np.polyfit(b_, a, 1)
            return a - np.polyval(s, b_)
        r_partial = np.corrcoef(resid(xx, oo), resid(yy, oo))[0, 1] if np.ptp(oo) > 0 else np.nan
        results.append({
            "band": b,
            "slope_vs_prev_conc": slope,
            "pearson_r": r,
            "partial_r_controlling_order": r_partial,
            "blank_mean": y[ok].mean(),
            "slope_as_pct_of_blank": slope / y[ok].mean() * 100 if y[ok].mean() else np.nan,
        })
    res = pd.DataFrame(results)
    res.attrs["order_conc_r"] = order_conc_r
    return df, res


def permutation_signal_test(samples, response="A", n_perm=5000, seed=0):
    """Is there ANY concentration-dependent signal, given replicate noise?

    One-way F-like statistic (between-concentration variance / within-
    concentration variance), with a permutation null built by shuffling the
    concentration labels across vials. Benjamini-Hochberg FDR across the 8
    bands. This is the honest 'is there a curve at all' test — far more
    informative than R² on 9 points."""
    rng = np.random.default_rng(seed)
    usable = samples[samples["n_samples"] > 0]
    out = []
    for b in BANDS:
        col = f"{b}_A" if response == "A" else f"{b}_mean"
        sub = usable[["fraction", col]].dropna()
        if sub["fraction"].nunique() < 3:
            continue
        y = sub[col].values.astype(float)
        g = sub["fraction"].values.astype(float)

        def stat(y_, g_):
            grand = y_.mean()
            ssb, ssw = 0.0, 0.0
            for lev in np.unique(g_):
                yy = y_[g_ == lev]
                ssb += len(yy) * (yy.mean() - grand) ** 2
                ssw += ((yy - yy.mean()) ** 2).sum()
            return ssb / ssw if ssw > 0 else np.inf

        obs = stat(y, g)
        null = np.array([stat(y, rng.permutation(g)) for _ in range(n_perm)])
        p = (np.sum(null >= obs) + 1) / (n_perm + 1)
        out.append({"band": b, "F_like_stat": obs, "p_perm": p})

    df = pd.DataFrame(out)
    if df.empty:
        return df
    # Benjamini-Hochberg
    df = df.sort_values("p_perm").reset_index(drop=True)
    m = len(df)
    df["p_bh"] = np.minimum.accumulate(
        (df["p_perm"].values * m / (np.arange(m) + 1))[::-1]
    )[::-1]
    df["p_bh"] = df["p_bh"].clip(upper=1.0)
    df["significant_FDR_0.05"] = df["p_bh"] < 0.05
    return df


def analyse_stability_control(spectra, start, end, outdir, label="stability"):
    """Protocol clause: 20-minute stability monitor on a 10% stock-fraction
    solution, pumps off, fixed light/stirring. Reports per-band drift over the
    window. Note: drift here could be optical drift, adsorption OR photochemical
    change — this does not identify the mechanism by itself."""
    sub = spectra.loc[(spectra.index >= start) & (spectra.index < end)]
    if sub.empty:
        print(f"[!] No scans found in stability window {start} .. {end}")
        return None
    t_min = (sub.index - sub.index[0]).total_seconds() / 60.0

    fig, ax = plt.subplots(figsize=(9, 5.5))
    rows = []
    for b in BANDS:
        y = sub[b].values
        ax.plot(t_min, y, label=f"{b} nm")
        slope = np.polyfit(t_min, y, 1)[0]
        rows.append({
            "band": b,
            "start_value": y[0],
            "end_value": y[-1],
            "total_change_pct": (y[-1] - y[0]) / y[0] * 100 if y[0] else np.nan,
            "slope_per_min": slope,
            "slope_pct_per_min": slope / y.mean() * 100 if y.mean() else np.nan,
        })
    ax.set_xlabel("Minutes into stability window")
    ax.set_ylabel("Reading")
    ax.set_title("Stability control — 10% stock fraction, pumps off")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"04_{label}_control.png"), dpi=150)
    plt.close(fig)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(outdir, f"{label}_control.csv"), index=False)
    print("\n--- Stability control (20 min, 10% fraction) ---")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("NOTE: drift here may be optical drift, adsorption or photochemical")
    print("      change — this control does not distinguish the mechanism.")
    return res


def linear_model(c, a, b):
    return a + b * c


def saturating_model(c, y0, amp, tau):
    tau = max(tau, 1e-6)
    return y0 + amp * (1 - np.exp(-c / tau))


def fit_model(model_fn, c, y, p0):
    try:
        popt, _ = curve_fit(model_fn, c, y, p0=p0, maxfev=20000)
        return popt
    except Exception:
        return None


def loocv_rmse(model_fn, c, y, p0_fn):
    c = np.asarray(c, dtype=float)
    y = np.asarray(y, dtype=float)
    errs = []
    for i in range(len(c)):
        mask = np.ones(len(c), dtype=bool)
        mask[i] = False
        popt = fit_model(model_fn, c[mask], y[mask], p0_fn(c[mask], y[mask]))
        if popt is None:
            continue
        pred = model_fn(c[i], *popt)
        errs.append((pred - y[i]) ** 2)
    if not errs:
        return np.nan
    return float(np.sqrt(np.mean(errs)))


def linear_p0(c, y):
    b = (y[-1] - y[0]) / (c[-1] - c[0] + 1e-9)
    a = y[0] - b * c[0]
    return [a, b]


def saturating_p0(c, y):
    y0 = y[0]
    amp = y[-1] - y[0]
    tau = (c[-1] - c[0]) / 2 + 1e-6
    return [y0, amp, tau]


def compare_models_per_band(agg, response="A"):
    results = []
    c = agg["fraction"].values
    for b in BANDS:
        y = agg[f"{b}_{response}_mean"].values
        if np.isnan(y).any():
            continue
        lin_popt = fit_model(linear_model, c, y, linear_p0(c, y))
        sat_popt = fit_model(saturating_model, c, y, saturating_p0(c, y))
        lin_rmse = loocv_rmse(linear_model, c, y, linear_p0)
        sat_rmse = loocv_rmse(saturating_model, c, y, saturating_p0)
        dyn_range = y.max() - y.min()
        results.append({
            "band": b,
            "dynamic_range": dyn_range,
            "linear_params": lin_popt,
            "saturating_params": sat_popt,
            "linear_loocv_rmse": lin_rmse,
            "saturating_loocv_rmse": sat_rmse,
            "better_model": "linear" if (not np.isnan(lin_rmse) and (np.isnan(sat_rmse) or lin_rmse <= sat_rmse)) else "saturating",
        })
    return pd.DataFrame(results).sort_values("dynamic_range", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# 7. Plots
# --------------------------------------------------------------------------

def plot_average_spectra(agg, outdir):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.cm.viridis
    fracs = agg["fraction"].values
    norm = (fracs - fracs.min()) / (fracs.max() - fracs.min() + 1e-9)
    for frac, n in zip(fracs, norm):
        row = agg[agg["fraction"] == frac].iloc[0]
        y = [row[f"{b}_I_mean"] for b in BANDS]
        ax.plot(BANDS, y, marker="o", color=cmap(n), label=f"{frac:.2f}")
    ax.set_xlabel("Band (nm)")
    ax.set_ylabel("Mean raw intensity")
    ax.set_title("Average spectrum per concentration (3 replicates each)")
    ax.legend(title="Fraction of S", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "01_average_spectra.png"), dpi=150)
    plt.close(fig)


def plot_blank_drift(blanks, outdir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for b in BANDS:
        ax.plot(blanks["window_midpoint"], blanks[f"{b}_mean"], marker="o", label=f"{b} nm")
    ax.set_xlabel("Time")
    ax.set_ylabel("Blank reading (raw)")
    ax.set_title("Water-blank stability across the run (drift check)")
    ax.legend(fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "02_blank_drift.png"), dpi=150)
    plt.close(fig)


def plot_calibration_curves(agg, samples, model_cmp, outdir, response="A"):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    c_fit = np.linspace(0, agg["fraction"].max(), 200)
    for ax, b in zip(axes.flat, BANDS):
        c = agg["fraction"].values
        y = agg[f"{b}_{response}_mean"].values
        yerr = agg[f"{b}_{response}_sd"].values
        ax.errorbar(c, y, yerr=yerr, fmt="o", color="black", capsize=3, label="replicate mean ± SD")

        row = model_cmp[model_cmp["band"] == b]
        if not row.empty:
            row = row.iloc[0]
            if row["linear_params"] is not None:
                ax.plot(c_fit, linear_model(c_fit, *row["linear_params"]), "--", color="tab:blue",
                        label=f"linear (LOOCV RMSE {row['linear_loocv_rmse']:.3g})")
            if row["saturating_params"] is not None:
                ax.plot(c_fit, saturating_model(c_fit, *row["saturating_params"]), "-", color="tab:orange",
                        label=f"saturating (LOOCV RMSE {row['saturating_loocv_rmse']:.3g})")
        ax.set_title(f"{b} nm")
        ax.set_xlabel("Fraction of S")
        ax.set_ylabel("Apparent A" if response == "A" else "Raw intensity")
        ax.legend(fontsize=6)
    fig.suptitle(f"Calibration curves — {'apparent absorbance' if response=='A' else 'raw intensity'} vs concentration")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"03_calibration_curves_{response}.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 8. Summary / verdict
# --------------------------------------------------------------------------

def print_summary(meas, samples, agg, blanks, model_cmp,
                  det_diag=None, q=None, full_scale=None,
                  carry_res=None, perm_res=None):
    print("\n" + "=" * 70)
    print("CALIBRATION DIAGNOSTIC SUMMARY")
    print("=" * 70)

    if det_diag is not None:
        print("\n--- Detector saturation / background ---")
        print(f"  Inferred quantisation step: {q:.3e}  ->  ADC full scale ~{full_scale:.0f} counts")
        print(det_diag[["band", "implied_counts_max", "pct_of_full_scale_at_max",
                        "n_at_ceiling", "n_at_zero"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        worst = det_diag["pct_of_full_scale_at_max"].max()
        if worst < 50:
            print(f"  [!] Brightest band peaks at only {worst:.0f}% of ADC full scale — "
                  "gain/integration time is set low, wasting dynamic range.")
        if (det_diag["n_at_zero"] > 0).any():
            print("  [!] Some readings are <=0 — possible background already subtracted.")
        else:
            print("  [OK] No zero/negative readings: no sign of a dark correction "
                  "having been applied in-pipeline (so no double subtraction here).")

    if carry_res is not None and not carry_res.empty:
        print("\n--- Carryover check (blank reading vs. preceding sample's concentration) ---")
        print(carry_res[["band", "pearson_r", "partial_r_controlling_order",
                         "slope_as_pct_of_blank"]]
              .to_string(index=False, float_format=lambda x: f"{x:.3g}"))
        strong = carry_res[carry_res["pearson_r"].abs() > 0.5]
        oc = carry_res.attrs.get("order_conc_r", np.nan)
        print(f"\n  Confound check: correlation between run order and preceding "
              f"concentration = {oc:.3f}")
        if abs(oc) > 0.4:
            print("  [!] CONFOUNDED: this session's randomisation happened to place higher")
            print("      concentrations later in the run, so 'blank vs preceding")
            print("      concentration' cannot be separated from 'blank vs time' (drift).")
            print("      Use the partial_r column (order controlled) instead of pearson_r,")
            print("      and do NOT conclude carryover from the raw correlation.")
            strong_p = carry_res[carry_res["partial_r_controlling_order"].abs() > 0.5]
            print(f"      After controlling for order, {len(strong_p)} band(s) still show "
                  f"|partial r| > 0.5.")
        elif len(strong):
            print(f"  [!] {len(strong)} band(s) show |r| > 0.5 — rinsing may be inadequate.")
        else:
            print("  [OK] No strong relationship — no evidence of carryover between vials.")

    if perm_res is not None and not perm_res.empty:
        print("\n--- Is there ANY concentration-dependent signal? "
              "(permutation test, BH-FDR across bands) ---")
        print(perm_res[["band", "F_like_stat", "p_perm", "p_bh", "significant_FDR_0.05"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        n_sig = int(perm_res["significant_FDR_0.05"].sum())
        if n_sig == 0:
            print("  [!] NO band shows a concentration effect that survives FDR correction.")
            print("      Between-concentration variation is not distinguishable from")
            print("      vial-to-vial noise. This calibration does not support a curve.")
        else:
            print(f"  [OK] {n_sig} band(s) show a concentration effect surviving FDR.")

    empty = meas[meas["n_samples"] == 0]
    sparse_nonempty = meas[(meas["n_samples"] > 0) & (meas["n_samples"] < MIN_SAMPLES_PER_WINDOW)]

    if len(empty):
        empty_desc = [f"order {int(o)} ({t})" for o, t in zip(empty["order"], empty["type"])]
        print(f"\n[!] {len(empty)} logged window(s) matched ZERO scans in the export "
              f"— missing data, not a code issue. These rows: {', '.join(empty_desc)}")
        print("    Check the export for a logging gap spanning these windows' record_start/record_end.")
    if len(sparse_nonempty):
        sparse_desc = [f"order {int(o)} (n={int(n)})" for o, n in
                        zip(sparse_nonempty["order"], sparse_nonempty["n_samples"])]
        print(f"[!] {len(sparse_nonempty)} window(s) matched fewer than {MIN_SAMPLES_PER_WINDOW} scans "
              f"— treat with caution: {', '.join(sparse_desc)}")
    if not len(empty) and not len(sparse_nonempty):
        print("\n[OK] Every logged window matched sufficient scans in the export.")

    print("\n--- Replicate consistency (raw intensity CV%) ---")
    for b in BANDS:
        cvs = agg[f"{b}_I_cv_pct"].dropna()
        if len(cvs):
            print(f"  {b:>4} nm : median CV {cvs.median():5.1f}%   max CV {cvs.max():5.1f}%")

    print("\n--- Blank drift over the run ---")
    for b in BANDS:
        vals = blanks[f"{b}_mean"].dropna()
        if len(vals) >= 2:
            drift_pct = (vals.iloc[-1] - vals.iloc[0]) / vals.iloc[0] * 100
            spread_pct = (vals.max() - vals.min()) / vals.mean() * 100
            print(f"  {b:>4} nm : first->last blank change {drift_pct:+5.1f}%   "
                  f"total spread {spread_pct:5.1f}% of mean")

    print("\n--- Dynamic range & best-fit model per band (apparent absorbance) ---")
    print(model_cmp[["band", "dynamic_range", "linear_loocv_rmse", "saturating_loocv_rmse", "better_model"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    best = model_cmp.sort_values("dynamic_range", ascending=False).iloc[0]
    print(f"\n--> Band with the largest dynamic range: {int(best['band'])} nm "
          f"(A ranges over {best['dynamic_range']:.3f} across the series), "
          f"better-fit model: {best['better_model']}.")

    print("\n" + "=" * 70)
    print("Read the plots in the output folder before trusting this verdict —")
    print("this text summary flags issues but the calibration curves are the real check.")
    print("=" * 70 + "\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def find_default(pattern):
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-log", default=None, help="Path to the run-log .xlsx")
    ap.add_argument("--spectra", default=None, help="Path to the AS7341 export .csv")
    ap.add_argument("--outdir", default="calibration_analysis_output")
    ap.add_argument("--stability-start", default=None,
                    help="Stability-control window start, e.g. '2026-09-22 14:00:00'")
    ap.add_argument("--stability-end", default=None,
                    help="Stability-control window end, e.g. '2026-09-22 14:20:00'")
    ap.add_argument("--n-perm", type=int, default=5000,
                    help="Permutation count for the signal test")
    args = ap.parse_args()

    run_log_path = args.run_log or find_default("*run_log*.xlsx") or find_default("*.xlsx")
    spectra_path = args.spectra or find_default("*as7341*.csv") or find_default("*.csv")
    if not run_log_path or not spectra_path:
        sys.exit("Could not find run-log .xlsx and/or spectra .csv — pass --run-log / --spectra explicitly.")

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Run log : {run_log_path}")
    print(f"Spectra : {spectra_path}")

    spectra_long = pd.read_csv(spectra_path)
    spectra = load_spectra(spectra_path)
    ref_date = spectra.index[0].normalize()

    run_log = load_run_log(run_log_path, ref_date)
    meas = build_measurements(run_log, spectra)
    meas, blanks, samples = add_local_blank_and_absorbance(meas)
    agg = aggregate_by_fraction(samples)

    meas.to_csv(os.path.join(args.outdir, "measurements_per_vial.csv"), index=False)
    agg.to_csv(os.path.join(args.outdir, "aggregated_by_fraction.csv"), index=False)

    det_diag, q, full_scale = detector_diagnostics(spectra_long, spectra)
    det_diag.to_csv(os.path.join(args.outdir, "detector_diagnostics.csv"), index=False)

    carry_df, carry_res = carryover_test(meas)
    if not carry_res.empty:
        carry_res.to_csv(os.path.join(args.outdir, "carryover_test.csv"), index=False)

    perm_res = permutation_signal_test(samples, response="A", n_perm=args.n_perm)
    if not perm_res.empty:
        perm_res.to_csv(os.path.join(args.outdir, "permutation_signal_test.csv"), index=False)

    model_cmp_A = compare_models_per_band(agg, response="A")
    model_cmp_I = compare_models_per_band(agg, response="I")

    plot_average_spectra(agg, args.outdir)
    plot_blank_drift(blanks, args.outdir)
    plot_calibration_curves(agg, samples, model_cmp_A, args.outdir, response="A")
    plot_calibration_curves(agg, samples, model_cmp_I, args.outdir, response="I")

    if args.stability_start and args.stability_end:
        analyse_stability_control(spectra, pd.Timestamp(args.stability_start),
                                  pd.Timestamp(args.stability_end), args.outdir)
    else:
        print("\n[i] Stability control not analysed — pass --stability-start / "
              "--stability-end once you've run the 20-minute 10% monitor.")

    print_summary(meas, samples, agg, blanks, model_cmp_A,
                  det_diag=det_diag, q=q, full_scale=full_scale,
                  carry_res=carry_res, perm_res=perm_res)
    print(f"All plots and CSVs written to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
