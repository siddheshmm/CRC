#!/usr/bin/env python3
"""
run_unified_protocol3_calibration.py

Unified, Best-of-Both Metrological & Statistical Analysis of Protocol 3:
Fixed-Volume Optical Dye Calibration in Pioreactor.

Combines:
  1. Strict 2.0-minute window extraction adhering to protocol constraints (from run_calibration_protocol3.py).
  2. Timestamp rounding to 1-second cadence & proper midnight day rollover handling.
  3. Dynamic local blank baseline interpolation across time (from analyze_calibration.py).
  4. Apparent absorbance calculation A_band = -log10(I_sample / I_blank_local).
  5. Metrological detector diagnostics (quantization step, headroom, ADC full-scale).
  6. Carryover testing with order-confound partial correlation analysis.
  7. Non-parametric Permutation Signal Test (5,000 label shuffles) with Benjamini-Hochberg FDR.
  8. Model selection via Leave-One-Out Cross-Validation (LOOCV RMSE): Linear vs Saturating Exponential.
  9. Beer-Lambert physical regression: apparent extinction coefficients (epsilon), intercepts, R^2, SEM.
  10. Publication-quality multi-panel visualization suite (8 figures).
"""

import argparse
import glob
import os
import sys
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BANDS = [415, 445, 480, 515, 555, 590, 630, 680]
BAND_COLORS = {
    415: "#6a0dad",  # violet
    445: "#0055d4",  # blue
    480: "#00a2e8",  # cyan
    515: "#008a00",  # green (peak dye absorption)
    555: "#8cb800",  # yellow-green
    590: "#e67e22",  # orange
    630: "#e74c3c",  # red
    680: "#880e4f",  # deep red / NIR
}
MIN_SAMPLES_PER_WINDOW = 10  # 10 scans @ 5s = 50s minimum to be considered valid


# --------------------------------------------------------------------------
# 1. Temporal Parsing & Window Extraction
# --------------------------------------------------------------------------

def parse_time_str(s):
    if pd.isna(s):
        return None
    s = str(s).strip().lower().replace(".", ":").replace(" ", "")
    for fmt in ("%I:%M%p", "%I:%M:%S%p", "%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).time()
        except ValueError:
            pass
    return None


def make_datetime(t, ref_start_date, start_hour_threshold=12):
    """Handles midnight date rollover dynamically relative to acquisition start."""
    if t is None:
        return None
    day_offset = 1 if t.hour < start_hour_threshold else 0
    base_date = ref_start_date + pd.Timedelta(days=day_offset)
    return pd.Timestamp.combine(base_date.date(), t)


def load_spectra(path):
    print(f"Loading spectrometer data: {path}")
    df = pd.read_csv(path)
    df["ts_local"] = pd.to_datetime(df["timestamp_localtime"])
    # AS7341 scans 8 bands nearly simultaneously with millisecond offsets.
    # Binning to nearest 1-second aligns all 8 bands cleanly into a single scan row.
    df["ts_sec"] = df["ts_local"].dt.round("1s")
    wide = df.pivot_table(index="ts_sec", columns="band", values="reading", aggfunc="mean")
    wide.columns = [int(c) for c in wide.columns]
    wide = wide.reindex(columns=BANDS).sort_index()
    return df, wide


def load_run_log(path, ref_start_date):
    print(f"Loading run log: {path}")
    df = pd.read_excel(path, sheet_name="Run Log")
    df = df[df["order"].apply(lambda v: isinstance(v, (int, float)) and not pd.isna(v))].copy()
    df = df[df["order"] != "EX"].copy()
    df["order"] = df["order"].astype(int)

    # Detect if experiment started in the evening to set rollover threshold
    first_time = parse_time_str(df["record_start"].iloc[0])
    threshold = first_time.hour - 2 if first_time and first_time.hour > 12 else 12

    for col in ("fill_time", "record_start", "record_end"):
        df[col + "_dt"] = df[col].apply(parse_time_str).apply(lambda t: make_datetime(t, ref_start_date, threshold))

    df["fraction"] = pd.to_numeric(df["fraction"], errors="coerce")
    df["replicate"] = pd.to_numeric(df["replicate"], errors="coerce")
    return df.reset_index(drop=True)


def extract_windows(run_log, wide):
    """Strict 2.0-minute window extraction starting at record_start."""
    print("Extracting strict 2.0-minute windows from acquisition data...")
    records = []
    for _, r in run_log.iterrows():
        s_dt = r["record_start_dt"]
        e_logged = r["record_end_dt"]
        if pd.isna(s_dt):
            continue
        # Strict 2.0-minute duration (capped by logged end if logged end is shorter)
        e_dt = min(s_dt + pd.Timedelta(minutes=2), e_logged) if pd.notna(e_logged) else s_dt + pd.Timedelta(minutes=2)
        
        sub = wide.loc[(wide.index >= s_dt) & (wide.index <= e_dt)]
        n_pts = len(sub)
        is_skipped = (n_pts < MIN_SAMPLES_PER_WINDOW)

        rec = {
            "order": r["order"],
            "type": r["type"],
            "fraction": float(r["fraction"]) if pd.notna(r["fraction"]) else 0.0,
            "replicate": int(r["replicate"]) if pd.notna(r["replicate"]) else np.nan,
            "record_start": r["record_start"],
            "record_end": r["record_end"],
            "start_dt": s_dt,
            "end_dt": e_dt,
            "midpoint_dt": s_dt + (e_dt - s_dt) / 2,
            "n_samples": n_pts,
            "status": "SKIPPED_PRE_RECORDING" if is_skipped else "VALID",
        }
        for b in BANDS:
            if not is_skipped:
                rec[f"{b}_mean"] = sub[b].mean()
                rec[f"{b}_sd"] = sub[b].std()
                rec[f"{b}_cv_pct"] = (sub[b].std() / sub[b].mean() * 100.0) if sub[b].mean() > 0 else 0.0
            else:
                rec[f"{b}_mean"] = np.nan
                rec[f"{b}_sd"] = np.nan
                rec[f"{b}_cv_pct"] = np.nan
        records.append(rec)

    df_meas = pd.DataFrame(records)
    n_valid = (df_meas["status"] == "VALID").sum()
    n_skip = (df_meas["status"] != "VALID").sum()
    print(f"Extracted {len(df_meas)} rows: {n_valid} VALID, {n_skip} SKIPPED (e.g. Orders pre-dating data logging).")
    return df_meas


# --------------------------------------------------------------------------
# 2. Dynamic Local Blank Baseline Interpolation & Absorbance
# --------------------------------------------------------------------------

def add_local_blank_and_absorbance(df_meas):
    """Interpolates water blank readings continuously over time to compensate for drift."""
    blanks = df_meas[(df_meas["type"] == "blank") & (df_meas["status"] == "VALID")].sort_values("midpoint_dt").copy()
    if blanks.empty:
        raise RuntimeError("No valid blank windows found in data!")

    blank_t = blanks["midpoint_dt"].values.astype("datetime64[ns]").astype(np.int64)
    all_t = df_meas["midpoint_dt"].values.astype("datetime64[ns]").astype(np.int64)

    for b in BANDS:
        blank_vals = blanks[f"{b}_mean"].values
        # Linear interpolation across time
        interp_blank = np.interp(all_t, blank_t, blank_vals)
        df_meas[f"{b}_blank_local"] = interp_blank
        
        # Calculate local transmission ratio and apparent absorbance
        ratio = df_meas[f"{b}_mean"] / interp_blank
        df_meas[f"{b}_ratio"] = ratio
        with np.errstate(invalid="ignore", divide="ignore"):
            df_meas[f"{b}_A"] = -np.log10(np.clip(ratio, 1e-6, None))

    samples = df_meas[(df_meas["type"] == "sample") & (df_meas["status"] == "VALID")].copy()
    return df_meas, blanks, samples


# --------------------------------------------------------------------------
# 3. Metrological & Sensor Diagnostics
# --------------------------------------------------------------------------

def detector_diagnostics(spectra_long):
    vals = spectra_long["reading"].values
    uniq = np.unique(vals)
    gaps = np.diff(np.sort(uniq))
    gaps = gaps[gaps > 0]
    q = np.min(gaps) if len(gaps) else np.nan
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


def carryover_test(df_meas):
    """Evaluates whether blanks carry residual dye from the preceding sample."""
    m = df_meas[df_meas["status"] == "VALID"].sort_values("order").reset_index(drop=True)
    rows = []
    for i, r in m.iterrows():
        if r["type"] != "blank":
            continue
        prev = m.iloc[:i]
        prev_samples = prev[prev["type"] == "sample"]
        if prev_samples.empty:
            continue
        prev_frac = prev_samples.iloc[-1]["fraction"]
        if pd.isna(prev_frac):
            continue
        row = {"order": r["order"], "preceding_fraction": prev_frac}
        for b in BANDS:
            row[f"{b}"] = r[f"{b}_mean"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df) < 3:
        return df, pd.DataFrame()

    order_conc_r = np.corrcoef(df["order"].values.astype(float),
                               df["preceding_fraction"].values.astype(float))[0, 1]

    results = []
    for b in BANDS:
        x = df["preceding_fraction"].values.astype(float)
        y = df[f"{b}"].values.astype(float)
        ok = ~np.isnan(y)
        if ok.sum() < 3 or np.ptp(x[ok]) == 0:
            continue
        slope, _ = np.polyfit(x[ok], y[ok], 1)
        r = np.corrcoef(x[ok], y[ok])[0, 1]

        # Partial correlation controlling for order
        o = df["order"].values.astype(float)[ok]
        def resid(a, b_):
            s = np.polyfit(b_, a, 1)
            return a - np.polyval(s, b_)
        r_partial = np.corrcoef(resid(x[ok], o), resid(y[ok], o))[0, 1] if np.ptp(o) > 0 else np.nan

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


def permutation_signal_test(samples, response="A", n_perm=5000, seed=42):
    """Permutation one-way ANOVA F-statistic test with Benjamini-Hochberg FDR."""
    rng = np.random.default_rng(seed)
    usable = samples[samples["n_samples"] > 0]
    out = []
    null_dists = {}

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
        null_dists[b] = (obs, null)
        p = (np.sum(null >= obs) + 1) / (n_perm + 1)
        out.append({"band": b, "F_like_stat": obs, "p_perm": p})

    df = pd.DataFrame(out)
    if df.empty:
        return df, null_dists
    # Benjamini-Hochberg adjustment
    df = df.sort_values("p_perm").reset_index(drop=True)
    m = len(df)
    df["p_bh"] = np.minimum.accumulate(
        (df["p_perm"].values * m / (np.arange(m) + 1))[::-1]
    )[::-1]
    df["p_bh"] = df["p_bh"].clip(upper=1.0)
    df["significant_FDR_0.05"] = df["p_bh"] < 0.05
    return df, null_dists


# --------------------------------------------------------------------------
# 4. Model Comparison via Leave-One-Out Cross-Validation (LOOCV)
# --------------------------------------------------------------------------

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
# 5. Replicate Aggregation & Beer-Lambert Linear Metrology
# --------------------------------------------------------------------------

def aggregate_replicates(samples):
    rows = []
    for frac, g in samples.groupby("fraction"):
        row = {
            "fraction": frac,
            "n_replicates": len(g),
            "orders": str(g["order"].tolist()),
        }
        for b in BANDS:
            # Transmission stats
            t_vals = g[f"{b}_mean"]
            row[f"{b}_I_mean"] = t_vals.mean()
            row[f"{b}_I_sd"] = t_vals.std(ddof=1) if len(t_vals) > 1 else 0.0
            row[f"{b}_I_sem"] = t_vals.sem() if len(t_vals) > 1 else 0.0
            row[f"{b}_I_cv_pct"] = (row[f"{b}_I_sd"] / row[f"{b}_I_mean"] * 100) if row[f"{b}_I_mean"] else 0.0

            # Apparent Absorbance stats
            a_vals = g[f"{b}_A"]
            row[f"{b}_A_mean"] = a_vals.mean()
            row[f"{b}_A_sd"] = a_vals.std(ddof=1) if len(a_vals) > 1 else 0.0
            row[f"{b}_A_sem"] = a_vals.sem() if len(a_vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("fraction").reset_index(drop=True)


def fit_beer_lambert(samples, agg, model_cmp_A):
    results = []
    for b in BANDS:
        c_vals = samples["fraction"].values
        a_vals = samples[f"{b}_A"].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(c_vals, a_vals)
        r2 = r_value ** 2
        pred_a = slope * c_vals + intercept
        residuals = a_vals - pred_a
        rmse = np.sqrt(np.mean(residuals ** 2))

        # Retrieve LOOCV model comparison
        m_row = model_cmp_A[model_cmp_A["band"] == b]
        better_model = m_row["better_model"].iloc[0] if not m_row.empty else "N/A"
        lin_loocv = m_row["linear_loocv_rmse"].iloc[0] if not m_row.empty else np.nan
        sat_loocv = m_row["saturating_loocv_rmse"].iloc[0] if not m_row.empty else np.nan

        results.append({
            "band": b,
            "epsilon_apparent": slope,
            "intercept": intercept,
            "R2": r2,
            "p_value": p_value,
            "std_err": std_err,
            "rmse": rmse,
            "linear_loocv_rmse": lin_loocv,
            "saturating_loocv_rmse": sat_loocv,
            "better_model": better_model,
        })
    return pd.DataFrame(results).sort_values("band").reset_index(drop=True)


# --------------------------------------------------------------------------
# 6. Visualization Suite (8 Figures)
# --------------------------------------------------------------------------

def generate_visualizations(wide, df_meas, blanks, samples, agg, df_fits, model_cmp_A,
                            carry_df, carry_res, null_dists, outdir):
    print("Generating comprehensive visualization suite (8 figures)...")

    # Figure 1: Full raw timeseries with extraction windows
    fig, ax = plt.subplots(figsize=(16, 6), dpi=150)
    for b in [445, 480, 515, 630]:
        ax.plot(wide.index, wide[b], label=f"{b} nm", color=BAND_COLORS[b], lw=1.2, alpha=0.85)

    for _, r in df_meas.iterrows():
        if r["status"] == "VALID":
            color = "#a1dab4" if r["type"] == "blank" else "#fed976"
            ax.axvspan(r["start_dt"], r["end_dt"], color=color, alpha=0.45, lw=0)
            ax.text(r["midpoint_dt"], 0.255, str(r["order"]), fontsize=6.5, ha="center", va="bottom", rotation=90)
        else:
            ax.axvspan(r["start_dt"], r["end_dt"], color="#fbb4ae", alpha=0.45, lw=0)
            ax.text(r["midpoint_dt"], 0.255, f"{r['order']}*", fontsize=6.5, ha="center", va="bottom", rotation=90, color="red")

    ax.set_ylabel("AS7341 Normalized Reading", fontsize=11)
    ax.set_title("Figure 1: Full Protocol 3 Acquisition Trace with Audited 2.0-Minute Extraction Windows\n"
                 "(Green: Interleaved Water Blanks | Gold: Sample Vials | Red*: Skipped Pre-recording)", fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "01_raw_timeseries_with_extraction_windows.png"))
    plt.close(fig)

    # Figure 2: Blank baseline stability & interpolated curves
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    t_full = pd.date_range(wide.index.min(), wide.index.max(), freq="30s")
    t_full_ns = t_full.values.astype("datetime64[ns]").astype(np.int64)
    blank_t_ns = blanks["midpoint_dt"].values.astype("datetime64[ns]").astype(np.int64)

    for b in BANDS:
        # Scatter points of measured blanks
        ax.plot(blanks["midpoint_dt"], blanks[f"{b}_mean"], "o", color=BAND_COLORS[b], markersize=5)
        # Interpolated continuous line
        interp_curve = np.interp(t_full_ns, blank_t_ns, blanks[f"{b}_mean"].values)
        ax.plot(t_full, interp_curve, "--", color=BAND_COLORS[b], lw=1.2, label=f"{b} nm")

    ax.set_ylabel("Raw Blank Reading", fontsize=11)
    ax.set_title("Figure 2: Interleaved Water Blank Stability & Dynamic Interpolated Baseline Across Session", fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(loc="upper right", ncol=2, fontsize=8.5, framealpha=0.9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "02_blank_baseline_stability_and_interpolation.png"))
    plt.close(fig)

    # Figure 3: Multiband spectral profiles vs concentration
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    cmap = plt.cm.viridis
    fracs = agg["fraction"].values
    norm = (fracs - fracs.min()) / (fracs.max() - fracs.min() + 1e-9)

    for frac, n in zip(fracs, norm):
        row = agg[agg["fraction"] == frac].iloc[0]
        y_I = [row[f"{b}_I_mean"] for b in BANDS]
        y_A = [row[f"{b}_A_mean"] for b in BANDS]
        ax1.plot(BANDS, y_I, marker="o", color=cmap(n), label=f"{frac:.2f}")
        ax2.plot(BANDS, y_A, marker="s", color=cmap(n), label=f"{frac:.2f}")

    ax1.set_xlabel("Wavelength (nm)", fontsize=11)
    ax1.set_ylabel("Transmission Intensity (I)", fontsize=11)
    ax1.set_title("Raw Multiband Transmission Spectra", fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Wavelength (nm)", fontsize=11)
    ax2.set_ylabel("Apparent Absorbance A = -log10(I/I_blank)", fontsize=11)
    ax2.set_title("Apparent Absorbance Spectra", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(title="Dye Fraction", bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8.5)

    fig.suptitle("Figure 3: AS7341 Optical Spectral Profiles vs Dye Concentration", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "03_spectral_profiles_vs_concentration.png"))
    plt.close(fig)

    # Figure 4: Beer-Lambert Calibration Curves (All 8 Bands)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=150, sharex=True)
    c_fit = np.linspace(0, agg["fraction"].max(), 200)

    for ax, b in zip(axes.flat, BANDS):
        # Raw vial points
        ax.scatter(samples["fraction"], samples[f"{b}_A"], color="#999999", alpha=0.6, s=25, label="Individual vials")
        # Replicate mean with SEM error bars
        ax.errorbar(agg["fraction"], agg[f"{b}_A_mean"], yerr=agg[f"{b}_A_sem"], fmt="o",
                    color=BAND_COLORS[b], ecolor="black", elinewidth=1.2, capsize=3.5, label="Replicate Mean ± SEM")

        # Fit parameters
        fit_row = df_fits[df_fits["band"] == b].iloc[0]
        ax.plot(c_fit, fit_row["epsilon_apparent"] * c_fit + fit_row["intercept"], "--", color="#0055d4", lw=1.5,
                label=f"Linear: ε={fit_row['epsilon_apparent']:.3f} (R²={fit_row['R2']:.3f})")

        m_row = model_cmp_A[model_cmp_A["band"] == b]
        if not m_row.empty and m_row.iloc[0]["saturating_params"] is not None:
            sat_p = m_row.iloc[0]["saturating_params"]
            ax.plot(c_fit, saturating_model(c_fit, *sat_p), ":", color="#e67e22", lw=1.5,
                    label=f"Saturating (LOOCV={m_row.iloc[0]['saturating_loocv_rmse']:.3f})")

        ax.set_title(f"{b} nm  [Best: {fit_row['better_model']}]", fontsize=10, fontweight="bold")
        ax.set_xlabel("Dye Stock Fraction (c)", fontsize=9)
        ax.set_ylabel("Absorbance A", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 4: Beer-Lambert Calibration Curves Across All 8 AS7341 Bands (A vs Concentration)\n"
                 "With Strict 2-Minute Extraction & Dynamic Baseline Subtraction", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "04_beer_lambert_calibration_curves.png"))
    plt.close(fig)

    # Figure 5: Transmission Calibration Curves (All 8 Bands)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=150, sharex=True)
    for ax, b in zip(axes.flat, BANDS):
        ax.scatter(samples["fraction"], samples[f"{b}_mean"], color="#999999", alpha=0.6, s=25, label="Vials")
        ax.errorbar(agg["fraction"], agg[f"{b}_I_mean"], yerr=agg[f"{b}_I_sem"], fmt="s",
                    color=BAND_COLORS[b], ecolor="black", elinewidth=1.2, capsize=3.5, label="Mean ± SEM")
        ax.set_title(f"{b} nm", fontsize=10, fontweight="bold")
        ax.set_xlabel("Dye Fraction", fontsize=9)
        ax.set_ylabel("Raw Reading (I)", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Figure 5: Raw Transmission Intensity Curves Across All 8 Channels", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "05_transmission_calibration_curves.png"))
    plt.close(fig)

    # Figure 6: Linear Fit Residuals & Linearity Diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    diag_bands = [480, 515, 555, 630]
    for ax, b in zip(axes.flat, diag_bands):
        c = samples["fraction"].values
        a = samples[f"{b}_A"].values
        fit_row = df_fits[df_fits["band"] == b].iloc[0]
        residuals = a - (fit_row["epsilon_apparent"] * c + fit_row["intercept"])
        ax.scatter(c, residuals, color=BAND_COLORS[b], s=40, edgecolors="black", zorder=3)
        ax.axhline(0, color="red", linestyle="--", lw=1)
        ax.set_title(f"{b} nm Residuals (RMSE = {fit_row['rmse']:.4f})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Dye Fraction", fontsize=9)
        ax.set_ylabel("Residual (A - A_pred)", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 6: Beer-Lambert Linear Fit Residuals Across Concentration (Linearity Check)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "06_fit_residuals_and_linearity.png"))
    plt.close(fig)

    # Figure 7: Carryover Confound Diagnostics
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    if not carry_df.empty:
        sc = ax.scatter(carry_df["preceding_fraction"], carry_df["515"], c=carry_df["order"],
                        cmap="plasma", s=70, edgecolors="black", zorder=3)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Run Order", fontsize=10)
        
        # Fit line
        slope, intercept, r_val, _, _ = stats.linregress(carry_df["preceding_fraction"], carry_df["515"])
        x_c = np.linspace(0, carry_df["preceding_fraction"].max(), 50)
        ax.plot(x_c, slope * x_c + intercept, "r--", lw=1.5,
                label=f"Linear Fit (r={r_val:.2f}, partial_r={carry_res[carry_res['band']==515]['partial_r_controlling_order'].iloc[0]:.2f})")
        ax.set_xlabel("Preceding Vial Dye Fraction", fontsize=11)
        ax.set_ylabel("Blank 515 nm Reading", fontsize=11)
        ax.set_title("Figure 7: Carryover Diagnostic (Blank Intensity vs Preceding Concentration)", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "07_carryover_confound_diagnostics.png"))
    plt.close(fig)

    # Figure 8: Permutation Test Null Distributions
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)
    perm_bands = [480, 515, 630]
    for ax, b in zip(axes, perm_bands):
        if b in null_dists:
            obs, null = null_dists[b]
            ax.hist(null, bins=40, color="#b0c4de", edgecolor="black", alpha=0.7, density=True, label="Null Dist (5000 shuffles)")
            ax.axvline(obs, color="red", lw=2, linestyle="--", label=f"Observed F = {obs:.2f}")
            p_val = (np.sum(null >= obs) + 1) / (len(null) + 1)
            ax.set_title(f"{b} nm (p_perm = {p_val:.4f})", fontsize=10, fontweight="bold")
            ax.set_xlabel("F-like Statistic", fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 8: Permutation Signal Test Null Distributions (Shuffled Concentration Labels)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "08_permutation_null_distributions.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 7. Main Execution & Verdict Reporting
# --------------------------------------------------------------------------

def print_final_verdict(df_meas, samples, agg, blanks, df_fits, model_cmp_A,
                        det_diag, q, full_scale, carry_res, perm_res):
    print("\n" + "=" * 78)
    print("           UNIFIED PROTOCOL 3 CALIBRATION: AUDITED VERDICT & METROLOGY")
    print("=" * 78)

    print("\n[1] WINDOW EXTRACTION & LOGGING INTEGRITY")
    n_valid = (df_meas["status"] == "VALID").sum()
    n_skip = (df_meas["status"] != "VALID").sum()
    print(f"  * Total Logged Windows: {len(df_meas)}")
    print(f"  * Valid 2.0-Minute Windows Extracted: {n_valid}")
    print(f"  * Skipped Windows (Pre-recording onset): {n_skip} (Orders: {df_meas[df_meas['status'] != 'VALID']['order'].tolist()})")
    print(f"  * Dynamic Interleaved Blanks Available: {len(blanks)} points across ~5 hours")

    print("\n[2] DETECTOR HEADROOM & ADC DYNAMICS")
    print(f"  * Inferred Quantization Step (q): {q:.3e}  -->  ADC Full Scale ~{full_scale:.0f} counts")
    max_util = det_diag["pct_of_full_scale_at_max"].max()
    print(f"  * Peak Channel Dynamic Headroom: {max_util:.1f}% of full scale (Channel {det_diag.loc[det_diag['pct_of_full_scale_at_max'].idxmax(), 'band']} nm)")
    print(f"  * ADC Ceiling Clipped: {det_diag['n_at_ceiling'].sum()} points | Floor Clipped (<=0): {det_diag['n_at_zero'].sum()} points")

    print("\n[3] CARRYOVER & SESSION CONFOUND ANALYSIS")
    if not carry_res.empty:
        oc = carry_res.attrs.get("order_conc_r", np.nan)
        print(f"  * Confound Correlation (Run Order vs Concentration): r = {oc:.3f}")
        for _, r in carry_res[carry_res["band"].isin([480, 515, 630])].iterrows():
            print(f"    - {int(r['band'])} nm: Raw Carryover Pearson r = {r['pearson_r']:+.3f} | Partial r (Order Controlled) = {r['partial_r_controlling_order']:+.3f}")
        if abs(oc) > 0.4:
            print("  --> Confound note: Run order correlates with concentration; partial correlation must be used.")

    print("\n[4] PERMUTATION SIGNAL TEST & BENJAMINI-HOCHBERG FDR")
    n_sig = perm_res["significant_FDR_0.05"].sum() if not perm_res.empty else 0
    print(f"  * Channels with concentration signal surviving FDR at alpha=0.05: {n_sig} / 8 bands")
    if not perm_res.empty:
        for _, r in perm_res.iterrows():
            sig_mark = "[PASS]" if r["significant_FDR_0.05"] else "[FAIL]"
            print(f"    {sig_mark} {int(r['band']):>4} nm : F = {r['F_like_stat']:6.2f} | p_perm = {r['p_perm']:.4f} | p_BH = {r['p_bh']:.4f}")

    print("\n[5] BEER-LAMBERT LINEARITY & MODEL SELECTION (LOOCV)")
    print(df_fits[["band", "epsilon_apparent", "intercept", "R2", "rmse", "linear_loocv_rmse", "saturating_loocv_rmse", "better_model"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    best_band_row = df_fits.loc[df_fits["epsilon_apparent"].abs().idxmax()]
    print(f"\n--> Primary Calibration Channel: {int(best_band_row['band'])} nm")
    print(f"    - Apparent Extinction Slope (epsilon): {best_band_row['epsilon_apparent']:.4f} Delta A / (fraction of stock)")
    print(f"    - Baseline Intercept: {best_band_row['intercept']:.4f} A")
    print(f"    - Goodness of Fit R^2: {best_band_row['R2']:.4f} (RMSE = {best_band_row['rmse']:.4f})")
    print(f"    - Preferred Model: {best_band_row['better_model']} (LOOCV RMSE = {best_band_row['linear_loocv_rmse']:.4f})")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Unified Protocol 3 Calibration Pipeline")
    parser.add_argument("--run-log", default=None, help="Path to dye calibration run log excel")
    parser.add_argument("--spectra", default=None, help="Path to AS7341 spectrometer readings csv")
    parser.add_argument("--outdir", default=os.path.join("outputs", "unified_calibration_analysis"), help="Output directory")
    parser.add_argument("--n-perm", type=int, default=5000, help="Number of label permutations")
    args = parser.parse_args()

    # Locate default files if not passed
    run_log_path = args.run_log or glob.glob("data/calibration/*run_log*.xlsx")[0]
    spectra_files = glob.glob("data/calibration/as7341_spectrum_readings/*.csv")
    spectra_path = args.spectra or spectra_files[0]
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # 1. Load Data
    spec_long, wide = load_spectra(spectra_path)
    ref_start_date = wide.index[0].normalize()
    run_log = load_run_log(run_log_path, ref_start_date)

    # 2. Window extraction (Strict 2-minute duration)
    df_meas = extract_windows(run_log, wide)

    # 3. Dynamic Local Blanking & Apparent Absorbance
    df_meas, blanks, samples = add_local_blank_and_absorbance(df_meas)

    # 4. Diagnostics
    det_diag, q, full_scale = detector_diagnostics(spec_long)
    carry_df, carry_res = carryover_test(df_meas)
    perm_res, null_dists = permutation_signal_test(samples, response="A", n_perm=args.n_perm)

    # 5. Replicate Aggregation & Model Comparisons
    agg = aggregate_replicates(samples)
    model_cmp_A = compare_models_per_band(agg, response="A")
    df_fits = fit_beer_lambert(samples, agg, model_cmp_A)

    # 6. Export Tables
    df_meas.to_csv(os.path.join(outdir, "per_vial_measurements.csv"), index=False)
    agg.to_csv(os.path.join(outdir, "aggregated_calibration_stats.csv"), index=False)
    df_fits.to_csv(os.path.join(outdir, "beer_lambert_fit_parameters.csv"), index=False)
    det_diag.to_csv(os.path.join(outdir, "detector_diagnostics.csv"), index=False)
    if not carry_res.empty:
        carry_res.to_csv(os.path.join(outdir, "carryover_test_results.csv"), index=False)
    if not perm_res.empty:
        perm_res.to_csv(os.path.join(outdir, "permutation_signal_test.csv"), index=False)

    # 7. Generate All Figures
    generate_visualizations(wide, df_meas, blanks, samples, agg, df_fits, model_cmp_A,
                            carry_df, carry_res, null_dists, outdir)

    # 8. Print Executive Verdict
    print_final_verdict(df_meas, samples, agg, blanks, df_fits, model_cmp_A,
                        det_diag, q, full_scale, carry_res, perm_res)
    print(f"All outputs and 8 figures generated in: {os.path.abspath(outdir)}")


if __name__ == "__main__":
    main()
