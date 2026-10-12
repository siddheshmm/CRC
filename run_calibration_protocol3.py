"""
run_calibration_protocol3.py

Rigorous Implementation of Protocol 3: Fixed-Volume Optical Dye Calibration
as defined in pioreactor_protocols/PROTOCOLS.md.

Analyzes data in data/calibration:
- as7341_spectrum_readings/as7341_spectrum_readings-Pulse_experiments-all_units-20260923025303.csv
- dye_calibration_run_log_scaled_15ml.xlsx

Methodology:
1. Uses `timestamp_localtime` as the operational analysis clock as instructed.
2. Audits each row from the Excel Run Log:
   - Extracts 2-minute windows starting at `record_start` (duration = 2.0 min, up to record_end).
   - Identifies skipped initial runs (Order 1, Order 2) occurring prior to acquisition onset.
3. Computes multiband spectral statistics (mean, SD, CV%) across all 8 AS7341 channels (415-680 nm).
4. Evaluates interleaved water blanks across time to determine baseline stability and drift.
5. Computes apparent absorbance: A_band(c) = -log10(I_band / I_blank_band).
6. Aggregates independent replicates (3 per concentration) to compute replicate mean and SEM.
7. Evaluates Beer-Lambert linearity: fits A_band(c) = epsilon * c + intercept and tests linearity (R^2, residuals).
8. Exports comprehensive CSV summaries and 6 publication-quality figures to outputs/calibration_analysis/.
"""

import os
import sys
import glob
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# AS7341 optical bands and visualization colors
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

OUT_DIR = os.path.join("outputs", "calibration_analysis")
os.makedirs(OUT_DIR, exist_ok=True)


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


def make_datetime(t):
    if t is None:
        return None
    # Data was collected on evening of 2026-09-22 spanning into morning of 2026-09-23
    day = 23 if t.hour < 12 else 22
    return pd.Timestamp(year=2026, month=9, day=day, hour=t.hour, minute=t.minute, second=t.second)


def main():
    print("=" * 70)
    print("PROTOCOL 3: AUDITED OPTICAL DYE CALIBRATION ANALYSIS")
    print("=" * 70)

    # 1. Load AS7341 data
    spec_files = glob.glob("data/calibration/as7341_spectrum_readings/*.csv")
    if not spec_files:
        print("ERROR: No AS7341 file found.")
        sys.exit(1)
    spec_path = spec_files[0]
    print(f"Loading AS7341 data from: {spec_path}")
    spec = pd.read_csv(spec_path)

    # Use timestamp_localtime as specified by user
    spec["ts_local"] = pd.to_datetime(spec["timestamp_localtime"])
    spec["ts_sec"] = spec["ts_local"].dt.round("1s")
    wide = spec.pivot_table(index="ts_sec", columns="band", values="reading", aggfunc="mean")
    wide.columns = [int(c) for c in wide.columns]
    wide = wide.sort_index()

    t_data_start = wide.index.min()
    t_data_end = wide.index.max()
    print(f"Acquisition duration: {t_data_start} to {t_data_end} (total {len(wide)} 5-second scans)")

    # 2. Load Excel Run Log
    excel_path = "data/calibration/dye_calibration_run_log_scaled_15ml.xlsx"
    print(f"Loading run log from: {excel_path}")
    log = pd.read_excel(excel_path, sheet_name="Run Log")
    log = log[log["order"] != "EX"].copy()  # Filter out example row

    log["fill_dt"] = log["fill_time"].apply(parse_time_str).apply(make_datetime)
    log["start_dt"] = log["record_start"].apply(parse_time_str).apply(make_datetime)
    log["end_dt"] = log["record_end"].apply(parse_time_str).apply(make_datetime)
    log["win_end_2m"] = log["start_dt"] + pd.Timedelta(minutes=2)

    # 3. Window extraction (2-minute window from record_start)
    print("\nAuditing 2-minute extraction windows across run log...")
    window_records = []
    for idx, r in log.iterrows():
        s_dt = r["start_dt"]
        e_dt = min(r["win_end_2m"], r["end_dt"])
        sub = wide.loc[(wide.index >= s_dt) & (wide.index <= e_dt)]
        n_pts = len(sub)
        is_skipped = (n_pts < 10)  # Fewer than 10 points (50s) considered incomplete/skipped

        rec = {
            "order": r["order"],
            "type": r["type"],
            "fraction": float(r["fraction"]) if pd.notna(r["fraction"]) else 0.0,
            "replicate": int(r["replicate"]) if pd.notna(r["replicate"]) else np.nan,
            "S_mL": r["S_mL"],
            "water_mL": r["water_mL"],
            "fill_time": r["fill_time"],
            "record_start": r["record_start"],
            "record_end": r["record_end"],
            "start_dt": s_dt,
            "end_dt": e_dt,
            "n_points": n_pts,
            "status": "SKIPPED_PRE_RECORDING" if is_skipped else "VALID",
        }

        for b in BANDS:
            if not is_skipped:
                rec[f"I_{b}_mean"] = sub[b].mean()
                rec[f"I_{b}_std"] = sub[b].std()
                rec[f"I_{b}_cv_pct"] = (sub[b].std() / sub[b].mean()) * 100.0 if sub[b].mean() > 0 else 0.0
            else:
                rec[f"I_{b}_mean"] = np.nan
                rec[f"I_{b}_std"] = np.nan
                rec[f"I_{b}_cv_pct"] = np.nan

        window_records.append(rec)

    df_windows = pd.DataFrame(window_records)
    df_windows.to_csv(os.path.join(OUT_DIR, "calibration_summary_table.csv"), index=False)
    print(f"Extracted {len(df_windows)} rows: {(df_windows['status'] == 'VALID').sum()} VALID, {(df_windows['status'] != 'VALID').sum()} SKIPPED.")

    valid_df = df_windows[df_windows["status"] == "VALID"].copy()

    # 4. Interleaved Blank Baseline Analysis
    blanks = valid_df[valid_df["type"] == "blank"].copy()
    print(f"\nAnalyzing {len(blanks)} valid interleaved water blanks...")
    blank_summary = {}
    for b in BANDS:
        b_mean = blanks[f"I_{b}_mean"].mean()
        b_std = blanks[f"I_{b}_mean"].std()
        b_cv = (b_std / b_mean) * 100.0 if b_mean > 0 else 0.0
        blank_summary[b] = {
            "mean_blank": b_mean,
            "std_blank": b_std,
            "cv_blank_pct": b_cv,
            "min_blank": blanks[f"I_{b}_mean"].min(),
            "max_blank": blanks[f"I_{b}_mean"].max(),
        }
    df_blank_summary = pd.DataFrame(blank_summary).T
    df_blank_summary.index.name = "band"
    df_blank_summary.to_csv(os.path.join(OUT_DIR, "blank_drift_summary.csv"))
    print("Blank drift summary across experiment:")
    print(df_blank_summary[["mean_blank", "std_blank", "cv_blank_pct"]])

    # 5. Apparent Absorbance Calculation
    # A_band(c) = -log10(I_band / I_blank_band)
    for b in BANDS:
        I_blank = blank_summary[b]["mean_blank"]
        valid_df[f"A_{b}_mean"] = -np.log10(valid_df[f"I_{b}_mean"] / I_blank)

    # 6. Concentration Aggregation
    samples = valid_df[valid_df["type"] == "sample"].copy()
    conc_stats = []
    for frac, grp in samples.groupby("fraction"):
        row = {
            "fraction": frac,
            "n_replicates": len(grp),
            "orders": str(grp["order"].tolist()),
        }
        for b in BANDS:
            # Transmission stats across replicates
            t_vals = grp[f"I_{b}_mean"]
            row[f"I_{b}_mean"] = t_vals.mean()
            row[f"I_{b}_std"] = t_vals.std(ddof=1) if len(t_vals) > 1 else 0.0
            row[f"I_{b}_sem"] = t_vals.sem() if len(t_vals) > 1 else 0.0

            # Absorbance stats across replicates
            a_vals = grp[f"A_{b}_mean"]
            row[f"A_{b}_mean"] = a_vals.mean()
            row[f"A_{b}_std"] = a_vals.std(ddof=1) if len(a_vals) > 1 else 0.0
            row[f"A_{b}_sem"] = a_vals.sem() if len(a_vals) > 1 else 0.0

        conc_stats.append(row)

    df_conc = pd.DataFrame(conc_stats).sort_values("fraction")
    df_conc.to_csv(os.path.join(OUT_DIR, "concentration_calibration_stats.csv"), index=False)
    print("\nConcentration Statistics Table:")
    print(df_conc[["fraction", "n_replicates", "I_515_mean", "A_515_mean", "A_480_mean", "A_630_mean"]])

    # 7. Beer-Lambert Linearity Fits
    fit_records = []
    for b in BANDS:
        c_vals = samples["fraction"].values
        a_vals = samples[f"A_{b}_mean"].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(c_vals, a_vals)
        r2 = r_value ** 2
        pred_a = slope * c_vals + intercept
        residuals = a_vals - pred_a
        rmse = np.sqrt(np.mean(residuals ** 2))

        fit_records.append({
            "band": b,
            "epsilon_apparent_slope": slope,
            "intercept": intercept,
            "R2": r2,
            "p_value": p_value,
            "std_err": std_err,
            "rmse": rmse,
        })

    df_fits = pd.DataFrame(fit_records).sort_values("band")
    df_fits.to_csv(os.path.join(OUT_DIR, "beer_lambert_fit_parameters.csv"), index=False)
    print("\nBeer-Lambert Linearity Fit Parameters:")
    print(df_fits[["band", "epsilon_apparent_slope", "intercept", "R2", "rmse"]])

    # 8. Generate Publication-Quality Plots
    print("\nGenerating figures...")

    # Plot 1: Full raw time series with extraction windows
    fig, ax = plt.subplots(figsize=(15, 6), dpi=200)
    for b in [445, 480, 515, 630]:
        ax.plot(wide.index, wide[b], label=f"{b} nm", color=BAND_COLORS[b], lw=1.2, alpha=0.85)

    # Highlight windows
    for idx, r in df_windows.iterrows():
        if r["status"] == "VALID":
            color = "#a1dab4" if r["type"] == "blank" else "#fed976"
            ax.axvspan(r["start_dt"], r["end_dt"], color=color, alpha=0.45, lw=0)
            ax.text(r["start_dt"], 0.26, str(r["order"]), fontsize=6, ha="center", va="bottom", rotation=90)
        else:
            ax.axvspan(r["start_dt"], r["end_dt"], color="#fbb4ae", alpha=0.45, lw=0)
            ax.text(r["start_dt"], 0.26, f"{r['order']}*", fontsize=6, ha="center", va="bottom", rotation=90, color="red")

    ax.set_title("Protocol 3: AS7341 Optical Trace with Audited 2-Minute Extraction Windows\n"
                 "(Green = Interleaved Blanks, Yellow = Calibrated Samples, Red = Skipped Pre-Recording Runs)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Local Time (timestamp_localtime)", fontsize=10)
    ax.set_ylabel("Optical Transmission Reading (AS7341)", fontsize=10)
    ax.set_ylim(0.0, 0.29)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "plot1_calibration_raw_timeseries_with_windows.png"))
    plt.close(fig)

    # Plot 2: Interleaved Blank Stability & Drift
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=200, sharex=True)
    for b in BANDS:
        ax1.plot(blanks["start_dt"], blanks[f"I_{b}_mean"], "o-", label=f"{b} nm", color=BAND_COLORS[b], lw=1.5, markersize=5)
        # Normalized drift % relative to first valid blank
        first_val = blanks[f"I_{b}_mean"].iloc[0]
        drift_pct = (blanks[f"I_{b}_mean"] - first_val) / first_val * 100.0
        ax2.plot(blanks["start_dt"], drift_pct, "o-", label=f"{b} nm", color=BAND_COLORS[b], lw=1.5, markersize=5)

    ax1.set_title("Interleaved Water Blank Baseline Stability (5-Hour Calibration Run)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Blank Transmission Reading I_blank", fontsize=10)
    ax1.legend(ncol=4, fontsize=8, loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.set_title("Baseline Relative Drift (%) from Initial Blank", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Local Time", fontsize=10)
    ax2.set_ylabel("Relative Drift (%)", fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "plot2_blank_baseline_stability.png"))
    plt.close(fig)

    # Plot 3: Spectral Profiles vs Dye Concentration
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(df_conc)))

    for idx, (_, r) in enumerate(df_conc.iterrows()):
        frac = r["fraction"]
        t_spec = [r[f"I_{b}_mean"] for b in BANDS]
        t_err = [r[f"I_{b}_sem"] for b in BANDS]
        a_spec = [r[f"A_{b}_mean"] for b in BANDS]
        a_err = [r[f"A_{b}_sem"] for b in BANDS]

        label = f"Frac {frac:.2f} (n={int(r['n_replicates'])})"
        ax1.errorbar(BANDS, t_spec, yerr=t_err, marker="o", lw=1.5, capsize=3, label=label, color=cmap[idx])
        ax2.errorbar(BANDS, a_spec, yerr=a_err, marker="s", lw=1.5, capsize=3, label=label, color=cmap[idx])

    ax1.set_title("A. Spectral Transmission Profile I(lambda)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Wavelength (nm)", fontsize=10)
    ax1.set_ylabel("Transmission Reading I", fontsize=10)
    ax1.set_xticks(BANDS)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=7, loc="lower left")

    ax2.set_title("B. Apparent Absorbance Profile A(lambda) = -log10(I / I_blank)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Wavelength (nm)", fontsize=10)
    ax2.set_ylabel("Apparent Absorbance A", fontsize=10)
    ax2.set_xticks(BANDS)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "plot3_spectral_profiles_vs_concentration.png"))
    plt.close(fig)

    # Plot 4: Beer-Lambert Calibration Curves (Apparent Absorbance vs Concentration)
    key_bands = [445, 480, 515, 555]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=200)
    axes = axes.flatten()

    for i, b in enumerate(key_bands):
        ax = axes[i]
        fit_row = df_fits[df_fits["band"] == b].iloc[0]
        slope = fit_row["epsilon_apparent_slope"]
        intercept = fit_row["intercept"]
        r2 = fit_row["R2"]

        # Replicate scatter
        ax.scatter(samples["fraction"], samples[f"A_{b}_mean"], color=BAND_COLORS[b], alpha=0.6, s=35, label="Individual Vials (Replicates)")

        # Aggregated mean with error bars
        ax.errorbar(df_conc["fraction"], df_conc[f"A_{b}_mean"], yerr=df_conc[f"A_{b}_sem"],
                     fmt="o", color=BAND_COLORS[b], ecolor="black", elinewidth=1.5, capsize=4, markersize=7, label="Mean ± SEM (n=3)")

        # Fit line
        c_grid = np.linspace(0, 1.0, 100)
        fit_line = slope * c_grid + intercept
        ax.plot(c_grid, fit_line, "--", color="black", lw=1.8,
                label=f"Fit: A = {slope:.3f}·c + {intercept:.3f}\n$R^2$ = {r2:.4f}, RMSE = {fit_row['rmse']:.4f}")

        ax.set_title(f"Channel {b} nm (Beer-Lambert Linearity)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Stock Dye Fraction c (fraction of S)", fontsize=10)
        ax.set_ylabel(f"Apparent Absorbance $A_{{{b}}}$", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, loc="upper left", frameon=True)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "plot4_calibration_curves_beer_lambert.png"))
    plt.close(fig)

    # Plot 5: Transmission Calibration Curves (I vs c)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=200)
    axes = axes.flatten()

    for i, b in enumerate(key_bands):
        ax = axes[i]
        I_blank = blank_summary[b]["mean_blank"]
        fit_row = df_fits[df_fits["band"] == b].iloc[0]
        slope = fit_row["epsilon_apparent_slope"]

        # Exponential decay curve: I = I_blank * 10^(-epsilon * c)
        c_grid = np.linspace(0, 1.0, 100)
        exp_pred = I_blank * (10.0 ** (-slope * c_grid))

        ax.scatter(samples["fraction"], samples[f"I_{b}_mean"], color=BAND_COLORS[b], alpha=0.6, s=35, label="Individual Vials")
        ax.errorbar(df_conc["fraction"], df_conc[f"I_{b}_mean"], yerr=df_conc[f"I_{b}_sem"],
                     fmt="o", color=BAND_COLORS[b], ecolor="black", elinewidth=1.5, capsize=4, markersize=7, label="Mean ± SEM")
        ax.plot(c_grid, exp_pred, "--", color="darkred", lw=1.8, label=f"Beer-Lambert: $I = I_0 \\cdot 10^{{-{slope:.2f}c}}$")

        ax.set_title(f"Channel {b} nm: Transmission Decay", fontsize=11, fontweight="bold")
        ax.set_xlabel("Stock Dye Fraction c", fontsize=10)
        ax.set_ylabel(f"Transmission Reading $I_{{{b}}}$", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, loc="upper right", frameon=True)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "plot5_transmission_calibration_curves.png"))
    plt.close(fig)

    # Plot 6: Linearity Residuals
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=200)
    axes = axes.flatten()

    for i, b in enumerate(key_bands):
        ax = axes[i]
        fit_row = df_fits[df_fits["band"] == b].iloc[0]
        slope = fit_row["epsilon_apparent_slope"]
        intercept = fit_row["intercept"]

        c_vals = samples["fraction"].values
        a_vals = samples[f"A_{b}_mean"].values
        pred = slope * c_vals + intercept
        res = a_vals - pred

        ax.scatter(c_vals, res, color=BAND_COLORS[b], s=40, edgecolors="black", lw=0.5)
        ax.axhline(0, color="black", linestyle="--", lw=1.2)
        ax.set_title(f"Channel {b} nm: Absorbance Residuals", fontsize=11, fontweight="bold")
        ax.set_xlabel("Stock Dye Fraction c", fontsize=10)
        ax.set_ylabel("Residual (Observed - Fitted A)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "plot6_fit_residuals_and_linearity_assessment.png"))
    plt.close(fig)

    print("\nAll figures generated successfully:")
    for f in glob.glob(os.path.join(OUT_DIR, "*.png")):
        print("  -", os.path.basename(f))

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
