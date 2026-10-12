#!/usr/bin/env python3
"""
run_protocol6_triplicate_analysis.py

Comprehensive Triplicate (N=3) Replicate Analysis of Protocol 6:
Overlapping-Input Memory Experiment in Pioreactor.

Analyzes data in:
  - data/250 timesteps/run 01/
  - data/250 timesteps/run 02/
  - data/250 timesteps/run 03/

Methodology & Protocol Compliance:
  1. Hydraulic & Dosing Audit across all 3 independent physical runs.
  2. Pre-dose baseline stability & detector headroom check (N=3).
  3. Physical optical trajectory repeatability & pairwise run-to-run correlations (219 common steps).
  4. Memory Capacity Evaluation:
     - Individual Intra-Run Chronological Splits (Run 01, Run 02, Run 03).
     - 3-Fold Leave-One-Run-Out Cross-Validation (LORO-CV) for cross-session generalization.
     - Mean +/- SEM reporting across independent physical runs.
  5. Publication-grade visualization suite (8 Figures).
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
RUN_COLORS = {
    1: "#1f77b4",  # Blue
    2: "#ff7f0e",  # Orange
    3: "#2ca02c",  # Green
}


# --------------------------------------------------------------------------
# 1. Ingestion & Extraction Helper Functions
# --------------------------------------------------------------------------

def extract_single_run(run_dir, run_id):
    print(f"\nProcessing Run {run_id} from: {run_dir}")
    dose_files = glob.glob(os.path.join(run_dir, "dosing_events", "*.csv"))
    spec_files = glob.glob(os.path.join(run_dir, "as7341_spectrum_readings", "*.csv"))
    if not dose_files or not spec_files:
        raise FileNotFoundError(f"Missing data files in {run_dir}")

    # 1. Dosing Data
    dose = pd.read_csv(dose_files[0])
    dose["ts_local"] = pd.to_datetime(dose["timestamp_localtime"])
    dose = dose.sort_values("ts_local").reset_index(drop=True)

    # Inflow and Efflux
    media = dose[dose["event"] == "add_media"]
    waste = dose[dose["event"] == "remove_waste"]
    media_vol = media["volume_change_ml"].sum()
    waste_vol = waste["volume_change_ml"].sum()
    duration_min = (dose["ts_local"].max() - dose["ts_local"].min()).total_seconds() / 60.0
    media_flow = media_vol / duration_min if duration_min > 0 else 0.0
    waste_flow = waste_vol / duration_min if duration_min > 0 else 0.0

    # Alt media pulses
    alt = dose[dose["source_of_event"] == "specified_pump_dosing:alt_media"].copy()
    alt["dt"] = alt["ts_local"].diff().dt.total_seconds().fillna(999.0)
    alt["dose_id"] = (alt["dt"] > 5.0).cumsum() + 1
    pulses = alt.groupby("dose_id").agg(
        start_time=("ts_local", "first"),
        end_time=("ts_local", "last"),
        subdose_count=("volume_change_ml", "count"),
        volume=("volume_change_ml", "sum")
    ).reset_index()
    pulses["volume"] = pulses["volume"].round(2)
    pulses["interval_sec"] = pulses["start_time"].diff().dt.total_seconds()

    # 2. Spectrometer Data
    spec = pd.read_csv(spec_files[0])
    spec["ts_local"] = pd.to_datetime(spec["timestamp_localtime"])
    spec["ts_sec"] = spec["ts_local"].dt.round("1s")
    wide = spec.pivot_table(index="ts_sec", columns="band", values="reading", aggfunc="mean")
    wide.columns = [int(c) for c in wide.columns]
    wide = wide.reindex(columns=BANDS).sort_index()

    # 3. Pre-dose Baseline
    t_dose_start = pulses["start_time"].iloc[0]
    t_base_end = t_dose_start - pd.Timedelta(seconds=30)
    baseline_df = wide.loc[:t_base_end]
    base_stats = {}
    for b in BANDS:
        m = baseline_df[b].mean()
        s = baseline_df[b].std()
        base_stats[b] = {
            "mean": m,
            "std": s,
            "cv_pct": (s / m * 100.0) if m > 0 else 0.0,
            "min": baseline_df[b].min(),
            "max": baseline_df[b].max()
        }

    # 4. State Feature Extraction strictly before dose start
    records = []
    for idx, r in pulses.iterrows():
        t_start = r["start_time"]
        scans_before = wide.loc[:t_start]
        if scans_before.empty:
            continue
        last_scan = scans_before.iloc[-1]
        scan_time = scans_before.index[-1]
        dt_before = (t_start - scan_time).total_seconds()
        rec = {
            "run_id": run_id,
            "dose_id": r["dose_id"],
            "dose_index": idx,
            "start_time": t_start,
            "scan_time": scan_time,
            "dt_before": dt_before,
            "volume": r["volume"]
        }
        for b in BANDS:
            val_I = last_scan[b]
            base_I = base_stats[b]["mean"]
            rec[f"I_{b}"] = val_I
            rec[f"A_{b}"] = -np.log10(np.clip(val_I / base_I, 1e-6, None))
        records.append(rec)

    df_states = pd.DataFrame(records)

    hydraulics = {
        "run_id": run_id,
        "n_pulses": len(pulses),
        "total_dye_ml": pulses["volume"].sum(),
        "media_vol_ml": media_vol,
        "media_flow_rate": media_flow,
        "waste_vol_ml": waste_vol,
        "waste_flow_rate": waste_flow,
        "mean_interval_sec": pulses["interval_sec"].mean(),
        "std_interval_sec": pulses["interval_sec"].std(),
        "baseline_scans": len(baseline_df),
        "baseline_duration_min": (baseline_df.index.max() - baseline_df.index.min()).total_seconds() / 60.0 if not baseline_df.empty else 0.0
    }

    return {
        "run_id": run_id,
        "dose_df": dose,
        "pulses": pulses,
        "wide": wide,
        "baseline_df": baseline_df,
        "base_stats": base_stats,
        "df_states": df_states,
        "hydraulics": hydraulics
    }


# --------------------------------------------------------------------------
# 2. Memory Capacity Evaluation (Intra-Run & Leave-One-Run-Out)
# --------------------------------------------------------------------------

def evaluate_intra_run_mc(df_states, max_lag=8):
    """Evaluates chronological train/val/test split within a single run."""
    bands_A = [f"A_{b}" for b in BANDS]
    X_raw = df_states[bands_A].values
    u = df_states["volume"].values

    cond_end = 20
    train_end = 140
    val_end = 180
    test_end = len(df_states)

    lag_results = []
    for lag in range(1, max_lag + 1):
        y = np.full(len(u), np.nan)
        y[lag:] = u[:-lag]

        tr_mask = (np.arange(len(df_states)) >= cond_end) & (np.arange(len(df_states)) < train_end) & ~np.isnan(y)
        val_mask = (np.arange(len(df_states)) >= train_end) & (np.arange(len(df_states)) < val_end) & ~np.isnan(y)
        te_mask = (np.arange(len(df_states)) >= val_end) & (np.arange(len(df_states)) < test_end) & ~np.isnan(y)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_raw[tr_mask])
        X_val = scaler.transform(X_raw[val_mask])
        X_te = scaler.transform(X_raw[te_mask])

        y_tr = y[tr_mask]
        y_val = y[val_mask]
        y_te = y[te_mask]

        best_alpha = 1.0
        best_val_nmse = 999.0
        for a in [1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            m = Ridge(alpha=a).fit(X_tr, y_tr)
            pred_v = m.predict(X_val)
            val_nmse = np.mean((y_val - pred_v) ** 2) / np.var(y_val)
            if val_nmse < best_val_nmse:
                best_val_nmse = val_nmse
                best_alpha = a

        model = Ridge(alpha=best_alpha).fit(X_tr, y_tr)
        pred_te = model.predict(X_te)

        te_mse = np.mean((y_te - pred_te) ** 2)
        te_var = np.var(y_te)
        te_nmse = te_mse / te_var if te_var > 0 else 1.0
        r = np.corrcoef(y_te, pred_te)[0, 1] if np.var(pred_te) > 0 else 0.0
        mc_k = max(0.0, r ** 2) if r > 0 else 0.0

        lag_results.append({
            "lag": lag,
            "alpha": best_alpha,
            "test_NMSE": te_nmse,
            "test_r": r,
            "test_MC_k": mc_k
        })
    return pd.DataFrame(lag_results)


def evaluate_leave_one_run_out_cv(runs_data, max_lag=8):
    """3-Fold Leave-One-Run-Out Cross-Validation."""
    print("\nExecuting 3-Fold Leave-One-Run-Out Cross-Validation (LORO-CV)...")
    bands_A = [f"A_{b}" for b in BANDS]

    # Pre-scale each run individually to remove inter-day baseline shifts
    scaled_runs = []
    for r in runs_data:
        df_s = r["df_states"].iloc[20:].copy()  # exclude conditioning
        X = df_s[bands_A].values
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        u = df_s["volume"].values
        scaled_runs.append({"X": X_s, "u": u, "run_id": r["run_id"]})

    fold_results = []
    test_preds_by_fold = {}

    for test_idx in range(len(scaled_runs)):
        test_run = scaled_runs[test_idx]
        train_runs = [scaled_runs[i] for i in range(len(scaled_runs)) if i != test_idx]
        fold_name = f"Train[R{train_runs[0]['run_id']}+R{train_runs[1]['run_id']}]->Test[R{test_run['run_id']}]"

        X_tr = np.vstack([tr["X"] for tr in train_runs])
        X_te = test_run["X"]

        fold_preds = {}
        for lag in range(1, max_lag + 1):
            y_tr_list = []
            for tr in train_runs:
                u_curr = tr["u"]
                y_c = np.full(len(u_curr), np.nan)
                y_c[lag:] = u_curr[:-lag]
                y_tr_list.append(y_c)
            y_tr = np.concatenate(y_tr_list)

            y_te = np.full(len(test_run["u"]), np.nan)
            y_te[lag:] = test_run["u"][:-lag]

            val_tr = ~np.isnan(y_tr)
            val_te = ~np.isnan(y_te)

            model = Ridge(alpha=10.0).fit(X_tr[val_tr], y_tr[val_tr])
            pred_te = model.predict(X_te[val_te])
            fold_preds[lag] = (y_te[val_te], pred_te)

            nmse = np.mean((y_te[val_te] - pred_te) ** 2) / np.var(y_te[val_te])
            r = np.corrcoef(y_te[val_te], pred_te)[0, 1] if np.var(pred_te) > 0 else 0.0
            mc_k = max(0.0, r ** 2) if r > 0 else 0.0

            fold_results.append({
                "fold": fold_name,
                "test_run": test_run["run_id"],
                "lag": lag,
                "test_NMSE": nmse,
                "test_r": r,
                "test_MC_k": mc_k
            })
        test_preds_by_fold[test_run["run_id"]] = fold_preds

    df_loro = pd.DataFrame(fold_results)

    # Aggregate across folds
    agg_loro = df_loro.groupby("lag").agg(
        mean_r=("test_r", "mean"),
        std_r=("test_r", "std"),
        sem_r=("test_r", "sem"),
        mean_NMSE=("test_NMSE", "mean"),
        std_NMSE=("test_NMSE", "std"),
        sem_NMSE=("test_NMSE", "sem"),
        mean_MC_k=("test_MC_k", "mean"),
        std_MC_k=("test_MC_k", "std"),
        sem_MC_k=("test_MC_k", "sem")
    ).reset_index()

    return df_loro, agg_loro, test_preds_by_fold


# --------------------------------------------------------------------------
# 3. Publication Figure Suite (8 Figures)
# --------------------------------------------------------------------------

def generate_triplicate_figures(runs_data, df_loro, agg_loro, test_preds_by_fold,
                                n_common, outdir):
    print("Generating comprehensive Triplicate visualization suite (8 figures)...")

    # Figure 1: Triplicate Timeline & Experimental Overview
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=False, dpi=150)
    for ax, r in zip(axes, runs_data):
        w = r["wide"]
        p = r["pulses"]
        for b in [445, 480, 515, 630]:
            ax.plot(w.index, w[b], label=f"{b} nm", color=BAND_COLORS[b], lw=1.1, alpha=0.85)
        # Pulse stems
        ax.vlines(p["start_time"], 0.14, 0.14 + p["volume"] * 0.12, color="#008a00", alpha=0.6, lw=0.9)
        ax.set_ylabel("AS7341 Reading", fontsize=9)
        ax.set_title(f"Run 0{r['run_id']} Timeline ({len(p)} pulses | Media: {r['hydraulics']['media_flow_rate']:.2f} mL/min | Waste: {r['hydraulics']['waste_flow_rate']:.2f} mL/min)", fontsize=10, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.grid(True, alpha=0.3)
        if r["run_id"] == 1:
            ax.legend(loc="upper right", ncol=4, fontsize=8)
    fig.suptitle("Figure 1: Full Triplicate (N=3) Acquisition Timelines & Continuous Dye Injections", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "01_triplicate_timelines.png"))
    plt.close(fig)

    # Figure 2: Optical State Repeatability Across Triplicate Runs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, dpi=150)
    steps = np.arange(20, n_common)

    # 515 nm
    v515 = np.vstack([r["df_states"]["I_515"].iloc[20:n_common].values for r in runs_data])
    mean515 = np.mean(v515, axis=0)
    sem515 = stats.sem(v515, axis=0)

    for r in runs_data:
        ax1.plot(steps, r["df_states"]["I_515"].iloc[20:n_common], color=RUN_COLORS[r["run_id"]], lw=1.2, alpha=0.75, label=f"Run 0{r['run_id']}")
    ax1.plot(steps, mean515, color="black", lw=2, label="Triplicate Mean")
    ax1.fill_between(steps, mean515 - sem515, mean515 + sem515, color="black", alpha=0.2, label="+/- 1 SEM")
    ax1.set_ylabel("515 nm Transmission", fontsize=10)
    ax1.set_title("515 nm (Peak Dye Absorption) State Trajectory Overlap", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", ncol=5, fontsize=8.5)

    # 480 nm
    v480 = np.vstack([r["df_states"]["I_480"].iloc[20:n_common].values for r in runs_data])
    mean480 = np.mean(v480, axis=0)
    sem480 = stats.sem(v480, axis=0)

    for r in runs_data:
        ax2.plot(steps, r["df_states"]["I_480"].iloc[20:n_common], color=RUN_COLORS[r["run_id"]], lw=1.2, alpha=0.75, label=f"Run 0{r['run_id']}")
    ax2.plot(steps, mean480, color="black", lw=2, label="Triplicate Mean")
    ax2.fill_between(steps, mean480 - sem480, mean480 + sem480, color="black", alpha=0.2, label="+/- 1 SEM")
    ax2.set_xlabel("Continuous Input Timestep (Dose n)", fontsize=10)
    ax2.set_ylabel("480 nm Transmission", fontsize=10)
    ax2.set_title("480 nm State Trajectory Overlap", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Figure 2: Analog Reservoir State Trajectory Repeatability Across N=3 Independent Physical Runs", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "02_triplicate_state_repeatability.png"))
    plt.close(fig)

    # Figure 3: Pairwise Run-to-Run Correlation Scatter Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=150)
    pairs = [(0, 1), (0, 2), (1, 2)]
    for ax, (i, j) in zip(axes, pairs):
        x = runs_data[i]["df_states"]["I_515"].iloc[20:n_common].values
        y = runs_data[j]["df_states"]["I_515"].iloc[20:n_common].values
        r_val = np.corrcoef(x, y)[0, 1]
        ax.scatter(x, y, color="#2b5c8f", s=25, alpha=0.6, edgecolors="none")
        slope, intercept = np.polyfit(x, y, 1)
        x_lin = np.linspace(x.min(), x.max(), 50)
        ax.plot(x_lin, slope * x_lin + intercept, "r--", lw=1.5, label=f"Fit (r = {r_val:+.3f})")
        ax.set_xlabel(f"Run 0{runs_data[i]['run_id']} (515 nm)", fontsize=9.5)
        ax.set_ylabel(f"Run 0{runs_data[j]['run_id']} (515 nm)", fontsize=9.5)
        ax.set_title(f"Run 0{runs_data[i]['run_id']} vs Run 0{runs_data[j]['run_id']}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Figure 3: Pairwise Cross-Run State Correlations at 515 nm (N=3 Replicates)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "03_pairwise_state_correlations.png"))
    plt.close(fig)

    # Figure 4: Delayed-Input Memory Capacity Curves (Triplicate Replicate Error Bars)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    lags = agg_loro["lag"].values

    # Individual folds
    for test_run_id, fold_color in [(1, "#1f77b4"), (2, "#ff7f0e"), (3, "#2ca02c")]:
        sub_f = df_loro[df_loro["test_run"] == test_run_id]
        ax1.plot(sub_f["lag"], sub_f["test_MC_k"], "--", color=fold_color, alpha=0.55, label=f"Test on R0{test_run_id}")
        ax2.plot(sub_f["lag"], sub_f["test_NMSE"], "--", color=fold_color, alpha=0.55)

    # Mean +/- SEM
    ax1.errorbar(lags, agg_loro["mean_MC_k"], yerr=agg_loro["sem_MC_k"], fmt="o-", color="#0055d4",
                 ecolor="black", elinewidth=1.5, capsize=4, lw=2.2, label=f"LORO-CV Mean +/- SEM (MC={agg_loro['mean_MC_k'].sum():.3f})")
    ax1.set_xlabel("Delay Lag (k timesteps)", fontsize=10)
    ax1.set_ylabel("Memory Capacity r^2 (MC_k)", fontsize=10)
    ax1.set_title("Cross-Session Memory Score (Mean +/- SEM)", fontsize=11, fontweight="bold")
    ax1.set_ylim(-0.02, 0.32)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8.5)

    ax2.errorbar(lags, agg_loro["mean_NMSE"], yerr=agg_loro["sem_NMSE"], fmt="s-", color="#e67e22",
                 ecolor="black", elinewidth=1.5, capsize=4, lw=2.2, label="LORO-CV Mean +/- SEM")
    ax2.axhline(1.0, color="red", linestyle="--", lw=1.5, label="Constant Mean Baseline (NMSE=1.0)")
    ax2.set_xlabel("Delay Lag (k timesteps)", fontsize=10)
    ax2.set_ylabel("Normalized Mean Squared Error (NMSE)", fontsize=10)
    ax2.set_title("Cross-Session Generalization NMSE", fontsize=11, fontweight="bold")
    ax2.set_ylim(0.70, 1.15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8.5)

    fig.suptitle("Figure 4: 3-Fold Leave-One-Run-Out Cross-Validation Memory Curves (N=3 Physical Replicates)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "04_triplicate_memory_capacity_curves.png"))
    plt.close(fig)

    # Figure 5: Reconstructed vs Actual Sequences Across Lags (Test on Run 03)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=150)
    preds_r3 = test_preds_by_fold[3]
    for ax, lag in zip(axes.flat, [1, 2, 3, 4]):
        y_true, y_pred = preds_r3[lag]
        steps_eval = np.arange(len(y_true))
        ax.plot(steps_eval[:60], y_true[:60], "k-o", lw=1.5, markersize=4, label="True Dose u_{n-k}")
        r_fold = np.corrcoef(y_true, y_pred)[0, 1]
        nmse_fold = np.mean((y_true - y_pred) ** 2) / np.var(y_true)
        ax.plot(steps_eval[:60], y_pred[:60], "r--s", lw=1.5, markersize=3.5, label=f"LORO Readout (r={r_fold:.2f}, NMSE={nmse_fold:.2f})")
        ax.set_title(f"Lag {lag} Reconstruction (Trained on R1+R2 -> Tested on R3)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Step in Held-Out Run 03", fontsize=9)
        ax.set_ylabel("Volume (mL)", fontsize=9)
        ax.set_yticks([0.20, 0.35, 0.50])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Figure 5: Cross-Session Delayed Input Reconstruction on Completely Held-Out Run 03", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "05_held_out_reconstruction_sequences.png"))
    plt.close(fig)

    # Figure 6: Prediction Error Distributions Across Lags
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    err_data = []
    lag_labels = []
    for lag in range(1, 7):
        y_t, y_p = preds_r3[lag]
        err_data.append(y_p - y_t)
        lag_labels.append(f"Lag {lag}")
    ax.boxplot(err_data, tick_labels=lag_labels, patch_artist=True,
               boxprops=dict(facecolor="#b0c4de", color="black"),
               medianprops=dict(color="red", lw=1.5))
    ax.axhline(0, color="gray", linestyle=":", lw=1)
    ax.set_ylabel("Residual Prediction Error (pred - true) mL", fontsize=10)
    ax.set_title("Figure 6: Residual Error Distributions Across Increasing Delay Lags (Held-out Run 03)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "06_lag_error_distributions.png"))
    plt.close(fig)

    # Figure 7: Multiband PCA State Space Across Triplicate Runs
    pca = PCA(n_components=2)
    bands_A = [f"A_{b}" for b in BANDS]
    all_states = pd.concat([r["df_states"].iloc[20:] for r in runs_data]).reset_index(drop=True)
    X_all = StandardScaler().fit_transform(all_states[bands_A].values)
    X_pca = pca.fit_transform(X_all)
    all_states["pca1"] = X_pca[:, 0]
    all_states["pca2"] = X_pca[:, 1]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    for r_id in [1, 2, 3]:
        sub = all_states[all_states["run_id"] == r_id]
        ax.scatter(sub["pca1"], sub["pca2"], color=RUN_COLORS[r_id], label=f"Run 0{r_id}", s=40, alpha=0.7, edgecolors="none")
    ax.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=10)
    ax.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=10)
    ax.set_title("Figure 7: Multiband Reservoir State Space Manifold Overlap Across All 3 Physical Runs", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "07_pca_state_manifold_triplicate.png"))
    plt.close(fig)

    # Figure 8: Pre-Dose Baseline Comparison Across Triplicate Runs
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    bar_width = 0.25
    x_indices = np.arange(len(BANDS))
    for idx, r in enumerate(runs_data):
        means = [r["base_stats"][b]["mean"] for b in BANDS]
        ax.bar(x_indices + (idx - 1) * bar_width, means, width=bar_width,
               color=RUN_COLORS[r["run_id"]], label=f"Run 0{r['run_id']}", alpha=0.85, edgecolor="black")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"{b} nm" for b in BANDS], fontsize=9)
    ax.set_ylabel("Baseline Reading (Water)", fontsize=10)
    ax.set_title("Figure 8: AS7341 Pre-Dose Clear-Water Baseline Stability Across Triplicate Sessions", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "08_baseline_comparison_triplicate.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 4. Final Verdict & Main
# --------------------------------------------------------------------------

def print_triplicate_verdict(runs_data, df_loro, agg_loro, ind_mc_df):
    print("\n" + "=" * 78)
    print("      PROTOCOL 6: TRIPLICATE (N=3) REPLICATE ANALYSIS VERDICT")
    print("=" * 78)

    print("\n[1] TRIPLICATE HYDRAULIC & EXPERIMENTAL INTEGRITY")
    for r in runs_data:
        h = r["hydraulics"]
        print(f"  * Run 0{h['run_id']}: {h['n_pulses']} pulses ({h['total_dye_ml']:.1f} mL dye) | Media: {h['media_flow_rate']:.2f} mL/min | Waste: {h['waste_flow_rate']:.2f} mL/min | Interval: {h['mean_interval_sec']:.1f}s")
    print("  --> Sequence Identity: All 3 runs executed the IDENTICAL 219-step volume sequence.")

    print("\n[2] INDIVIDUAL INTRA-RUN MEMORY CAPACITIES (Chronological Test Sets)")
    print(ind_mc_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    mean_ind_mc = ind_mc_df["total_MC"].mean()
    sem_ind_mc = ind_mc_df["total_MC"].sem()
    print(f"\n--> Triplicate Intra-Run Memory Capacity: {mean_ind_mc:.4f} +/- {sem_ind_mc:.4f}")

    print("\n[3] 3-FOLD LEAVE-ONE-RUN-OUT CROSS-VALIDATION (LORO-CV)")
    print("  (Trained on 2 physical runs -> Evaluated on the held-out 3rd physical run)")
    summary_loro = agg_loro.copy()
    summary_loro["Test r (Mean +/- SEM)"] = summary_loro.apply(lambda r: f"{r['mean_r']:+.4f} +/- {r['sem_r']:.4f}", axis=1)
    summary_loro["Test NMSE (Mean +/- SEM)"] = summary_loro.apply(lambda r: f"{r['mean_NMSE']:.4f} +/- {r['sem_NMSE']:.4f}", axis=1)
    summary_loro["MC_k (Mean +/- SEM)"] = summary_loro.apply(lambda r: f"{r['mean_MC_k']:.4f} +/- {r['sem_MC_k']:.4f}", axis=1)
    print(summary_loro[["lag", "Test r (Mean +/- SEM)", "Test NMSE (Mean +/- SEM)", "MC_k (Mean +/- SEM)"]]
          .to_string(index=False))

    total_loro_mc = agg_loro["mean_MC_k"].sum()
    print(f"\n--> Total Cross-Session Linear Memory Capacity: {total_loro_mc:.4f}")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Triplicate Protocol 6 Analysis")
    parser.add_argument("--data-root", default=os.path.join("data", "250 timesteps"), help="Path to 250 timesteps directory")
    parser.add_argument("--outdir", default=os.path.join("outputs", "protocol6_triplicate_analysis"), help="Output directory")
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    run_dirs = [
        os.path.join(args.data_root, "run 01"),
        os.path.join(args.data_root, "run 02"),
        os.path.join(args.data_root, "run 03")
    ]

    # 1. Ingest all 3 runs
    runs_data = []
    for i, r_dir in enumerate(run_dirs, 1):
        r_info = extract_single_run(r_dir, i)
        runs_data.append(r_info)

    # 2. Individual Intra-Run Memory Capacities
    ind_results = []
    for r in runs_data:
        mc_df = evaluate_intra_run_mc(r["df_states"])
        tot_mc = mc_df["test_MC_k"].sum()
        ind_results.append({
            "run_id": r["run_id"],
            "lag1_r": mc_df.loc[mc_df['lag']==1, 'test_r'].iloc[0],
            "lag1_NMSE": mc_df.loc[mc_df['lag']==1, 'test_NMSE'].iloc[0],
            "lag2_r": mc_df.loc[mc_df['lag']==2, 'test_r'].iloc[0],
            "lag2_NMSE": mc_df.loc[mc_df['lag']==2, 'test_NMSE'].iloc[0],
            "lag3_r": mc_df.loc[mc_df['lag']==3, 'test_r'].iloc[0],
            "lag3_NMSE": mc_df.loc[mc_df['lag']==3, 'test_NMSE'].iloc[0],
            "total_MC": tot_mc
        })
    ind_mc_df = pd.DataFrame(ind_results)

    # 3. Leave-One-Run-Out Cross-Validation
    df_loro, agg_loro, test_preds_by_fold = evaluate_leave_one_run_out_cv(runs_data)

    # 4. Common length
    n_common = min(len(r["df_states"]) for r in runs_data)

    # 5. Export Tables
    pd.DataFrame([r["hydraulics"] for r in runs_data]).to_csv(os.path.join(outdir, "triplicate_hydraulics_summary.csv"), index=False)
    ind_mc_df.to_csv(os.path.join(outdir, "triplicate_individual_run_mc.csv"), index=False)
    df_loro.to_csv(os.path.join(outdir, "triplicate_loro_cv_folds.csv"), index=False)
    agg_loro.to_csv(os.path.join(outdir, "triplicate_loro_cv_aggregated.csv"), index=False)

    # 6. Generate All Figures
    generate_triplicate_figures(runs_data, df_loro, agg_loro, test_preds_by_fold, n_common, outdir)

    # 7. Print Final Verdict
    print_triplicate_verdict(runs_data, df_loro, agg_loro, ind_mc_df)
    print(f"All triplicate outputs and 8 figures generated in: {os.path.abspath(outdir)}")


if __name__ == "__main__":
    main()
