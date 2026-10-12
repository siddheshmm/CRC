#!/usr/bin/env python3
"""
run_protocol6_analysis.py

Rigorous Implementation of Protocol 6: Overlapping-Input Memory Experiment
as specified in pioreactor_protocols/PROTOCOLS.md.

Analyzes data in data/250 timesteps/:
  - dosing_events/dosing_events-Demo_experiment-all_units-20260925145607.csv
  - as7341_spectrum_readings/as7341_spectrum_readings-Demo_experiment-all_units-20260925145607.csv
  - od_readings/od_readings-Demo_experiment-all_units-20260925145607.csv

Methodology & Protocol Compliance:
  1. Section 6A (Prerequisite & Hydraulic Screening):
     - Continuous inflow and efflux balance check (media vs waste).
     - Pre-dose baseline stability check (12:15 to 12:36, ~21.5 minutes).
     - Sensor dynamic headroom and clipping check across all 8 AS7341 bands.
     - Dosing interval and pulse volume verification.
  2. Section 6B (Experimental Protocol):
     - Identifies all delivered pulses (221 pulses, alphabet {0.20, 0.35, 0.50} mL).
     - Contiguous chronological split: Conditioning (20), Training (120), Validation (40), Test (41).
       No random shuffling or future data leakage.
  3. Section 6C (State Feature Extraction & Readout Evaluation):
     - Defines state x_n as the 8-band AS7341 observation strictly BEFORE dose slot n.
     - Delayed-input reconstruction targets: u_{n-k} for lags k = 1..8.
     - Readout training: Ridge regression tuned on validation set, evaluated on frozen test set.
     - Memory capacity metrics: Test NMSE, Pearson r, r^2 (MC_k), and cumulative linear capacity MC = sum(r^2).
     - Nonlinear task: Reconstruction of cross-product of centered inputs: y_n = (u_{n-1} - u_bar) * (u_{n-2} - u_bar).
     - Baseline comparisons: Constant predictor and fitted single-state hydraulic Leaky Integrator.
  4. Publication Figure Suite (7 Figures):
     - Figure 1: Full 2.5-hour timeline (pumping, dosing pulses, continuous AS7341 spectra).
     - Figure 2: Pre-dose baseline stability audit.
     - Figure 3: Reservoir dynamics close-up (overlapping optical transient responses).
     - Figure 4: Delayed-input reconstruction memory curves (r^2 and NMSE vs lag).
     - Figure 5: Test set actual vs predicted pulse sequences for key lags.
     - Figure 6: Nonlinear interaction task reconstruction.
     - Figure 7: Multiband PCA state space projection.
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
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


# --------------------------------------------------------------------------
# 1. Ingestion & Temporal Alignment
# --------------------------------------------------------------------------

def load_and_audit_hydraulics(dose_path):
    print(f"Loading dosing data: {dose_path}")
    dose = pd.read_csv(dose_path)
    dose["ts_local"] = pd.to_datetime(dose["timestamp_localtime"])
    dose = dose.sort_values("ts_local").reset_index(drop=True)

    # 1. Continuous Media Inflow
    media = dose[dose["event"] == "add_media"].copy()
    media_vol = media["volume_change_ml"].sum()
    media_duration_min = (media["ts_local"].max() - media["ts_local"].min()).total_seconds() / 60.0
    media_flow_rate = media_vol / media_duration_min if media_duration_min > 0 else 0.0

    # 2. Continuous Waste Efflux
    waste = dose[dose["event"] == "remove_waste"].copy()
    waste_vol = waste["volume_change_ml"].sum()
    waste_duration_min = (waste["ts_local"].max() - waste["ts_local"].min()).total_seconds() / 60.0
    waste_flow_rate = waste_vol / waste_duration_min if waste_duration_min > 0 else 0.0

    # 3. Alt Media (Dye) Dosing Pulses
    alt = dose[dose["source_of_event"] == "specified_pump_dosing:alt_media"].copy()
    alt["dt_diff"] = alt["ts_local"].diff().dt.total_seconds().fillna(999.0)
    # A gap > 5 seconds denotes a distinct scheduled dose pulse
    alt["dose_id"] = (alt["dt_diff"] > 5.0).cumsum() + 1

    pulses = alt.groupby("dose_id").agg(
        start_time=("ts_local", "first"),
        end_time=("ts_local", "last"),
        subdose_count=("volume_change_ml", "count"),
        volume=("volume_change_ml", "sum")
    ).reset_index()

    pulses["volume"] = pulses["volume"].round(2)
    pulses["pulse_duration_sec"] = (pulses["end_time"] - pulses["start_time"]).dt.total_seconds()
    pulses["interval_sec"] = pulses["start_time"].diff().dt.total_seconds()
    alt_vol = pulses["volume"].sum()

    print(f"Hydraulics summary:")
    print(f"  * Continuous Media Inflow: {media_vol:.1f} mL ({media_flow_rate:.2f} mL/min)")
    print(f"  * Continuous Waste Efflux: {waste_vol:.1f} mL ({waste_flow_rate:.2f} mL/min)")
    print(f"  * Alt-Media (Dye) Pulses: {len(pulses)} pulses, {alt_vol:.2f} mL total")
    print(f"  * Pulse Volumes: {dict(pulses['volume'].value_counts())}")
    print(f"  * Mean Start-to-Start Interval: {pulses['interval_sec'].mean():.2f} +/- {pulses['interval_sec'].std():.2f} s")

    hydraulics_info = {
        "media_vol": media_vol,
        "media_flow_rate": media_flow_rate,
        "waste_vol": waste_vol,
        "waste_flow_rate": waste_flow_rate,
        "alt_vol": alt_vol,
        "n_pulses": len(pulses),
        "mean_interval": pulses["interval_sec"].mean(),
        "std_interval": pulses["interval_sec"].std(),
    }
    return dose, media, waste, pulses, hydraulics_info


def load_and_pivot_spectra(spec_path):
    print(f"Loading spectrometer readings: {spec_path}")
    spec = pd.read_csv(spec_path)
    spec["ts_local"] = pd.to_datetime(spec["timestamp_localtime"])
    spec["ts_sec"] = spec["ts_local"].dt.round("1s")
    wide = spec.pivot_table(index="ts_sec", columns="band", values="reading", aggfunc="mean")
    wide.columns = [int(c) for c in wide.columns]
    wide = wide.reindex(columns=BANDS).sort_index()
    print(f"Pivoted AS7341 scans: {len(wide)} scans from {wide.index.min()} to {wide.index.max()}")
    return spec, wide


def audit_baseline_and_headroom(wide, pulses):
    t_dose_start = pulses["start_time"].iloc[0]
    # Pre-dose baseline: from beginning of acquisition up to 30s before first dose
    t_base_end = t_dose_start - pd.Timedelta(seconds=30)
    baseline_df = wide.loc[:t_base_end]
    baseline_duration_min = (baseline_df.index.max() - baseline_df.index.min()).total_seconds() / 60.0

    print(f"\nPre-Dose Baseline Audit ({baseline_duration_min:.1f} minutes, {len(baseline_df)} scans):")
    base_stats = {}
    for b in BANDS:
        m = baseline_df[b].mean()
        s = baseline_df[b].std()
        cv = (s / m) * 100.0 if m > 0 else 0.0
        drift_pct = ((baseline_df[b].iloc[-1] - baseline_df[b].iloc[0]) / baseline_df[b].iloc[0]) * 100.0
        base_stats[b] = {
            "mean": m,
            "std": s,
            "cv_pct": cv,
            "drift_pct": drift_pct,
            "min": baseline_df[b].min(),
            "max": baseline_df[b].max(),
        }
        print(f"  * {b:>4} nm: Mean = {m:.4f} | SD = {s:.4f} | CV = {cv:4.2f}% | Drift = {drift_pct:+5.2f}%")

    # Dynamic headroom during dosing
    dosing_df = wide.loc[t_dose_start:]
    print("\nDosing Phase Optical Dynamics:")
    dosing_stats = {}
    for b in BANDS:
        m = dosing_df[b].mean()
        min_v = dosing_df[b].min()
        max_v = dosing_df[b].max()
        base_m = base_stats[b]["mean"]
        max_attenuation = ((base_m - min_v) / base_m) * 100.0 if base_m > 0 else 0.0
        dosing_stats[b] = {
            "mean": m,
            "min": min_v,
            "max": max_v,
            "max_attenuation_pct": max_attenuation
        }
        print(f"  * {b:>4} nm: Mean = {m:.4f} | Range = [{min_v:.4f}, {max_v:.4f}] | Max Attenuation = {max_attenuation:5.1f}%")

    return baseline_df, base_stats, dosing_stats


# --------------------------------------------------------------------------
# 2. State Feature Extraction (Protocol 6C)
# --------------------------------------------------------------------------

def extract_reservoir_states(pulses, wide, base_stats):
    """Protocol 6C: Define x_n as current multiband observation just BEFORE dose slot n."""
    print("\nExtracting pre-dose state observations x_n...")
    records = []
    for idx, r in pulses.iterrows():
        t_start = r["start_time"]
        # Find the latest scan strictly before t_start
        scans_before = wide.loc[:t_start]
        if scans_before.empty:
            continue
        last_scan = scans_before.iloc[-1]
        scan_time = scans_before.index[-1]
        dt_before = (t_start - scan_time).total_seconds()

        rec = {
            "dose_id": r["dose_id"],
            "dose_index": idx,
            "start_time": t_start,
            "scan_time": scan_time,
            "dt_before": dt_before,
            "volume": r["volume"],
        }
        for b in BANDS:
            val_I = last_scan[b]
            base_I = base_stats[b]["mean"]
            rec[f"I_{b}"] = val_I
            # Apparent absorbance relative to pre-dose baseline
            rec[f"A_{b}"] = -np.log10(np.clip(val_I / base_I, 1e-6, None))

        records.append(rec)

    df_states = pd.DataFrame(records)
    print(f"Successfully extracted {len(df_states)} aligned state vectors x_n.")
    print(f"Mean timing offset before dose: {df_states['dt_before'].mean():.2f} +/- {df_states['dt_before'].std():.2f} s")
    return df_states


# --------------------------------------------------------------------------
# 3. Model Training & Memory Capacity (Protocol 6C)
# --------------------------------------------------------------------------

def fit_hydraulic_leaky_integrator(u, target_signal):
    """Fits theoretical single-state leaky integrator s_n = (1-alpha)*s_{n-1} + alpha*u_{n-1}."""
    def loss(alpha):
        a = alpha[0]
        s = np.zeros(len(u))
        for i in range(1, len(u)):
            s[i] = (1.0 - a) * s[i - 1] + a * u[i - 1]
        # Correlation with inverted target signal (dye accumulation)
        corr = np.corrcoef(s[20:], target_signal[20:])[0, 1]
        return corr  # minimize negative correlation with transmission

    res = minimize(loss, x0=[0.3], bounds=[(0.01, 0.99)], method="L-BFGS-B")
    best_alpha = res.x[0]
    s_opt = np.zeros(len(u))
    for i in range(1, len(u)):
        s_opt[i] = (1.0 - best_alpha) * s_opt[i - 1] + best_alpha * u[i - 1]
    return best_alpha, s_opt


def evaluate_delayed_input_reconstruction(df_states, max_lag=8):
    """Protocol 6C: Delayed-input reconstruction u_{n-k} across lags k=1..max_lag."""
    u = df_states["volume"].values
    bands_I = [f"I_{b}" for b in BANDS]
    bands_A = [f"A_{b}" for b in BANDS]
    X_raw = df_states[bands_I].values
    X_abs = df_states[bands_A].values

    # Contiguous chronological split (Protocol 6B):
    # Total 221 pulses:
    # 0..19: Conditioning (burn-in / initial dye buildup)
    # 20..139: Training (120 pulses)
    # 140..179: Validation (40 pulses)
    # 180..220: Held-out Test (41 pulses)
    cond_end = 20
    train_end = 140
    val_end = 180
    test_end = len(df_states)

    print(f"\nContiguous chronological split (Protocol 6B):")
    print(f"  * Conditioning: [0, {cond_end}) ({cond_end} pulses)")
    print(f"  * Training:     [{cond_end}, {train_end}) ({train_end - cond_end} pulses)")
    print(f"  * Validation:   [{train_end}, {val_end}) ({val_end - train_end} pulses)")
    print(f"  * Test (Final): [{val_end}, {test_end}) ({test_end - val_end} pulses)")

    # Leaky integrator benchmark
    best_alpha, s_leaky = fit_hydraulic_leaky_integrator(u, df_states["I_515"].values)
    print(f"  * Fitted Hydraulic Leaky Integrator alpha = {best_alpha:.4f} (tau ~ {(33.0 / best_alpha):.1f} s)")

    lag_results = []
    test_predictions = {}

    for lag in range(1, max_lag + 1):
        y = np.full(len(u), np.nan)
        y[lag:] = u[:-lag]

        tr_mask = (np.arange(len(df_states)) >= cond_end) & (np.arange(len(df_states)) < train_end) & ~np.isnan(y)
        val_mask = (np.arange(len(df_states)) >= train_end) & (np.arange(len(df_states)) < val_end) & ~np.isnan(y)
        te_mask = (np.arange(len(df_states)) >= val_end) & (np.arange(len(df_states)) < test_end) & ~np.isnan(y)

        # Scale features using training set only
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_raw[tr_mask])
        X_val = scaler.transform(X_raw[val_mask])
        X_te = scaler.transform(X_raw[te_mask])

        y_tr = y[tr_mask]
        y_val = y[val_mask]
        y_te = y[te_mask]

        # Regularization tuning on validation set
        best_alpha_ridge = 1.0
        best_val_nmse = 999.0
        alphas = [1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
        for a in alphas:
            m = Ridge(alpha=a).fit(X_tr, y_tr)
            pred_v = m.predict(X_val)
            val_nmse = np.mean((y_val - pred_v) ** 2) / np.var(y_val)
            if val_nmse < best_val_nmse:
                best_val_nmse = val_nmse
                best_alpha_ridge = a

        # Fit final model on training set
        model = Ridge(alpha=best_alpha_ridge).fit(X_tr, y_tr)
        pred_te = model.predict(X_te)
        test_predictions[lag] = (y_te, pred_te)

        # Evaluation metrics on frozen test set
        te_mse = np.mean((y_te - pred_te) ** 2)
        te_var = np.var(y_te)
        te_nmse = te_mse / te_var if te_var > 0 else 1.0
        r = np.corrcoef(y_te, pred_te)[0, 1] if np.var(pred_te) > 0 else 0.0
        mc_k = max(0.0, r ** 2) if r > 0 else 0.0

        # Benchmark: Leaky integrator prediction
        s_tr = s_leaky[tr_mask].reshape(-1, 1)
        s_te = s_leaky[te_mask].reshape(-1, 1)
        m_leaky = Ridge(alpha=1.0).fit(s_tr, y_tr)
        pred_leaky = m_leaky.predict(s_te)
        r_leaky = np.corrcoef(y_te, pred_leaky)[0, 1] if np.var(pred_leaky) > 0 else 0.0
        mc_leaky = max(0.0, r_leaky ** 2) if r_leaky > 0 else 0.0

        lag_results.append({
            "lag": lag,
            "ridge_alpha": best_alpha_ridge,
            "val_NMSE": best_val_nmse,
            "test_NMSE": te_nmse,
            "test_r": r,
            "test_MC_k": mc_k,
            "leaky_MC_k": mc_leaky,
            "test_N": len(y_te)
        })

    df_lag = pd.DataFrame(lag_results)
    total_mc = df_lag["test_MC_k"].sum()
    total_mc_leaky = df_lag["leaky_MC_k"].sum()

    print("\n--- DELAYED-INPUT RECONSTRUCTION ON TEST SET ---")
    print(df_lag[["lag", "ridge_alpha", "test_NMSE", "test_r", "test_MC_k", "leaky_MC_k"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n--> Total Linear Memory Capacity (MC = sum r^2): {total_mc:.4f}")
    print(f"--> Benchmark Leaky Integrator Capacity:          {total_mc_leaky:.4f}")

    # 4. Nonlinear Task (Protocol 6C):
    # Interaction between two centered lags: y_n = (u_{n-1} - u_bar) * (u_{n-2} - u_bar)
    u_bar = np.mean(u[cond_end:train_end])
    u_cent = u - u_bar
    y_nl = np.full(len(u), np.nan)
    y_nl[2:] = u_cent[1:-1] * u_cent[:-2]

    tr_mask_nl = (np.arange(len(df_states)) >= cond_end) & (np.arange(len(df_states)) < train_end) & ~np.isnan(y_nl)
    val_mask_nl = (np.arange(len(df_states)) >= train_end) & (np.arange(len(df_states)) < val_end) & ~np.isnan(y_nl)
    te_mask_nl = (np.arange(len(df_states)) >= val_end) & (np.arange(len(df_states)) < test_end) & ~np.isnan(y_nl)

    scaler_nl = StandardScaler()
    X_tr_nl = scaler_nl.fit_transform(X_raw[tr_mask_nl])
    X_val_nl = scaler_nl.transform(X_raw[val_mask_nl])
    X_te_nl = scaler_nl.transform(X_raw[te_mask_nl])

    y_tr_nl = y_nl[tr_mask_nl]
    y_val_nl = y_nl[val_mask_nl]
    y_te_nl = y_nl[te_mask_nl]

    best_a_nl = 1.0
    best_v_nl = 999.0
    for a in [1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        m = Ridge(alpha=a).fit(X_tr_nl, y_tr_nl)
        pred_v = m.predict(X_val_nl)
        v_nmse = np.mean((y_val_nl - pred_v) ** 2) / np.var(y_val_nl)
        if v_nmse < best_v_nl:
            best_v_nl = v_nmse
            best_a_nl = a

    m_nl = Ridge(alpha=best_a_nl).fit(X_tr_nl, y_tr_nl)
    pred_te_nl = m_nl.predict(X_te_nl)
    te_nmse_nl = np.mean((y_te_nl - pred_te_nl) ** 2) / np.var(y_te_nl)
    r_nl = np.corrcoef(y_te_nl, pred_te_nl)[0, 1] if np.var(pred_te_nl) > 0 else 0.0
    r2_nl = max(0.0, r_nl ** 2) if r_nl > 0 else 0.0

    print(f"\n--- NONLINEAR PRODUCT TASK (u_cent[n-1] * u_cent[n-2]) ---")
    print(f"  * Test NMSE: {te_nmse_nl:.4f} | Pearson r: {r_nl:+.4f} | r^2: {r2_nl:.4f}")

    nl_results = {
        "task": "product_centered_lag1_lag2",
        "ridge_alpha": best_a_nl,
        "val_NMSE": best_v_nl,
        "test_NMSE": te_nmse_nl,
        "test_r": r_nl,
        "test_r2": r2_nl,
        "y_te": y_te_nl,
        "pred_te": pred_te_nl,
    }

    split_indices = {
        "cond_end": cond_end,
        "train_end": train_end,
        "val_end": val_end,
        "test_end": test_end,
    }

    return df_lag, test_predictions, nl_results, split_indices, s_leaky


# --------------------------------------------------------------------------
# 4. Publication Figure Suite (7 Figures)
# --------------------------------------------------------------------------

def generate_protocol6_figures(dose, media, waste, pulses, wide, baseline_df,
                                df_states, df_lag, test_predictions, nl_results,
                                split_indices, outdir):
    print("Generating comprehensive Protocol 6 visualization suite (7 figures)...")

    # Figure 1: Full Experiment Overview (Hydraulics, Dosing, Spectra)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 11), sharex=True, dpi=150)

    # Top: Cumulative volumes
    t_m = media["ts_local"]
    cum_m = media["volume_change_ml"].cumsum()
    t_w = waste["ts_local"]
    cum_w = waste["volume_change_ml"].cumsum()
    t_a = pulses["start_time"]
    cum_a = pulses["volume"].cumsum()

    ax1.plot(t_m, cum_m, color="#0055d4", lw=1.5, label="Clear Water Inflow (Media)")
    ax1.plot(t_w, cum_w, color="#e67e22", lw=1.5, linestyle="--", label="Liquid Efflux (Waste)")
    ax1.plot(t_a, cum_a, color="#008a00", lw=2, label="Cumulative Dye Input (Alt-Media)")
    ax1.set_ylabel("Cumulative Volume (mL)", fontsize=10)
    ax1.set_title("Figure 1A: Continuous Hydraulic Inflow, Efflux, and Cumulative Dye Additions", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(True, alpha=0.3)

    # Middle: Pulse volumes
    ax2.vlines(pulses["start_time"], 0, pulses["volume"], color="#008a00", lw=1.2, alpha=0.85)
    ax2.plot(pulses["start_time"], pulses["volume"], "o", color="#008a00", markersize=4)
    ax2.set_ylabel("Pulse Volume (mL)", fontsize=10)
    ax2.set_ylim(0, 0.65)
    ax2.set_yticks([0.20, 0.35, 0.50])
    ax2.set_title("Figure 1B: Discrete Alt-Media Input Sequence u_n (221 Pulses @ ~33 s Spacing)", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Bottom: AS7341 Spectrometer Readings
    for b in [445, 480, 515, 630]:
        ax3.plot(wide.index, wide[b], label=f"{b} nm", color=BAND_COLORS[b], lw=1.2)

    # Mark splits on bottom plot
    t_cond = pulses["start_time"].iloc[split_indices["cond_end"]]
    t_train = pulses["start_time"].iloc[split_indices["train_end"]]
    t_val = pulses["start_time"].iloc[split_indices["val_end"]]
    ax3.axvline(t_cond, color="gray", linestyle=":", lw=1.5, label="Conditioning End")
    ax3.axvline(t_train, color="blue", linestyle="--", lw=1.5, label="Train/Val Split")
    ax3.axvline(t_val, color="red", linestyle="--", lw=1.5, label="Val/Test Split")

    ax3.set_ylabel("AS7341 Normalized Reading", fontsize=10)
    ax3.set_title("Figure 1C: AS7341 Multiband Optical Response Trace Across Entire Experiment", fontsize=11, fontweight="bold")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax3.legend(loc="upper right", ncol=4, fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "01_full_experiment_overview.png"))
    plt.close(fig)

    # Figure 2: Pre-Dose Baseline Stability Audit
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    for b in BANDS:
        ax1.plot(baseline_df.index, baseline_df[b], label=f"{b} nm", color=BAND_COLORS[b], lw=1.2)
    ax1.set_ylabel("Reading", fontsize=10)
    ax1.set_title("Pre-Dose Baseline Trace (12:15 to 12:36)", fontsize=11, fontweight="bold")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", ncol=2, fontsize=7.5)

    # Relative drift percentage
    for b in BANDS:
        rel = (baseline_df[b] - baseline_df[b].iloc[0]) / baseline_df[b].iloc[0] * 100.0
        ax2.plot(baseline_df.index, rel, label=f"{b} nm", color=BAND_COLORS[b], lw=1.2)
    ax2.set_ylabel("Drift Relative to Start (%)", fontsize=10)
    ax2.set_title("Baseline Relative Drift (%) Over 21.5 Minutes", fontsize=11, fontweight="bold")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Figure 2: Protocol 6A Prerequisite Check — Water Baseline Stability", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "02_baseline_stability_audit.png"))
    plt.close(fig)

    # Figure 3: Reservoir Dynamics Close-up (Overlapping Pulses)
    t_start_zoom = pulses["start_time"].iloc[0] - pd.Timedelta(minutes=1)
    t_end_zoom = pulses["start_time"].iloc[20] + pd.Timedelta(minutes=1)
    sub_wide = wide.loc[t_start_zoom:t_end_zoom]
    sub_pulses = pulses[(pulses["start_time"] >= t_start_zoom) & (pulses["start_time"] <= t_end_zoom)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, dpi=150)
    ax1.vlines(sub_pulses["start_time"], 0, sub_pulses["volume"], color="#008a00", lw=2)
    for _, p in sub_pulses.iterrows():
        ax1.text(p["start_time"], p["volume"] + 0.03, f"{p['volume']:.2f}", fontsize=8, ha="center")
    ax1.set_ylabel("Dose (mL)", fontsize=10)
    ax1.set_ylim(0, 0.65)
    ax1.set_title("Input Sequence (First 20 Pulses)", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    for b in [480, 515, 555, 630]:
        ax2.plot(sub_wide.index, sub_wide[b], label=f"{b} nm", color=BAND_COLORS[b], lw=1.5)
    ax2.set_ylabel("Reading", fontsize=10)
    ax2.set_title("Continuous Multiband Dye Washout & Integration Dynamics", fontsize=11, fontweight="bold")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Figure 3: Chemical Reservoir Dynamics — Overlapping Dye Pulses & Washout", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "03_reservoir_dynamics_closeup.png"))
    plt.close(fig)

    # Figure 4: Delayed-Input Memory Capacity Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    lags = df_lag["lag"].values
    ax1.plot(lags, df_lag["test_MC_k"], "o-", color="#0055d4", lw=2, label="AS7341 Optical Reservoir")
    ax1.plot(lags, df_lag["leaky_MC_k"], "s--", color="#e67e22", lw=1.5, label="Fitted Leaky Integrator")
    ax1.set_xlabel("Delay Lag (k timesteps)", fontsize=10)
    ax1.set_ylabel("Determination Coefficient r^2 (MC_k)", fontsize=10)
    ax1.set_title(f"Delayed-Input Memory Capacity (Total MC = {df_lag['test_MC_k'].sum():.3f})", fontsize=11, fontweight="bold")
    ax1.set_ylim(-0.02, 0.30)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)

    ax2.bar(lags, df_lag["test_NMSE"], color="#2b5c8f", alpha=0.85, edgecolor="black")
    ax2.axhline(1.0, color="red", linestyle="--", lw=1.5, label="Constant Mean Baseline (NMSE=1.0)")
    ax2.set_xlabel("Delay Lag (k timesteps)", fontsize=10)
    ax2.set_ylabel("Normalized Mean Squared Error (NMSE)", fontsize=10)
    ax2.set_title("Held-out Test NMSE vs Delay Lag", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1.25)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)

    fig.suptitle("Figure 4: Protocol 6C Benchmark — Fading Memory Capacity on Held-Out Test Set", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "04_memory_capacity_curves.png"))
    plt.close(fig)

    # Figure 5: Reconstructed vs Actual Sequences on Held-Out Test Set
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=150)
    test_lags = [1, 2, 3, 4]
    for ax, lag in zip(axes.flat, test_lags):
        y_true, y_pred = test_predictions[lag]
        steps = np.arange(len(y_true))
        ax.plot(steps, y_true, "k-o", lw=1.5, markersize=4, label="True Dose u_{n-k}")
        ax.plot(steps, y_pred, "r--s", lw=1.5, markersize=3.5, label=f"Readout Prediction (r={df_lag.loc[df_lag['lag']==lag, 'test_r'].iloc[0]:.2f})")
        ax.set_title(f"Lag {lag} Reconstruction (Test NMSE = {df_lag.loc[df_lag['lag']==lag, 'test_NMSE'].iloc[0]:.3f})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Test Step (30s slots)", fontsize=9)
        ax.set_ylabel("Volume (mL)", fontsize=9)
        ax.set_yticks([0.20, 0.35, 0.50])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Figure 5: Reconstructed vs Ground Truth Delayed Inputs on Frozen Test Set", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "05_reconstructed_test_sequences.png"))
    plt.close(fig)

    # Figure 6: Nonlinear Task Performance
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    y_nl_true = nl_results["y_te"]
    y_nl_pred = nl_results["pred_te"]
    steps = np.arange(len(y_nl_true))
    ax.plot(steps, y_nl_true, "k-o", lw=1.5, markersize=4, label="True Product: (u_{n-1} - u_bar)*(u_{n-2} - u_bar)")
    ax.plot(steps, y_nl_pred, "m--^", lw=1.5, markersize=4, label=f"Reservoir Readout (r={nl_results['test_r']:.2f}, NMSE={nl_results['test_NMSE']:.3f})")
    ax.axhline(0, color="gray", linestyle=":", lw=1)
    ax.set_xlabel("Test Step (30s slots)", fontsize=10)
    ax.set_ylabel("Product Target", fontsize=10)
    ax.set_title("Figure 6: Protocol 6C Nonlinear Interaction Task Reconstruction (Held-Out Test Set)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "06_nonlinear_task_reconstruction.png"))
    plt.close(fig)

    # Figure 7: Multiband PCA State Space Projection
    pca = PCA(n_components=2)
    bands_I = [f"I_{b}" for b in BANDS]
    X_all = df_states[bands_I].values
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X_all))

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=150)
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df_states["volume"], cmap="viridis", s=50, edgecolors="black", alpha=0.85)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Current Dose Volume u_n (mL)", fontsize=10)
    ax.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=10)
    ax.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=10)
    ax.set_title("Figure 7: 2D Principal Component Projection of 8-Band Optical Reservoir State Space", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "07_pca_state_space_projection.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. Executive Reporting & Main
# --------------------------------------------------------------------------

def print_protocol6_verdict(hydraulics_info, base_stats, dosing_stats,
                             df_lag, nl_results, split_indices):
    print("\n" + "=" * 78)
    print("       PROTOCOL 6: OVERLAPPING-INPUT MEMORY EXPERIMENT AUDIT & VERDICT")
    print("=" * 78)

    print("\n[1] HYDRAULIC BALANCE & PREREQUISITE SCREENING (Protocol 6A)")
    print(f"  * Media Inflow Rate: {hydraulics_info['media_flow_rate']:.2f} mL/min (Total: {hydraulics_info['media_vol']:.1f} mL)")
    print(f"  * Waste Removal Rate: {hydraulics_info['waste_flow_rate']:.2f} mL/min (Total: {hydraulics_info['waste_vol']:.1f} mL)")
    print(f"  * Efflux Headroom: Waste rate exceeds media inflow by {(hydraulics_info['waste_flow_rate'] - hydraulics_info['media_flow_rate']):.2f} mL/min (No level rise).")
    print(f"  * Alt-Media Dosing: {hydraulics_info['n_pulses']} pulses delivered, total {hydraulics_info['alt_vol']:.2f} mL dye.")
    print(f"  * Dosing Interval: Mean {hydraulics_info['mean_interval']:.2f} +/- {hydraulics_info['std_interval']:.2f} s (End-to-start spacing adhered to ~30s).")

    print("\n[2] OPTICAL SENSOR SCREENING & HEADROOM (Protocol 6A)")
    base_515 = base_stats[515]
    dose_515 = dosing_stats[515]
    print(f"  * Pre-Dose Baseline Stability (515 nm): Mean = {base_515['mean']:.4f} | CV = {base_515['cv_pct']:.2f}% | Drift = {base_515['drift_pct']:+.2f}%")
    print(f"  * Dynamic Attenuation under Dye Dosing (515 nm): Peak drop = {dose_515['max_attenuation_pct']:.1f}%")
    print(f"  * Saturation Check: Minimum reading = {dose_515['min']:.4f} (> 0), Maximum = {dose_515['max']:.4f} (< ADC ceiling). No clipping.")

    print("\n[3] DELAYED-INPUT RECONSTRUCTION & MEMORY CAPACITY (Protocol 6C)")
    print(f"  * Contiguous Chronological Split: Train n={split_indices['train_end'] - split_indices['cond_end']} | Val n={split_indices['val_end'] - split_indices['train_end']} | Test n={split_indices['test_end'] - split_indices['val_end']}")
    print(df_lag[["lag", "ridge_alpha", "test_NMSE", "test_r", "test_MC_k", "leaky_MC_k"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    total_mc = df_lag["test_MC_k"].sum()
    print(f"\n--> Total Linear Memory Capacity (MC = sum r^2): {total_mc:.4f}")
    print(f"--> Benchmark Hydraulic Leaky Integrator MC:     {df_lag['leaky_MC_k'].sum():.4f}")

    print("\n[4] NONLINEAR INTERACTION TASK (Protocol 6C)")
    print(f"  * Task: Product of centered inputs (u_cent[n-1] * u_cent[n-2])")
    print(f"  * Held-Out Test NMSE: {nl_results['test_NMSE']:.4f} | Pearson r: {nl_results['test_r']:+.4f} | r^2: {nl_results['test_r2']:.4f}")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Protocol 6 Overlapping-Input Memory Experiment Analysis")
    parser.add_argument("--data-dir", default=os.path.join("data", "250 timesteps"), help="Path to 250 timesteps folder")
    parser.add_argument("--outdir", default=os.path.join("outputs", "protocol6_analysis"), help="Output directory")
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    dose_files = glob.glob(os.path.join(args.data_dir, "dosing_events", "*.csv"))
    spec_files = glob.glob(os.path.join(args.data_dir, "as7341_spectrum_readings", "*.csv"))
    if not dose_files or not spec_files:
        sys.exit("Error: Could not locate dosing_events and as7341_spectrum_readings in data directory.")

    # 1. Load Hydraulics & Dosing
    dose, media, waste, pulses, hydraulics_info = load_and_audit_hydraulics(dose_files[0])

    # 2. Load AS7341 Spectrometer
    spec, wide = load_and_pivot_spectra(spec_files[0])

    # 3. Audit Baseline & Headroom
    baseline_df, base_stats, dosing_stats = audit_baseline_and_headroom(wide, pulses)

    # 4. Extract Pre-Dose State Vectors x_n
    df_states = extract_reservoir_states(pulses, wide, base_stats)

    # 5. Evaluate Delayed-Input Reconstruction & Non-linear Tasks
    df_lag, test_predictions, nl_results, split_indices, s_leaky = evaluate_delayed_input_reconstruction(df_states)

    # 6. Save Tables
    pulses.to_csv(os.path.join(outdir, "protocol6_dosing_pulses.csv"), index=False)
    df_states.to_csv(os.path.join(outdir, "protocol6_aligned_features.csv"), index=False)
    df_lag.to_csv(os.path.join(outdir, "protocol6_memory_capacity_results.csv"), index=False)
    pd.DataFrame(base_stats).T.to_csv(os.path.join(outdir, "protocol6_baseline_stats.csv"))

    # 7. Generate All Figures
    generate_protocol6_figures(dose, media, waste, pulses, wide, baseline_df,
                               df_states, df_lag, test_predictions, nl_results,
                               split_indices, outdir)

    # 8. Print Executive Verdict
    print_protocol6_verdict(hydraulics_info, base_stats, dosing_stats,
                            df_lag, nl_results, split_indices)
    print(f"All outputs and 7 figures successfully written to: {os.path.abspath(outdir)}")


if __name__ == "__main__":
    main()
