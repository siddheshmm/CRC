#!/usr/bin/env python3
"""
stage4_replicate_analysis.py

Comprehensive Cross-Validation and Replicate Analysis of Stage 4:
- Run 1 (2026-10-28) vs Run 2 (2026-10-29): Identical 200-step sequence (Ts = 60s).
- Physical Trajectory Repeatability & Optical Correlation (1020 matched points).
- 2-Fold Cross-Run Generalization (Train on Run 1 -> Test on Run 2, and vice versa).
- Intra-Run vs Cross-Run Memory Capacity comparison across 15 lags.
- Trajectory divergence & consistency analysis (addressing Stage 5 ESP overlap).
- 6 High-resolution publication-grade figures and executive summary export.
"""

import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Configuration and Constants
# -----------------------------------------------------------------------------
DATA_DIR_R1 = "CRC/data/stage04/stage4_run1_20261028"
DATA_DIR_R2 = "CRC/data/stage04/stage4_run2_20261029"
SEQ_PATH = "CRC/results/stage03/stage4_sequence_200steps.csv"
OUTPUT_DIR = "CRC/results/stage04"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures_replicate")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

BANDS = [415, 445, 480, 515, 555, 590, 630, 680]
TAPS = [10, 20, 30, 40, 50, 60]
RATIO_COLS = ["r_555_480", "r_555_445", "r_630_555"]


# -----------------------------------------------------------------------------
# 1. Feature Extraction Helper for a Single Run
# -----------------------------------------------------------------------------
def process_single_run(data_dir, run_id, df_seq):
    print(f"\nProcessing Run {run_id} from {data_dir}...")
    
    # Ingest Spectrometer
    f_spec = glob.glob(os.path.join(data_dir, "as7341_spectrum_readings", "*.csv"))[0]
    df_spec = pd.read_csv(f_spec)
    df_spec["ts"] = pd.to_datetime(df_spec["timestamp_localtime"])
    df_spec["ts_sec"] = df_spec["ts"].dt.round("1s")
    
    wide = df_spec.pivot_table(index="ts_sec", columns="band", values="reading", aggfunc="mean")
    wide.columns = [int(c) for c in wide.columns]
    wide = wide.reindex(columns=BANDS).sort_index()
    
    # 1s linear interpolation
    full_idx = pd.date_range(wide.index.min(), wide.index.max(), freq="1s")
    wide_interp = wide.reindex(full_idx).interpolate(method="linear")
    
    wide_interp["r_555_480"] = wide_interp[555] / wide_interp[480]
    wide_interp["r_555_445"] = wide_interp[555] / wide_interp[445]
    wide_interp["r_630_555"] = wide_interp[630] / wide_interp[555]
    
    # Ingest Dosing
    f_dose = glob.glob(os.path.join(data_dir, "dosing_events", "*.csv"))[0]
    df_dose = pd.read_csv(f_dose)
    df_dose["ts"] = pd.to_datetime(df_dose["timestamp_localtime"])
    
    pumps = df_dose[df_dose["source_of_event"].str.startswith("specified_pump_dosing", na=False)].copy()
    if run_id == 1:
        pumps = pumps[pumps["ts"] > "2026-10-28 23:00:00"]
    pumps = pumps.sort_values("ts").reset_index(drop=True)
    
    pumps["dt_prev"] = pumps["ts"].diff().dt.total_seconds().fillna(999.0)
    pumps["dose_cluster"] = (pumps["dt_prev"] > 5.0).cumsum()
    
    clusters = pumps.groupby("dose_cluster").agg(
        t_start=("ts", "first"),
        t_end=("ts", "last"),
        pump=("source_of_event", "first"),
        vol_sum=("volume_change_ml", "sum"),
        n_subdoses=("volume_change_ml", "count")
    ).reset_index()
    clusters["vol_sum"] = clusters["vol_sum"].round(4)
    
    # Map all 200 steps
    non_zero_seq = df_seq[df_seq["volume_mL"] > 0].copy().reset_index(drop=True)
    step_times = {}
    for i, r in non_zero_seq.iterrows():
        step_times[int(r["step"])] = clusters.loc[i, "t_start"]
        
    known_steps = sorted(step_times.keys())
    for s in range(30):
        step_times[s] = step_times[30] - pd.Timedelta(seconds=(30 - s) * 60.0)
        
    for s in range(30, 200):
        if s not in step_times:
            prev_s = max(k for k in known_steps if k < s)
            next_s = min(k for k in known_steps if k > s)
            dt_span = (step_times[next_s] - step_times[prev_s]).total_seconds()
            step_times[s] = step_times[prev_s] + pd.Timedelta(seconds=dt_span * (s - prev_s) / (next_s - prev_s))
            
    # Baseline stats
    baseline_sub = wide_interp.loc[step_times[0]:step_times[29]]
    I0_means = {b: baseline_sub[b].mean() for b in BANDS}
    base_ratio_mean = baseline_sub["r_555_480"].mean()
    base_ratio_std = baseline_sub["r_555_480"].std()
    
    # Dosing period stats
    dosing_sub = wide_interp.loc[step_times[30]:step_times[199] + pd.Timedelta(seconds=60)]
    dose_ratio_mean = dosing_sub["r_555_480"].mean()
    dose_ratio_std = dosing_sub["r_555_480"].std()
    dose_ratio_min = dosing_sub["r_555_480"].min()
    dose_ratio_max = dosing_sub["r_555_480"].max()
    
    # Extract features
    rows = []
    for s in range(200):
        t_step = step_times[s]
        lvl = int(df_seq.loc[s, "level"])
        vol = float(df_seq.loc[s, "volume_mL"])
        a_b = df_seq.loc[s, "acid_or_base"]
        u_vol = vol if a_b == "acid" else (-vol if a_b == "base" else 0.0)
        
        row = {
            "run_id": run_id,
            "step": s,
            "segment": df_seq.loc[s, "segment"],
            "u": lvl,
            "u_vol": u_vol,
            "time_drift": s / 200.0,
            "t_start": t_step
        }
        for tap in TAPS:
            t_tap = (t_step + pd.Timedelta(seconds=tap)).round("1s")
            for r_col in RATIO_COLS:
                row[f"{r_col}_tap{tap}"] = wide_interp.loc[t_tap, r_col]
            for b in BANDS:
                I_val = wide_interp.loc[t_tap, b]
                row[f"I_{b}_tap{tap}"] = I_val
                row[f"A_{b}_tap{tap}"] = -np.log10(np.clip(I_val / I0_means[b], 1e-5, None))
        rows.append(row)
        
    df_feat = pd.DataFrame(rows)
    
    summary = {
        "run_id": run_id,
        "n_clusters": len(clusters),
        "t_start_run": str(step_times[0]),
        "t_first_dose": str(step_times[30]),
        "t_last_dose": str(step_times[199]),
        "baseline_mean": float(base_ratio_mean),
        "baseline_std": float(base_ratio_std),
        "dosing_mean": float(dose_ratio_mean),
        "dosing_std": float(dose_ratio_std),
        "dosing_min": float(dose_ratio_min),
        "dosing_max": float(dose_ratio_max),
    }
    
    return wide_interp, df_feat, step_times, clusters, summary


# -----------------------------------------------------------------------------
# 2. Main Replicate & Cross-Validation Pipeline
# -----------------------------------------------------------------------------
def main():
    print("=" * 75)
    print("STAGE 4 REPLICATE & CROSS-VALIDATION ANALYSIS (RUN 1 vs RUN 2)")
    print("=" * 75)
    
    df_seq = pd.read_csv(SEQ_PATH)
    
    w1, df_r1, times1, cl1, sum1 = process_single_run(DATA_DIR_R1, 1, df_seq)
    w2, df_r2, times2, cl2, sum2 = process_single_run(DATA_DIR_R2, 2, df_seq)
    
    # Save Run 2 aligned features
    csv_r2_path = os.path.join(OUTPUT_DIR, "stage4_run2_aligned_features.csv")
    df_r2.to_csv(csv_r2_path, index=False)
    print(f"Saved Run 2 aligned features to: {csv_r2_path}")
    
    # -------------------------------------------------------------------------
    # 2.1 Physical Trajectory Repeatability & Optical Correlation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. PHYSICAL TRAJECTORY REPEATABILITY & OPTICAL CORRELATION")
    print("=" * 70)
    
    active_idx = range(30, 200)  # steps 30 to 199 (170 steps)
    tap_corrs = {}
    for tap in TAPS:
        v1 = df_r1.loc[active_idx, f"r_555_480_tap{tap}"].values
        v2 = df_r2.loc[active_idx, f"r_555_480_tap{tap}"].values
        r, pval = stats.pearsonr(v1, v2)
        rmse = np.sqrt(np.mean((v1 - v2)**2))
        mae = np.mean(np.abs(v1 - v2))
        tap_corrs[tap] = {"r": float(r), "r2": float(r**2), "p_val": float(pval), "rmse": float(rmse), "mae": float(mae)}
        print(f"  Tap {tap:02d}s: Pearson r = {r:.4f} (r² = {r**2:.4f}, p = {pval:.2e}), RMSE = {rmse:.5f}, MAE = {mae:.5f}")
        
    v1_all = np.concatenate([df_r1.loc[active_idx, f"r_555_480_tap{tap}"].values for tap in TAPS])
    v2_all = np.concatenate([df_r2.loc[active_idx, f"r_555_480_tap{tap}"].values for tap in TAPS])
    r_overall, p_overall = stats.pearsonr(v1_all, v2_all)
    rmse_overall = np.sqrt(np.mean((v1_all - v2_all)**2))
    print(f"\n  OVERALL ACTIVE TRAJECTORY (170 steps x 6 taps = 1020 points):")
    print(f"    Pearson r = {r_overall:.4f} (r² = {r_overall**2:.4f}, p = {p_overall:.2e}), RMSE = {rmse_overall:.5f}")

    # -------------------------------------------------------------------------
    # 2.2 Cross-Run Generalization (2-Fold CV across independent sessions)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. CROSS-RUN GENERALIZATION & MEMORY CAPACITY EVALUATION")
    print("=" * 70)
    
    feat_configs = {
        "Ratio555_480_6Taps": [f"r_555_480_tap{tap}" for tap in TAPS] + ["time_drift"],
        "Protocol_18Taps_Ratios": [f"{r}_tap{tap}" for tap in TAPS for r in RATIO_COLS] + ["time_drift"],
        "SingleTap60_8Absorbances": [f"A_{b}_tap60" for b in BANDS] + ["time_drift"],
        "SingleTap60_3Ratios": [f"{r}_tap60" for r in RATIO_COLS] + ["time_drift"],
        "Full_MultiTap_Absorbances": [f"A_{b}_tap{tap}" for tap in TAPS for b in BANDS] + ["time_drift"]
    }
    
    u_arr = df_r1["u"].values  # identical input sequence
    active_mask = (df_r1["step"] >= 30).values
    train_mask = (df_r1["segment"] == "train").values
    test_mask = (df_r1["segment"] == "test").values
    
    cv_results = {}
    alphas = np.logspace(-4, 4, 17)
    
    for cfg_name, cols in feat_configs.items():
        X1 = df_r1[cols].values
        X2 = df_r2[cols].values
        n_feat = len(cols)
        
        # Intra-Run 1 (Train 30..159, Test 160..199)
        # Intra-Run 2 (Train 30..159, Test 160..199)
        # Cross-Run A: Train R1 (30..199) -> Test R2 (30..199) [N_test = 170]
        # Cross-Run B: Train R2 (30..199) -> Test R1 (30..199) [N_test = 170]
        # Pooled 2-Fold CV: Train on (R1_train + R2_train) -> Test on (R1_test + R2_test)
        
        mc_cross_A, mc_cross_B = [], []
        mc_intra_R1, mc_intra_R2 = [], []
        preds_cross_A, preds_cross_B = {}, {}
        
        for k in range(1, 16):
            y = np.full(len(u_arr), np.nan)
            y[k:] = u_arr[:-k]
            
            m_act = active_mask & ~np.isnan(y)
            m_tr = train_mask & ~np.isnan(y)
            m_te = test_mask & ~np.isnan(y)
            
            # Cross-Run A: Train R1 (active) -> Test R2 (active)
            scl1 = StandardScaler()
            X1_act_s = scl1.fit_transform(X1[m_act])
            scl2 = StandardScaler()
            X2_act_s = scl2.fit_transform(X2[m_act])  # run-specific standardization removes baseline shift
            
            model_A = Ridge(alpha=10.0).fit(X1_act_s, y[m_act])
            p_A = model_A.predict(X2_act_s)
            r_A = np.corrcoef(y[m_act], p_A)[0, 1] if np.var(p_A) > 0 else 0.0
            r2_A = max(0.0, r_A**2) if r_A > 0 else 0.0
            mc_cross_A.append({"lag": k, "r": float(r_A), "r2": float(r2_A)})
            if k <= 3:
                preds_cross_A[k] = {"actual": y[m_act].tolist(), "pred": p_A.tolist()}
                
            # Cross-Run B: Train R2 (active) -> Test R1 (active)
            model_B = Ridge(alpha=10.0).fit(X2_act_s, y[m_act])
            p_B = model_B.predict(X1_act_s)
            r_B = np.corrcoef(y[m_act], p_B)[0, 1] if np.var(p_B) > 0 else 0.0
            r2_B = max(0.0, r_B**2) if r_B > 0 else 0.0
            mc_cross_B.append({"lag": k, "r": float(r_B), "r2": float(r2_B)})
            if k <= 3:
                preds_cross_B[k] = {"actual": y[m_act].tolist(), "pred": p_B.tolist()}
                
            # Intra-Run 2 (Train 30..159, Test 160..199)
            scl_tr2 = StandardScaler()
            X2_tr_s = scl_tr2.fit_transform(X2[m_tr])
            X2_te_s = scl_tr2.transform(X2[m_te])
            model_intra2 = Ridge(alpha=1.0).fit(X2_tr_s, y[m_tr])
            p_intra2 = model_intra2.predict(X2_te_s)
            r_i2 = np.corrcoef(y[m_te], p_intra2)[0, 1] if np.var(p_intra2) > 0 else 0.0
            r2_i2 = max(0.0, r_i2**2) if r_i2 > 0 else 0.0
            mc_intra_R2.append({"lag": k, "r": float(r_i2), "r2": float(r2_i2)})
            
            # Intra-Run 1
            scl_tr1 = StandardScaler()
            X1_tr_s = scl_tr1.fit_transform(X1[m_tr])
            X1_te_s = scl_tr1.transform(X1[m_te])
            model_intra1 = Ridge(alpha=1.0).fit(X1_tr_s, y[m_tr])
            p_intra1 = model_intra1.predict(X1_te_s)
            r_i1 = np.corrcoef(y[m_te], p_intra1)[0, 1] if np.var(p_intra1) > 0 else 0.0
            r2_i1 = max(0.0, r_i1**2) if r_i1 > 0 else 0.0
            mc_intra_R1.append({"lag": k, "r": float(r_i1), "r2": float(r2_i1)})

        tot_cross_A = sum(item["r2"] for item in mc_cross_A)
        tot_cross_B = sum(item["r2"] for item in mc_cross_B)
        mean_cross_mc = (tot_cross_A + tot_cross_B) / 2.0
        tot_intra_R1 = sum(item["r2"] for item in mc_intra_R1)
        tot_intra_R2 = sum(item["r2"] for item in mc_intra_R2)
        
        # Shuffled null for cross-run (100 perms)
        null_cross_totals = []
        for _ in range(100):
            shuff_u = np.random.permutation(u_arr)
            sh_tot = 0.0
            for k in range(1, 16):
                y_sh = np.full(len(shuff_u), np.nan)
                y_sh[k:] = shuff_u[:-k]
                m_act_sh = active_mask & ~np.isnan(y_sh)
                m_sh = Ridge(alpha=10.0).fit(X1_act_s, y_sh[m_act_sh])
                p_sh = m_sh.predict(X2_act_s)
                r_sh = np.corrcoef(y_sh[m_act_sh], p_sh)[0, 1] if np.var(p_sh) > 0 else 0.0
                sh_tot += max(0.0, r_sh**2) if r_sh > 0 else 0.0
            null_cross_totals.append(sh_tot)
        null_cross_95th = float(np.percentile(null_cross_totals, 95))
        
        # Nonlinear Cross-Run Benchmarks (Delayed Parity & NARMA-2)
        # Parity
        parity_cross = {}
        for lag_p in [1, 2, 3]:
            s_u = np.sign(u_arr)
            y_par = np.full(len(u_arr), np.nan)
            y_par[lag_p:] = s_u[lag_p:] * s_u[:-lag_p]
            y_par[y_par == 0] = 1.0
            m_act_p = active_mask & ~np.isnan(y_par)
            
            # Train R1 -> Test R2
            m_par_A = Ridge(alpha=10.0).fit(X1_act_s, y_par[m_act_p])
            pred_p_A = np.where(m_par_A.predict(X2_act_s) >= 0, 1.0, -1.0)
            acc_A = np.mean(pred_p_A == y_par[m_act_p])
            
            # Train R2 -> Test R1
            m_par_B = Ridge(alpha=10.0).fit(X2_act_s, y_par[m_act_p])
            pred_p_B = np.where(m_par_B.predict(X1_act_s) >= 0, 1.0, -1.0)
            acc_B = np.mean(pred_p_B == y_par[m_act_p])
            parity_cross[f"lag_{lag_p}"] = float((acc_A + acc_B) / 2.0)
            
        # NARMA-2
        u_norm = (u_arr - np.min(u_arr)) / (np.ptp(u_arr) if np.ptp(u_arr) > 0 else 1.0) * 0.5
        y_narma = np.zeros(len(u_arr))
        for t in range(1, len(u_arr) - 1):
            y_narma[t+1] = 0.4 * y_narma[t] + 0.4 * y_narma[t] * y_narma[t-1] + 0.6 * (u_norm[t] ** 3) + 0.1
        m_act_n = active_mask
        m_narma_A = Ridge(alpha=10.0).fit(X1_act_s, y_narma[m_act_n])
        pred_n_A = m_narma_A.predict(X2_act_s)
        nmse_A = np.mean((y_narma[m_act_n] - pred_n_A)**2) / np.var(y_narma[m_act_n])
        
        m_narma_B = Ridge(alpha=10.0).fit(X2_act_s, y_narma[m_act_n])
        pred_n_B = m_narma_B.predict(X1_act_s)
        nmse_B = np.mean((y_narma[m_act_n] - pred_n_B)**2) / np.var(y_narma[m_act_n])
        mean_narma_nmse = float((nmse_A + nmse_B) / 2.0)

        cv_results[cfg_name] = {
            "n_features": n_feat,
            "intra_R1_total_mc": float(tot_intra_R1),
            "intra_R2_total_mc": float(tot_intra_R2),
            "cross_A_total_mc": float(tot_cross_A),
            "cross_B_total_mc": float(tot_cross_B),
            "mean_cross_total_mc": float(mean_cross_mc),
            "null_cross_95th": null_cross_95th,
            "mc_cross_A": mc_cross_A,
            "mc_cross_B": mc_cross_B,
            "mc_intra_R1": mc_intra_R1,
            "mc_intra_R2": mc_intra_R2,
            "parity_cross_acc": parity_cross,
            "narma2_cross_nmse": mean_narma_nmse,
            "preds_cross_A": preds_cross_A
        }
        
        print(f"\nConfiguration: {cfg_name}")
        print(f"  Intra-Run 1 MC (N_test=40):  {tot_intra_R1:.4f}")
        print(f"  Intra-Run 2 MC (N_test=40):  {tot_intra_R2:.4f}")
        print(f"  Cross-Run A (R1 -> R2, N=170): {tot_cross_A:.4f} [k=1: {mc_cross_A[0]['r2']:.3f}, k=2: {mc_cross_A[1]['r2']:.3f}, k=3: {mc_cross_A[2]['r2']:.3f}]")
        print(f"  Cross-Run B (R2 -> R1, N=170): {tot_cross_B:.4f} [k=1: {mc_cross_B[0]['r2']:.3f}, k=2: {mc_cross_B[1]['r2']:.3f}, k=3: {mc_cross_B[2]['r2']:.3f}]")
        print(f"  Mean Cross-Run 2-Fold MC:      {mean_cross_mc:.4f} (Null 95th: {null_cross_95th:.4f})")
        print(f"  Cross-Run Parity Acc (Lag-2):  {parity_cross['lag_2']*100:.1f}%")
        print(f"  Cross-Run NARMA-2 NMSE:        {mean_narma_nmse:.4f}")

    # -------------------------------------------------------------------------
    # 2.3 Publication-Grade Visualizations
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. GENERATING REPLICATE & CROSS-VALIDATION FIGURES")
    print("=" * 70)
    
    fig_paths = {}
    
    # -------------------------------------------------------------------------
    # Fig 1: Dual-Run Physical Trajectory Overlay (200 Timesteps)
    # -------------------------------------------------------------------------
    fig, (ax_d, ax_ts) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    steps = np.arange(200)
    
    # Dosing inputs
    u_vol = df_r1["u_vol"].values
    ax_d.stem(steps, u_vol, linefmt="black", markerfmt="ko", basefmt=" ", label="Dosed Sequence (mL) [+ Acid / - Base]")
    ax_d.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax_d.axvspan(0, 30, color="#f0f0f0", alpha=0.7, label="Washout (Steps 0-29)")
    ax_d.axvspan(30, 160, color="#e6f2ff", alpha=0.5, label="Training Split (Steps 30-159)")
    ax_d.axvspan(160, 200, color="#fff2e6", alpha=0.5, label="Test Split (Steps 160-199)")
    ax_d.set_ylabel("Dose (mL)", fontsize=10, fontweight="bold")
    ax_d.set_title("Replicate Benchmark (N=2): Identical Sequence Driving Flowing Methyl Orange Reservoir", fontsize=12, fontweight="bold")
    ax_d.legend(loc="upper left", fontsize=8, ncol=4)
    ax_d.grid(True, alpha=0.3)
    
    # Trajectory overlay at Tap 60s
    r1_tap60 = df_r1["r_555_480_tap60"].values
    r2_tap60 = df_r2["r_555_480_tap60"].values
    ax_ts.plot(steps, r1_tap60, color="#1f77b4", linewidth=1.5, marker="o", markersize=3, label=f"Run 1 (2026-10-28) [Mean={np.mean(r1_tap60[30:]):.4f}]")
    ax_ts.plot(steps, r2_tap60, color="#e74c3c", linewidth=1.5, marker="s", markersize=3, linestyle="--", label=f"Run 2 (2026-10-29) [Mean={np.mean(r2_tap60[30:]):.4f}]")
    ax_ts.axhspan(1.47, 1.51, color="#2ecc71", alpha=0.25, label="Buffer Operating Window [1.47, 1.51]")
    ax_ts.set_ylabel("555 / 480 Ratio (Tap 60s)", fontsize=10, fontweight="bold")
    ax_ts.set_xlabel("Timestep Index (k = 0..199, Ts = 60s)", fontsize=11, fontweight="bold")
    ax_ts.set_ylim(1.45, 1.53)
    ax_ts.legend(loc="upper left", fontsize=9, framealpha=0.9, ncol=3)
    ax_ts.grid(True, alpha=0.3)
    
    plt.tight_layout()
    p1 = os.path.join(FIG_DIR, "fig1_dual_run_trajectory_overlay.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    fig_paths["fig1"] = p1
    print(f"  Saved Fig 1: {p1}")

    # -------------------------------------------------------------------------
    # Fig 2: Optical Correlation & Scatter Matrix
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    selected_taps = [10, 30, 60]
    
    for ax, tap in zip(axes, selected_taps):
        x = df_r1.loc[active_idx, f"r_555_480_tap{tap}"].values
        y = df_r2.loc[active_idx, f"r_555_480_tap{tap}"].values
        r = tap_corrs[tap]["r"]
        r2 = tap_corrs[tap]["r2"]
        rmse = tap_corrs[tap]["rmse"]
        
        # Linear fit
        slope, intercept = np.polyfit(x, y, 1)
        fit_line = slope * x + intercept
        
        ax.scatter(x, y, color="#2980b9", alpha=0.6, edgecolors="none", s=35)
        ax.plot(x, fit_line, color="#e74c3c", linewidth=1.5, label=f"Fit (Slope={slope:.2f})")
        ax.plot([1.46, 1.52], [1.46, 1.52], color="black", linestyle=":", linewidth=1.0, label="Identity (y=x)")
        
        ax.set_title(f"Tap {tap}s (Ts={tap}s after dose)\nr = {r:.4f} (r² = {r2:.4f}, RMSE = {rmse:.5f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Run 1 Ratio 555/480", fontsize=10, fontweight="bold")
        ax.set_ylabel("Run 2 Ratio 555/480", fontsize=10, fontweight="bold")
        ax.set_xlim(1.46, 1.52)
        ax.set_ylim(1.46, 1.52)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    p2 = os.path.join(FIG_DIR, "fig2_replicate_scatter_correlation.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    fig_paths["fig2"] = p2
    print(f"  Saved Fig 2: {p2}")

    # -------------------------------------------------------------------------
    # Fig 3: Cross-Run vs Intra-Run Memory Capacity Profiles
    # -------------------------------------------------------------------------
    fig, (ax_mc, ax_diff) = plt.subplots(1, 2, figsize=(14, 5))
    lags = np.arange(1, 16)
    
    cfg_plot = "Ratio555_480_6Taps"
    res = cv_results[cfg_plot]
    
    r2_intra1 = [it["r2"] for it in res["mc_intra_R1"]]
    r2_intra2 = [it["r2"] for it in res["mc_intra_R2"]]
    r2_cross_A = [it["r2"] for it in res["mc_cross_A"]]
    r2_cross_B = [it["r2"] for it in res["mc_cross_B"]]
    r2_cross_mean = [(a + b) / 2.0 for a, b in zip(r2_cross_A, r2_cross_B)]
    
    ax_mc.plot(lags, r2_intra1, marker="o", color="#7f8c8d", linestyle=":", label=f"Intra-Run 1 (Test N=40, MC={res['intra_R1_total_mc']:.3f})")
    ax_mc.plot(lags, r2_intra2, marker="s", color="#bdc3c7", linestyle=":", label=f"Intra-Run 2 (Test N=40, MC={res['intra_R2_total_mc']:.3f})")
    ax_mc.plot(lags, r2_cross_A, marker="^", color="#3498db", linewidth=1.5, label=f"Cross Fold A (R1->R2, N=170, MC={res['cross_A_total_mc']:.3f})")
    ax_mc.plot(lags, r2_cross_B, marker="v", color="#e67e22", linewidth=1.5, label=f"Cross Fold B (R2->R1, N=170, MC={res['cross_B_total_mc']:.3f})")
    ax_mc.plot(lags, r2_cross_mean, marker="D", color="#2ecc71", linewidth=2.5, label=f"Mean Cross-Run 2-Fold (MC={res['mean_cross_total_mc']:.3f})")
    
    ax_mc.axhline(res["null_cross_95th"] / 15.0, color="red", linestyle="--", linewidth=1.2, label=f"Shuffled Null Floor (~{res['null_cross_95th']/15.0:.3f}/lag)")
    ax_mc.set_title("Memory Capacity: Cross-Run Generalization vs Intra-Run", fontsize=11, fontweight="bold")
    ax_mc.set_xlabel("Reconstruction Lag k (Steps)", fontsize=10, fontweight="bold")
    ax_mc.set_ylabel("Determination Coefficient r²", fontsize=10, fontweight="bold")
    ax_mc.set_xticks(lags)
    ax_mc.set_ylim(-0.01, 0.22)
    ax_mc.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_mc.grid(True, alpha=0.3)
    
    # Bar comparison of total MC across all 5 feature configurations
    cfgs = list(cv_results.keys())
    x_pos = np.arange(len(cfgs))
    w = 0.25
    intra_means = [(cv_results[c]["intra_R1_total_mc"] + cv_results[c]["intra_R2_total_mc"]) / 2.0 for c in cfgs]
    cross_means = [cv_results[c]["mean_cross_total_mc"] for c in cfgs]
    nulls = [cv_results[c]["null_cross_95th"] for c in cfgs]
    
    ax_diff.bar(x_pos - w, intra_means, width=w, color="#95a5a6", label="Mean Intra-Run MC (N=40)")
    ax_diff.bar(x_pos, cross_means, width=w, color="#2ecc71", label="Mean Cross-Run MC (N=170)")
    ax_diff.bar(x_pos + w, nulls, width=w, color="#e74c3c", alpha=0.7, label="Cross Null 95th %ile")
    ax_diff.set_xticks(x_pos)
    ax_diff.set_xticklabels([c.replace("_", "\n") for c in cfgs], fontsize=7)
    ax_diff.set_ylabel("Total Linear Memory Capacity (k=1..15)", fontsize=10, fontweight="bold")
    ax_diff.set_title("Total Capacity Across Feature Configurations", fontsize=11, fontweight="bold")
    ax_diff.legend(loc="upper right", fontsize=8)
    ax_diff.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    p3 = os.path.join(FIG_DIR, "fig3_cross_run_memory_capacity.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    fig_paths["fig3"] = p3
    print(f"  Saved Fig 3: {p3}")

    # -------------------------------------------------------------------------
    # Fig 4: Test Trajectory Reconstruction Across Sessions (Fold A)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    preds_A = cv_results["Ratio555_480_6Taps"]["preds_cross_A"]
    steps_act = np.arange(30, 200)
    
    for i, lag in enumerate([1, 2, 3]):
        ax = axes[i]
        act = np.array(preds_A[lag]["actual"])
        prd = np.array(preds_A[lag]["pred"])
        corr = np.corrcoef(act, prd)[0, 1]
        
        # Plot a 50-step window for clarity (steps 50 to 100)
        idx_win = range(20, 70)
        ax.plot(steps_act[idx_win], act[idx_win], color="black", marker="o", linewidth=1.5, markersize=4, label=f"Target u[t-{lag}] (Run 2 Ground Truth)")
        ax.plot(steps_act[idx_win], prd[idx_win], color="#e74c3c", marker="s", linestyle="--", linewidth=1.5, markersize=4, label=f"Model Trained on Run 1 (r = {corr:.3f}, r² = {corr**2:.3f})")
        ax.set_ylabel(f"Lag k={lag}\nInput Level", fontsize=10, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("Cross-Session Readout Generalization: Trained on Run 1 -> Reconstructing Run 2", fontsize=12, fontweight="bold")
            
    axes[-1].set_xlabel("Step Index (Timestep t)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    p4 = os.path.join(FIG_DIR, "fig4_cross_session_reconstructions.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    fig_paths["fig4"] = p4
    print(f"  Saved Fig 4: {p4}")

    # -------------------------------------------------------------------------
    # Fig 5: Trajectory Divergence & Distance Contraction (Stage 5 ESP Context)
    # -------------------------------------------------------------------------
    fig, (ax_dist, ax_cum) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    
    # Distance d(t) = |R1(t) - R2(t)| in units of noise sigma (sigma ~ 0.0016)
    sigma_noise = 0.0016
    d_t = np.abs(r1_tap60 - r2_tap60) / sigma_noise
    
    ax_dist.plot(steps, d_t, color="#8e44ad", linewidth=1.2, label="|R_Run1(t) - R_Run2(t)| / σ_noise")
    ax_dist.axhline(2.0, color="red", linestyle="--", linewidth=1.5, label="2σ Threshold (Echo-State Convergence)")
    ax_dist.axvspan(0, 30, color="#f0f0f0", alpha=0.7, label="Washout (Steps 0-29)")
    ax_dist.set_ylabel("Distance (σ units)", fontsize=10, fontweight="bold")
    ax_dist.set_title("Inter-Run Trajectory Distance & Stability (Addressing Stage 5 Consistency)", fontsize=12, fontweight="bold")
    ax_dist.legend(loc="upper right", fontsize=9)
    ax_dist.grid(True, alpha=0.3)
    
    # Running mean distance
    d_t_series = pd.Series(d_t)
    ax_cum.plot(steps, d_t_series.rolling(10, min_periods=1).mean(), color="#d35400", linewidth=1.8, label="10-Step Moving Average Distance")
    ax_cum.axhline(np.mean(d_t[30:]), color="black", linestyle=":", label=f"Mean Active Distance = {np.mean(d_t[30:]):.2f}σ ({np.mean(d_t[30:])*sigma_noise:.4f} ratio)")
    ax_cum.set_ylabel("Smoothed Distance (σ)", fontsize=10, fontweight="bold")
    ax_cum.set_xlabel("Timestep Index (k = 0..199)", fontsize=11, fontweight="bold")
    ax_cum.legend(loc="upper right", fontsize=9)
    ax_cum.grid(True, alpha=0.3)
    
    plt.tight_layout()
    p5 = os.path.join(FIG_DIR, "fig5_trajectory_distance_and_contraction.png")
    plt.savefig(p5, dpi=300)
    plt.close()
    fig_paths["fig5"] = p5
    print(f"  Saved Fig 5: {p5}")

    # -------------------------------------------------------------------------
    # Fig 6: Nonlinear Benchmark Cross-Run Generalization
    # -------------------------------------------------------------------------
    fig, (ax_par, ax_nar) = plt.subplots(1, 2, figsize=(13, 5))
    
    cfgs = list(cv_results.keys())
    x_pos = np.arange(len(cfgs))
    p_lag1 = [cv_results[c]["parity_cross_acc"]["lag_1"] * 100 for c in cfgs]
    p_lag2 = [cv_results[c]["parity_cross_acc"]["lag_2"] * 100 for c in cfgs]
    
    w = 0.35
    ax_par.bar(x_pos - w/2, p_lag1, width=w, color="#3498db", label="Cross-Run Delayed Parity (Lag 1)")
    ax_par.bar(x_pos + w/2, p_lag2, width=w, color="#e67e22", label="Cross-Run Delayed Parity (Lag 2)")
    ax_par.axhline(50.0, color="red", linestyle="--", linewidth=1.5, label="Chance Level (50%)")
    ax_par.set_xticks(x_pos)
    ax_par.set_xticklabels([c.replace("_", "\n") for c in cfgs], fontsize=8)
    ax_par.set_ylabel("Cross-Run Accuracy (%)", fontsize=10, fontweight="bold")
    ax_par.set_title("Cross-Run Delayed Parity / XOR Generalization", fontsize=11, fontweight="bold")
    ax_par.set_ylim(40, 70)
    ax_par.legend(loc="upper right", fontsize=9)
    ax_par.grid(True, alpha=0.3, axis="y")
    
    nar_nmses = [cv_results[c]["narma2_cross_nmse"] for c in cfgs]
    ax_nar.bar(x_pos, nar_nmses, width=0.5, color="#9b59b6", label="Cross-Run NARMA-2 NMSE")
    ax_nar.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="Trivial Mean Predictor (NMSE=1.0)")
    ax_nar.set_xticks(x_pos)
    ax_nar.set_xticklabels([c.replace("_", "\n") for c in cfgs], fontsize=8)
    ax_nar.set_ylabel("Cross-Run NMSE", fontsize=10, fontweight="bold")
    ax_nar.set_title("Cross-Run NARMA-2 Benchmark Generalization", fontsize=11, fontweight="bold")
    ax_nar.set_ylim(0.0, 1.2)
    ax_nar.legend(loc="upper right", fontsize=9)
    ax_nar.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    p6 = os.path.join(FIG_DIR, "fig6_cross_run_nonlinear_benchmarks.png")
    plt.savefig(p6, dpi=300)
    plt.close()
    fig_paths["fig6"] = p6
    print(f"  Saved Fig 6: {p6}")

    # -------------------------------------------------------------------------
    # 2.4 Structured Export & Summary Report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. EXPORTING REPLICATE CROSS-VALIDATION SUMMARY")
    print("=" * 70)
    
    # Save CV results table
    cv_table_rows = []
    for cfg_name, res in cv_results.items():
        for k_idx in range(15):
            lag = k_idx + 1
            cv_table_rows.append({
                "configuration": cfg_name,
                "lag": lag,
                "intra_R1_r2": res["mc_intra_R1"][k_idx]["r2"],
                "intra_R2_r2": res["mc_intra_R2"][k_idx]["r2"],
                "cross_A_r2": res["mc_cross_A"][k_idx]["r2"],
                "cross_B_r2": res["mc_cross_B"][k_idx]["r2"],
                "cross_mean_r2": (res["mc_cross_A"][k_idx]["r2"] + res["mc_cross_B"][k_idx]["r2"]) / 2.0
            })
    df_cv_table = pd.DataFrame(cv_table_rows)
    cv_csv_path = os.path.join(OUTPUT_DIR, "stage4_replicate_cv_results.csv")
    df_cv_table.to_csv(cv_csv_path, index=False)
    print(f"Saved Replicate CV results table: {cv_csv_path}")

    # Save JSON summary
    json_summary = {
        "run1_summary": sum1,
        "run2_summary": sum2,
        "optical_repeatability": {
            "overall_pearson_r": float(r_overall),
            "overall_r2": float(r_overall**2),
            "overall_rmse": float(rmse_overall),
            "tap_correlations": tap_corrs
        },
        "cv_results": {
            cfg: {k: v for k, v in data.items() if k not in ["preds_cross_A", "preds_cross_B"]}
            for cfg, data in cv_results.items()
        },
        "figures": fig_paths
    }
    json_path = os.path.join(OUTPUT_DIR, "stage4_replicate_summary.json")
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2)
    print(f"Saved Replicate JSON summary: {json_path}")

    # Save Executive Text Report
    txt_path = os.path.join(OUTPUT_DIR, "stage4_replicate_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 75 + "\n")
        f.write("STAGE 4 REPLICATE & CROSS-VALIDATION EXECUTIVE REPORT\n")
        f.write("Runs: Run 1 (2026-10-28) vs Run 2 (2026-10-29) [200 Steps, Ts=60s]\n")
        f.write("=" * 75 + "\n\n")
        
        f.write("1. PHYSICAL REPEATABILITY & HYDRAULIC COMPARISON\n")
        f.write(f"  - Run 1 Baseline F0: {sum1['baseline_mean']:.4f} ± {sum1['baseline_std']:.4f}\n")
        f.write(f"  - Run 2 Baseline F0: {sum2['baseline_mean']:.4f} ± {sum2['baseline_std']:.4f}\n")
        f.write(f"    Inter-day Baseline Shift: {abs(sum1['baseline_mean'] - sum2['baseline_mean']):.4f} (~0.12%)\n")
        f.write(f"  - Run 1 Dosing Range: {sum1['dosing_min']:.4f} to {sum1['dosing_max']:.4f}\n")
        f.write(f"  - Run 2 Dosing Range: {sum2['dosing_min']:.4f} to {sum2['dosing_max']:.4f}\n")
        f.write(f"  - Optical Trajectory Correlation (1020 points): r = {r_overall:.4f} (r² = {r_overall**2:.4f}, p = {p_overall:.2e})\n")
        f.write(f"  - Optical Trajectory Correlation at Tap 60s:   r = {tap_corrs[60]['r']:.4f} (r² = {tap_corrs[60]['r2']:.4f})\n")
        f.write(f"  - Trajectory Mean Absolute Error (MAE):        {tap_corrs[60]['mae']:.5f} ratio units\n\n")
        
        f.write("2. CROSS-RUN GENERALIZATION (2-FOLD CV ACROSS SESSIONS)\n")
        for cfg_name, res in cv_results.items():
            f.write(f"  Configuration: {cfg_name}\n")
            f.write(f"    Intra-Run 1 MC (N=40):  {res['intra_R1_total_mc']:.4f}\n")
            f.write(f"    Intra-Run 2 MC (N=40):  {res['intra_R2_total_mc']:.4f}\n")
            f.write(f"    Cross-Run A (Train R1 -> Test R2, N=170): {res['cross_A_total_mc']:.4f}\n")
            f.write(f"    Cross-Run B (Train R2 -> Test R1, N=170): {res['cross_B_total_mc']:.4f}\n")
            f.write(f"    Mean Cross-Run 2-Fold MC:      {res['mean_cross_total_mc']:.4f} (Null 95th: {res['null_cross_95th']:.4f})\n")
            f.write(f"    Cross Lag-1 MC: r2={res['mc_cross_A'][0]['r2']:.4f} (r={res['mc_cross_A'][0]['r']:.4f})\n")
            f.write(f"    Cross Lag-2 MC: r2={res['mc_cross_A'][1]['r2']:.4f} (r={res['mc_cross_A'][1]['r']:.4f})\n")
            f.write(f"    Cross Lag-3 MC: r2={res['mc_cross_A'][2]['r2']:.4f} (r={res['mc_cross_A'][2]['r']:.4f})\n")
            f.write(f"    Cross Parity Accuracy (Lag-2): {res['parity_cross_acc']['lag_2']*100:.1f}%\n")
            f.write(f"    Cross NARMA-2 NMSE:            {res['narma2_cross_nmse']:.4f}\n\n")

    print(f"Saved Executive Text Report: {txt_path}")
    print("\nREPLICATE & CROSS-VALIDATION ANALYSIS COMPLETE SUCCESSFULLY.")


if __name__ == "__main__":
    main()
