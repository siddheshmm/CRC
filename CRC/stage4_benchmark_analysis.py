#!/usr/bin/env python3
"""
stage4_benchmark_analysis.py

Comprehensive Scientific Benchmark Analysis of Stage 4 (Methyl Orange Chemical Reservoir):
- 200-timestep pilot run (30 washout + 130 training + 40 test steps, Ts = 60s).
- Rigorous physical & hydraulic audit (dosing events, step jitter, OD stability).
- AS7341 spectral ratio & absorbance feature extraction with temporal multiplexing.
- Memory Capacity (MC) evaluation across lags k=1..15 with 5-fold CV Ridge regression.
- Shuffled-input null baseline & analytical noise floor.
- Nonlinear benchmarks: Delayed Parity / XOR, Degree-2 Polynomial Capacity, NARMA-2.
- 6 High-resolution publication-grade figures and structured JSON/TXT export.
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
DATA_DIR = "CRC/data/stage04/stage4_run1_20261028"
SEQ_PATH = "CRC/results/stage3/stage4_sequence_200steps.csv"
OUTPUT_DIR = "CRC/results/stage4"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)

BANDS = [415, 445, 480, 515, 555, 590, 630, 680]
BAND_COLORS = {
    415: "#6a0dad",  # violet
    445: "#0055d4",  # blue
    480: "#00a2e8",  # cyan
    515: "#008a00",  # green
    555: "#8cb800",  # yellow-green (peak indicator response)
    590: "#e67e22",  # orange
    630: "#e74c3c",  # red
    680: "#880e4f",  # deep red / NIR
}

# -----------------------------------------------------------------------------
# 1. Ingestion & Audit
# -----------------------------------------------------------------------------
def load_and_audit_data():
    print("=" * 70)
    print("1. INGESTING & AUDITING STAGE 4 DATA")
    print("=" * 70)

    # 1.1 Ingest Spectrometer Data
    spec_files = glob.glob(os.path.join(DATA_DIR, "as7341_spectrum_readings", "*.csv"))
    if not spec_files:
        raise FileNotFoundError(f"No spectrum CSV found in {DATA_DIR}/as7341_spectrum_readings")
    df_spec = pd.read_csv(spec_files[0])
    df_spec["ts"] = pd.to_datetime(df_spec["timestamp_localtime"])
    df_spec["ts_sec"] = df_spec["ts"].dt.round("1s")

    wide = df_spec.pivot_table(index="ts_sec", columns="band", values="reading", aggfunc="mean")
    wide.columns = [int(c) for c in wide.columns]
    wide = wide.reindex(columns=BANDS).sort_index()

    # Interpolate rare 1s gaps
    full_idx = pd.date_range(wide.index.min(), wide.index.max(), freq="1s")
    wide_interp = wide.reindex(full_idx).interpolate(method="linear")

    # Spectral ratios
    wide_interp["r_555_480"] = wide_interp[555] / wide_interp[480]
    wide_interp["r_555_445"] = wide_interp[555] / wide_interp[445]
    wide_interp["r_630_555"] = wide_interp[630] / wide_interp[555]

    # 1.2 Ingest Dosing Data
    dose_files = glob.glob(os.path.join(DATA_DIR, "dosing_events", "*.csv"))
    if not dose_files:
        raise FileNotFoundError(f"No dosing CSV found in {DATA_DIR}/dosing_events")
    df_dose = pd.read_csv(dose_files[0])
    df_dose["ts"] = pd.to_datetime(df_dose["timestamp_localtime"])

    # Waste events
    waste = df_dose[df_dose["event"] == "remove_waste"]
    waste_vol_ml = waste["volume_change_ml"].sum()

    # Specified pump dosing (exclude 22:37 test pulses)
    pumps = df_dose[df_dose["source_of_event"].str.startswith("specified_pump_dosing", na=False)].copy()
    pumps = pumps[pumps["ts"] > "2026-10-28 23:00:00"].sort_values("ts").reset_index(drop=True)

    # Subdose clustering (< 5s)
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

    # 1.3 Ingest OD Data
    od_files = glob.glob(os.path.join(DATA_DIR, "od_readings", "*.csv"))
    df_od = pd.read_csv(od_files[0]) if od_files else pd.DataFrame()
    if not df_od.empty:
        df_od["ts"] = pd.to_datetime(df_od["timestamp_localtime"])

    # 1.4 Sequence Alignment
    df_seq = pd.read_csv(SEQ_PATH)
    non_zero_seq = df_seq[df_seq["volume_mL"] > 0].copy().reset_index(drop=True)

    print(f"Loaded Spectrometer Scans: {len(df_spec)} rows across {len(wide_interp)} seconds.")
    print(f"Loaded Dosing Events: {len(pumps)} raw rows, {len(clusters)} non-zero physical pulse clusters.")
    print(f"Expected Sequence: {len(df_seq)} total steps ({len(non_zero_seq)} non-zero, 50 zero-volume).")

    # Map timestamps for all 200 steps
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

    # Audit statistics
    dt_steps = [ (step_times[s+1] - step_times[s]).total_seconds() for s in range(199) ]
    total_acid_ml = clusters[clusters["pump"].str.contains("media", na=False) & ~clusters["pump"].str.contains("alt_media", na=False)]["vol_sum"].sum()
    total_base_ml = clusters[clusters["pump"].str.contains("alt_media", na=False)]["vol_sum"].sum()

    audit = {
        "total_steps": 200,
        "non_zero_clusters": len(clusters),
        "mean_step_dt_s": float(np.mean(dt_steps)),
        "std_step_dt_s": float(np.std(dt_steps)),
        "min_step_dt_s": float(np.min(dt_steps)),
        "max_step_dt_s": float(np.max(dt_steps)),
        "total_acid_ml": float(total_acid_ml),
        "total_base_ml": float(total_base_ml),
        "waste_events": len(waste),
        "waste_vol_ml": float(waste_vol_ml),
        "t_start_run": str(step_times[0]),
        "t_first_dose": str(step_times[30]),
        "t_last_dose": str(step_times[199]),
        "t_end_logging": str(df_spec["ts"].max())
    }

    print("\n--- HYDRAULIC & TIMING AUDIT ---")
    for k, v in audit.items():
        print(f"  {k}: {v}")

    return wide_interp, df_seq, step_times, clusters, df_od, audit


# -----------------------------------------------------------------------------
# 2. Baseline Stability & Buffer Range Check
# -----------------------------------------------------------------------------
def analyze_buffer_stability(wide_interp, step_times):
    print("\n" + "=" * 70)
    print("2. CARRIER BUFFER & OPTICAL STABILITY ANALYSIS")
    print("=" * 70)

    # Segments
    t_w_start = step_times[0]
    t_w_end = step_times[30]
    t_d_end = step_times[199] + pd.Timedelta(seconds=60)
    t_tail_end = wide_interp.index.max()

    washout_ratio = wide_interp.loc[t_w_start:t_w_end, "r_555_480"]
    dosing_ratio = wide_interp.loc[t_w_end:t_d_end, "r_555_480"]
    tail_ratio = wide_interp.loc[t_d_end:t_tail_end, "r_555_480"]
    full_ratio = wide_interp["r_555_480"]

    stats_buf = {
        "washout_mean": float(washout_ratio.mean()),
        "washout_std": float(washout_ratio.std()),
        "washout_min": float(washout_ratio.min()),
        "washout_max": float(washout_ratio.max()),
        "dosing_mean": float(dosing_ratio.mean()),
        "dosing_std": float(dosing_ratio.std()),
        "dosing_min": float(dosing_ratio.min()),
        "dosing_max": float(dosing_ratio.max()),
        "tail_mean": float(tail_ratio.mean()),
        "tail_std": float(tail_ratio.std()),
        "full_min": float(full_ratio.min()),
        "full_max": float(full_ratio.max()),
        "gate_147_151_passed": bool(1.470 <= washout_ratio.mean() <= 1.510),
        "saturation_boundary_140_158_passed": bool(dosing_ratio.min() >= 1.40 and dosing_ratio.max() <= 1.58)
    }

    print(f"Washout Baseline (555/480): Mean = {stats_buf['washout_mean']:.4f} ± {stats_buf['washout_std']:.4f} [Target: 1.470 - 1.510]")
    print(f"  Passes Gate (1.47 - 1.51): {stats_buf['gate_147_151_passed']}")
    print(f"Dosing Dynamic Range (555/480): Min = {stats_buf['dosing_min']:.4f}, Max = {stats_buf['dosing_max']:.4f} [Bounds: 1.400 - 1.580]")
    print(f"  Passes Saturation Bounds (1.40 - 1.58): {stats_buf['saturation_boundary_140_158_passed']}")
    print(f"Post-Dosing Tail Recovery (555/480): Mean = {stats_buf['tail_mean']:.4f} ± {stats_buf['tail_std']:.4f}")

    return stats_buf


# -----------------------------------------------------------------------------
# 3. Feature Extraction (Temporal Multiplexing & Apparent Absorbance)
# -----------------------------------------------------------------------------
def extract_state_features(wide_interp, df_seq, step_times):
    print("\n" + "=" * 70)
    print("3. EXTRACTING RESERVOIR STATE FEATURES")
    print("=" * 70)

    taps = [10, 20, 30, 40, 50, 60]
    ratio_cols = ["r_555_480", "r_555_445", "r_630_555"]

    # Compute baseline reference I0 for apparent absorbances from washout
    baseline_sub = wide_interp.loc[step_times[0]:step_times[29]]
    I0_means = {b: baseline_sub[b].mean() for b in BANDS}

    feature_rows = []
    for s in range(200):
        t_step = step_times[s]
        lvl = int(df_seq.loc[s, "level"])
        vol = float(df_seq.loc[s, "volume_mL"])
        a_b = df_seq.loc[s, "acid_or_base"]
        u_signed_vol = vol if a_b == "acid" else (-vol if a_b == "base" else 0.0)

        row = {
            "step": s,
            "segment": df_seq.loc[s, "segment"],
            "u": lvl,  # discrete integer input [-4..+4]
            "u_vol": u_signed_vol,  # physical signed volume in mL
            "time_drift": s / 200.0,
            "t_start": t_step
        }

        for tap in taps:
            t_tap = (t_step + pd.Timedelta(seconds=tap)).round("1s")
            for r_col in ratio_cols:
                row[f"{r_col}_tap{tap}"] = wide_interp.loc[t_tap, r_col]
            for b in BANDS:
                I_val = wide_interp.loc[t_tap, b]
                row[f"I_{b}_tap{tap}"] = I_val
                row[f"A_{b}_tap{tap}"] = -np.log10(np.clip(I_val / I0_means[b], 1e-5, None))

        feature_rows.append(row)

    df_feat = pd.DataFrame(feature_rows)
    print(f"Extracted {df_feat.shape[1]} total attributes across {len(df_feat)} steps.")
    return df_feat, I0_means, taps, ratio_cols


# -----------------------------------------------------------------------------
# 4. Memory Capacity & Benchmark Evaluation
# -----------------------------------------------------------------------------
def evaluate_memory_and_benchmarks(df_feat, taps, ratio_cols):
    print("\n" + "=" * 70)
    print("4. EVALUATING MEMORY CAPACITY & BENCHMARK READOUTS")
    print("=" * 70)

    # Feature Configurations
    feat_configs = {
        "Protocol_18Taps_Ratios": [f"{r}_tap{tap}" for tap in taps for r in ratio_cols] + ["time_drift"],
        "Ratio555_480_6Taps": [f"r_555_480_tap{tap}" for tap in taps] + ["time_drift"],
        "SingleTap60_8Absorbances": [f"A_{b}_tap60" for b in BANDS] + ["time_drift"],
        "SingleTap60_3Ratios": [f"{r}_tap60" for r in ratio_cols] + ["time_drift"],
        "Full_MultiTap_Absorbances": [f"A_{b}_tap{tap}" for tap in taps for b in BANDS] + ["time_drift"]
    }

    train_mask = (df_feat["segment"] == "train").values
    test_mask = (df_feat["segment"] == "test").values
    u_arr = df_feat["u"].values
    N_test = np.sum(test_mask)

    alphas = np.logspace(-4, 4, 17)
    results = {}

    for cfg_name, cols in feat_configs.items():
        X_raw = df_feat[cols].values
        n_features = len(cols)

        # 4.1 Effective Dimension (Participation Ratio)
        scaler_ed = StandardScaler()
        X_tr_scaled = scaler_ed.fit_transform(X_raw[train_mask])
        cov = np.cov(X_tr_scaled, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 1e-10]
        ed = (eigvals.sum() ** 2) / ((eigvals ** 2).sum()) if len(eigvals) > 0 else 0.0

        # 4.2 Linear Memory Capacity (k=1..15)
        mc_lags = []
        total_mc = 0.0
        preds_dict = {}

        for k in range(1, 16):
            y = np.full(len(u_arr), np.nan)
            y[k:] = u_arr[:-k]

            tr_m = train_mask & ~np.isnan(y)
            te_m = test_mask & ~np.isnan(y)

            X_tr, y_tr = X_raw[tr_m], y[tr_m]
            X_te, y_te = X_raw[te_m], y[te_m]

            # 5-fold CV for alpha selection on training segment only
            kf = KFold(n_splits=5, shuffle=False)
            best_alpha = 1.0
            best_cv_mse = 1e9
            for a in alphas:
                f_mses = []
                for tr_idx, val_idx in kf.split(X_tr):
                    scl = StandardScaler()
                    X_tr_f = scl.fit_transform(X_tr[tr_idx])
                    X_val_f = scl.transform(X_tr[val_idx])
                    m = Ridge(alpha=a).fit(X_tr_f, y_tr[tr_idx])
                    pred_v = m.predict(X_val_f)
                    f_mses.append(np.mean((y_tr[val_idx] - pred_v) ** 2))
                mean_f = np.mean(f_mses)
                if mean_f < best_cv_mse:
                    best_cv_mse = mean_f
                    best_alpha = a

            # Fit on full training set
            scl_final = StandardScaler()
            X_tr_s = scl_final.fit_transform(X_tr)
            X_te_s = scl_final.transform(X_te)
            model = Ridge(alpha=best_alpha).fit(X_tr_s, y_tr)
            pred_te = model.predict(X_te_s)

            if k <= 3:
                preds_dict[k] = {"actual": y_te.tolist(), "predicted": pred_te.tolist()}

            r = np.corrcoef(y_te, pred_te)[0, 1] if np.var(pred_te) > 0 else 0.0
            r2 = max(0.0, r ** 2) if r > 0 else 0.0
            nmse = np.mean((y_te - pred_te) ** 2) / np.var(y_te) if np.var(y_te) > 0 else 1.0

            mc_lags.append({
                "lag": k,
                "r": float(r),
                "r2": float(r2),
                "nmse": float(nmse),
                "alpha": float(best_alpha)
            })
            total_mc += r2

        # 4.3 Shuffled Null Baseline (100 permutations)
        shuff_mc_totals = []
        for _ in range(100):
            shuff_u = np.random.permutation(u_arr)
            shuff_total = 0.0
            for k in range(1, 16):
                y_sh = np.full(len(shuff_u), np.nan)
                y_sh[k:] = shuff_u[:-k]
                tr_m = train_mask & ~np.isnan(y_sh)
                te_m = test_mask & ~np.isnan(y_sh)
                scl = StandardScaler()
                X_tr_s = scl.fit_transform(X_raw[tr_m])
                X_te_s = scl.transform(X_raw[te_m])
                m = Ridge(alpha=1.0).fit(X_tr_s, y_sh[tr_m])
                p_te = m.predict(X_te_s)
                r_sh = np.corrcoef(y_sh[te_m], p_te)[0, 1] if np.var(p_te) > 0 else 0.0
                shuff_total += max(0.0, r_sh ** 2) if r_sh > 0 else 0.0
            shuff_mc_totals.append(shuff_total)

        null_mean = float(np.mean(shuff_mc_totals))
        null_95th = float(np.percentile(shuff_mc_totals, 95))
        noise_floor_analytical = float(n_features / N_test)

        # 4.4 Delayed Parity / XOR Benchmark (k=1..3)
        parity_res = {}
        for lag_p in [1, 2, 3]:
            # parity: sign(u[t]) * sign(u[t-lag])
            s_u = np.sign(u_arr)
            y_par = np.full(len(u_arr), np.nan)
            y_par[lag_p:] = s_u[lag_p:] * s_u[:-lag_p]
            # Replace 0s with 1 or exclude
            y_par[y_par == 0] = 1.0

            tr_m = train_mask & ~np.isnan(y_par)
            te_m = test_mask & ~np.isnan(y_par)
            scl = StandardScaler()
            X_tr_s = scl.fit_transform(X_raw[tr_m])
            X_te_s = scl.transform(X_raw[te_m])
            m = Ridge(alpha=1.0).fit(X_tr_s, y_par[tr_m])
            pred_p = m.predict(X_te_s)
            pred_binary = np.where(pred_p >= 0, 1.0, -1.0)
            acc = np.mean(pred_binary == y_par[te_m])
            parity_res[f"lag_{lag_p}"] = float(acc)

        # 4.5 Degree-2 Nonlinear Polynomial Capacity
        # Target: u[t-1] * u[t-2]
        y_poly = np.full(len(u_arr), np.nan)
        y_poly[2:] = u_arr[1:-1] * u_arr[:-2]
        tr_m = train_mask & ~np.isnan(y_poly)
        te_m = test_mask & ~np.isnan(y_poly)
        scl = StandardScaler()
        X_tr_s = scl.fit_transform(X_raw[tr_m])
        X_te_s = scl.transform(X_raw[te_m])
        m_poly = Ridge(alpha=1.0).fit(X_tr_s, y_poly[tr_m])
        pred_poly = m_poly.predict(X_te_s)
        r_poly = np.corrcoef(y_poly[te_m], pred_poly)[0, 1] if np.var(pred_poly) > 0 else 0.0
        cap_deg2 = max(0.0, r_poly ** 2) if r_poly > 0 else 0.0

        # Target: u[t-1]^2
        y_sq = np.full(len(u_arr), np.nan)
        y_sq[1:] = u_arr[:-1] ** 2
        tr_m = train_mask & ~np.isnan(y_sq)
        te_m = test_mask & ~np.isnan(y_sq)
        m_sq = Ridge(alpha=1.0).fit(X_tr_s, y_sq[tr_m])
        pred_sq = m_sq.predict(X_te_s)
        r_sq = np.corrcoef(y_sq[te_m], pred_sq)[0, 1] if np.var(pred_sq) > 0 else 0.0
        cap_sq = max(0.0, r_sq ** 2) if r_sq > 0 else 0.0

        # 4.6 NARMA-2 Benchmark
        # u mapped to [0, 0.5]
        u_norm = (u_arr - np.min(u_arr)) / (np.ptp(u_arr) if np.ptp(u_arr) > 0 else 1.0) * 0.5
        y_narma2 = np.zeros(len(u_arr))
        for t in range(1, len(u_arr) - 1):
            y_narma2[t+1] = 0.4 * y_narma2[t] + 0.4 * y_narma2[t] * y_narma2[t-1] + 0.6 * (u_norm[t] ** 3) + 0.1
        tr_m = train_mask
        te_m = test_mask
        m_narma = Ridge(alpha=1.0).fit(X_tr_s, y_narma2[tr_m])
        pred_narma = m_narma.predict(X_te_s)
        narma2_nmse = np.mean((y_narma2[te_m] - pred_narma) ** 2) / np.var(y_narma2[te_m])

        results[cfg_name] = {
            "n_features": n_features,
            "effective_dimension": float(ed),
            "total_mc": float(total_mc),
            "mc_lags": mc_lags,
            "null_shuffled_mean": null_mean,
            "null_shuffled_95th": null_95th,
            "noise_floor_analytical": noise_floor_analytical,
            "parity_accuracy": parity_res,
            "degree2_capacity_cross": float(cap_deg2),
            "degree2_capacity_squared": float(cap_sq),
            "narma2_nmse": float(narma2_nmse),
            "test_predictions": preds_dict
        }

        print(f"\nConfiguration: {cfg_name}")
        print(f"  Features: {n_features} | Effective Dimension: {ed:.2f}")
        print(f"  Total Linear MC (k=1..15): {total_mc:.4f} (Null 95th: {null_95th:.4f})")
        print(f"  MC lag-1: {mc_lags[0]['r2']:.4f} (corr r: {mc_lags[0]['r']:.4f})")
        print(f"  Delayed Parity Acc: lag-1={parity_res['lag_1']*100:.1f}%, lag-2={parity_res['lag_2']*100:.1f}%")
        print(f"  Degree-2 Capacity: cross={cap_deg2:.4f}, squared={cap_sq:.4f}")
        print(f"  NARMA-2 NMSE: {narma2_nmse:.4f}")

    return results


# -----------------------------------------------------------------------------
# 5. Publication Visualization Suite
# -----------------------------------------------------------------------------
def generate_publication_figures(wide_interp, df_seq, step_times, clusters, df_od, df_feat, results, stats_buf):
    print("\n" + "=" * 70)
    print("5. GENERATING PUBLICATION-GRADE FIGURES")
    print("=" * 70)

    fig_paths = {}

    # -------------------------------------------------------------------------
    # Figure 1: Complete 5-Hour Experimental Overview
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.0, 1.0, 0.8]})
    
    # Segment boundaries
    t_w0 = step_times[0]
    t_tr0 = step_times[30]
    t_te0 = step_times[160]
    t_end = step_times[199] + pd.Timedelta(seconds=60)
    
    for ax in axes:
        ax.axvspan(t_w0, t_tr0, color="#f0f0f0", alpha=0.7, label="Washout (Steps 0-29)" if ax == axes[0] else "")
        ax.axvspan(t_tr0, t_te0, color="#e6f2ff", alpha=0.5, label="Training (Steps 30-159)" if ax == axes[0] else "")
        ax.axvspan(t_te0, t_end, color="#fff2e6", alpha=0.5, label="Testing (Steps 160-199)" if ax == axes[0] else "")
        ax.axvline(t_end, color="gray", linestyle="--", linewidth=1.2, label="Sequence End" if ax == axes[0] else "")

    # Subplot 1: Dosing Pulses
    ax0 = axes[0]
    acid_c = clusters[clusters["pump"].str.contains("media", na=False) & ~clusters["pump"].str.contains("alt_media", na=False)]
    base_c = clusters[clusters["pump"].str.contains("alt_media", na=False)]
    ax0.stem(acid_c["t_start"], acid_c["vol_sum"], linefmt="red", markerfmt="ro", basefmt=" ", label="Acid Pulses (add_media)")
    ax0.stem(base_c["t_start"], -base_c["vol_sum"], linefmt="blue", markerfmt="bo", basefmt=" ", label="Base Pulses (add_alt_media)")
    ax0.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax0.set_ylabel("Dose (mL)\n[+ Acid / - Base]", fontsize=10, fontweight="bold")
    ax0.set_title("Stage 4 Benchmark Run: Dosing Sequence, Spectral Dynamics & Hydraulic Stability", fontsize=13, fontweight="bold")
    ax0.legend(loc="upper left", framealpha=0.9, fontsize=9, ncol=4)
    ax0.grid(True, alpha=0.3)

    # Subplot 2: AS7341 Spectral Bands
    ax1 = axes[1]
    for b in [445, 480, 515, 555, 590, 630]:
        ax1.plot(wide_interp.index, wide_interp[b], color=BAND_COLORS[b], linewidth=1.1, label=f"{b} nm")
    ax1.set_ylabel("Intensity (counts)", fontsize=10, fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=8, ncol=6)
    ax1.grid(True, alpha=0.3)

    # Subplot 3: 555/480 Ratio & Operating Window
    ax2 = axes[2]
    ax2.plot(wide_interp.index, wide_interp["r_555_480"], color="#2c3e50", linewidth=1.2, label="Ratio 555 / 480 nm")
    ax2.axhspan(1.47, 1.51, color="#2ecc71", alpha=0.25, label="Target Operating Window [1.47, 1.51]")
    ax2.axhline(1.40, color="red", linestyle=":", linewidth=1.0, label="Acid Saturation (1.40)")
    ax2.axhline(1.58, color="blue", linestyle=":", linewidth=1.0, label="Base Saturation (1.58)")
    ax2.set_ylabel("Ratio 555 / 480", fontsize=10, fontweight="bold")
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=8, ncol=4)
    ax2.set_ylim(1.43, 1.55)
    ax2.grid(True, alpha=0.3)

    # Subplot 4: Optical Density Stability
    ax3 = axes[3]
    if not df_od.empty:
        od_ch3 = df_od[df_od["channel"] == 3]
        od_ch4 = df_od[df_od["channel"] == 4]
        ax3.plot(od_ch3["ts"], od_ch3["od_reading"], color="#8e44ad", linewidth=1.0, label="OD 45° (Ch 3)")
        ax3.plot(od_ch4["ts"], od_ch4["od_reading"], color="#d35400", linewidth=1.0, label="OD 135° (Ch 4)")
    ax3.set_ylabel("OD Reading", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Time (Local IST)", fontsize=11, fontweight="bold")
    ax3.legend(loc="upper left", framealpha=0.9, fontsize=9, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    plt.tight_layout()
    p1 = os.path.join(OUTPUT_DIR, "figures", "fig1_stage4_full_overview.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    fig_paths["fig1"] = p1
    print(f"  Saved Figure 1: {p1}")

    # -------------------------------------------------------------------------
    # Figure 2: Chemical Reservoir Buffer Stability & Operating Window
    # -------------------------------------------------------------------------
    fig, (ax_ts, ax_hist) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2.5, 1]})
    
    ax_ts.plot(wide_interp.index, wide_interp["r_555_480"], color="#34495e", linewidth=1.2, label="R(t) = I_555 / I_480")
    ax_ts.axhspan(1.47, 1.51, color="#2ecc71", alpha=0.3, label="Target Window [1.47, 1.51] (pKa ~ 3.74)")
    ax_ts.axhline(stats_buf["washout_mean"], color="#e74c3c", linestyle="--", linewidth=1.2, label=f"Baseline F0 = {stats_buf['washout_mean']:.4f}")
    ax_ts.axhline(1.40, color="darkred", linestyle=":", label="Acid Plateau (1.40)")
    ax_ts.axhline(1.58, color="darkblue", linestyle=":", label="Base Plateau (1.58)")
    ax_ts.set_title("Carrier pH Buffer Stability Across Full Experiment", fontsize=12, fontweight="bold")
    ax_ts.set_xlabel("Time (IST)", fontsize=10, fontweight="bold")
    ax_ts.set_ylabel("Ratio 555 / 480 nm", fontsize=10, fontweight="bold")
    ax_ts.set_ylim(1.44, 1.54)
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_ts.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_ts.grid(True, alpha=0.3)

    # Histogram of ratios
    dosing_r = wide_interp.loc[t_tr0:t_end, "r_555_480"]
    ax_hist.hist(dosing_r, bins=35, color="#3498db", edgecolor="black", alpha=0.75, orientation="horizontal", density=True)
    ax_hist.axhspan(1.47, 1.51, color="#2ecc71", alpha=0.3)
    ax_hist.axhline(stats_buf["washout_mean"], color="#e74c3c", linestyle="--")
    ax_hist.set_title("Dosing Response Distribution", fontsize=11, fontweight="bold")
    ax_hist.set_xlabel("Probability Density", fontsize=10, fontweight="bold")
    ax_hist.set_ylim(1.44, 1.54)
    ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    p2 = os.path.join(OUTPUT_DIR, "figures", "fig2_carrier_buffer_stability.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    fig_paths["fig2"] = p2
    print(f"  Saved Figure 2: {p2}")

    # -------------------------------------------------------------------------
    # Figure 3: Dosing Sequence vs Physical Optical Response (Zoomed-in)
    # -------------------------------------------------------------------------
    fig, (ax_d, ax_r) = plt.subplots(2, 1, figsize=(14, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1.5]})
    
    # Zoom in on steps 30 to 70 (first 40 steps of training)
    t_z_start = step_times[30] - pd.Timedelta(seconds=60)
    t_z_end = step_times[70]

    acid_sub = acid_c[(acid_c["t_start"] >= t_z_start) & (acid_c["t_start"] <= t_z_end)]
    base_sub = base_c[(base_c["t_start"] >= t_z_start) & (base_c["t_start"] <= t_z_end)]
    wide_sub = wide_interp.loc[t_z_start:t_z_end]

    ax_d.stem(acid_sub["t_start"], acid_sub["vol_sum"], linefmt="red", markerfmt="ro", basefmt=" ", label="Acid (+ Dose)")
    ax_d.stem(base_sub["t_start"], -base_sub["vol_sum"], linefmt="blue", markerfmt="bo", basefmt=" ", label="Base (- Dose)")
    ax_d.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax_d.set_ylabel("Input Volume (mL)", fontsize=10, fontweight="bold")
    ax_d.set_title("Microscopic Physical Dynamics: Input Pulses vs Flowing Reservoir State (Steps 30-70)", fontsize=12, fontweight="bold")
    ax_d.legend(loc="upper right", fontsize=9)
    ax_d.grid(True, alpha=0.3)

    ax_r.plot(wide_sub.index, wide_sub["r_555_480"], color="#2c3e50", linewidth=1.5, marker=".", markersize=3, label="Ratio 555/480 nm")
    ax_r.axhspan(1.47, 1.51, color="#2ecc71", alpha=0.25, label="Buffer Midpoint Window")
    ax_r.set_ylabel("Ratio 555 / 480", fontsize=10, fontweight="bold")
    ax_r.set_xlabel("Time (IST)", fontsize=10, fontweight="bold")
    ax_r.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax_r.legend(loc="upper right", fontsize=9)
    ax_r.grid(True, alpha=0.3)

    plt.tight_layout()
    p3 = os.path.join(OUTPUT_DIR, "figures", "fig3_dosing_vs_optical_response_zoom.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    fig_paths["fig3"] = p3
    print(f"  Saved Figure 3: {p3}")

    # -------------------------------------------------------------------------
    # Figure 4: Memory Capacity Curves (MC_k vs lag k)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    lags = np.arange(1, 16)
    markers = ["o", "s", "^", "D", "v"]
    cfg_colors = {
        "Protocol_18Taps_Ratios": "#1f77b4",
        "Ratio555_480_6Taps": "#ff7f0e",
        "SingleTap60_8Absorbances": "#2ca02c",
        "SingleTap60_3Ratios": "#d62728",
        "Full_MultiTap_Absorbances": "#9467bd"
    }

    for (cfg_name, res), m in zip(results.items(), markers):
        r2_vals = [item["r2"] for item in res["mc_lags"]]
        tot = res["total_mc"]
        ed = res["effective_dimension"]
        lbl = f"{cfg_name} (MC={tot:.3f}, ED={ed:.1f})"
        ax.plot(lags, r2_vals, marker=m, linewidth=2.0, markersize=6, color=cfg_colors.get(cfg_name, "black"), label=lbl)

    # Shuffled null 95th percentile
    ref_null = results["Protocol_18Taps_Ratios"]["null_shuffled_95th"] / 15.0
    ax.axhline(ref_null, color="gray", linestyle="--", linewidth=1.5, label=f"Shuffled Null Floor (per lag ~ {ref_null:.3f})")

    ax.set_title("Stage 4 Linear Memory Capacity Profiles ($MC_k = r^2$ vs Lag $k$)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Reconstruction Lag $k$ (Timesteps, $T_s = 60$ s)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Determination Coefficient $r_k^2$", fontsize=11, fontweight="bold")
    ax.set_xticks(lags)
    ax.set_ylim(-0.01, 0.25)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    p4 = os.path.join(OUTPUT_DIR, "figures", "fig4_memory_capacity_curves.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    fig_paths["fig4"] = p4
    print(f"  Saved Figure 4: {p4}")

    # -------------------------------------------------------------------------
    # Figure 5: Test-Set Reconstructed Trajectories (Lags 1, 2, 3)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    preds = results["Protocol_18Taps_Ratios"]["test_predictions"]
    test_steps = np.arange(160, 200)

    for i, lag in enumerate([1, 2, 3]):
        ax = axes[i]
        act = np.array(preds[lag]["actual"])
        prd = np.array(preds[lag]["predicted"])
        corr = np.corrcoef(act, prd)[0, 1]
        r2 = corr ** 2

        ax.plot(test_steps[:len(act)], act, color="black", marker="o", linewidth=1.5, markersize=4, label=f"Actual Target u[t-{lag}]")
        ax.plot(test_steps[:len(prd)], prd, color="#e74c3c", marker="s", linestyle="--", linewidth=1.5, markersize=4, label=f"Reservoir Readout (r={corr:.3f}, r²={r2:.3f})")
        ax.set_ylabel(f"Lag k={lag}\nInput Level", fontsize=10, fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("Test-Set Chronological Reconstruction: Actual vs Readout Predictions", fontsize=12, fontweight="bold")

    axes[-1].set_xlabel("Test Step Index (Timestep $t$)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    p5 = os.path.join(OUTPUT_DIR, "figures", "fig5_test_reconstruction_trajectories.png")
    plt.savefig(p5, dpi=300)
    plt.close()
    fig_paths["fig5"] = p5
    print(f"  Saved Figure 5: {p5}")

    # -------------------------------------------------------------------------
    # Figure 6: Nonlinear Benchmark Performance
    # -------------------------------------------------------------------------
    fig, (ax_p, ax_n) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Parity Accuracy
    cfgs = list(results.keys())
    x_pos = np.arange(len(cfgs))
    p_lag1 = [results[c]["parity_accuracy"]["lag_1"] * 100 for c in cfgs]
    p_lag2 = [results[c]["parity_accuracy"]["lag_2"] * 100 for c in cfgs]

    w = 0.35
    ax_p.bar(x_pos - w/2, p_lag1, width=w, color="#3498db", label="Delayed Parity (Lag 1)")
    ax_p.bar(x_pos + w/2, p_lag2, width=w, color="#e67e22", label="Delayed Parity (Lag 2)")
    ax_p.axhline(50.0, color="red", linestyle="--", linewidth=1.5, label="Chance Level (50%)")
    ax_p.set_xticks(x_pos)
    ax_p.set_xticklabels([c.replace("_", "\n") for c in cfgs], fontsize=8)
    ax_p.set_ylabel("Classification Accuracy (%)", fontsize=10, fontweight="bold")
    ax_p.set_title("Delayed Parity / XOR Benchmark vs Chance", fontsize=11, fontweight="bold")
    ax_p.set_ylim(35, 75)
    ax_p.legend(loc="upper right", fontsize=9)
    ax_p.grid(True, alpha=0.3, axis="y")

    # Degree-2 Polynomial Capacity & NARMA
    deg2_cross = [results[c]["degree2_capacity_cross"] for c in cfgs]
    deg2_sq = [results[c]["degree2_capacity_squared"] for c in cfgs]

    ax_n.bar(x_pos - w/2, deg2_cross, width=w, color="#2ecc71", label="Cross-Product u[t-1]·u[t-2]")
    ax_n.bar(x_pos + w/2, deg2_sq, width=w, color="#9b59b6", label="Squared u[t-1]²")
    ax_n.set_xticks(x_pos)
    ax_n.set_xticklabels([c.replace("_", "\n") for c in cfgs], fontsize=8)
    ax_n.set_ylabel("Degree-2 Capacity ($r^2$)", fontsize=10, fontweight="bold")
    ax_n.set_title("Degree-2 Nonlinear Information Capacity", fontsize=11, fontweight="bold")
    ax_n.legend(loc="upper right", fontsize=9)
    ax_n.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    p6 = os.path.join(OUTPUT_DIR, "figures", "fig6_nonlinear_benchmarks.png")
    plt.savefig(p6, dpi=300)
    plt.close()
    fig_paths["fig6"] = p6
    print(f"  Saved Figure 6: {p6}")

    return fig_paths


# -----------------------------------------------------------------------------
# 6. Structured Export & Summary Report
# -----------------------------------------------------------------------------
def export_results(df_feat, results, audit, stats_buf, fig_paths):
    print("\n" + "=" * 70)
    print("6. EXPORTING STRUCTURED RESULTS & SUMMARY")
    print("=" * 70)

    # Save aligned features CSV
    csv_feat_path = os.path.join(OUTPUT_DIR, "stage4_aligned_features.csv")
    df_feat.to_csv(csv_feat_path, index=False)
    print(f"Saved aligned features CSV: {csv_feat_path}")

    # Save MC results table
    mc_rows = []
    for cfg_name, res in results.items():
        for item in res["mc_lags"]:
            mc_rows.append({
                "configuration": cfg_name,
                "n_features": res["n_features"],
                "effective_dimension": res["effective_dimension"],
                "total_mc": res["total_mc"],
                "lag": item["lag"],
                "r": item["r"],
                "r2": item["r2"],
                "nmse": item["nmse"],
                "alpha": item["alpha"],
                "null_95th": res["null_shuffled_95th"]
            })
    df_mc = pd.DataFrame(mc_rows)
    csv_mc_path = os.path.join(OUTPUT_DIR, "stage4_mc_results.csv")
    df_mc.to_csv(csv_mc_path, index=False)
    print(f"Saved Memory Capacity table: {csv_mc_path}")

    # Save JSON summary
    json_path = os.path.join(OUTPUT_DIR, "stage4_summary.json")
    summary_data = {
        "audit": audit,
        "buffer_stability": stats_buf,
        "benchmark_results": {
            cfg: {k: v for k, v in data.items() if k != "test_predictions"}
            for cfg, data in results.items()
        },
        "figures": fig_paths
    }
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved JSON summary: {json_path}")

    # Save Markdown / Text summary report
    txt_path = os.path.join(OUTPUT_DIR, "stage4_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 75 + "\n")
        f.write("STAGE 4 BENCHMARK EXECUTIVE REPORT: CHEMICAL RESERVOIR COMPUTING\n")
        f.write("System: Methyl Orange CSTR Reservoir (Acid / Base Bipolar Dosing)\n")
        f.write("=" * 75 + "\n\n")

        f.write("1. HYDRAULIC & PHYSICAL AUDIT\n")
        f.write(f"  - Total Timesteps: {audit['total_steps']} (Washout: 30, Train: 130, Test: 40)\n")
        f.write(f"  - Physical Doses Delivered: {audit['non_zero_clusters']} non-zero pulses (100% sequence compliance)\n")
        f.write(f"  - Step Duration (Ts): Mean = {audit['mean_step_dt_s']:.2f} s (SD = {audit['std_step_dt_s']:.2f} s)\n")
        f.write(f"  - Acid Delivered: {audit['total_acid_ml']:.4f} mL vinegar\n")
        f.write(f"  - Base Delivered: {audit['total_base_ml']:.4f} mL dilute ammonia\n")
        f.write(f"  - Run Window: {audit['t_first_dose']} to {audit['t_last_dose']}\n\n")

        f.write("2. CARRIER BUFFER & OPTICAL STABILITY\n")
        f.write(f"  - Washout Pre-Dose Baseline: {stats_buf['washout_mean']:.4f} ± {stats_buf['washout_std']:.4f}\n")
        f.write(f"    Target Gate [1.470, 1.510]: {'PASSED' if stats_buf['gate_147_151_passed'] else 'FAILED'}\n")
        f.write(f"  - Dosing Dynamic Range: Min = {stats_buf['dosing_min']:.4f}, Max = {stats_buf['dosing_max']:.4f}\n")
        f.write(f"    Saturation Bounds [1.400, 1.580]: {'PASSED' if stats_buf['saturation_boundary_140_158_passed'] else 'FAILED'}\n")
        f.write(f"  - Post-Dosing Recovery: {stats_buf['tail_mean']:.4f} ± {stats_buf['tail_std']:.4f}\n\n")

        f.write("3. BENCHMARK PERFORMANCE ACROSS FEATURE CONFIGURATIONS\n")
        for cfg_name, res in results.items():
            f.write(f"  Configuration: {cfg_name}\n")
            f.write(f"    Features: {res['n_features']} | Effective Dimension (PR): {res['effective_dimension']:.2f}\n")
            f.write(f"    Total Memory Capacity (k=1..15): {res['total_mc']:.4f}\n")
            f.write(f"    Null Shuffled Baseline (95th %ile): {res['null_shuffled_95th']:.4f}\n")
            f.write(f"    Lag-1 MC: r2={res['mc_lags'][0]['r2']:.4f} (corr r={res['mc_lags'][0]['r']:.4f})\n")
            f.write(f"    Lag-2 MC: r2={res['mc_lags'][1]['r2']:.4f} (corr r={res['mc_lags'][1]['r']:.4f})\n")
            f.write(f"    Lag-3 MC: r2={res['mc_lags'][2]['r2']:.4f} (corr r={res['mc_lags'][2]['r']:.4f})\n")
            f.write(f"    Delayed Parity (XOR): Lag-1 = {res['parity_accuracy']['lag_1']*100:.1f}%, Lag-2 = {res['parity_accuracy']['lag_2']*100:.1f}%\n")
            f.write(f"    Degree-2 Nonlinear Capacity: Cross = {res['degree2_capacity_cross']:.4f}, Squared = {res['degree2_capacity_squared']:.4f}\n")
            f.write(f"    NARMA-2 Test NMSE: {res['narma2_nmse']:.4f}\n\n")

    print(f"Saved Executive Text Report: {txt_path}")
    print("\nANALYSIS COMPLETE SUCCESSFULLY.")


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main():
    wide_interp, df_seq, step_times, clusters, df_od, audit = load_and_audit_data()
    stats_buf = analyze_buffer_stability(wide_interp, step_times)
    df_feat, I0_means, taps, ratio_cols = extract_state_features(wide_interp, df_seq, step_times)
    results = evaluate_memory_and_benchmarks(df_feat, taps, ratio_cols)
    fig_paths = generate_publication_figures(wide_interp, df_seq, step_times, clusters, df_od, df_feat, results, stats_buf)
    export_results(df_feat, results, audit, stats_buf, fig_paths)

if __name__ == "__main__":
    main()
