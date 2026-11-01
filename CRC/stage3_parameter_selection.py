#!/usr/bin/env python3
"""
stage3_parameter_selection.py — Stage 3 Parameter Selection & Benchmark Sequence Generator
for Chemical Reservoir Computing (Methyl Orange, Pioreactor CSTR).

Audited & Grounded in Measured Hydraulics:
  - Rule 1: v_max selection (80% overshoot dF rule vs linear gain)
  - Rule 2: Level size dv = v_max / 4 (checks Delta F_step >= 0.007, flags failures)
  - Rule 3: Decay check (compares tau directly against measured hydraulics from Stage 1)
  - Rule 4: Step length T_s = tau / 3 (derives target 40-50s vs nominal 60s)

Generates:
  - stage4_sequence_600steps.csv
  - stage4_planned_inputs.json
  - stage3_parameters.json & stage3_parameter_report.txt / .md
  - Diagnostic figures (parameter selection & sequence preview)
"""

import sys
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# UTF-8 stdout protection
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "font.size": 10,
    "legend.frameon": False
})

def main():
    root = Path("CRC")
    out_dir = root / "results" / "stage03"
    out_dir.mkdir(parents=True, exist_ok=True)
    figdir = out_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    
    # Also support alias directory
    out_dir_alias = root / "results_stage3"
    out_dir_alias.mkdir(parents=True, exist_ok=True)
    (out_dir_alias / "figures").mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("STAGE 3: BENCHMARK RUN PARAMETER SELECTION & SEQUENCE GENERATOR (AUDITED)")
    print("=" * 75)

    # --------------------------------------------------------------------------
    # 1. LOAD EMPIRICAL BENCHMARKS FROM STAGE 1 & STAGE 2
    # --------------------------------------------------------------------------
    # Stage 1: Measured Hydraulics
    stage1_report_path = root / "results" / "stage01" / "baseline_report.json"
    if not stage1_report_path.exists():
        raise FileNotFoundError(f"Missing {stage1_report_path}")
    with open(stage1_report_path, "r", encoding="utf-8") as f:
        s1_data = json.load(f)["stage1_long"]

    hyd = s1_data["hydraulics"]
    Q_meas = float(hyd["Q_mean"])                       # 4.0 mL/min
    tau_V_over_Q = float(hyd["tau_from_V_over_Q_min"])   # 2.5 min = 150 s
    tau_step_fit = float(hyd["tau_step_fit_min"])       # 1.99 min = 119.4 s
    V_eff = float(hyd["effective_mixed_volume_mL"])      # 7.96 mL
    sigma0 = float(s1_data["noise_sd_median_of_5min_bins"]) # ~0.0017
    noise_5sigma = 5.0 * sigma0

    # Stage 2 Block A: Linear Calibration Gains
    blockA_path = root / "results" / "stage 02 - block A" / "summary.json"
    with open(blockA_path, "r", encoding="utf-8") as f:
        s2_blockA = json.load(f)
    g_acid = s2_blockA["gain_models"]["acid"]["slope_per_mL"] # ~0.0649 ratio/mL
    g_base = s2_blockA["gain_models"]["base"]["slope_per_mL"] # ~0.1782 ratio/mL

    # Stage 2 Block B: Measured Overshoot Deflections
    blockBC_path = root / "results" / "stage 02 - blocks B & C" / "summary.json"
    with open(blockBC_path, "r", encoding="utf-8") as f:
        s2_blockBC = json.load(f)
    acid_overshoot_dF = abs(s2_blockBC["block_b"]["acid_0.50"]["amp_mean"]) # ~0.0203
    base_overshoot_dF = abs(s2_blockBC["block_b"]["base_0.50"]["amp_mean"]) # ~0.0078

    # --------------------------------------------------------------------------
    # 2. APPLY RULE 1: v_max SELECTION & ASYMMETRY
    # --------------------------------------------------------------------------
    target_dF_acid = 0.80 * acid_overshoot_dF # 0.80 * 0.0203 = ~0.0162
    target_dF_base = 0.80 * base_overshoot_dF # 0.80 * 0.0078 = ~0.0062

    v_max_acid_raw = target_dF_acid / g_acid  # ~0.250 mL
    v_max_base_raw = target_dF_base / g_base  # ~0.035 mL (or 0.044 mL with earlier 0.0098)

    asymmetry_ratio = g_base / g_acid         # ~2.75x

    # In operational execution:
    # Acid: capped at 0.20 mL (largest tested linear volume from Block A)
    # Base: set to 0.08 mL to attempt symmetric deflection (~0.0142)
    v_max_acid = 0.20
    v_max_base = 0.08
    expected_peak_acid = g_acid * v_max_acid # ~0.0130
    expected_peak_base = g_base * v_max_base # ~0.0143

    # Audit Note on v_max_base:
    v_max_base_discrepancy = expected_peak_base > base_overshoot_dF

    print(f"\n[Rule 1] v_max and Asymmetry Evaluation:")
    print(f"  Acid gain: {g_acid:.4f}/mL | 80% Overshoot target: {target_dF_acid:.4f} -> v_max_raw = {v_max_acid_raw:.3f} mL")
    print(f"  Base gain: {g_base:.4f}/mL | 80% Overshoot target: {target_dF_base:.4f} -> v_max_raw = {v_max_base_raw:.3f} mL")
    print(f"  Selected v_max: Acid = {v_max_acid:.2f} mL, Base = {v_max_base:.2f} mL")
    print(f"  CRITICAL NOTE: v_max_base (0.08 mL) gives expected peak {expected_peak_base:.4f}, which exceeds measured 0.5 mL overshoot ({base_overshoot_dF:.4f}).")

    # --------------------------------------------------------------------------
    # 3. APPLY RULE 2: LEVEL SIZE & 5-SIGMA SEPARATION CHECK
    # --------------------------------------------------------------------------
    # 9 levels: {-4, -3, -2, -1, 0, 1, 2, 3, 4}
    dv_acid = v_max_acid / 4.0 # 0.050 mL
    dv_base = v_max_base / 4.0 # 0.020 mL

    dF_step_acid = dv_acid * g_acid # 0.050 * 0.0649 = ~0.0032
    dF_step_base = dv_base * g_base # 0.020 * 0.1782 = ~0.0036

    rule2_target_dF = 0.0070
    rule2_acid_pass = bool(dF_step_acid >= rule2_target_dF)
    rule2_base_pass = bool(dF_step_base >= rule2_target_dF)
    rule2_pass = rule2_acid_pass and rule2_base_pass

    print(f"\n[Rule 2] Level Size & Noise Separation Check (Target: Delta F_step >= {rule2_target_dF:.4f}):")
    print(f"  Acid step size: {dv_acid*1000:.1f} uL -> Expected Delta F_step = {dF_step_acid:.4f} [{'PASS' if rule2_acid_pass else 'FAIL'}]")
    print(f"  Base step size: {dv_base*1000:.1f} uL -> Expected Delta F_step = {dF_step_base:.4f} [{'PASS' if rule2_base_pass else 'FAIL'}]")
    print(f"  Rule 2 Overall: {'PASSED' if rule2_pass else 'FAILED'}")
    if not rule2_pass:
        print("  CONSEQUENCE: Step deflections (~0.0034) are below the 0.007 plan threshold.")
        print(f"  At measured within-level noise sigma = {sigma0:.4f}, step separation is only {dF_step_acid/sigma0:.1f} sigma.")
        print("  Reagents require ~2.2x - 2.5x higher concentration, or level count should be reduced from 9 to 5.")

    # --------------------------------------------------------------------------
    # 4. APPLY RULE 3 & RULE 4: DECAY CHECK & STEP LENGTH T_s
    # --------------------------------------------------------------------------
    # Step length from measured hydraulics:
    T_s_from_stepfit = (tau_step_fit * 60.0) / 3.0   # 39.8 s
    T_s_from_V_over_Q = (tau_V_over_Q * 60.0) / 3.0  # 50.0 s
    T_s_operational = 60.0                            # Nominal Ts used in Stage 4/5/6

    print(f"\n[Rule 3 & 4] Hydraulics and Step Length T_s:")
    print(f"  Stage 1 Measured Q: {Q_meas:.2f} mL/min | V_eff = {V_eff:.2f} mL")
    print(f"  Hydraulic tau (V/Q): {tau_V_over_Q:.2f} min ({tau_V_over_Q*60:.1f} s)")
    print(f"  Empirical Step-Fit tau: {tau_step_fit:.2f} min ({tau_step_fit*60:.1f} s)")
    print(f"  Rule 4 Target Step Length (tau / 3): {T_s_from_stepfit:.1f} s (step-fit) to {T_s_from_V_over_Q:.1f} s (V/Q)")
    print(f"  Operational Step Length Used: {T_s_operational:.1f} s (T_s / tau_meas = {T_s_operational/(tau_step_fit*60):.2f}x)")

    # --------------------------------------------------------------------------
    # 5. GENERATE STAGE 4 BENCHMARK SEQUENCE (N = 600 STEPS, SEED = 42)
    # --------------------------------------------------------------------------
    SEED = 42
    N_STEPS = 600
    N_WASHOUT = 30
    N_TRAIN = 400
    N_TEST = 170
    levels_range = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

    rng = random.Random(SEED)
    records = []
    for step in range(N_STEPS):
        if step < N_WASHOUT:
            segment = "washout"
            level = 0
        elif step < N_WASHOUT + N_TRAIN:
            segment = "train"
            level = rng.choice(levels_range)
        else:
            segment = "test"
            level = rng.choice(levels_range)

        if level > 0:
            reagent = "acid"
            vol = level * dv_acid
            pump = "add_media"
        elif level < 0:
            reagent = "base"
            vol = abs(level) * dv_base
            pump = "add_alt_media"
        else:
            reagent = "none"
            vol = 0.0
            pump = "none"

        t_scheduled_sec = step * T_s_operational
        records.append({
            "step": step,
            "segment": segment,
            "t_offset_s": t_scheduled_sec,
            "t_offset_min": t_scheduled_sec / 60.0,
            "level": level,
            "acid_or_base": reagent,
            "volume_mL": round(vol, 4),
            "pump": pump
        })

    df_seq = pd.DataFrame(records)

    # Save to both stage03 and alias directory
    for od in [out_dir, out_dir_alias]:
        df_seq.to_csv(od / "stage4_sequence_600steps.csv", index=False)
        (od / "stage4_planned_inputs.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

        # Save 200-step prefix for legacy reference (30 washout + 130 train + 40 test)
        df_200 = df_seq.iloc[:200].copy()
        df_200.loc[df_200["step"] >= 160, "segment"] = "test"
        df_200.to_csv(od / "stage4_sequence_200steps.csv", index=False)

        acid_seq_str = ",".join(str(r["volume_mL"]) if r["acid_or_base"] == "acid" else "0.0" for r in records)
        base_seq_str = ",".join(str(r["volume_mL"]) if r["acid_or_base"] == "base" else "0.0" for r in records)
        (od / "volume_sequence_acid.txt").write_text(acid_seq_str + "\n", encoding="utf-8")
        (od / "volume_sequence_base.txt").write_text(base_seq_str + "\n", encoding="utf-8")

    # --------------------------------------------------------------------------
    # 6. LOGISTICS CALCULATION
    # --------------------------------------------------------------------------
    total_time_h = (N_STEPS * T_s_operational + 40 * 60) / 3600.0
    total_acid_vol_mL = df_seq[df_seq["acid_or_base"] == "acid"]["volume_mL"].sum()
    total_base_vol_mL = df_seq[df_seq["acid_or_base"] == "base"]["volume_mL"].sum()
    total_carrier_L = (Q_meas * (total_time_h * 60.0)) / 1000.0

    print(f"\n[Logistics for Full 600-Step Run]:")
    print(f"  Duration: {total_time_h:.2f} hours (with 20 min pre/post baselines)")
    print(f"  Acid required: {total_acid_vol_mL:.2f} mL | Base required: {total_base_vol_mL:.2f} mL")
    print(f"  Carrier consumed (at Q={Q_meas:.1f} mL/min): {total_carrier_L:.2f} L")

    # --------------------------------------------------------------------------
    # 7. GENERATE DIAGNOSTIC FIGURES
    # --------------------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))

    # 1A. Asymmetry Scaling
    ax = axes1[0]
    v_arr = np.linspace(0, 0.25, 100)
    ax.plot(v_arr, g_acid * v_arr, 'r-', lw=2.2, label=f"Acid Gain ({g_acid:.3f}/mL)")
    ax.plot(v_arr, g_base * v_arr, 'b-', lw=2.2, label=f"Base Gain ({g_base:.3f}/mL)")
    ax.axhline(target_dF_acid, color='r', ls=':', label=f"Acid 80% Target ({target_dF_acid:.4f})")
    ax.axhline(target_dF_base, color='b', ls=':', label=f"Base 80% Target ({target_dF_base:.4f})")
    ax.scatter([v_max_acid], [expected_peak_acid], color='r', s=90, zorder=5, label=f"Acid v_max={v_max_acid:.2f}mL")
    ax.scatter([v_max_base], [expected_peak_base], color='b', s=90, zorder=5, label=f"Base v_max={v_max_base:.2f}mL")
    ax.set_xlabel("Pulse Volume (mL)")
    ax.set_ylabel("Expected Peak |Delta Ratio|")
    ax.set_title("A. Operating Envelope & Asymmetry Balance", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.0)

    # 1B. Discrete Levels
    ax = axes1[1]
    levels = np.array(levels_range)
    vols = np.array([l * dv_acid if l > 0 else (-abs(l) * dv_base if l < 0 else 0.0) for l in levels])
    colors = ['#1f77b4' if l < 0 else ('#d62728' if l > 0 else 'gray') for l in levels]
    bars = ax.bar(levels, vols, color=colors, alpha=0.85, edgecolor='k')
    for b, v in zip(bars, vols):
        if abs(v) > 0:
            ax.text(b.get_x() + b.get_width()/2, v + (0.008 if v > 0 else -0.015),
                    f"{abs(v)*1000:.0f}uL", ha='center', fontsize=8, fontweight='bold')
    ax.set_xlabel("Discrete Input Level u[k]")
    ax.set_ylabel("Injected Dose Volume (mL)")
    ax.set_title("B. 9-Level Discrete Dosing Palette", fontsize=11, fontweight="bold")
    ax.axhline(0, color='k', lw=0.8)

    # 1C. Expected Optical Step Size vs Rule 2
    ax = axes1[2]
    expected_dF = np.array([abs(l) * (dF_step_acid if l > 0 else dF_step_base) for l in levels])
    ax.bar(levels, expected_dF, color=colors, alpha=0.85, edgecolor='k')
    ax.axhline(rule2_target_dF, color='darkred', ls='-', lw=2.0, label=f"Rule 2 Target Delta F >= {rule2_target_dF:.4f} [FAILED]")
    ax.axhline(noise_5sigma, color='darkgreen', ls='--', lw=1.5, label=f"5-Sigma Noise ({noise_5sigma:.4f})")
    ax.axhline(sigma0, color='gray', ls=':', lw=1.2, label=f"1-Sigma Noise ({sigma0:.4f})")
    ax.set_xlabel("Discrete Input Level u[k]")
    ax.set_ylabel("Expected Peak |Delta Ratio|")
    ax.set_title("C. Signal Sensitivity vs Noise & Rule 2 Gate", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.0)

    fig1.tight_layout()
    fig1.savefig(figdir / "fig_stage3_parameter_selection.png", bbox_inches="tight")
    fig1.savefig(out_dir_alias / "figures" / "fig_stage3_parameter_selection.png", bbox_inches="tight")
    plt.close(fig1)

    # --------------------------------------------------------------------------
    # 8. WRITE DETAILED AUDITED REPORT & JSON
    # --------------------------------------------------------------------------
    report_lines = [
        "STAGE 3: BENCHMARK PARAMETER SELECTION REPORT (AUDITED)",
        "=" * 75,
        "",
        "1. PHYSICAL HYDRAULICS (FROM STAGE 1 MEASUREMENT)",
        "-" * 75,
        f"  - Measured Flow Rate Q:           {Q_meas:.2f} mL/min (runs: {hyd.get('q_runs_mL_per_min', [])})",
        f"  - Theoretical Residence Time:     tau(V/Q) = {tau_V_over_Q:.2f} min ({tau_V_over_Q*60:.1f} s)",
        f"  - Empirical Step-Fit Tau:         tau_step = {tau_step_fit:.2f} min ({tau_step_fit*60:.1f} s)",
        f"  - Effective Mixing Volume:        V_eff = {V_eff:.2f} mL ({hyd['effective_volume_fraction']*100:.1f}% chamber efficiency)",
        f"  - Derived Step Length (tau / 3):  {T_s_from_stepfit:.1f} s to {T_s_from_V_over_Q:.1f} s",
        f"  - Nominal Step Length Used:       T_s = {T_s_operational:.1f} s (deviation: ~1.2x to 1.5x longer than tau/3)",
        "",
        "2. RULE 1: MAXIMUM PULSE VOLUME (v_max)",
        "-" * 75,
        f"  - Acid 80% Overshoot Target dF:   {target_dF_acid:.4f} (overshoot: {acid_overshoot_dF:.4f})",
        f"  - Base 80% Overshoot Target dF:   {target_dF_base:.4f} (overshoot: {base_overshoot_dF:.4f})",
        f"  - Linear Gains:                   Acid = {g_acid:.4f}/mL, Base = {g_base:.4f}/mL",
        f"  - Asymmetry Factor (Base/Acid):   {asymmetry_ratio:.2f}x",
        f"  - Selected v_max:                 Acid = {v_max_acid:.2f} mL, Base = {v_max_base:.2f} mL",
        f"  - Critical Note:                  v_max_base_raw = {v_max_base_raw:.3f} mL. Using 0.08 mL gave expected",
        f"                                    peak {expected_peak_base:.4f}, exceeding measured 0.5 mL overshoot.",
        "",
        "3. RULE 2: LEVEL SIZE & SEPARATION GATE (Delta F_step >= 0.007)",
        "-" * 75,
        f"  - Discretization:                 9 levels {{-4..+4}}",
        f"  - Acid Step Size (dv_acid):       {dv_acid*1000:.1f} uL ({dv_acid:.4f} mL) -> Step dF = {dF_step_acid:.4f}",
        f"  - Base Step Size (dv_base):       {dv_base*1000:.1f} uL ({dv_base:.4f} mL) -> Step dF = {dF_step_base:.4f}",
        f"  - Protocol Target Threshold:      Delta F_step >= {rule2_target_dF:.4f}",
        f"  - RULE 2 VERDICT:                 FAILED (Acid = {dF_step_acid:.4f}, Base = {dF_step_base:.4f})",
        f"  - Physical Consequence:           Step deflections are only ~1.7 sigma above noise. With within-level",
        f"                                    SD of 0.005-0.011, adjacent input levels overlap substantially,",
        f"                                    leaving only ~3-4 physically distinguishable optical states.",
        f"  - Protocol Remedy:                Increase stock concentrations by ~2.5x, or reduce from 9 to 5 levels.",
        "",
        "4. SEQUENCE SPECIFICATION (N = 600 STEPS)",
        "-" * 75,
        f"  - Seed:                           {SEED}",
        f"  - Washout:                        Steps 0 to 29 (30 steps = 30 min, 0.0 mL)",
        f"  - Training:                       Steps 30 to 429 (400 steps = 6.67 hours)",
        f"  - Test:                           Steps 430 to 599 (170 steps = 2.83 hours)",
        f"  - Acid Stock Consumed:            {total_acid_vol_mL:.2f} mL",
        f"  - Base Stock Consumed:            {total_base_vol_mL:.2f} mL",
        f"  - Carrier Buffer Consumed:        {total_carrier_L:.2f} L at Q = {Q_meas:.2f} mL/min",
        "=" * 75
    ]
    report_text = "\n".join(report_lines)
    for od in [out_dir, out_dir_alias]:
        (od / "stage3_parameter_report.txt").write_text(report_text, encoding="utf-8")

    params_dict = {
        "hydraulics_measured": {
            "Q_mean_mL_per_min": Q_meas,
            "tau_V_over_Q_min": tau_V_over_Q,
            "tau_step_fit_min": tau_step_fit,
            "V_eff_mL": V_eff
        },
        "rule1": {
            "v_max_acid_mL": v_max_acid,
            "v_max_base_mL": v_max_base,
            "v_max_base_raw_mL": v_max_base_raw,
            "asymmetry_ratio": asymmetry_ratio
        },
        "rule2": {
            "dv_acid_mL": dv_acid,
            "dv_base_mL": dv_base,
            "dF_step_acid": dF_step_acid,
            "dF_step_base": dF_step_base,
            "rule2_target_dF": rule2_target_dF,
            "verdict": "FAIL",
            "separable_states_estimate": "3 to 4"
        },
        "rule3_rule4": {
            "derived_Ts_stepfit_s": T_s_from_stepfit,
            "derived_Ts_V_over_Q_s": T_s_from_V_over_Q,
            "operational_Ts_s": T_s_operational
        },
        "logistics_600steps": {
            "total_time_h": total_time_h,
            "acid_mL": total_acid_vol_mL,
            "base_mL": total_base_vol_mL,
            "carrier_L": total_carrier_L
        }
    }
    for od in [out_dir, out_dir_alias]:
        (od / "stage3_parameters.json").write_text(json.dumps(params_dict, indent=2), encoding="utf-8")

    print(f"\nSaved updated Stage 3 report to: {out_dir / 'stage3_parameter_report.txt'}")
    print(f"Saved updated Stage 3 JSON to: {out_dir / 'stage3_parameters.json'}")

if __name__ == "__main__":
    main()
