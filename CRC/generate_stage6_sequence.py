#!/usr/bin/env python3
"""
generate_stage6_sequence.py — Stage 6 Hold-Out Benchmark Sequence Generator
for Chemical Reservoir Computing (Methyl Orange, Pioreactor CSTR).

Follows Section 9 of the Benchmarking Protocol:
- Hold-out run: 200 steps (30 washout + 170 test)
- Seed: 43 (independent random seed)
- Operating parameters identical to Stage 4:
    dv_acid = 0.05 mL, dv_base = 0.02 mL, T_s = 60.0 s
    Levels {-4..+4} uniform discrete alphabet
"""

import os
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    root = Path("CRC")
    out_dir = root / "results" / "stage06"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("STAGE 6 HOLD-OUT SEQUENCE GENERATOR (SEED 43, 200 TIMESTEPS)")
    print("=" * 75)

    # --------------------------------------------------------------------------
    # Parameters Matching Stage 4
    # --------------------------------------------------------------------------
    SEED = 43
    N_STEPS = 200
    N_WASHOUT = 30
    N_TEST = 170
    T_s_sec = 60.0
    T_s_min = 1.0

    dv_acid = 0.05   # 50 uL per level
    dv_base = 0.02   # 20 uL per level
    levels_range = list(range(-4, 5))  # -4, -3, -2, -1, 0, 1, 2, 3, 4

    # Generate sequence
    rng = random.Random(SEED)
    records = []

    for step in range(N_STEPS):
        if step < N_WASHOUT:
            segment = "washout"
            level = 0
        else:
            segment = "holdout_test"
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

        t_scheduled_sec = step * T_s_sec
        t_scheduled_min = t_scheduled_sec / 60.0

        records.append({
            "step": step,
            "segment": segment,
            "t_offset_s": t_scheduled_sec,
            "t_offset_min": t_scheduled_min,
            "level": level,
            "acid_or_base": reagent,
            "volume_mL": round(vol, 4),
            "pump": pump
        })

    df_seq = pd.DataFrame(records)

    # Save CSV
    seq_csv = out_dir / "stage6_sequence_200steps.csv"
    df_seq.to_csv(seq_csv, index=False)
    print(f"Saved Stage 6 CSV sequence to: {seq_csv}")

    # Save JSON
    seq_json = out_dir / "stage6_planned_inputs.json"
    seq_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved Stage 6 JSON schedule to: {seq_json}")

    # Save bipolar signed volume tokens string
    # Bipolar mode in specified_pump_dosing: positive = acid (media), negative = base (alt_media)
    bipolar_vols = []
    for r in records:
        if r["acid_or_base"] == "acid":
            bipolar_vols.append(f"{r['volume_mL']:.4f}")
        elif r["acid_or_base"] == "base":
            bipolar_vols.append(f"-{r['volume_mL']:.4f}")
        else:
            bipolar_vols.append("0.0")

    seq_txt = out_dir / "volume_sequence_bipolar_200.txt"
    seq_txt.write_text(",".join(bipolar_vols) + "\n", encoding="utf-8")
    print(f"Saved comma-separated sequence string to: {seq_txt}")

    # Logistics calculations
    total_acid_vol_mL = df_seq[df_seq["acid_or_base"] == "acid"]["volume_mL"].sum()
    total_base_vol_mL = df_seq[df_seq["acid_or_base"] == "base"]["volume_mL"].sum()
    carrier_flow_rate = 3.33  # mL/min
    total_time_min = N_STEPS * T_s_min + 20.0 + 20.0  # sequence + 20m pre + 20m post
    total_time_h = total_time_min / 60.0
    carrier_consumed_L = (carrier_flow_rate * total_time_min) / 1000.0

    n_acid_pulses = len(df_seq[df_seq["acid_or_base"] == "acid"])
    n_base_pulses = len(df_seq[df_seq["acid_or_base"] == "base"])
    n_zero_steps = len(df_seq[df_seq["acid_or_base"] == "none"])

    summary_text = (
        "===========================================================================\n"
        "STAGE 6 HOLD-OUT SEQUENCE SPECIFICATIONS\n"
        "===========================================================================\n"
        f"Design Seed: {SEED} (Independent hold-out test seed)\n"
        f"Total Steps: {N_STEPS} (30 Washout, 170 Hold-out Test)\n"
        f"Step Duration (Ts): {T_s_sec:.1f} s ({T_s_min:.1f} min)\n"
        f"Total Sequence Duration: {N_STEPS * T_s_min / 60.0:.2f} hours (3 h 20 min)\n"
        f"Full Run Duration (with 20 min pre/post baselines): {total_time_h:.2f} hours\n\n"
        "PALETTE & CHEMICAL INVENTORY:\n"
        f"  - Acid Dosing Unit: Media pump (50 uL/level, v_max = 0.20 mL)\n"
        f"  - Base Dosing Unit: Alt-media pump (20 uL/level, v_max = 0.08 mL)\n"
        f"  - Acid Pulses: {n_acid_pulses} pulses, Total Volume = {total_acid_vol_mL:.4f} mL\n"
        f"  - Base Pulses: {n_base_pulses} pulses, Total Volume = {total_base_vol_mL:.4f} mL\n"
        f"  - Zero / Idle Steps: {n_zero_steps} steps\n"
        f"  - Carrier Flow: ~3.33 mL/min continuous inflow\n"
        f"  - Carrier Buffer Consumed: {carrier_consumed_L:.2f} Liters (ensure >= 1.5 L bottle reserve)\n"
        f"  - Waste Efflux Pump: Continuous / active waste removal mandatory\n\n"
        "LEVEL DISTRIBUTION (Test Segment Steps 30..199):\n"
        f"{df_seq.loc[30:, 'level'].value_counts().sort_index().to_string()}\n"
    )

    summary_path = out_dir / "stage6_sequence_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"\n{summary_text}")

    # --------------------------------------------------------------------------
    # Generate Diagnostic Preview Figure
    # --------------------------------------------------------------------------
    fig, (ax_stem, ax_hist) = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [2, 1]})

    # Stem plot of sequence
    steps = df_seq["step"].values
    vols_signed = [r["volume_mL"] if r["acid_or_base"] == "acid" else (-r["volume_mL"] if r["acid_or_base"] == "base" else 0.0) for r in records]

    ax_stem.axvspan(0, 30, color="#f0f0f0", alpha=0.8, label="Washout (Steps 0-29, 0 mL)")
    ax_stem.axvspan(30, 200, color="#fff2e6", alpha=0.5, label="Hold-Out Test (Steps 30-199, Seed 43)")

    acid_steps = df_seq[df_seq["acid_or_base"] == "acid"]["step"]
    acid_vols = df_seq[df_seq["acid_or_base"] == "acid"]["volume_mL"]
    base_steps = df_seq[df_seq["acid_or_base"] == "base"]["step"]
    base_vols = df_seq[df_seq["acid_or_base"] == "base"]["volume_mL"]

    ax_stem.stem(acid_steps, acid_vols, linefmt="red", markerfmt="ro", basefmt=" ", label="Acid Pulses (add_media, +Level)")
    ax_stem.stem(base_steps, -base_vols, linefmt="blue", markerfmt="bo", basefmt=" ", label="Base Pulses (add_alt_media, -Level)")
    ax_stem.axhline(0, color="black", linestyle=":", linewidth=0.8)

    ax_stem.set_ylabel("Injected Volume (mL)\n[+ Acid / - Base]", fontsize=10, fontweight="bold")
    ax_stem.set_title("Stage 6 Hold-Out Benchmark Sequence Preview (Seed 43, 200 Timesteps)", fontsize=12, fontweight="bold")
    ax_stem.legend(loc="upper left", fontsize=9, ncol=4)
    ax_stem.grid(True, alpha=0.3)

    # Level distribution histogram
    test_levels = df_seq.loc[30:, "level"]
    counts = test_levels.value_counts().sort_index()
    colors = ["#1f77b4" if l < 0 else ("#d62728" if l > 0 else "gray") for l in counts.index]

    ax_hist.bar(counts.index, counts.values, color=colors, edgecolor="black", alpha=0.85)
    for l, c in zip(counts.index, counts.values):
        ax_hist.text(l, c + 0.5, str(c), ha="center", fontweight="bold", fontsize=9)
    ax_hist.set_xlabel("Discrete Input Level u[k] (Levels -4..+4)", fontsize=10, fontweight="bold")
    ax_hist.set_ylabel("Pulse Count", fontsize=10, fontweight="bold")
    ax_hist.set_title("Hold-Out Test Input Level Distribution (Uniform Random, N=170)", fontsize=11, fontweight="bold")
    ax_hist.set_xticks(range(-4, 5))
    ax_hist.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig_path = fig_dir / "fig_stage6_sequence_preview.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved diagnostic preview figure: {fig_path}")

if __name__ == "__main__":
    main()
