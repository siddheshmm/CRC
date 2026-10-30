#!/usr/bin/env python3
"""
generate_stage4_450steps.py — Official 450-Timestep Benchmark Sequence Generator
for Chemical Reservoir Computing (Methyl Orange, Pioreactor CSTR).

Parameters:
- Total Steps: 450 (Ts = 60.0 s = 1.0 min, total dosing duration: 7.50 hours)
- Segment Split:
    1. Washout: Steps 0 to 29 (30 min, 0 mL)
    2. Training: Steps 30 to 329 (300 min = 5.0 hours)
    3. Held-out Test: Steps 330 to 449 (120 min = 2.0 hours)
- Seed: 42 (Prefix strictly identical to 200-step and 600-step sequences)
- Dosing Palette:
    Acid (+1 to +4): 0.05, 0.10, 0.15, 0.20 mL (Media pump)
    Base (-1 to -4): 0.02, 0.04, 0.06, 0.08 mL (Alt-media pump)
    Zero (0): 0.0 mL
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    root = Path("CRC")
    out_stage04 = root / "results" / "stage04"
    fig_dir = out_stage04 / "figures"
    out_stage04.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("STAGE 4 BENCHMARK SEQUENCE GENERATOR (450 TIMESTEPS, Ts = 60s)")
    print("=" * 75)

    # 1. Load the official 600-step sequence (Seed 42)
    s600_path = root / "stage4_sequence_600steps.csv"
    if not s600_path.exists():
        s600_path = root / "results" / "stage03" / "stage4_sequence_600steps.csv"
    
    df600 = pd.read_csv(s600_path)
    print(f"Loaded source sequence with {len(df600)} steps.")

    # Slice first 450 steps
    df450 = df600.iloc[:450].copy().reset_index(drop=True)

    # Update segment column for the 30 / 300 / 120 split
    # Steps 0..29: washout (30)
    # Steps 30..329: train (300)
    # Steps 330..449: test (120)
    segments = []
    for s in range(450):
        if s < 30:
            segments.append("washout")
        elif s < 330:
            segments.append("train")
        else:
            segments.append("test")
    df450["segment"] = segments

    # 2. Save CSV files
    # Primary in root of CRC
    csv_root = root / "stage4_sequence_450steps.csv"
    df450.to_csv(csv_root, index=False)
    print(f"Saved primary CSV to: {csv_root}")

    # Copy in stage04 results
    csv_stage04 = out_stage04 / "stage4_sequence_450steps.csv"
    df450.to_csv(csv_stage04, index=False)
    print(f"Saved stage04 CSV to: {csv_stage04}")

    # 3. Save Bipolar comma-separated sequence string
    bipolar_tokens = []
    for _, r in df450.iterrows():
        if r["acid_or_base"] == "acid":
            bipolar_tokens.append(f"{r['volume_mL']:.4f}")
        elif r["acid_or_base"] == "base":
            bipolar_tokens.append(f"-{r['volume_mL']:.4f}")
        else:
            bipolar_tokens.append("0.0")

    txt_path = out_stage04 / "volume_sequence_bipolar_450.txt"
    txt_path.write_text(",".join(bipolar_tokens) + "\n", encoding="utf-8")
    print(f"Saved bipolar string to: {txt_path}")

    # 4. Save JSON schedule
    json_path = out_stage04 / "stage4_sequence_450steps.json"
    json_path.write_text(json.dumps(df450.to_dict(orient="records"), indent=2), encoding="utf-8")
    print(f"Saved JSON schedule to: {json_path}")

    # 5. Inventory & Logistics Calculations
    acid_df = df450[df450["acid_or_base"] == "acid"]
    base_df = df450[df450["acid_or_base"] == "base"]
    idle_df = df450[df450["acid_or_base"] == "none"]

    total_acid_mL = acid_df["volume_mL"].sum()
    total_base_mL = base_df["volume_mL"].sum()

    carrier_rate = 3.33  # mL/min
    total_run_min = 450 + 20 + 20  # dosing + 20m pre + 20m post
    total_run_h = total_run_min / 60.0
    carrier_consumed_L = (total_run_min * carrier_rate) / 1000.0

    summary_text = (
        "===========================================================================\n"
        "STAGE 4 BENCHMARK RUN SPECIFICATIONS (450 TIMESTEPS)\n"
        "===========================================================================\n"
        "Design Seed: 42 (Exact prefix match to Run 1 and Run 2)\n"
        "Total Steps: 450 timesteps (Ts = 60.0 s = 1.0 min)\n"
        f"Dosing Duration: {450 / 60.0:.2f} hours (450 minutes)\n"
        f"Total Experiment Duration: {total_run_h:.2f} hours (including 20m pre + 20m post baselines)\n\n"
        "CHRONOLOGICAL SEGMENT SPLIT (Contiguous, Never Shuffled):\n"
        "  1. Washout Segment: Steps 0 to 29 (30 minutes, 0.0 mL)\n"
        "     Allows continuous carrier flow to settle into dynamic steady state.\n"
        "  2. Training Segment: Steps 30 to 329 (300 minutes = 5.0 hours, 9-level palette)\n"
        "     Overdetermined training set (300 samples for 7-19 features).\n"
        "  3. Held-Out Test Segment: Steps 330 to 449 (120 minutes = 2.0 hours, 9-level palette)\n"
        "     High-power evaluation set (120 samples, chance noise floor ~0.058).\n\n"
        "CHEMICAL INVENTORY REQUIRED:\n"
        f"  - Acid Stock (Vinegar): Total delivered = {total_acid_mL:.4f} mL ({len(acid_df)} pulses)\n"
        "    Ensure >= 35 mL in the acid supply bottle.\n"
        f"  - Base Stock (Ammonia): Total delivered = {total_base_mL:.4f} mL ({len(base_df)} pulses)\n"
        "    Ensure >= 20 mL in the base supply bottle.\n"
        f"  - Idle / Zero Steps: {len(idle_df)} steps\n"
        f"  - Continuous Carrier Buffer Flow: Consumes ~{carrier_consumed_L:.2f} L at 3.33 mL/min\n"
        "    Ensure >= 2.0 L bottle reserve (safely fits in a standard 2.0 L bottle!).\n"
        "  - Waste Carboy: Ensure >= 2.0 L empty capacity before starting.\n\n"
        "EXECUTION COMMAND ON PIOREACTOR:\n"
        "  nohup pio run specified_pump_dosing \\\n"
        "    --pump bipolar \\\n"
        "    --csv-file \"stage4_sequence_450steps.csv\" \\\n"
        "    --duration 1.0 \\\n"
        "    --max-cycles 450 > stage4_450steps.log 2>&1 &\n"
        "===========================================================================\n"
    )

    summary_file = out_stage04 / "stage4_sequence_450steps_summary.txt"
    summary_file.write_text(summary_text, encoding="utf-8")
    print(f"\n{summary_text}")

    # 6. Render Diagnostic Preview Figure
    fig, (ax_stem, ax_dist) = plt.subplots(2, 1, figsize=(15, 7), gridspec_kw={"height_ratios": [1.8, 1]})

    t_hrs = df450["t_offset_min"] / 60.0
    vols_signed = np.array(bipolar_tokens, dtype=float)

    # Highlight segments
    t_w_end = 30 / 60.0
    t_tr_end = 330 / 60.0
    t_end = 450 / 60.0

    ax_stem.axvspan(0, t_w_end, color="#f0f0f0", alpha=0.8, label="Washout (Steps 0-29, 0 mL)")
    ax_stem.axvspan(t_w_end, t_tr_end, color="#e6f2ff", alpha=0.5, label="Training Split (Steps 30-329, 300 min)")
    ax_stem.axvspan(t_tr_end, t_end, color="#fff2e6", alpha=0.5, label="Held-Out Test (Steps 330-449, 120 min)")

    acid_m = vols_signed > 0
    base_m = vols_signed < 0

    ax_stem.stem(t_hrs[acid_m], vols_signed[acid_m], linefmt="red", markerfmt="ro", basefmt=" ", label="Acid Pulses (+ Dose)")
    ax_stem.stem(t_hrs[base_m], vols_signed[base_m], linefmt="blue", markerfmt="bo", basefmt=" ", label="Base Pulses (- Dose)")
    ax_stem.axhline(0, color="black", linestyle=":", linewidth=0.8)

    ax_stem.set_ylabel("Injected Dose (mL)\n[+ Acid / - Base]", fontweight="bold")
    ax_stem.set_title("Stage 4 Benchmark Sequence Preview: 450 Timesteps (7.50 Hours of Dosing)", fontsize=12, fontweight="bold")
    ax_stem.legend(loc="upper right", fontsize=8.5, ncol=4)
    ax_stem.grid(True, alpha=0.3)
    ax_stem.set_ylim(-0.12, 0.24)

    # Input level distribution across train and test
    tr_levels = df450.loc[30:329, "level"].value_counts().sort_index()
    te_levels = df450.loc[330:449, "level"].value_counts().sort_index()
    all_levels = np.arange(-4, 5)

    w = 0.35
    ax_dist.bar(all_levels - w/2, [tr_levels.get(l, 0) for l in all_levels], width=w, color="#3498db", edgecolor="black", label="Training Segment (N=300)")
    ax_dist.bar(all_levels + w/2, [te_levels.get(l, 0) for l in all_levels], width=w, color="#e67e22", edgecolor="black", label="Held-Out Test (N=120)")

    ax_dist.set_xticks(all_levels)
    ax_dist.set_xlabel("Discrete Input Level u[k] (Levels -4..+4)", fontweight="bold")
    ax_dist.set_ylabel("Pulse Count", fontweight="bold")
    ax_dist.set_title("Discrete Level Balance Across Training & Test Splits", fontsize=11, fontweight="bold")
    ax_dist.legend(loc="upper right", fontsize=8.5)
    ax_dist.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig_path = fig_dir / "fig_stage4_sequence_450steps_preview.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved diagnostic preview figure: {fig_path}")

if __name__ == "__main__":
    main()
