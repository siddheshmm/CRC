#!/usr/bin/env python3
"""
run_stage4_benchmark.py — Automated Bipolar Benchmark Runner for Stage 4
Executes the acid/base pulse schedule on Pioreactor with background resilience.

Terminal / Disconnect Resilience:
  - Ignores SIGHUP (signal.SIG_IGN): Keeps running even if SSH terminal disconnects,
    times out, or laptop goes to sleep.
  - Line-buffered stdout: Logs write in real time to stage4.log when run with nohup or in background.
  - Saves running PID to stage4_running.pid so you can check or stop it anytime.

Usage on Pioreactor:
    # 1. Ultra-fast 15-second hardware test (1 acid + 1 base, 15s apart):
    python run_stage4_benchmark.py --seq stage4_test_sequence.csv --skip-baseline --step-s 15

    # 2. Main 10-Hour Benchmark Run (runs in background, immune to terminal closing):
    nohup python run_stage4_benchmark.py --seq stage4_sequence_600steps.csv > stage4.log 2>&1 &
    # To monitor: tail -f stage4.log
"""

import os
import sys
import time
import signal
import argparse
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

# Force unbuffered output so logs write immediately when redirected
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Ignore SIGHUP on Linux/Pioreactor: prevents script from dying when SSH closes/sleeps
if hasattr(signal, "SIGHUP"):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

try:
    from pioreactor.actions.pump import add_media, add_alt_media
    from pioreactor.whoami import get_unit_name, get_latest_experiment_name
    PIO_AVAILABLE = True
except ImportError:
    PIO_AVAILABLE = False
    get_unit_name = lambda: "pioreactor_sim"
    get_latest_experiment_name = lambda: "demo_experiment"

def main():
    parser = argparse.ArgumentParser(description="Stage 4 Bipolar Benchmark Runner")
    parser.add_argument("--seq", default="stage4_sequence_600steps.csv", help="Path to sequence CSV")
    parser.add_argument("--unit", default=None, help="Pioreactor unit name (auto-detected if omitted)")
    parser.add_argument("--experiment", default=None, help="Experiment name (auto-detected if omitted)")
    parser.add_argument("--step-s", type=float, default=None, help="Step duration T_s in seconds (auto-detected from CSV if omitted)")
    parser.add_argument("--pre-baseline-min", type=float, default=20.0, help="Minutes to record pre-run baseline (default 20)")
    parser.add_argument("--post-baseline-min", type=float, default=20.0, help="Minutes to record post-run baseline (default 20)")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip both pre and post baseline waiting periods (for quick tests)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate execution without firing hardware pumps")
    args = parser.parse_args()

    seq_path = Path(args.seq)
    if not seq_path.exists():
        sys.exit(f"Error: sequence file '{seq_path}' not found.")

    df_seq = pd.read_csv(seq_path)
    total_steps = len(df_seq)

    # Determine step duration T_s
    if args.step_s is not None:
        step_s = args.step_s
    elif "t_offset_s" in df_seq.columns and total_steps > 1:
        step_s = float(df_seq["t_offset_s"].iloc[1] - df_seq["t_offset_s"].iloc[0])
    else:
        step_s = 60.0

    unit = args.unit or get_unit_name()
    experiment = args.experiment or get_latest_experiment_name()

    pre_min = 0.0 if args.skip_baseline else args.pre_baseline_min
    post_min = 0.0 if args.skip_baseline else args.post_baseline_min

    # Save PID file for easy process tracking
    pid_file = Path("stage4_running.pid")
    pid_file.write_text(str(os.getpid()), encoding="utf-8")

    print("="*75, flush=True)
    print("PIOREACTOR STAGE 4 BENCHMARK RUNNER (DISCONNECT-PROOF)", flush=True)
    print("="*75, flush=True)
    print(f"Process PID: {os.getpid()} (saved to stage4_running.pid)", flush=True)
    print(f"Unit:        {unit}", flush=True)
    print(f"Experiment:  {experiment}", flush=True)
    print(f"Sequence:    {seq_path.name} ({total_steps} steps, T_s = {step_s:.1f}s)", flush=True)
    print(f"Baselines:   Pre: {pre_min:.1f} min | Post: {post_min:.1f} min", flush=True)
    print(f"Hardware:    {'Connected (Pioreactor library active)' if (PIO_AVAILABLE and not args.dry_run) else 'SIMULATION / DRY-RUN'}", flush=True)
    print(f"Terminal:    SIGHUP ignored -> Will continue running even if SSH drops or laptop sleeps.", flush=True)
    print("="*75, flush=True)

    # 1. Baseline period (Pre-run)
    if pre_min > 0:
        print(f"\n[Phase 1] Pre-Run Baseline: {pre_min:.1f} minutes with continuous carrier flow...", flush=True)
        wait_s = pre_min * 60.0
        t_base_start = time.time()
        while time.time() - t_base_start < wait_s:
            rem = wait_s - (time.time() - t_base_start)
            print(f"  Recording baseline: {rem/60.0:.1f} min remaining...", end="\r", flush=True)
            time.sleep(10)
        print("\n  Pre-run baseline period complete.", flush=True)
    else:
        print("\n[Phase 1] Pre-Run Baseline: Skipped.", flush=True)

    # 2. Main Sequence Execution
    print(f"\n[Phase 2] Starting Sequence Execution ({total_steps} steps)...", flush=True)
    t0 = time.time()
    t0_dt = datetime.now(timezone.utc)
    print(f"  Start time t0 (UTC):   {t0_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}", flush=True)
    print(f"  Start time t0 (Local): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    log_rows = []

    for k, row in df_seq.iterrows():
        step = int(row["step"])
        lvl = int(row["level"])
        vol = float(row["volume_mL"])
        reagent = str(row["acid_or_base"]).lower()
        pump = str(row["pump"]).lower()
        seg = str(row.get("segment", "run"))

        # Calculate exact scheduled time based on CSV or step_s
        if "t_offset_s" in row:
            target_time = t0 + float(row["t_offset_s"])
        else:
            target_time = t0 + step * step_s

        now = time.time()
        sleep_dur = target_time - now
        if sleep_dur > 0:
            time.sleep(sleep_dur)

        actual_t = datetime.now(timezone.utc)
        print(f"[{actual_t.strftime('%H:%M:%S')}] Step {step:03d}/{total_steps-1:03d} | Seg: {seg:7s} | Level {lvl:+2d} | {reagent:4s} {vol:.3f} mL", end="", flush=True)

        dosed = False
        error_msg = ""
        if vol > 0:
            if not args.dry_run and PIO_AVAILABLE:
                try:
                    if pump == "add_media" or reagent == "acid":
                        add_media(ml=vol, unit=unit, experiment=experiment)
                        dosed = True
                    elif pump == "add_alt_media" or reagent == "base":
                        add_alt_media(ml=vol, unit=unit, experiment=experiment)
                        dosed = True
                except Exception as e:
                    error_msg = str(e)
                    print(f" -> ERROR: {e}", end="", flush=True)
            else:
                dosed = True
                print(" (simulated)", end="", flush=True)
        print(flush=True)

        log_rows.append({
            "step": step,
            "segment": seg,
            "scheduled_time_s": target_time - t0,
            "actual_time_utc": actual_t.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            "level": lvl,
            "reagent": reagent,
            "volume_mL": vol,
            "dosed": dosed,
            "error": error_msg
        })

    # Save execution log
    log_file = Path(f"stage4_execution_log_{t0_dt.strftime('%Y%m%d_%H%M%S')}.csv")
    pd.DataFrame(log_rows).to_csv(log_file, index=False)
    print(f"\nExecution log written to: {log_file}", flush=True)

    # 3. Post-run baseline
    if post_min > 0:
        print(f"\n[Phase 3] Post-Sequence Washout: {post_min:.1f} minutes...", flush=True)
        t_end_start = time.time()
        wait_end = post_min * 60.0
        while time.time() - t_end_start < wait_end:
            rem = wait_end - (time.time() - t_end_start)
            print(f"  Recording washout: {rem/60.0:.1f} min remaining...", end="\r", flush=True)
            time.sleep(10)
        print("\n  Post-run washout complete.", flush=True)
    else:
        print("\n[Phase 3] Post-Sequence Washout: Skipped.", flush=True)

    if pid_file.exists():
        try:
            pid_file.unlink()
        except Exception:
            pass

    print("\n" + "="*75, flush=True)
    print("STAGE 4 BENCHMARK RUN COMPLETE!", flush=True)
    print("Export the AS7341 spectrometer CSV from the Pioreactor UI.", flush=True)
    print("="*75, flush=True)

if __name__ == "__main__":
    main()
