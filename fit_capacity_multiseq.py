#!/usr/bin/env python3
"""
fit_capacity_multiseq.py

Protocol 6D: capacity estimation across INDEPENDENT random-input sequences (not repeats of
one sequence), with an explicit freeze step and a stability-vs-amount-of-data check.

Layout expected (each subfolder is one independently-generated sequence, run once):
  <data-root>/<seq_label>/dosing_events/*.csv
  <data-root>/<seq_label>/as7341_spectrum_readings/*.csv

Procedure:
  1. Load each sequence; build causal multiband states and design-alphabet symbols exactly as
     in the Protocol 6 triplicate analysis (last complete scan strictly before each dose).
  2. Split each sequence's own timeline into conditioning/train/val/test using the protocol
     6B fractions (40/336/112/112 of 600, scaled to each sequence's length).
  3. FREEZE: fit ridge weights + scaler on the train segments of the designated freeze
     sequences (default: all but the last one loaded), with alpha chosen on their own val
     segments. Save the frozen parameters to a JSON file with source sequence IDs and a
     timestamp, before any test segment is evaluated.
  4. Evaluate the frozen model, unchanged, on every sequence's own test segment (this
     includes freeze sequences' test segments too, since those were never used for fitting
     or tuning) plus a rolled-input null using the same frozen model.
  5. Stability curve: recompute the pooled test-segment memory curve using the test segments
     of the first 1..K sequences (in the order given on the command line / directory sort
     order), so you can see whether the estimate has stopped moving as more sequences are
     added.

This intentionally reuses the causal extraction and ridge/null machinery from
run_protocol6_triplicate_analysis_v2.py, simplified to the frozen (no re-tuning per
evaluation) procedure this protocol calls for.

Usage:
  python fit_capacity_multiseq.py --data-root capacity_seqs --outdir out_6d
  python fit_capacity_multiseq.py --data-root capacity_seqs --outdir out_6d \
      --freeze-sequences seq01,seq02,seq03
"""

import argparse
import glob
import json
import os
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BANDS = [415, 445, 480, 515, 555, 590, 630, 680]
ALPHAS = [1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
FRAC_COND, FRAC_TRAIN, FRAC_VAL = 40 / 600, 336 / 600, 112 / 600
SEQ_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf", "#bcbd22"]
FLAGS = []


def flag(msg):
    print(f"  [FLAG] {msg}")
    FLAGS.append(msg)


# ---- ingestion (same causal-extraction logic as the v2 triplicate script) ----------

def _to_dt(series, utc):
    for kw in ({"format": "ISO8601"}, {}):
        try:
            t = pd.to_datetime(series, utc=utc, errors="coerce", **kw)
        except (ValueError, TypeError):
            continue
        if t.notna().mean() > 0.99:
            if utc:
                t = t.dt.tz_convert(None)
            elif getattr(t.dt, "tz", None) is not None:
                t = t.dt.tz_localize(None)
            return t
    raise ValueError("could not parse timestamps")


def read_first_csv(d, sub):
    files = sorted(glob.glob(os.path.join(d, sub, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV in {os.path.join(d, sub)}")
    return pd.read_csv(files[0])


def extract_sequence(seq_dir, seq_id, args, levels):
    tag = seq_id
    dose = read_first_csv(seq_dir, "dosing_events")
    spec = read_first_csv(seq_dir, "as7341_spectrum_readings")
    if "timestamp" in dose.columns and "timestamp" in spec.columns:
        clock, utc = "timestamp", True
    else:
        flag(f"{tag}: UTC 'timestamp' missing; using timestamp_localtime for BOTH streams")
        clock, utc = "timestamp_localtime", False
    dose["ts"] = _to_dt(dose[clock], utc)
    spec["ts"] = _to_dt(spec[clock], utc)
    dose = dose.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    alt = dose[dose["source_of_event"] == args.alt_source].copy()
    if alt.empty:
        raise ValueError(f"{tag}: no rows with source_of_event == '{args.alt_source}'")
    gap = alt["ts"].diff().dt.total_seconds().fillna(1e9)
    alt["pulse_id"] = (gap > args.gap_sec).cumsum()
    pulses = (alt.groupby("pulse_id")
              .agg(start_time=("ts", "first"), end_time=("ts", "last"),
                   volume_raw=("volume_change_ml", "sum")).reset_index(drop=True))
    dist = np.abs(pulses["volume_raw"].values[:, None] - levels[None, :])
    pulses["symbol"] = levels[dist.argmin(axis=1)]
    interval = pulses["start_time"].diff().dt.total_seconds()
    med_int = float(np.nanmedian(interval))
    inc = np.ones(len(pulses), dtype=int)
    inc[1:] = np.maximum(1, np.rint(interval.values[1:] / med_int).astype(int))
    pulses["slot"] = np.cumsum(inc) - 1
    n_slots = int(pulses["slot"].iloc[-1]) + 1
    if n_slots - len(pulses):
        flag(f"{tag}: {n_slots - len(pulses)} slot(s) missing on the grid; kept as NaN")

    spec["band"] = pd.to_numeric(spec["band"], errors="coerce")
    spec["reading"] = pd.to_numeric(spec["reading"], errors="coerce")
    spec = spec[spec["band"].isin(BANDS)].dropna(subset=["ts", "reading"]).sort_values("ts")
    gsec = spec["ts"].diff().dt.total_seconds().fillna(1e9)
    spec["scan_id"] = (gsec > args.scan_gap_sec).cumsum()
    wide = spec.pivot_table(index="scan_id", columns="band", values="reading", aggfunc="mean")
    wide.columns = [int(c) for c in wide.columns]
    wide = wide.reindex(columns=BANDS)
    stime = spec.groupby("scan_id")["ts"].max()
    wide.index = pd.DatetimeIndex(stime.reindex(wide.index).values, name="scan_time")
    wide = wide.dropna(how="any").groupby(level=0).mean().sort_index()

    t0 = pulses["start_time"].iloc[0]
    base = wide[(wide.index >= t0 - pd.Timedelta(minutes=args.baseline_window_min) - pd.Timedelta(seconds=args.baseline_guard_sec)) &
               (wide.index < t0 - pd.Timedelta(seconds=args.baseline_guard_sec))]
    if len(base) < 10:
        raise ValueError(f"{tag}: only {len(base)} baseline scans before first dose")
    bmean = base.mean()

    K = max(1, args.avg_scans)
    rows = []
    for _, p in pulses.iterrows():
        pos = wide.index.searchsorted(p["start_time"], side="left")
        if pos == 0:
            continue
        sel = wide.iloc[max(0, pos - K):pos]
        Im = sel.mean()
        A = -np.log10((Im / bmean).clip(lower=1e-6))
        rec = {"slot": int(p["slot"])}
        for b in BANDS:
            rec[f"A_{b}"] = A[b]
        rows.append(rec)
    df = pd.DataFrame(rows).set_index("slot").reindex(range(n_slots))
    sym = pulses.set_index("slot")["symbol"].reindex(range(n_slots))
    df["symbol"] = sym.values
    print(f"  {tag}: {len(pulses)} pulses -> {n_slots} slots, interval {med_int:.1f}s")
    return df, n_slots


def load_sequences(root, args, levels, order):
    seqs = {}
    dirs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
    if not dirs:
        raise FileNotFoundError(f"No sequence folders in {root}")
    labels = [os.path.basename(d) for d in dirs]
    if order:
        want = [x.strip() for x in order.split(",")]
        missing = [w for w in want if w not in labels]
        if missing:
            raise ValueError(f"--sequence-order names not found: {missing}")
        labels = want
        dirs = [os.path.join(root, w) for w in want]
    for d, lab in zip(dirs, labels):
        df, n = extract_sequence(d, lab, args, levels)
        seqs[lab] = {"states": df, "n_slots": n}
    return seqs, labels


# ---- ridge machinery (frozen: fit once, apply everywhere, no per-eval re-tuning) ---

def split_bounds(n):
    cond = int(round(n * FRAC_COND))
    tr_e = cond + int(round(n * FRAC_TRAIN))
    va_e = tr_e + int(round(n * FRAC_VAL))
    return cond, tr_e, va_e


def lagged(u, k):
    y = np.full(len(u), np.nan)
    if k < len(u):
        y[k:] = u[:len(u) - k]
    return y


def task_target(u, kind, lags, uc0):
    if kind == "lag":
        return lagged(u, lags[0])
    return lagged(u - uc0, lags[0]) * lagged(u - uc0, lags[1])


def build_tasks(max_lag):
    tasks = [(f"lag{k}", "lag", (k,)) for k in range(1, max_lag + 1)]
    tasks += [(f"prod_{i}_{j}", "prod", (i, j)) for i, j in [(1, 2), (1, 3), (2, 3), (1, 4)] if max(i, j) <= max_lag]
    return tasks


def ridge_fit(X, y, alpha):
    mx, sx = X.mean(0), X.std(0)
    sx = np.where(sx == 0, 1.0, sx)
    Z = (X - mx) / sx
    my = y.mean()
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - my))
    return {"mx": mx.tolist(), "sx": sx.tolist(), "my": float(my), "w": w.tolist(), "alpha": alpha}


def ridge_predict(m, X):
    mx, sx, my, w = np.array(m["mx"]), np.array(m["sx"]), m["my"], np.array(m["w"])
    return ((X - mx) / sx) @ w + my


def metrics(y, p, y_train_mean):
    var = np.var(y)
    nmse = np.mean((y - p) ** 2) / var
    r = np.corrcoef(y, p)[0, 1] if np.var(p) > 0 else 0.0
    return {"NMSE": nmse, "r": r, "r2": r * r if r > 0 else 0.0,
            "skill": 1.0 - nmse, "cap": max(0.0, 1.0 - nmse), "n_test": len(y)}


def seg(feats, target, lo, hi):
    X, y = feats[lo:hi], target[lo:hi]
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[ok], y[ok]


def freeze_model(seqs, freeze_ids, tasks, uc0):
    """Fit + alpha-tune on the freeze sequences' train+val only. Returns {task: frozen_model}."""
    frozen = {}
    for name, kind, lags in tasks:
        tr_X, tr_y, va_X, va_y = [], [], [], []
        for sid in freeze_ids:
            st = seqs[sid]["states"]
            n = seqs[sid]["n_slots"]
            cond, tr_e, va_e = split_bounds(n)
            u = st["symbol"].values
            feats = st[[f"A_{b}" for b in BANDS]].values
            tg = task_target(u, kind, lags, uc0)
            X, y = seg(feats, tg, cond, tr_e)
            tr_X.append(X); tr_y.append(y)
            X, y = seg(feats, tg, tr_e, va_e)
            va_X.append(X); va_y.append(y)
        Xtr, ytr = np.vstack(tr_X), np.concatenate(tr_y)
        Xva, yva = np.vstack(va_X), np.concatenate(va_y)
        if min(len(ytr), len(yva)) < 20 or np.var(yva) == 0:
            frozen[name] = None
            continue
        best_a, best_e = None, np.inf
        for a in ALPHAS:
            e = np.mean((yva - ridge_predict(ridge_fit(Xtr, ytr, a), Xva)) ** 2) / np.var(yva)
            if e < best_e:
                best_e, best_a = e, a
        frozen[name] = ridge_fit(Xtr, ytr, best_a)
        frozen[name]["val_NMSE"] = float(best_e)
        frozen[name]["n_train"] = len(ytr)
    return frozen


def evaluate_frozen(seqs, frozen, tasks, uc0, seq_ids):
    """Apply the frozen model to each sequence's own test segment. No re-tuning."""
    out = {name: {} for name, _, _ in tasks}
    for name, kind, lags in tasks:
        m = frozen.get(name)
        if m is None:
            continue
        for sid in seq_ids:
            st = seqs[sid]["states"]
            n = seqs[sid]["n_slots"]
            _, _, va_e = split_bounds(n)
            u = st["symbol"].values
            feats = st[[f"A_{b}" for b in BANDS]].values
            tg = task_target(u, kind, lags, uc0)
            X, y = seg(feats, tg, va_e, n)
            if len(y) < 5 or np.var(y) == 0:
                continue
            p = ridge_predict(m, X)
            out[name][sid] = metrics(y, p, m["my"])
    return out


def null_eval(seqs, frozen, tasks, uc0, seq_ids, rng, n_null, min_shift=20):
    skills = {name: np.full(n_null, np.nan) for name, _, _ in tasks}
    for d in range(n_null):
        for name, kind, lags in tasks:
            m = frozen.get(name)
            if m is None:
                continue
            vals = []
            for sid in seq_ids:
                st = seqs[sid]["states"]
                n = seqs[sid]["n_slots"]
                _, _, va_e = split_bounds(n)
                u = st["symbol"].values
                sh = int(rng.integers(min_shift, max(min_shift + 1, n - min_shift)))
                tg = task_target(np.roll(u, sh), kind, lags, uc0)
                feats = st[[f"A_{b}" for b in BANDS]].values
                X, y = seg(feats, tg, va_e, n)
                if len(y) < 5 or np.var(y) == 0:
                    continue
                p = ridge_predict(m, X)
                vals.append(metrics(y, p, m["my"])["skill"])
            if vals:
                skills[name][d] = np.mean(vals)
    return skills


def bh_adjust(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def main():
    ap = argparse.ArgumentParser(description="Protocol 6D: multi-sequence frozen-model capacity analysis")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--outdir", default="out_6d")
    ap.add_argument("--levels", default="0.20,0.35,0.50")
    ap.add_argument("--alt-source", default="specified_pump_dosing:alt_media")
    ap.add_argument("--gap-sec", type=float, default=5.0)
    ap.add_argument("--scan-gap-sec", type=float, default=2.0)
    ap.add_argument("--avg-scans", type=int, default=1)
    ap.add_argument("--baseline-window-min", type=float, default=10.0)
    ap.add_argument("--baseline-guard-sec", type=float, default=30.0)
    ap.add_argument("--max-lag", type=int, default=8)
    ap.add_argument("--sequence-order", default=None,
                    help="comma-separated sequence folder names, in collection order; default = sorted folder names")
    ap.add_argument("--freeze-sequences", default=None,
                    help="comma-separated sequence names to use for fit+alpha-tuning (default: all but the last loaded sequence)")
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    levels = np.array(sorted(float(x) for x in args.levels.split(",")))
    uc0 = float(levels.mean())

    print("Loading sequences...")
    seqs, order = load_sequences(args.data_root, args, levels, args.sequence_order)
    print(f"Sequences (collection order): {order}")
    if len(order) < 2:
        raise ValueError("Need at least 2 independent sequences (>=4 recommended per Protocol 6D)")

    freeze_ids = ([x.strip() for x in args.freeze_sequences.split(",")] if args.freeze_sequences
                  else order[:-1])
    if set(freeze_ids) - set(order):
        raise ValueError(f"--freeze-sequences names not among loaded sequences: {set(freeze_ids) - set(order)}")
    print(f"Freeze sequences (used for fit + alpha tuning): {freeze_ids}")

    tasks = build_tasks(args.max_lag)
    frozen = freeze_model(seqs, freeze_ids, tasks, uc0)
    freeze_record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sequences": freeze_ids,
        "levels": levels.tolist(), "uc0": uc0, "max_lag": args.max_lag,
        "models": frozen,
    }
    with open(os.path.join(args.outdir, "frozen_model.json"), "w") as f:
        json.dump(freeze_record, f, indent=1)
    print(f"Frozen model saved to {os.path.join(args.outdir, 'frozen_model.json')}")
    for name, _, _ in tasks:
        if frozen[name] is not None:
            print(f"  {name}: alpha={frozen[name]['alpha']:g}, n_train={frozen[name]['n_train']}, "
                  f"val_NMSE={frozen[name]['val_NMSE']:.3f}")

    print("\nEvaluating frozen model on every sequence's own test segment...")
    per_seq = evaluate_frozen(seqs, frozen, tasks, uc0, order)

    null = None
    if args.n_null > 0:
        print(f"Rolled-input null ({args.n_null} draws, frozen model, no re-tuning)...")
        null = null_eval(seqs, frozen, tasks, uc0, order, rng, args.n_null)

    rows = []
    for name, _, _ in tasks:
        vals = [per_seq[name][s]["cap"] for s in order if s in per_seq[name]]
        skills = [per_seq[name][s]["skill"] for s in order if s in per_seq[name]]
        if not vals:
            continue
        row = {"task": name, "n_sequences": len(vals),
               "cap_mean": np.mean(vals), "cap_sd": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
               "cap_sem": np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan,
               "skill_mean": np.mean(skills),
               "n_test_total": sum(per_seq[name][s]["n_test"] for s in order if s in per_seq[name])}
        if null is not None:
            nv = null[name][np.isfinite(null[name])]
            if len(nv):
                row["null_skill_p95"] = np.percentile(nv, 95)
                row["p_perm"] = (1 + np.sum(nv >= row["skill_mean"])) / (1 + len(nv))
        rows.append(row)
    memcurve = pd.DataFrame(rows)
    if null is not None and "p_perm" in memcurve:
        memcurve["p_BH"] = bh_adjust(memcurve["p_perm"].fillna(1.0).values)
    memcurve.to_csv(os.path.join(args.outdir, "memory_curve.csv"), index=False)

    per_seq_rows = []
    for name, _, _ in tasks:
        for s in order:
            if s in per_seq[name]:
                per_seq_rows.append({"task": name, "sequence": s, **per_seq[name][s]})
    pd.DataFrame(per_seq_rows).to_csv(os.path.join(args.outdir, "per_sequence_results.csv"), index=False)

    # ---- stability curve: cumulative test-segment pooling over first K sequences
    lag_tasks = [t[0] for t in tasks if t[1] == "lag"]
    stab_rows = []
    for K in range(1, len(order) + 1):
        cum_ids = order[:K]
        n_test_cum = 0
        for name in lag_tasks:
            vals = [per_seq[name][s]["cap"] for s in cum_ids if s in per_seq[name]]
            nts = sum(per_seq[name][s]["n_test"] for s in cum_ids if s in per_seq[name])
            if name == lag_tasks[0]:
                n_test_cum = nts
            if vals:
                stab_rows.append({"K_sequences": K, "n_test_slots_this_task": nts, "task": name,
                                 "cap_mean": np.mean(vals),
                                 "cap_sem": np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan})
    stab = pd.DataFrame(stab_rows)
    stab.to_csv(os.path.join(args.outdir, "stability_curve.csv"), index=False)

    print("\n" + "=" * 78)
    print(" PROTOCOL 6D — MULTI-SEQUENCE MEMORY CURVE (frozen model)")
    print("=" * 78)
    cols = ["task", "n_sequences", "cap_mean", "cap_sem", "n_test_total"]
    if "p_BH" in memcurve:
        cols += ["null_skill_p95", "p_perm", "p_BH"]
    print(memcurve[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if "p_BH" in memcurve:
        sig = memcurve[memcurve["p_BH"] < 0.05]["task"].tolist()
        print(f"\nSignificant after BH-FDR (q<0.05): {sig if sig else 'none'}")
    lag_curve = memcurve[memcurve["task"].isin(lag_tasks)]
    print(f"\nSummed capacity, lags 1-{args.max_lag}: {lag_curve['cap_mean'].sum():.3f} "
          f"(pooled over {lag_curve['n_test_total'].iloc[0] if len(lag_curve) else 0} test slots/task)")

    print("\nFLAGS:")
    for f in FLAGS:
        print("  - " + f)
    if not FLAGS:
        print("  none")

    # ---- figures
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=130)
    lag_mc = memcurve[memcurve["task"].isin(lag_tasks)].copy()
    lag_mc["lag"] = lag_mc["task"].str.replace("lag", "").astype(int)
    lag_mc = lag_mc.sort_values("lag")
    ax.errorbar(lag_mc["lag"], lag_mc["cap_mean"], yerr=lag_mc["cap_sem"].fillna(0), fmt="o-",
               color="#0055d4", capsize=4, lw=2, label=f"pooled over {len(order)} sequences")
    if "null_skill_p95" in lag_mc:
        ax.plot(lag_mc["lag"], lag_mc["null_skill_p95"], "r--", label="rolled-input null 95th pct")
    ax.set_xlabel("lag k")
    ax.set_ylabel("capacity = max(0, 1-NMSE)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Fig 1: memory curve, frozen model applied to every sequence's own test segment", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "01_memory_curve.png"))
    plt.close(fig)

    n_show = min(4, len(lag_tasks))
    fig, axes = plt.subplots(1, n_show, figsize=(4.5 * n_show, 4.5), dpi=130, squeeze=False)
    for ax, name in zip(axes[0], lag_tasks[:n_show]):
        sub = stab[stab["task"] == name]
        ax.errorbar(sub["K_sequences"], sub["cap_mean"], yerr=sub["cap_sem"].fillna(0), fmt="s-",
                   color="#e67e22", capsize=4)
        ax.set_xlabel("sequences pooled (K)")
        ax.set_ylabel("capacity")
        ax.set_title(name, fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xticks(sub["K_sequences"])
    fig.suptitle("Fig 2: capacity estimate vs number of independent sequences pooled (stability check)",
                fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "02_stability_curve.png"))
    plt.close(fig)

    print(f"\nOutputs in: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
