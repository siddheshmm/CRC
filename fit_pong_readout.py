#!/usr/bin/env python3
"""
fit_pong_readout.py — fit and freeze the lane-pong readout from normal Pioreactor exports.

Task the readout learns (default "wander" ball): the ball sits on a hidden AIM lane but is knocked
one lane off with some probability; guess the lane it is aimed at in the NEXT slot. A moving
average of recent lanes does this - which is what a leaky dye state physically computes.
Mechanically: given the 8-band absorbance vector of the last scan BEFORE the dose in
slot n ("pre"), and/or the scan `observe-delay` seconds after it ("post"), predict the ball's
lane in slot n+1.  For the zigzag ball, next lane = (n_lanes-1) - lane(n-1), so the useful
information is the most recent dose already sitting in the vial: "pre" is usually the best set.  Linear ridge regression on the lane index; paddle = round(prediction).

Usage
  # 1. training sequence for the ball (feed to specified_pump_dosing --volume-sequence):
  python fit_pong_readout.py --make-sequence 250 --levels 0.1,0.3,0.5 --seed 3
  #    (play later with a DIFFERENT seed, same --ball/--hold/--jitter)

  # 2. fit from one or more export folders (each has dosing_events/ and as7341_spectrum_readings/):
  python fit_pong_readout.py --runs ball_run1 ball_run2 --levels 0.1,0.3,0.5 --out pong_readout.json
  (add --alt-source reservoir_pong:alt_media if the doses came from the plugin)

What it reports (blocked CV, never scored on rows it trained on):
  * accuracy at predicting the ball's next lane, and an exact binomial p-value that it beats the
    memoryless "copy the current lane" paddle
  * references: always-most-common-lane, copy-current-lane, (previous,current) lookup, and the
    ceiling for ONE leaky state (ideal moving average of the true lanes)
Use the same ball settings (--ball/--hold/--jitter) in training and play, but different seeds.
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BANDS = [415, 445, 480, 515, 555, 590, 630, 680]
ALPHAS = [1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]


# ── ball (identical copy of the one in reservoir_pong.py) ─────────────────────

def make_ball(n, n_lanes=3, seed=0, kind="wander", hold=20.0, jitter=0.6, start_lane=0, start_dir=1):
    """-> (lanes, aims), one entry per slot.  `lanes` is where the ball is (what gets dosed);
    `aims` is the lane the ball is aimed at (what the paddle should guess).
    kind="wander": the aim re-draws with prob 1/hold each slot; each slot the ball is on the aim,
                   except with prob `jitter` it is knocked one lane up/down (clipped at the walls).
                   Estimating the aim = a moving average of recent lanes = what a leaky dye state computes.
    kind="zigzag": bounces 0,1,2,1,0,... with aim = lane (needs a DIFFERENCE of past inputs, which one
                   leaky state only supplies on a knife-edge - kept for comparison)."""
    rng = random.Random(seed)
    lanes, aims = [], []
    if kind == "zigzag":
        lane, d = int(start_lane), int(start_dir)
        for _ in range(n):
            lanes.append(lane)
            nxt = lane + d
            if nxt < 0 or nxt >= n_lanes:
                d = -d
                nxt = lane + d
            lane = nxt
        return lanes, list(lanes)
    target = rng.randrange(n_lanes)
    for _ in range(n):
        if rng.random() < 1.0 / max(1.0, float(hold)):
            target = rng.randrange(n_lanes)
        lane = target
        if rng.random() < jitter:
            lane = min(n_lanes - 1, max(0, target + rng.choice((-1, 1))))
        lanes.append(lane)
        aims.append(target)
    return lanes, aims


def make_ball_lanes(*args, **kwargs):
    return make_ball(*args, **kwargs)[0]


# ── ingestion (same causal rules as the capacity analysis) ────────────────────

def _first_csv(d, sub):
    f = sorted(glob.glob(os.path.join(d, sub, "*.csv")))
    if not f:
        raise FileNotFoundError(f"no CSV in {os.path.join(d, sub)}")
    return pd.read_csv(f[0])


def _ts(s):
    return pd.to_datetime(s, utc=True).dt.tz_convert(None)


def load_run(run_dir, levels, alt_source, delay_s, baseline_min, guard_s, pulse_gap_s=5.0, scan_gap_s=2.0):
    dose = _first_csv(run_dir, "dosing_events")
    spec = _first_csv(run_dir, "as7341_spectrum_readings")
    dose["ts"] = _ts(dose["timestamp"])
    spec["ts"] = _ts(spec["timestamp"])
    alt = dose[dose["source_of_event"] == alt_source].sort_values("ts").copy()
    if alt.empty:
        have = sorted(dose["source_of_event"].astype(str).unique())
        raise ValueError(f"{run_dir}: no doses with source_of_event == '{alt_source}'. Found: {have}")
    gap = alt["ts"].diff().dt.total_seconds().fillna(1e9)
    alt["pid"] = (gap > pulse_gap_s).cumsum()
    P = alt.groupby("pid").agg(start=("ts", "first"), vol=("volume_change_ml", "sum")).reset_index(drop=True)
    lv = np.asarray(levels)
    P["lane"] = np.abs(P["vol"].values[:, None] - lv[None, :]).argmin(1)
    err = np.abs(P["vol"].values - lv[P["lane"].values])
    if err.max() > 0.4 * np.min(np.diff(np.sort(lv))):
        print(f"  [FLAG] {run_dir}: logged pulse volume is up to {err.max():.3f} mL from the nearest level")
    iv = P["start"].diff().dt.total_seconds()
    med = float(np.nanmedian(iv))
    inc = np.ones(len(P), dtype=int)
    inc[1:] = np.maximum(1, np.rint(iv.values[1:] / med).astype(int))
    P["slot"] = np.cumsum(inc) - 1
    n_slots = int(P["slot"].iloc[-1]) + 1

    spec["band"] = pd.to_numeric(spec["band"], errors="coerce")
    spec = spec[spec["band"].isin(BANDS)].dropna(subset=["ts", "reading"]).sort_values("ts")
    spec["scan"] = (spec["ts"].diff().dt.total_seconds().fillna(1e9) > scan_gap_s).cumsum()
    W = spec.pivot_table(index="scan", columns="band", values="reading", aggfunc="mean")[BANDS]
    W.index = pd.DatetimeIndex(spec.groupby("scan")["ts"].max().reindex(W.index).values)
    W = W.dropna().sort_index()

    t0 = P["start"].iloc[0]
    base = W[(W.index >= t0 - pd.Timedelta(minutes=baseline_min) - pd.Timedelta(seconds=guard_s))
             & (W.index < t0 - pd.Timedelta(seconds=guard_s))]
    if len(base) < 10:
        raise ValueError(f"{run_dir}: only {len(base)} baseline scans before the first dose")
    A = -np.log10((W / base.mean()).clip(lower=1e-6))

    rows = {}
    for _, p in P.iterrows():
        pos_pre = A.index.searchsorted(p["start"], side="left")
        pos_post = A.index.searchsorted(p["start"] + pd.Timedelta(seconds=delay_s), side="left")
        rec = {"lane": int(p["lane"])}
        pre = A.iloc[pos_pre - 1].values if pos_pre > 0 else np.full(8, np.nan)
        post = A.iloc[pos_post].values if pos_post < len(A) else np.full(8, np.nan)
        for i, b in enumerate(BANDS):
            rec[f"pre_{b}"] = pre[i]
            rec[f"post_{b}"] = post[i]
        rows[int(p["slot"])] = rec
    df = pd.DataFrame.from_dict(rows, orient="index").reindex(range(n_slots))
    df.index.name = "slot"
    print(f"  {run_dir}: {len(P)} doses -> {n_slots} slots, interval {med:.1f}s, baseline {len(base)} scans, "
          f"tail {(W.index.max() - P['start'].iloc[-1]).total_seconds():.0f}s")
    return df


def build_rows(dfs, feature_set):
    cols = []
    if "pre" in feature_set.split("+"):
        cols += [f"pre_{b}" for b in BANDS]
    if "post" in feature_set.split("+"):
        cols += [f"post_{b}" for b in BANDS]
    out = []
    for rid, df in enumerate(dfs):
        d = df.copy()
        d["next"] = d["aim"].shift(-1)
        d["run"] = rid
        d["slot"] = d.index
        out.append(d)
    D = pd.concat(out, ignore_index=True)
    ok = D[cols + ["lane", "next"]].notna().all(axis=1)
    return D[ok].reset_index(drop=True), cols


# ── ridge + blocked cross-validation ──────────────────────────────────────────

def ridge_fit(X, y, alpha):
    mx, sx = X.mean(0), X.std(0)
    sx = np.where(sx == 0, 1.0, sx)
    Z = (X - mx) / sx
    my = y.mean()
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - my))
    return mx, sx, my, w


def ridge_predict(m, X):
    mx, sx, my, w = m
    return ((X - mx) / sx) @ w + my


def folds(D, K, gap):
    """Yield (train_idx, test_idx): contiguous held-out chunks per run, with a gap either side."""
    out = []
    for k in range(K):
        test, train = [], []
        for rid, g in D.groupby("run"):
            slots = np.sort(g["slot"].unique())
            chunk = np.array_split(slots, K)[k]
            if len(chunk) == 0:
                train += list(g.index)
                continue
            a, b = chunk.min(), chunk.max()
            test += list(g.index[(g["slot"] >= a) & (g["slot"] <= b)])
            train += list(g.index[(g["slot"] < a - gap) | (g["slot"] > b + gap)])
        if test and train:
            out.append((np.array(train), np.array(test)))
    return out


def pick_alpha(D, X, y, K=4, gap=5):
    best, best_e = ALPHAS[0], np.inf
    for a in ALPHAS:
        errs = []
        for tr, te in folds(D, K, gap):
            m = ridge_fit(X[tr], y[tr], a)
            errs.append(np.mean((y[te] - ridge_predict(m, X[te])) ** 2))
        e = float(np.mean(errs)) if errs else np.inf
        if e < best_e:
            best, best_e = a, e
    return best


def outer_cv(D, X, y, K=5, gap=5):
    pred = np.full(len(D), np.nan)
    for tr, te in folds(D, K, gap):
        a = pick_alpha(D.iloc[tr].reset_index(drop=True), X[tr], y[tr])
        pred[te] = ridge_predict(ridge_fit(X[tr], y[tr], a), X[te])
    return pred


def binom_sf(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p), exact."""
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def oracle_leaky(dfs, D, n_lanes):
    """Ceiling for ONE leaky state: exponential moving average of the TRUE observed lanes, passed
    through the best affine map to the next aim, then rounded. Reference only, not a trained model."""
    best = (0.0, 0.0)
    nxt = D["next"].values
    for lam in np.linspace(0.0, 0.95, 20):
        ema = {}
        for rid, df in enumerate(dfs):
            e, lanes = 0.0, df["lane"].values
            for i, l in enumerate(lanes):
                e = lam * e + (1 - lam) * (l if np.isfinite(l) else e)
                ema[(rid, i)] = e
        x = np.array([ema[(r, sl)] for r, sl in zip(D["run"], D["slot"])])
        A = np.c_[x, np.ones(len(x))]
        w = np.linalg.lstsq(A, nxt, rcond=None)[0]
        acc = float(np.mean(np.clip(np.rint(A @ w), 0, n_lanes - 1) == nxt))
        if acc > best[0]:
            best = (acc, float(lam))
    return best


def report(D, pred, n_lanes, dfs):
    ok = np.isfinite(pred)
    d = D[ok].copy()
    d["hat"] = np.clip(np.rint(pred[ok]), 0, n_lanes - 1).astype(int)
    nxt, cur = d["next"].astype(int).values, d["lane"].astype(int).values
    n = int(ok.sum())
    k = int(np.sum(d["hat"].values == nxt))
    acc = k / n
    skill = 1.0 - np.mean((nxt - pred[ok]) ** 2) / np.var(nxt)
    base_major = float(np.mean(nxt == int(pd.Series(nxt).mode().iloc[0])))
    base_copy = float(np.mean(cur == nxt))
    look = d.groupby("lane")["next"].agg(lambda s: s.mode().iloc[0])
    base_look = float(np.mean(look.reindex(cur).values == nxt))
    leaky_acc, leaky_lam = oracle_leaky(dfs, D[ok], n_lanes)
    return {
        "n_scored": n, "accuracy": acc, "skill_1_minus_nmse": float(skill),
        "p_vs_memoryless": binom_sf(k, n, max(base_copy, base_look)),
        "baseline_always_most_common": base_major, "baseline_copy_current_lane": base_copy,
        "baseline_best_lookup_from_current_lane": base_look,
        "oracle_leaky_average": leaky_acc, "oracle_leaky_lambda": leaky_lam,
    }


def main():
    ap = argparse.ArgumentParser(description="Fit/freeze the reservoir lane-pong readout")
    ap.add_argument("--runs", nargs="*", default=[], help="export folders (dosing_events/ + as7341_spectrum_readings/)")
    ap.add_argument("--levels", default="0.1,0.3,0.5")
    ap.add_argument("--alt-source", default="specified_pump_dosing:alt_media")
    ap.add_argument("--observe-delay", type=float, default=5.0)
    ap.add_argument("--baseline-min", type=float, default=10.0)
    ap.add_argument("--baseline-guard-sec", type=float, default=30.0)
    ap.add_argument("--pulse-gap-sec", type=float, default=5.0, help="ticks closer than this belong to one dose")
    ap.add_argument("--scan-gap-sec", type=float, default=2.0, help="band rows closer than this belong to one scan")
    ap.add_argument("--feature-set", choices=["pre", "post", "pre+post", "auto"], default="auto",
                    help="pre = last scan before the dose, post = scan observe-delay s after it")
    ap.add_argument("--out", default="pong_readout.json")
    ap.add_argument("--make-sequence", type=int, default=0, help="print a ball volume sequence of this length and exit")
    ap.add_argument("--seed", type=int, default=3, help="ball seed for --make-sequence (play with a different one)")
    ap.add_argument("--ball", choices=["wander", "zigzag"], default="wander")
    ap.add_argument("--hold", type=float, default=20.0)
    ap.add_argument("--seeds", type=int, nargs="*", default=[],
                    help="ball seed used for each run in --runs, same order (needed to recover the hidden aims)")
    ap.add_argument("--jitter", type=float, default=0.6)
    args = ap.parse_args()

    levels = [float(x) for x in args.levels.split(",")]
    n_lanes = len(levels)
    if args.make_sequence:
        lanes = make_ball_lanes(args.make_sequence, n_lanes, args.seed, args.ball, args.hold, args.jitter)
        print(",".join(f"{levels[l]:g}" for l in lanes))
        return
    if not args.runs:
        sys.exit("give --runs <folder> ... (or --make-sequence N)")

    print("Loading runs...")
    sets = ["pre", "post", "pre+post"] if args.feature_set == "auto" else [args.feature_set]
    dfs = [load_run(r, levels, args.alt_source, args.observe_delay, args.baseline_min, args.baseline_guard_sec,
                   args.pulse_gap_sec, args.scan_gap_sec)
           for r in args.runs]

    if args.ball == "wander" and len(args.seeds) != len(args.runs):
        sys.exit("--ball wander needs --seeds with one seed per run (the hidden aim is regenerated from it)")
    for rid, df in enumerate(dfs):
        seed = args.seeds[rid] if args.seeds else 0
        lanes, aims = make_ball(len(df) + 1, n_lanes, seed, args.ball, args.hold, args.jitter)
        obs = df["lane"].values
        m = np.isfinite(obs)
        if not np.array_equal(np.asarray(lanes[:len(df)])[m], obs[m].astype(int)):
            sys.exit(f"{args.runs[rid]}: the doses do not match the ball made by seed {seed} "
                     f"({args.ball}, hold={args.hold}, jitter={args.jitter}). Check seed/order/settings.")
        df["aim"] = aims[:len(df)]
    print("  doses match the regenerated ball in every run (hidden aims recovered).")

    results = {}
    for fs in sets:
        D, cols = build_rows(dfs, fs)
        if len(D) < 40:
            sys.exit(f"only {len(D)} usable rows - need more slots")
        X, y = D[cols].values, D["next"].astype(float).values
        pred = outer_cv(D, X, y)
        results[fs] = (D, cols, X, y, report(D, pred, n_lanes, dfs))

    best_fs = max(results, key=lambda k: results[k][4]["skill_1_minus_nmse"])
    print("\nBlocked cross-validation (out-of-fold only)")
    for fs, (_, _, _, _, r) in results.items():
        star = "  <- chosen" if fs == best_fs else ""
        print(f"  features={fs:9s} skill={r['skill_1_minus_nmse']:+.3f} accuracy={r['accuracy']:.3f} "
              f"(p={r['p_vs_memoryless']:.3f} that it beats the memoryless lookup){star}")
    D, cols, X, y, r = results[best_fs]
    print(f"\nReference accuracies at guessing the NEXT aimed lane, same {r['n_scored']} rows")
    print(f"  always guess the most common lane                     : {r['baseline_always_most_common']:.3f}")
    print(f"  copy the current ball lane (memoryless paddle)        : {r['baseline_copy_current_lane']:.3f}")
    print(f"  best lookup from the current ball lane (memoryless)   : {r['baseline_best_lookup_from_current_lane']:.3f}")
    print(f"  ideal leaky average of TRUE lanes (lambda={r['oracle_leaky_lambda']:.2f})       : {r['oracle_leaky_average']:.3f}   <- ceiling for one leaky state")
    if r["oracle_leaky_average"] - max(r["baseline_copy_current_lane"], r["baseline_best_lookup_from_current_lane"]) < 0.03:
        print("  NOTE: even an ideal moving average barely beats the memoryless paddle on these data,")
        print("        so they cannot show a memory benefit (random sequence, or ball settings with no headroom).")

    alpha = pick_alpha(D, X, y, K=5)
    mx, sx, my, w = ridge_fit(X, y, alpha)
    model = {
        "version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "feature_set": best_fs,
        "observe_delay_s": args.observe_delay, "bands": BANDS, "levels": levels, "n_lanes": n_lanes,
        "alpha": alpha, "mx": mx.tolist(), "sx": sx.tolist(), "w": w.tolist(), "b": float(my),
        "trained_on": args.runs, "n_rows": int(len(D)), "cv": r,
        "ball": {"kind": args.ball, "hold": args.hold, "jitter": args.jitter},
        "train_seeds": args.seeds,
    }
    with open(args.out, "w") as f:
        json.dump(model, f, indent=1)
    print(f"\nFrozen readout saved to {args.out} (alpha={alpha:g}, features={best_fs}).")
    print("Do not refit after looking at play results - that would break the held-out test.")


if __name__ == "__main__":
    main()
