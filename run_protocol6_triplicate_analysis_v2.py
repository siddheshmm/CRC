#!/usr/bin/env python3
"""
run_protocol6_triplicate_analysis_v2.py

Protocol 6 (overlapping-input memory) analysis for N replicate Pioreactor runs.
Revised against PROTOCOLS.md sections 6A/6C.  Key differences from v1 are listed
in the review notes; the short version:

  * UTC clock used for both streams (falls back to local time only with a flag)
  * strictly causal observation: last COMPLETE scan strictly BEFORE dose slot n
  * design-alphabet symbols (snapped) and slot grid, so a skipped/merged pulse
    cannot silently shift every lag target
  * scalers / ridge alpha fitted on training data only, alpha tuned on validation
  * protocol split proportions (40/336/112/112 of 600) scaled to n slots
  * "pooled chronological" test = train on early segments of all runs, test on the
    late segment nobody trained on (valid even if runs share one input sequence)
  * LORO-CV kept, but with nested alpha selection, training-only scaling and an
    explicit SAME-SEQUENCE warning (with identical sequences it is NOT a
    generalisation test)
  * capacity = max(0, 1 - NMSE) (r^2 kept as a secondary column)
  * rolled-input null + BH-FDR, constant predictor, single-state leaky integrator
    (oracle and noise-matched), product-of-centred-inputs nonlinear task,
    optional water-control runs
  * baseline / tail / cadence / clipping / SNR audits that actually run
"""

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BANDS = [415, 445, 480, 515, 555, 590, 630, 680]
BAND_COLORS = {415: "#6a0dad", 445: "#0055d4", 480: "#00a2e8", 515: "#008a00",
               555: "#8cb800", 590: "#e67e22", 630: "#e74c3c", 680: "#880e4f"}
RUN_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
ALPHAS = [1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
# protocol 6B split (40 conditioning / 336 train / 112 val / 112 test of 600)
FRAC_COND, FRAC_TRAIN, FRAC_VAL = 40 / 600, 336 / 600, 112 / 600

FLAGS = []


def flag(msg):
    print(f"  [FLAG] {msg}")
    FLAGS.append(msg)


# --------------------------------------------------------------------------
# 1. Ingestion
# --------------------------------------------------------------------------

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


def read_first_csv(run_dir, sub):
    files = sorted(glob.glob(os.path.join(run_dir, sub, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV in {os.path.join(run_dir, sub)}")
    if len(files) > 1:
        flag(f"{run_dir}/{sub}: {len(files)} CSVs found, using only {os.path.basename(files[0])}")
    return pd.read_csv(files[0])


def extract_single_run(run_dir, run_id, args, levels):
    tag = f"Run {run_id:02d}"
    print(f"\nProcessing {tag} from: {run_dir}")
    dose = read_first_csv(run_dir, "dosing_events")
    spec = read_first_csv(run_dir, "as7341_spectrum_readings")

    # ---- clock: UTC for BOTH streams (protocol: never shift only one stream)
    if "timestamp" in dose.columns and "timestamp" in spec.columns:
        clock, utc = "timestamp", True
    else:
        flag(f"{tag}: UTC 'timestamp' column missing in one stream; using timestamp_localtime for BOTH")
        clock, utc = "timestamp_localtime", False
    dose["ts"] = _to_dt(dose[clock], utc)
    spec["ts"] = _to_dt(spec[clock], utc)
    dose = dose.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    # ---- dosing-event audit (what is actually in the log)
    audit = (dose.groupby(["event", "source_of_event"])
             .agg(n=("volume_change_ml", "size"), vol_ml=("volume_change_ml", "sum"))
             .reset_index())
    print("  dosing-log audit (event / source / n / logged mL):")
    print("    " + audit.to_string(index=False).replace("\n", "\n    "))

    alt = dose[dose["source_of_event"] == args.alt_source].copy()
    if alt.empty:
        raise ValueError(f"{tag}: no rows with source_of_event == '{args.alt_source}'. "
                         f"Available: {sorted(dose['source_of_event'].astype(str).unique())}")
    gap = alt["ts"].diff().dt.total_seconds().fillna(1e9)
    alt["pulse_id"] = (gap > args.gap_sec).cumsum()
    pulses = (alt.groupby("pulse_id")
              .agg(start_time=("ts", "first"), end_time=("ts", "last"),
                   n_subdoses=("volume_change_ml", "size"),
                   volume_raw=("volume_change_ml", "sum"))
              .reset_index(drop=True))
    pulses["dur_sec"] = (pulses["end_time"] - pulses["start_time"]).dt.total_seconds()
    pulses["interval_sec"] = pulses["start_time"].diff().dt.total_seconds()

    # ---- symbols: snap logged pulse volume to the design alphabet
    dist = np.abs(pulses["volume_raw"].values[:, None] - levels[None, :])
    pulses["symbol"] = levels[dist.argmin(axis=1)]
    pulses["snap_err"] = pulses["volume_raw"] - pulses["symbol"]
    spacing = np.min(np.diff(np.sort(levels))) if len(levels) > 1 else 1.0
    if np.abs(pulses["snap_err"]).max() > 0.4 * spacing:
        flag(f"{tag}: logged pulse volume deviates up to {np.abs(pulses['snap_err']).max():.3f} mL from the "
             f"nearest design level (>40% of level spacing) - symbol assignment may be ambiguous")
    for lv in levels:
        sub = pulses.loc[pulses["symbol"] == lv, "volume_raw"]
        if len(sub) and abs(sub.mean() / lv - 1) > 0.10:
            print(f"  NOTE: level {lv:.2f} mL logs a mean of {sub.mean():.3f} mL ({(sub.mean() / lv - 1) * 100:+.0f}%) "
                  f"- pump/micro-tick quantisation; needs Protocol 1B gravimetric validation")

    # ---- slot grid (robust to a skipped / merged pulse)
    med_int = float(np.nanmedian(pulses["interval_sec"]))
    inc = np.ones(len(pulses), dtype=int)
    inc[1:] = np.maximum(1, np.rint(pulses["interval_sec"].values[1:] / med_int).astype(int))
    pulses["slot"] = np.cumsum(inc) - 1
    n_slots = int(pulses["slot"].iloc[-1]) + 1
    n_missing = n_slots - len(pulses)
    if len(pulses) != args.expected_pulses:
        flag(f"{tag}: {len(pulses)} pulses clustered, expected {args.expected_pulses} "
             f"(check --gap-sec / aborted run)")
    if n_missing:
        flag(f"{tag}: {n_missing} slot(s) missing on the grid (gap > 1.5x median interval); kept as NaN")
    es = (pulses["start_time"].shift(-1) - pulses["end_time"]).dt.total_seconds()
    med_es = float(np.nanmedian(es))
    if abs(med_es - args.interval_sec) <= 2.0 and abs(med_int - args.interval_sec) > 2.0:
        print(f"  NOTE: intervals are END-to-START ({med_es:.1f}s planned {args.interval_sec:.0f}s); start-to-start "
              f"= {med_int:.1f}s and varies with pulse size (dur by symbol: "
              f"{pulses.groupby('symbol')['dur_sec'].median().round(1).to_dict()}). Slot spacing is therefore NOT uniform.")
    elif abs(med_int - args.interval_sec) > 2.0:
        flag(f"{tag}: median start-to-start {med_int:.1f}s / end-to-start {med_es:.1f}s vs planned {args.interval_sec:.0f}s")

    # ---- spectra -> complete scans (band rows grouped by time gap, not by rounded second)
    spec["band"] = pd.to_numeric(spec["band"], errors="coerce")
    spec["reading"] = pd.to_numeric(spec["reading"], errors="coerce")
    spec = spec[spec["band"].isin(BANDS)].dropna(subset=["ts", "reading"]).sort_values("ts")
    gsec = spec["ts"].diff().dt.total_seconds().fillna(1e9)
    spec["scan_id"] = (gsec > args.scan_gap_sec).cumsum()
    wide = spec.pivot_table(index="scan_id", columns="band", values="reading", aggfunc="mean")
    wide.columns = [int(c) for c in wide.columns]
    wide = wide.reindex(columns=BANDS)
    stime = spec.groupby("scan_id")["ts"].max()          # scan COMPLETION time (conservative)
    wide.index = pd.DatetimeIndex(stime.reindex(wide.index).values, name="scan_time")
    n_all = len(wide)
    wide = wide.dropna(how="any")
    if n_all - len(wide):
        flag(f"{tag}: {n_all - len(wide)} incomplete scan(s) (missing band) dropped")
    wide = wide.groupby(level=0).mean().sort_index()

    dts = wide.index.to_series().diff().dt.total_seconds().dropna()
    cadence = float(dts.median())
    n_gap = int((dts > 2 * cadence).sum())
    if args.scan_gap_sec >= 0.5 * cadence:
        flag(f"{tag}: --scan-gap-sec ({args.scan_gap_sec}) is >= half the scan cadence ({cadence:.1f}s)")
    if n_gap:
        flag(f"{tag}: {n_gap} data gap(s) > 2x cadence (max {dts.max():.1f}s)")

    # ---- baseline: last N minutes before the first dose (minus guard)
    t0 = pulses["start_time"].iloc[0]
    t_end = t0 - pd.Timedelta(seconds=args.baseline_guard_sec)
    t_beg = t_end - pd.Timedelta(minutes=args.baseline_window_min)
    base = wide[(wide.index >= t_beg) & (wide.index < t_end)]
    if len(base) < 10:
        raise ValueError(f"{tag}: only {len(base)} baseline scans before first dose")
    pre_rec_min = (t0 - wide.index.min()).total_seconds() / 60.0
    if pre_rec_min < args.baseline_window_min + args.baseline_guard_sec / 60.0 - 0.5:
        flag(f"{tag}: only {pre_rec_min:.1f} min of pre-dose recording (protocol asks for 10 min baseline)")
    bmean = base.mean()
    tsec = (base.index - base.index[0]).total_seconds().values
    base_stats = {}
    for b in BANDS:
        slope = np.polyfit(tsec, base[b].values, 1)[0] * 600.0 / bmean[b] * 100.0
        resid = base[b].values - np.polyval(np.polyfit(tsec, base[b].values, 1), tsec)
        base_stats[b] = {"mean": bmean[b], "std": base[b].std(),
                         "cv_pct": resid.std(ddof=2) / bmean[b] * 100.0,
                         "drift_pct_per_10min": slope}
    Aw = -np.log10((wide / bmean).clip(lower=1e-6))          # apparent absorbance vs run baseline
    # noise = residual SD after removing a linear trend in the baseline window (drift reported separately)
    A_base = Aw.loc[base.index]
    A_noise = A_base.apply(lambda c: np.std(c.values - np.polyval(np.polyfit(tsec, c.values, 1), tsec), ddof=2))

    # ---- clipping / headroom
    if args.saturation_level:
        sat_frac = (wide >= args.saturation_level).mean()
        headroom = args.saturation_level / wide.max()
        if (sat_frac > 0).any():
            flag(f"{tag}: readings at/above --saturation-level in bands {list(sat_frac[sat_frac > 0].index)}")
        print(f"  min headroom (sat/max reading): {headroom.min():.2f}x")
    else:
        for b in BANDS:
            mx = wide[b].max()
            if (wide[b] == mx).sum() >= 5:
                flag(f"{tag}: band {b} equals its max ({mx:g}) in >=5 scans - possible clipping "
                     f"(pass --saturation-level to check properly)")

    # ---- signal vs baseline noise during dosing
    t_last = pulses["end_time"].iloc[-1]
    dsel = Aw[(Aw.index >= t0 + pd.Timedelta(minutes=5)) & (Aw.index <= t_last)]
    snr = (dsel.std() / A_noise)
    if (snr < 3).all():
        flag(f"{tag}: dosing-window absorbance variability is <3x baseline noise in EVERY band "
             f"- little signal to read out")

    # ---- baseline drift / tail / recovery
    tail_min = (wide.index.max() - t_last).total_seconds() / 60.0
    late_pct, late_tol = {}, {}
    if tail_min < args.min_tail_min:
        flag(f"{tag}: tail after last dose is {tail_min:.1f} min (< {args.min_tail_min:.0f} min)")
    if tail_min >= 3.0:
        late = wide[wide.index >= wide.index.max() - pd.Timedelta(minutes=2)]
        for b in BANDS:
            late_pct[b] = (late[b].mean() / bmean[b] - 1.0) * 100.0
            late_tol[b] = max(0.5, 3.0 * base_stats[b]["cv_pct"])
        bad = [b for b in BANDS if abs(late_pct[b]) > late_tol[b]]
        if bad:
            flag(f"{tag}: final-2-min level differs from pre-dose baseline beyond max(0.5%, 3x noise) "
                 f"in bands {bad} (dye accumulation, level change or drift; not a mechanism call)")
    bad_d = [b for b in BANDS if abs(base_stats[b]["drift_pct_per_10min"]) > max(0.5, 3 * base_stats[b]["cv_pct"])]
    if bad_d:
        flag(f"{tag}: baseline still drifting (up to {max(abs(base_stats[b]['drift_pct_per_10min']) for b in bad_d):.1f}%/10min, "
             f"bands {bad_d}) - baseline may be a warm-up/transient, so absolute A offsets are unreliable")

    # ---- causal state extraction: last K complete scans STRICTLY before dose start
    K = max(1, args.avg_scans)
    rows = []
    for _, p in pulses.iterrows():
        pos = wide.index.searchsorted(p["start_time"], side="left")     # scans with time < start
        if pos == 0:
            continue
        sel = wide.iloc[max(0, pos - K):pos]
        Im = sel.mean()
        A = -np.log10((Im / bmean).clip(lower=1e-6))
        rec = {"slot": int(p["slot"]), "start_time": p["start_time"], "scan_time": sel.index[-1],
               "dt_before": (p["start_time"] - sel.index[-1]).total_seconds(),
               "volume_raw": p["volume_raw"], "symbol": p["symbol"]}
        for b in BANDS:
            rec[f"I_{b}"] = Im[b]
            rec[f"A_{b}"] = A[b]
        rows.append(rec)
    df_states = pd.DataFrame(rows).set_index("slot").reindex(range(n_slots))
    df_states["run_id"] = run_id
    sym = pulses.set_index("slot")["symbol"].reindex(range(n_slots))
    df_states["symbol"] = sym.values
    if df_states["dt_before"].max() > 2 * cadence:
        flag(f"{tag}: some observations are >2x cadence old at dose time (max {df_states['dt_before'].max():.1f}s)")
    print(f"  observation age at dose time: median {df_states['dt_before'].median():.1f}s, "
          f"max {df_states['dt_before'].max():.1f}s (scan cadence {cadence:.1f}s)")

    lvl_tab = (pulses.groupby("symbol")["volume_raw"].agg(["count", "mean", "std"]).round(4))
    print("  logged pulse volume per design level (count / mean / sd):")
    print("    " + lvl_tab.to_string().replace("\n", "\n    "))

    win_min = (t_last - t0).total_seconds() / 60.0
    inwin = dose[(dose["ts"] >= t0) & (dose["ts"] <= t_last)]
    media_vol = inwin.loc[inwin["event"] == "add_media", "volume_change_ml"].sum()
    waste_vol = inwin.loc[inwin["event"] == "remove_waste", "volume_change_ml"].sum()
    q_in = (media_vol + pulses["volume_raw"].sum()) / win_min if win_min > 0 else np.nan
    tau_exp = args.vial_ml / q_in * 60.0 if q_in and q_in > 0 else np.nan
    hyd = {"run_id": run_id, "n_pulses": len(pulses), "median_end_to_start_sec": med_es,
           "logged_total_inflow_ml_per_min": q_in, "tau_expected_sec_logged": tau_exp, "n_slots": n_slots, "n_missing_slots": n_missing,
           "alt_stock_ml_logged": pulses["volume_raw"].sum(),
           "dosing_window_min": win_min,
           "logged_media_ml_in_window": media_vol, "logged_waste_ml_in_window": waste_vol,
           "logged_media_ml_per_min": media_vol / win_min if win_min > 0 else np.nan,
           "logged_waste_ml_per_min": waste_vol / win_min if win_min > 0 else np.nan,
           "median_interval_sec": med_int, "sd_interval_sec": pulses["interval_sec"].std(),
           "max_interval_sec": pulses["interval_sec"].max(),
           "median_pulse_dur_sec": pulses["dur_sec"].median(),
           "scan_cadence_sec": cadence, "n_data_gaps": n_gap,
           "pre_dose_recording_min": pre_rec_min, "baseline_scans": len(base),
           "tail_min": tail_min, "clock": clock,
           "median_dt_before_sec": df_states["dt_before"].median(),
           "max_dt_before_sec": df_states["dt_before"].max()}
    for b in BANDS:
        hyd[f"snr_{b}"] = snr[b]
        hyd[f"base_cv_pct_{b}"] = base_stats[b]["cv_pct"]
        hyd[f"base_drift_pct10_{b}"] = base_stats[b]["drift_pct_per_10min"]
        if late_pct:
            hyd[f"late_offset_pct_{b}"] = late_pct[b]
    print("  NOTE: logged media/waste volumes are calibration-derived records, not liquid measurements.")
    return {"run_id": run_id, "dose_df": dose, "pulses": pulses, "wide": wide, "Aw": Aw,
            "base": base, "base_stats": base_stats, "df_states": df_states, "hydraulics": hyd,
            "audit": audit}


def load_runs(root, args, levels):
    dirs = sorted(d for d in glob.glob(os.path.join(root, args.run_glob)) if os.path.isdir(d))
    if not dirs:
        raise FileNotFoundError(f"No run folders matching '{args.run_glob}' in {root}")
    return [extract_single_run(d, i, args, levels) for i, d in enumerate(dirs, 1)]


# --------------------------------------------------------------------------
# 2. Readout machinery (numpy ridge, scaler fit on training rows only)
# --------------------------------------------------------------------------

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
    return lagged(u - uc0, lags[0]) * lagged(u - uc0, lags[1])      # product of centred inputs


def build_tasks(max_lag):
    tasks = [(f"lag{k}", "lag", (k,)) for k in range(1, max_lag + 1)]
    tasks += [(f"prod_{i}_{j}", "prod", (i, j)) for i, j in [(1, 2), (1, 3), (2, 3), (1, 4)]
              if max(i, j) <= max_lag]
    return tasks


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


def metrics(y, p, y_train_mean):
    var = np.var(y)
    nmse = np.mean((y - p) ** 2) / var
    nmse_c = np.mean((y - y_train_mean) ** 2) / var
    r = np.corrcoef(y, p)[0, 1] if np.var(p) > 0 else 0.0
    return {"NMSE": nmse, "NMSE_const": nmse_c, "r": r,
            "r2": r * r if r > 0 else 0.0, "skill": 1.0 - nmse, "cap": max(0.0, 1.0 - nmse)}


def _seg(feats, targets, rid, lo, hi):
    X, y = feats[rid][lo:hi], targets[rid][lo:hi]
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[ok], y[ok]


def chrono_eval(feats, targets, bounds, train_ids, test_ids, keep_preds=False):
    """Fit on early segments of train_ids, tune alpha on their validation segments,
    freeze, then test each test_id on ITS OWN final segment."""
    cond, tr_e, va_e = bounds
    tr = [_seg(feats, targets, i, cond, tr_e) for i in train_ids]
    va = [_seg(feats, targets, i, tr_e, va_e) for i in train_ids]
    Xtr, ytr = np.vstack([a for a, _ in tr]), np.concatenate([b for _, b in tr])
    Xva, yva = np.vstack([a for a, _ in va]), np.concatenate([b for _, b in va])
    out = {rid: None for rid in test_ids}
    if min(len(ytr), len(yva)) < 20 or np.var(yva) == 0:
        return out
    best_a, best_e = None, np.inf
    for a in ALPHAS:
        e = np.mean((yva - ridge_predict(ridge_fit(Xtr, ytr, a), Xva)) ** 2) / np.var(yva)
        if e < best_e:
            best_e, best_a = e, a
    model = ridge_fit(Xtr, ytr, best_a)
    for rid in test_ids:
        Xte, yte = _seg(feats, targets, rid, va_e, feats[rid].shape[0])
        if len(yte) < 10 or np.var(yte) == 0:
            continue
        p = ridge_predict(model, Xte)
        m = metrics(yte, p, ytr.mean())
        m.update(alpha=best_a, val_NMSE=best_e, n_test=len(yte))
        if keep_preds:
            m["y"], m["p"] = yte, p
        out[rid] = m
    return out


def loro_eval(feats, targets, cond, ids):
    """Leave-one-run-out. alpha chosen by INNER leave-one-training-run-out; scaler fit on
    pooled training runs only; conditioning rows excluded."""
    res = {}
    for test in ids:
        train = [i for i in ids if i != test]
        segs = {i: _seg(feats, targets, i, cond, feats[i].shape[0]) for i in ids}
        best_a, best_e = None, np.inf
        for a in ALPHAS:
            errs = []
            for v in train:
                inner = [i for i in train if i != v]
                if not inner:
                    continue
                Xi = np.vstack([segs[i][0] for i in inner])
                yi = np.concatenate([segs[i][1] for i in inner])
                Xv, yv = segs[v]
                if len(yv) < 10 or np.var(yv) == 0:
                    continue
                errs.append(np.mean((yv - ridge_predict(ridge_fit(Xi, yi, a), Xv)) ** 2) / np.var(yv))
            if errs and np.mean(errs) < best_e:
                best_e, best_a = np.mean(errs), a
        if best_a is None:
            best_a = 10.0
        Xtr = np.vstack([segs[i][0] for i in train])
        ytr = np.concatenate([segs[i][1] for i in train])
        Xte, yte = segs[test]
        if len(yte) < 10 or np.var(yte) == 0:
            res[test] = None
            continue
        m = metrics(yte, ridge_predict(ridge_fit(Xtr, ytr, best_a), Xte), ytr.mean())
        m.update(alpha=best_a, n_test=len(yte))
        res[test] = m
    return res


def bh_adjust(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def sequences_identical(u):
    ids = list(u)
    a = np.nan_to_num(u[ids[0]], nan=-1)
    return all(np.array_equal(a, np.nan_to_num(u[i], nan=-1)) for i in ids[1:])


def pooled_table(feats, u, tasks, bounds, uc0, ids, keep_preds_for=None):
    """{task_name: {run_id: metrics}} using pooled chronological training."""
    res = {}
    for name, kind, lags in tasks:
        tg = {i: task_target(u[i], kind, lags, uc0) for i in ids}
        res[name] = chrono_eval(feats, tg, bounds, ids, ids,
                                keep_preds=(keep_preds_for is not None and name in keep_preds_for))
    return res


def null_caps(feats, u, tasks, bounds, uc0, ids, same_seq, rng, n_null, min_shift=20):
    """Rolled-input null.  Returns (clipped capacity, unclipped skill=1-NMSE), each n_null x n_tasks,
    averaged over runs.  Significance is tested on the UNCLIPPED skill (clipping at 0 turns the
    null into a point mass and inflates apparent significance)."""
    n = next(iter(u.values())).shape[0]
    caps = np.full((n_null, len(tasks)), np.nan)
    skills = np.full((n_null, len(tasks)), np.nan)
    for d in range(n_null):
        s0 = int(rng.integers(min_shift, n - min_shift))
        sh = {i: (s0 if same_seq else int(rng.integers(min_shift, n - min_shift))) for i in ids}
        for ti, (name, kind, lags) in enumerate(tasks):
            tg = {i: task_target(np.roll(u[i], sh[i]), kind, lags, uc0) for i in ids}
            r = chrono_eval(feats, tg, bounds, ids, ids)
            v = [r[i] for i in ids if r[i]]
            if v:
                caps[d, ti] = np.mean([m["cap"] for m in v])
                skills[d, ti] = np.mean([m["skill"] for m in v])
    return caps, skills


# --------------------------------------------------------------------------
# 3. Single-state leaky-integrator baseline (same input timing)
# --------------------------------------------------------------------------

def integrator_state(u, alpha, uc0, delay=0):
    """s_n = alpha*s_{n-1} + u_{n-1-delay}; delay in slots (mixing/transport lag)."""
    uu = np.nan_to_num(u - uc0, nan=0.0)
    if delay:
        uu = np.concatenate([np.zeros(delay), uu[:-delay]])
    s = np.zeros(len(u))
    for n in range(1, len(u)):
        s[n] = alpha * s[n - 1] + uu[n - 1]
    return s


def _detrend(y, t):
    ok = np.isfinite(y)
    if ok.sum() < 5:
        return y
    return y - np.polyval(np.polyfit(t[ok], y[ok], 1), t)


def integrator_baselines(feats, u, ids, bounds, uc0, tasks, rng, slot_sec, n_noise=10):
    """Single-state LINEAR leaky integrator (fitted alpha and transport delay, same input timing)
    observed through a fitted static quadratic sensor curve; one channel + linear readout.
    Parameters are identified on TRAINING rows, using linearly detrended PCs so that slow
    drift / warm-up does not masquerade as dynamics.  Any product-task score it reaches is
    attributable to static sensor curvature alone."""
    cond, tr_e, _ = bounds
    tt = np.arange(feats[ids[0]].shape[0], dtype=float)
    det = {i: np.column_stack([_detrend(feats[i][cond:tr_e, j], tt[cond:tr_e]) for j in range(feats[i].shape[1])])
           for i in ids}
    Xd = np.vstack([det[i] for i in ids])
    Xd = Xd[np.isfinite(Xd).all(axis=1)]
    sxd = np.where(Xd.std(0) == 0, 1.0, Xd.std(0))
    _, S, Vt = np.linalg.svd(Xd / sxd, full_matrices=False)
    ev = S ** 2 / np.sum(S ** 2)
    pr = (np.sum(S ** 2)) ** 2 / np.sum(S ** 4)
    pcs = {i: (det[i] / sxd) @ Vt[:3].T for i in ids}          # n_train x 3, detrended
    best = (-1.0, 0.0, 0, 0)
    for k in range(3):
        for d in range(0, 4):
            for a in np.linspace(0.0, 0.99, 100):
                xs, ys = [], []
                for i in ids:
                    s_ = integrator_state(u[i], a, uc0, d)[cond:tr_e]
                    s_ = _detrend(s_, tt[cond:tr_e])
                    p_ = pcs[i][:, k]
                    ok = np.isfinite(p_)
                    xs.append(s_[ok])
                    ys.append(p_[ok])
                x, y = np.concatenate(xs), np.concatenate(ys)
                if x.std() == 0:
                    continue
                r = abs(np.corrcoef(x, y)[0, 1])
                if r > best[0]:
                    best = (r, a, d, k)
    _, alpha, delay, k_used = best
    tau_s = -slot_sec / np.log(alpha) if 0 < alpha < 1 else np.nan
    s_all = {i: integrator_state(u[i], alpha, uc0, delay) for i in ids}
    xs = np.concatenate([_detrend(s_all[i][cond:tr_e], tt[cond:tr_e])[np.isfinite(pcs[i][:, k_used])] for i in ids])
    ys = np.concatenate([pcs[i][:, k_used][np.isfinite(pcs[i][:, k_used])] for i in ids])
    coef = np.linalg.lstsq(np.column_stack([np.ones_like(xs), xs, xs ** 2]), ys, rcond=None)[0]
    g_all = {i: coef[0] + coef[1] * s_all[i] + coef[2] * s_all[i] ** 2 for i in ids}
    g_ref = np.concatenate([g_all[i][cond:tr_e] for i in ids])
    gmu, gsd = g_ref.mean(), g_ref.std()
    gfit = np.concatenate([g_all[i][cond:tr_e][np.isfinite(pcs[i][:, k_used])] for i in ids])
    r_fit = min(0.99, abs(np.corrcoef(gfit, ys)[0, 1]))
    quad_share = abs(coef[2]) * np.std(xs ** 2) / (abs(coef[1]) * np.std(xs) + abs(coef[2]) * np.std(xs ** 2) + 1e-12)

    def make(noise):
        out = {}
        for i in ids:
            g = (g_all[i] - gmu) / gsd
            if noise:
                g = r_fit * g + np.sqrt(max(0.0, 1 - r_fit ** 2)) * rng.standard_normal(len(g))
            out[i] = g[:, None]
        return out

    oracle, noisy = {}, {}
    for name, kind, lags in tasks:
        tg = {i: task_target(u[i], kind, lags, uc0) for i in ids}
        r0 = chrono_eval(make(False), tg, bounds, ids, ids)
        oracle[name] = np.mean([r0[i]["cap"] for i in ids if r0[i]]) if any(r0.values()) else np.nan
        vals = []
        for _ in range(n_noise):
            r1 = chrono_eval(make(True), tg, bounds, ids, ids)
            vals += [r1[i]["cap"] for i in ids if r1[i]]
        noisy[name] = np.mean(vals) if vals else np.nan
    info = {"alpha": alpha, "delay": delay, "tau_sec": tau_s, "pc1_corr": r_fit, "pc_used": f"PC{k_used + 1}",
            "pc_var_ratio": ev, "participation_ratio": pr, "quad_share": quad_share}
    return oracle, noisy, info


# --------------------------------------------------------------------------
# 4. Figures
# --------------------------------------------------------------------------

def rc(run_id):
    return RUN_COLORS[(run_id - 1) % len(RUN_COLORS)]


def make_figures(runs, levels, tasks, summ, bounds, preds, loro_agg, base_info, n_common,
                 same_seq, outdir):
    cond = bounds[0]
    nR = len(runs)
    print("Generating figures...")

    # 1 timelines
    fig, axes = plt.subplots(nR, 1, figsize=(16, 3.6 * nR), dpi=130, squeeze=False)
    for ax, r in zip(axes[:, 0], runs):
        w, p = r["wide"], r["pulses"]
        for b in [445, 480, 515, 630]:
            ax.plot(w.index, w[b], color=BAND_COLORS[b], lw=1.0, alpha=0.85, label=f"{b} nm")
        lo, hi = ax.get_ylim()
        ax.vlines(p["start_time"], lo, lo + 0.25 * (hi - lo) * p["symbol"] / levels.max(),
                  color="k", alpha=0.35, lw=0.8)
        h = r["hydraulics"]
        ax.set_title(f"Run {r['run_id']:02d}: {h['n_pulses']} pulses | logged media "
                     f"{h['logged_media_ml_per_min']:.2f} / waste {h['logged_waste_ml_per_min']:.2f} mL/min "
                     f"(calibration-derived)", fontsize=9, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.grid(alpha=0.3)
        if r["run_id"] == 1:
            ax.legend(ncol=4, fontsize=8, loc="upper right")
    fig.suptitle("Fig 1: acquisition timelines (black stems = dose start, height ~ volume)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "01_timelines.png"))
    plt.close(fig)

    # 2 repeatability
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), dpi=130, sharex=True)
    steps = np.arange(cond, n_common)
    for ax, b in zip(axes, [515, 480]):
        M = np.vstack([r["df_states"][f"I_{b}"].values[cond:n_common] for r in runs])
        for r, row in zip(runs, M):
            ax.plot(steps, row, color=rc(r["run_id"]), lw=1.0, alpha=0.75, label=f"Run {r['run_id']:02d}")
        mu = np.nanmean(M, 0)
        sem = np.nanstd(M, 0, ddof=1) / np.sqrt(nR) if nR > 1 else 0 * mu
        ax.plot(steps, mu, "k", lw=1.8, label="mean")
        ax.fill_between(steps, mu - sem, mu + sem, color="k", alpha=0.2)
        ax.set_ylabel(f"{b} nm reading (pre-dose scan)")
        ax.grid(alpha=0.3)
    axes[0].legend(ncol=nR + 1, fontsize=8)
    axes[1].set_xlabel("dose slot n")
    ttl = "identical input sequence -> overlap measures repeatability" if same_seq else "sequences differ"
    fig.suptitle(f"Fig 2: state-trajectory repeatability ({ttl})", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "02_state_repeatability.png"))
    plt.close(fig)

    # 3 pairwise correlations (raw and linearly detrended)
    pairs = [(i, j) for i in range(nR) for j in range(i + 1, nR)]
    if pairs:
        fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.6), dpi=130, squeeze=False)
        for ax, (i, j) in zip(axes[0], pairs):
            x = runs[i]["df_states"]["I_515"].values[cond:n_common]
            y = runs[j]["df_states"]["I_515"].values[cond:n_common]
            ok = np.isfinite(x) & np.isfinite(y)
            t = np.arange(ok.sum())
            dx = x[ok] - np.polyval(np.polyfit(t, x[ok], 1), t)
            dy = y[ok] - np.polyval(np.polyfit(t, y[ok], 1), t)
            ax.scatter(x[ok], y[ok], s=18, alpha=0.6, edgecolors="none")
            ax.set_title(f"Run {runs[i]['run_id']:02d} vs {runs[j]['run_id']:02d}: r={np.corrcoef(x[ok], y[ok])[0, 1]:+.3f}, "
                         f"detrended r={np.corrcoef(dx, dy)[0, 1]:+.3f}", fontsize=9)
            ax.grid(alpha=0.3)
        fig.suptitle("Fig 3: cross-run 515 nm agreement (raw / detrended)", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "03_pairwise_correlations.png"))
        plt.close(fig)

    # 4 capacity vs lag with null + baselines
    lag_rows = summ[summ["task"].str.startswith("lag")]
    lags = np.arange(1, len(lag_rows) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=130)
    ax1.errorbar(lags, lag_rows["cap_mean"], yerr=lag_rows["cap_sem"].fillna(0), fmt="o-", color="#0055d4",
                 capsize=4, lw=2, label="reservoir (pooled chrono test)")
    if "null_p95" in lag_rows:
        ax1.plot(lags, lag_rows["null_p95"], "r--", label="rolled-input null 95th pct")
        ax1.plot(lags, lag_rows["null_mean"], "r:", label="null mean")
    ax1.plot(lags, lag_rows["cap_oracle_1state"], "g-s", ms=4, label="1-state integrator + static sensor curve (oracle)")
    ax1.plot(lags, lag_rows["cap_noisy_1state"], "m-^", ms=4, label="1-state integrator + sensor curve (noise-matched)")
    if "cap_water" in lag_rows and lag_rows["cap_water"].notna().any():
        ax1.plot(lags, lag_rows["cap_water"], "k-d", ms=4, label="water-control readout")
    ax1.set_xlabel("lag k")
    ax1.set_ylabel("capacity = max(0, 1-NMSE)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    ax2.errorbar(lags, lag_rows["NMSE_mean"], yerr=lag_rows["cap_sem"].fillna(0), fmt="s-", color="#e67e22", capsize=4)
    ax2.plot(lags, lag_rows["NMSE_const_mean"], "r--", label="constant predictor (train mean)")
    ax2.set_xlabel("lag k")
    ax2.set_ylabel("test NMSE")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.suptitle("Fig 4: delayed-input reconstruction on unseen final segment (error bars = SEM over runs; "
                 "repeatability only if sequences identical)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "04_memory_capacity_curves.png"))
    plt.close(fig)

    # 5 / 6 reconstruction + residuals (last run)
    rid = runs[-1]["run_id"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=130)
    for ax, k in zip(axes.flat, [1, 2, 3, 4]):
        m = preds.get(f"lag{k}", {}).get(rid)
        if not m:
            continue
        ax.plot(m["y"][:60], "k-o", ms=4, label=f"true u_(n-{k})")
        ax.plot(m["p"][:60], "r--s", ms=3.5, label=f"readout (r={m['r']:.2f}, NMSE={m['NMSE']:.2f})")
        ax.set_yticks(levels)
        ax.set_title(f"lag {k}, Run {rid:02d} final segment", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Fig 5: held-out reconstruction (final chronological segment)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "05_heldout_reconstruction.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
    errs, labs = [], []
    for k in range(1, 7):
        m = preds.get(f"lag{k}", {}).get(rid)
        if m:
            errs.append(m["p"] - m["y"])
            labs.append(f"lag {k}")
    if errs:
        try:
            ax.boxplot(errs, tick_labels=labs, patch_artist=True)
        except TypeError:
            ax.boxplot(errs, labels=labs, patch_artist=True)
    ax.axhline(0, color="gray", ls=":")
    ax.set_ylabel("pred - true (mL)")
    ax.set_title(f"Fig 6: residuals by lag (Run {rid:02d}, final segment)", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "06_residuals.png"))
    plt.close(fig)

    # 7 PCA (descriptive only)
    allA = pd.concat([r["df_states"].iloc[cond:n_common] for r in runs]).dropna(subset=[f"A_{b}" for b in BANDS])
    Z = (allA[[f"A_{b}" for b in BANDS]].values - allA[[f"A_{b}" for b in BANDS]].values.mean(0)) / \
        allA[[f"A_{b}" for b in BANDS]].values.std(0)
    pca = PCA(n_components=min(8, Z.shape[1])).fit(Z)
    T = pca.transform(Z)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), dpi=130)
    a1.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_ * 100)
    a1.set_xlabel("PC")
    a1.set_ylabel("% variance")
    a1.set_title(f"scree (participation ratio {base_info['participation_ratio']:.2f} on training rows)", fontsize=9)
    for r in runs:
        mk = (allA["run_id"] == r["run_id"]).values
        a2.scatter(T[mk, 0], T[mk, 1], s=20, alpha=0.65, color=rc(r["run_id"]), label=f"Run {r['run_id']:02d}")
    a2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    a2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3)
    fig.suptitle("Fig 7: multiband state dimensionality (descriptive; PCA fit on all rows here)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "07_pca_dimensionality.png"))
    plt.close(fig)

    # 8 baselines
    fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
    bw = 0.8 / nR
    x = np.arange(len(BANDS))
    for idx, r in enumerate(runs):
        ax.bar(x + (idx - (nR - 1) / 2) * bw, [r["base_stats"][b]["mean"] for b in BANDS], width=bw,
               yerr=[r["base_stats"][b]["std"] for b in BANDS], color=rc(r["run_id"]), capsize=2,
               label=f"Run {r['run_id']:02d}", edgecolor="k", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}" for b in BANDS])
    ax.set_ylabel("pre-dose baseline reading (mean +/- SD of scans)")
    ax.set_title("Fig 8: pre-dose clear-water baseline", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "08_baselines.png"))
    plt.close(fig)

    # 9 nonlinear tasks
    nl = summ[summ["task"].str.startswith("prod")]
    if len(nl):
        fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
        x = np.arange(len(nl))
        w = 0.2
        ax.bar(x - 1.5 * w, nl["cap_mean"], w, yerr=nl["cap_sem"].fillna(0), label="reservoir (linear readout)", color="#0055d4")
        ax.bar(x - 0.5 * w, nl["cap_oracle_1state"], w, label="1-state + sensor curve (oracle)", color="g")
        ax.bar(x + 0.5 * w, nl["cap_noisy_1state"], w, label="1-state + sensor curve (noise-matched)", color="m")
        if "null_p95" in nl:
            ax.bar(x + 1.5 * w, nl["null_p95"], w, label="rolled-input null 95th pct", color="r", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(nl["task"])
        ax.set_ylabel("capacity")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_title("Fig 9: product-of-centred-inputs tasks vs linear 1-state + static sensor curve", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "09_nonlinear_tasks.png"))
        plt.close(fig)


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Protocol 6 replicate analysis (v2)")
    ap.add_argument("--data-root", default=os.path.join("data", "250 timesteps"))
    ap.add_argument("--run-glob", default="run *")
    ap.add_argument("--water-root", default=None, help="optional matched water-control runs (same folder layout)")
    ap.add_argument("--outdir", default=os.path.join("outputs", "protocol6_analysis_v2"))
    ap.add_argument("--levels", default="0.20,0.35,0.50", help="design alphabet, mL")
    ap.add_argument("--expected-pulses", type=int, default=250)
    ap.add_argument("--vial-ml", type=float, default=10.0, help="nominal working volume for V/Q (protocol: 10 mL)")
    ap.add_argument("--cond-slots", type=int, default=None,
                    help="conditioning slots to discard (default: max(protocol fraction, ~5 tau_expected))")
    ap.add_argument("--interval-sec", type=float, default=30.0,
                    help="planned interval; real exports show END-to-START semantics (start-to-start = this + pulse time)")
    ap.add_argument("--alt-source", default="specified_pump_dosing:alt_media")
    ap.add_argument("--gap-sec", type=float, default=5.0, help="max gap between sub-doses of ONE pulse")
    ap.add_argument("--scan-gap-sec", type=float, default=2.0, help="max gap between band rows of ONE scan")
    ap.add_argument("--avg-scans", type=int, default=1, help="causal trailing mean of last K scans")
    ap.add_argument("--baseline-window-min", type=float, default=10.0)
    ap.add_argument("--baseline-guard-sec", type=float, default=30.0)
    ap.add_argument("--min-tail-min", type=float, default=10.0)
    ap.add_argument("--saturation-level", type=float, default=None)
    ap.add_argument("--max-lag", type=int, default=8)
    ap.add_argument("--n-null", type=int, default=200, help="rolled-input null draws (0 = skip)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    levels = np.array(sorted(float(x) for x in args.levels.split(",")))
    uc0 = float(levels.mean())                       # centre = design-alphabet mean

    runs = load_runs(args.data_root, args, levels)
    ids = [r["run_id"] for r in runs]
    n_common = min(len(r["df_states"]) for r in runs)
    bounds = split_bounds(n_common)
    tau_exp = float(np.nanmedian([r["hydraulics"]["tau_expected_sec_logged"] for r in runs]))
    slot_sec = float(np.nanmedian([r["hydraulics"]["median_interval_sec"] for r in runs]))
    cond_auto = int(np.ceil(5 * tau_exp / slot_sec)) if np.isfinite(tau_exp) else 0
    cond_use = args.cond_slots if args.cond_slots is not None else max(bounds[0], cond_auto)
    if cond_use != bounds[0]:
        bounds = (cond_use, bounds[1], bounds[2])
        print(f"Conditioning extended {split_bounds(n_common)[0]} -> {cond_use} slots (~5 x tau_expected {tau_exp:.0f}s / slot {slot_sec:.1f}s)")
    cond, tr_e, va_e = bounds
    print(f"\nCommon slots: {n_common} | split: cond [0,{cond}) train [{cond},{tr_e}) "
          f"val [{tr_e},{va_e}) test [{va_e},{n_common})")

    feats = {r["run_id"]: r["df_states"][[f"A_{b}" for b in BANDS]].values[:n_common] for r in runs}
    u = {r["run_id"]: r["df_states"]["symbol"].values[:n_common] for r in runs}
    same_seq = sequences_identical(u)
    if same_seq:
        flag("All runs executed the IDENTICAL input sequence. Cross-run agreement = repeatability, and "
             "leave-one-run-out is a SAME-SEQUENCE transfer test, not generalisation to new inputs. "
             "Protocol 6C asks for independent sequences for repeat/test runs.")
    else:
        print("Input sequences differ between runs (good for LORO generalisation).")

    tasks = build_tasks(args.max_lag)
    lag_tasks = [t for t in tasks if t[1] == "lag"]
    n_test = n_common - va_e
    print(f"Test segment ~{n_test} slots -> chance-level |r| ~ {1 / np.sqrt(max(n_test, 1)):.2f}; "
          f"expect small lags only to be resolvable.")

    # ---- primary: pooled chronological (train early segments of all runs -> test final segment)
    print("\nEvaluating pooled chronological readouts ...")
    preds_keep = {f"lag{k}" for k in range(1, 7)}
    res = pooled_table(feats, u, tasks, bounds, uc0, ids, keep_preds_for=preds_keep)

    # ---- intra-run chronological (each run alone; original v1 analysis, corrected)
    intra_rows = []
    for rid in ids:
        for name, kind, lags in lag_tasks:
            tg = {rid: task_target(u[rid], kind, lags, uc0)}
            m = chrono_eval({rid: feats[rid]}, tg, bounds, [rid], [rid])[rid]
            if m:
                intra_rows.append({"run_id": rid, "task": name, **{k: m[k] for k in
                                   ["alpha", "cap", "r", "r2", "NMSE", "NMSE_const", "n_test"]}})
    df_intra = pd.DataFrame(intra_rows)

    # ---- LORO
    loro_rows = []
    for name, kind, lags in lag_tasks:
        tg = {i: task_target(u[i], kind, lags, uc0) for i in ids}
        for test, m in loro_eval(feats, tg, cond, ids).items():
            if m:
                loro_rows.append({"test_run": test, "task": name, **{k: m[k] for k in
                                  ["alpha", "cap", "r", "r2", "NMSE", "NMSE_const", "n_test"]}})
    df_loro = pd.DataFrame(loro_rows)
    loro_agg = (df_loro.groupby("task").agg(cap_mean=("cap", "mean"), cap_sem=("cap", "sem"),
                                            r_mean=("r", "mean"), NMSE_mean=("NMSE", "mean")).reset_index()
                if len(df_loro) else pd.DataFrame())

    # ---- baselines
    print("Fitting single-state leaky-integrator baselines ...")
    oracle, noisy, base_info = integrator_baselines(feats, u, ids, bounds, uc0, tasks, rng, slot_sec)

    # ---- null
    null, null_skill = None, None
    if args.n_null > 0:
        print(f"Rolled-input null ({args.n_null} draws) ...")
        null, null_skill = null_caps(feats, u, tasks, bounds, uc0, ids, same_seq, rng, args.n_null)

    # ---- optional water control
    water_caps = {}
    if args.water_root:
        print("\nWater-control runs:")
        wruns = load_runs(args.water_root, args, levels)
        wn = min(min(len(r["df_states"]) for r in wruns), n_common)
        wids = [r["run_id"] for r in wruns]
        wf = {r["run_id"]: r["df_states"][[f"A_{b}" for b in BANDS]].values[:wn] for r in wruns}
        wu = {r["run_id"]: r["df_states"]["symbol"].values[:wn] for r in wruns}
        wres = pooled_table(wf, wu, tasks, split_bounds(wn), uc0, wids)
        for name in wres:
            v = [m["cap"] for m in wres[name].values() if m]
            water_caps[name] = np.mean(v) if v else np.nan

    # ---- summary table
    rows = []
    for ti, (name, kind, lags) in enumerate(tasks):
        per = [m for m in res[name].values() if m]
        if not per:
            continue
        caps = np.array([m["cap"] for m in per])
        row = {"task": name, "n_runs": len(per), "cap_mean": caps.mean(),
               "cap_sd": caps.std(ddof=1) if len(per) > 1 else np.nan,
               "cap_sem": caps.std(ddof=1) / np.sqrt(len(per)) if len(per) > 1 else np.nan,
               "skill_mean": np.mean([m["skill"] for m in per]), "r_mean": np.mean([m["r"] for m in per]), "r2_mean": np.mean([m["r2"] for m in per]),
               "NMSE_mean": np.mean([m["NMSE"] for m in per]),
               "NMSE_const_mean": np.mean([m["NMSE_const"] for m in per]),
               "alpha": per[0]["alpha"], "n_test_each": per[0]["n_test"],
               "cap_oracle_1state": oracle[name], "cap_noisy_1state": noisy[name]}
        if null is not None:
            nv = null[:, ti]
            nv = nv[np.isfinite(nv)]
            row["null_mean"] = nv.mean()
            row["null_p95"] = np.percentile(nv, 95)
            ns = null_skill[:, ti]
            ns = ns[np.isfinite(ns)]
            row["null_skill_p95"] = np.percentile(ns, 95)
            row["p_perm"] = (1 + np.sum(ns >= row["skill_mean"])) / (1 + len(ns))
        if water_caps:
            row["cap_water"] = water_caps.get(name, np.nan)
        rows.append(row)
    summ = pd.DataFrame(rows)
    if null is not None:
        summ["p_BH"] = bh_adjust(summ["p_perm"].values)

    # ---- export
    pd.DataFrame([r["hydraulics"] for r in runs]).to_csv(os.path.join(args.outdir, "run_audit.csv"), index=False)
    pd.concat([r["pulses"].assign(run_id=r["run_id"]) for r in runs]).to_csv(
        os.path.join(args.outdir, "pulses_all_runs.csv"), index=False)
    pd.concat([r["df_states"].assign(slot=r["df_states"].index) for r in runs]).to_csv(
        os.path.join(args.outdir, "states_all_runs.csv"), index=False)
    summ.to_csv(os.path.join(args.outdir, "pooled_chrono_summary.csv"), index=False)
    df_intra.to_csv(os.path.join(args.outdir, "intra_run_results.csv"), index=False)
    df_loro.to_csv(os.path.join(args.outdir, "loro_folds.csv"), index=False)
    if null is not None:
        pd.DataFrame(null_skill, columns=[t[0] for t in tasks]).to_csv(os.path.join(args.outdir, "null_skill_draws.csv"), index=False)

    make_figures(runs, levels, tasks, summ, bounds, {k: v for k, v in res.items() if k in preds_keep},
                 loro_agg, base_info, n_common, same_seq, args.outdir)

    # ---- verdict
    line = "=" * 80
    print("\n" + line + "\n PROTOCOL 6 - REPLICATE ANALYSIS (v2)\n" + line)
    print("\n[1] RUN AUDIT")
    show = ["run_id", "n_pulses", "n_missing_slots", "median_interval_sec", "max_interval_sec",
            "pre_dose_recording_min", "tail_min", "scan_cadence_sec", "n_data_gaps", "max_dt_before_sec"]
    print(pd.DataFrame([r["hydraulics"] for r in runs])[show].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"  identical input sequence across runs: {same_seq}")

    print(f"\n[2] STATE DIMENSIONALITY (training rows): PC variance ratios "
          f"{np.round(base_info['pc_var_ratio'][:4], 3)}, participation ratio {base_info['participation_ratio']:.2f} of 8")
    print(f"    fitted integrator (delay {base_info['delay']} slot(s)): alpha={base_info['alpha']:.2f} -> tau ~ {base_info['tau_sec']:.0f}s "
          f"(fit corr {base_info['pc1_corr']:.2f} on {base_info['pc_used']}, quadratic share {base_info['quad_share']:.2f}); expected V/Q from LOGGED inflow ~ {tau_exp:.0f}s")

    print("\n[3] POOLED CHRONOLOGICAL TEST (capacity = max(0,1-NMSE); skill = unclipped 1-NMSE, used for p-values; SEM over runs)")
    cols = ["task", "cap_mean", "skill_mean", "cap_sem", "r_mean", "NMSE_mean", "NMSE_const_mean",
            "cap_oracle_1state", "cap_noisy_1state"]
    if null is not None:
        cols += ["null_p95", "p_perm", "p_BH"]
    if water_caps:
        cols += ["cap_water"]
    print(summ[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    lag_s = summ[summ["task"].str.startswith("lag")]
    tot = lag_s["cap_mean"].sum()
    msg = f"    sum of lag capacities: {tot:.3f}"
    if null is not None:
        nsum = np.nansum(null[:, [i for i, t in enumerate(tasks) if t[1] == "lag"]], axis=1)
        msg += f" | null sum mean {nsum.mean():.3f}, 95th pct {np.percentile(nsum, 95):.3f} -> excess {tot - nsum.mean():+.3f}"
        sig = summ[(summ["p_BH"] < 0.05)]["task"].tolist()
        msg += f"\n    tasks significant after BH-FDR (q<0.05): {sig if sig else 'none'}"
    print(msg)
    if null is not None:
        print(f"    (p from unclipped skill vs rolled null; min attainable p = {1 / (len(ns) + 1):.4f}; BH over {len(summ)} tasks)")

    print("\n[4] INTRA-RUN CHRONOLOGICAL (each run alone)")
    if len(df_intra):
        tot_i = df_intra.groupby("run_id")["cap"].sum()
        print(df_intra[df_intra["task"].isin(["lag1", "lag2", "lag3"])]
              .pivot(index="run_id", columns="task", values="cap").to_string(float_format=lambda x: f"{x:.3f}"))
        print(f"    total capacity per run: { {int(k): round(float(v), 3) for k, v in tot_i.items()} }")

    print("\n[5] LEAVE-ONE-RUN-OUT" + ("  ** SAME-SEQUENCE TRANSFER, NOT GENERALISATION **" if same_seq else ""))
    if len(loro_agg):
        print(loro_agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n[6] FLAGS RAISED")
    if FLAGS:
        for f in FLAGS:
            print("  - " + f)
    else:
        print("  none")
    print("\nInterpretation guardrails:")
    print("  * SEM over runs reflects physical/sensor repeatability; with one shared input sequence it is NOT")
    print("    an estimate of variability across inputs. Independent sequences are needed for that.")
    print("  * Capacities on ~%d test slots are noisy; trust the null-referenced excess, not raw sums." % n_test)
    print("  * Nonlinear-task score only counts if it beats BOTH the rolled null and the 1-state + sensor-curve baseline")
    print("    (one linear state seen through a static sensor curve). The curve here is fitted from the data; the")
    print("    Protocol 3 calibrated curve is the stronger version of that check.")
    print("  * Water-control readout: " + ("included" if water_caps else "NOT run (pass --water-root)"))
    print(line)
    print(f"Outputs in: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
