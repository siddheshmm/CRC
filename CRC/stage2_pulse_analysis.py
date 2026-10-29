#!/usr/bin/env python3
"""
stage2_pulse_analysis.py – pulse-response analysis for Stage 2 (acid and base blocks)
==============================================================================================

Pulse TYPE comes from the logged dosing event, not from the A/B letter in the folder name:
    add_media -> acid pulse,  add_alt_media -> base pulse     (swap with --acid-event / --base-event if the bottles are swapped)
so it does not matter which block label you gave which reagent.  The script checks the direction of every response
(acid should lower 555/480, base should raise it) and warns if a run disagrees with its logged type.

Reads the Pioreactor exports of every Stage 2 run, finds each pulse from `dosing_events`, and measures the
response of the 555/480 ratio (plus 555/445, 555/680, 630/555, OD 45° and OD 135°) to it:

    peak size (dF) · time to peak · half-recovery time · decay time constant · recovery before the next pulse ·
    area under the response · pH shift (via the ladder sigmoid)

then pools the pulses by volume to give the gain curve, timing, run-to-run reproducibility, carry-over between
pulses and the numbers Stage 3 needs (v_max, level size).

Input layout (exactly as you exported; folders or .zip, names parsed for condition/run/time)
    <root>/stage2_A1_run01_1305-1330_IST/            (or the same name as .zip)
        od_readings/od_readings-....csv
        as7341_spectrum_readings/as7341_spectrum_readings-....csv
        dosing_events/dosing_events-....csv

Usage
    python stage2_pulse_analysis.py ~/Downloads/stage02
    python stage2_pulse_analysis.py ~/Downloads/stage02 --only A          # only the runs labelled A1..A3
    python stage2_pulse_analysis.py ~/Downloads/stage02 --only-type base  # only runs whose pulses are base
    python stage2_pulse_analysis.py ~/Downloads/stage02 --inspect         # show dosing-event columns/values and exit
    python stage2_pulse_analysis.py ~/Downloads/stage02 --acid-event add_media --base-event add_alt_media

Output  <out>/figures/*.png  <out>/tables/*.csv  summary.json  summary.txt  inputs_manifest.csv  results zip
All clock times in the plots are IST (UTC + 5:30); the analysis itself aligns on the UTC `timestamp` columns.
Requires: numpy, pandas, scipy, matplotlib
"""
import argparse
import hashlib
import json
import platform
import re
import sys
import tempfile
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=RuntimeWarning)
SCRIPT_VERSION = "1.0"
_trapz = getattr(np, "trapezoid", None) or np.trapz     # numpy 2.x renamed trapz

# ------------------------------------------------------------------ configuration
FEATURE = "555/480"
OTHER_FEATURES = ["555/445", "555/680", "630/555"]
CALIB = dict(ya=1.3586, yb=1.6198, pka=3.741)     # ladder sigmoid for 555/480 (2026-10-22)
LIMITS = dict(acid_floor=1.40, base_ceiling=1.58)  # keep pulses inside the transition
PLAN = {"A1": [0.05, 0.10, 0.20], "A2": [0.20, 0.10, 0.05], "A3": [0.10, 0.05, 0.20],
        "B1": [0.05, 0.10, 0.20], "B2": [0.20, 0.10, 0.05], "B3": [0.10, 0.05, 0.20]}
RUN_RE = re.compile(r"(?:.*_)?([AB]\d)_run(\d+)_(\d{4})-(\d{4})_IST", re.I)
IST = pd.Timedelta(hours=5, minutes=30)
BANDS = [415, 445, 480, 515, 555, 590, 630, 680]

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11,
                     "axes.titleweight": "bold", "legend.frameon": False})
VOL_COLORS = {0.05: "#9ecae1", 0.10: "#4292c6", 0.20: "#08306b"}


def vcol(v):
    return VOL_COLORS.get(round(v, 2), "0.4")


# ------------------------------------------------------------------ loading
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def discover_runs(root):
    root = Path(root)
    runs = []
    for p in sorted(root.iterdir()):
        m = RUN_RE.search(p.name)
        if m and (p.is_dir() or p.suffix.lower() == ".zip"):
            runs.append(dict(path=p, cond=m.group(1).upper(), run=int(m.group(2)), hhmm=f"{m.group(3)}-{m.group(4)}",
                             name=f"{m.group(1).upper()}_run{int(m.group(2)):02d}"))
    return runs


def read_run_csvs(path):
    """Return {kind: DataFrame} for as7341 / od / dosing plus the list of files used (for the manifest)."""
    path = Path(path)
    kinds = {"as7341": [], "od_readings": [], "dosing_events": []}
    files = []
    if path.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp)
        base = tmp
    else:
        base = path
    for f in sorted(base.rglob("*.csv")):
        for k in kinds:
            if k in f.name.lower() or k in str(f.parent.name).lower():
                kinds[k].append(pd.read_csv(f))
                files.append(f)
                break
    out = {k: (pd.concat(v, ignore_index=True) if v else None) for k, v in kinds.items()}
    return out, files


def utc_naive(s):
    return pd.to_datetime(s, utc=True).dt.tz_localize(None)


def to_cycles(df, key, value):
    df = df.copy()
    df["t"] = utc_naive(df["timestamp"])
    df = df.sort_values("t")
    cyc = (df["t"].diff() > pd.Timedelta("1s")).cumsum()
    w = df.assign(c=cyc).pivot_table(index="c", columns=key, values=value, aggfunc="mean")
    w.index = pd.DatetimeIndex(df.assign(c=cyc).groupby("c")["t"].first().values)
    w.columns.name = None
    return w.dropna()


def feature_series(spec):
    out = {}
    for f in [FEATURE] + OTHER_FEATURES:
        a, b = (int(x) for x in f.split("/"))
        out[f] = spec[a] / spec[b]
    return pd.DataFrame(out, index=spec.index)


def parse_dosing(df, acid_event, base_event, max_pulse_ml):
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    tcol = low.get("timestamp") or next((c for c in cols if "time" in c.lower()), None)
    evcol = low.get("event") or next((c for c in cols if "event" in c.lower() and "source" not in c.lower()), None)
    vcol_ = next((c for c in cols if "volume" in c.lower()), None)
    if not (tcol and evcol and vcol_):
        raise ValueError(f"cannot find time/event/volume columns in dosing_events: {cols}")
    ev = pd.DataFrame({"t": utc_naive(df[tcol]), "event": df[evcol].astype(str).str.lower(),
                       "vol": pd.to_numeric(df[vcol_], errors="coerce")}).dropna().sort_values("t")
    sel = ev[ev.event.isin([acid_event, base_event]) & (ev.vol.abs() > 0) & (ev.vol.abs() <= max_pulse_ml)].copy()
    sel["type"] = np.where(sel.event == acid_event, "acid", "base")
    sel["vol"] = sel.vol.abs()
    # merge duplicate log lines of the same pulse (within 2 s, same type)
    rows = []
    prev_t = None
    for _, r in sel.iterrows():
        if rows and rows[-1]["type"] == r["type"] and prev_t is not None and (r["t"] - prev_t).total_seconds() < 5.0:
            rows[-1]["vol"] += r["vol"]
        else:
            rows.append(dict(t=r["t"], vol=r["vol"], type=r["type"]))
        prev_t = r["t"]
    return pd.DataFrame(rows), ev


# ------------------------------------------------------------------ pulse analysis
def sigmoid_ph(y):
    y = float(np.clip(y, CALIB["ya"] + 0.002, CALIB["yb"] - 0.002))
    return CALIB["pka"] + np.log10((y - CALIB["ya"]) / (CALIB["yb"] - y))


def mad_sd(x):
    x = np.asarray(x, float)
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def decay_fit(tt_min, dd):
    """dd(t) = c + (a - c)*exp(-t/tau); returns tau, se, c, r2 (nan if the fit fails)."""
    if len(tt_min) < 15:
        return (np.nan,) * 4
    f = lambda t, a, c, tau: c + (a - c) * np.exp(-t / tau)
    try:
        p, cov = curve_fit(f, tt_min, dd, p0=[dd[0], 0.0, 2.0], bounds=([-1, -1, 0.2], [1, 1, 40]), maxfev=20000)
        res = dd - f(tt_min, *p)
        r2 = 1 - np.sum(res ** 2) / np.sum((dd - dd.mean()) ** 2)
        return p[2], float(np.sqrt(cov[2, 2])), p[1], r2
    except Exception:
        return (np.nan,) * 4


def analyse_pulse(t, y, t_p, t_next, ptype, cfg):
    """t: epoch seconds (np), y: feature values; t_p / t_next epoch seconds of this / next pulse (or None)."""
    sgn = -1.0 if ptype == "acid" else 1.0
    pre = (t >= t_p - 60) & (t < t_p - 5)
    pre_long = (t >= t_p - 150) & (t < t_p - 5)
    if pre.sum() < 6:
        return None
    F0 = float(np.median(y[pre]))
    sig = float(mad_sd(y[pre_long]))
    t_end = t_p + cfg["max_window_s"]
    if t_next is not None:
        t_end = min(t_end, t_next - 5)
    post = (t >= t_p) & (t <= t_end)
    if post.sum() < 8:
        return None
    tp, yp = t[post], y[post]
    ys = pd.Series(yp).rolling(3, center=True, min_periods=1).median().values
    d = ys - F0
    search = (tp - t_p) <= cfg["peak_window_s"]
    idx = int(np.argmin(d[search])) if sgn < 0 else int(np.argmax(d[search]))
    amp = float(d[idx])
    onset_thr = max(5 * sig, 0.003)
    on = np.where(np.abs(d) > onset_thr)[0]
    half = np.where(np.abs(d[: idx + 1]) >= 0.5 * abs(amp))[0]
    after = np.arange(idx, len(d))
    h_dec = after[np.abs(d[after]) <= 0.5 * abs(amp)]
    n_dec = after[np.abs(d[after]) <= 0.1 * abs(amp)]
    tt = (tp[idx:] - tp[idx]) / 60
    tau, tau_se, c_inf, r2 = decay_fit(tt, (yp - F0)[idx:])
    last = tp >= tp[-1] - 30
    d_end = float(np.median(yp[last]) - F0)
    return dict(F0=F0, sigma0=sig, amp=amp, F_peak=F0 + amp, direction_ok=bool(np.sign(amp) == sgn),
                t_onset_s=float(tp[on[0]] - t_p) if len(on) else np.nan, t_peak_s=float(tp[idx] - t_p),
                t_half_rise_s=float(tp[half[0]] - t_p) if len(half) else np.nan,
                t_half_decay_min=float((tp[h_dec[0]] - tp[idx]) / 60) if len(h_dec) else np.nan,
                t_90_recovery_min=float((tp[n_dec[0]] - tp[idx]) / 60) if len(n_dec) else np.nan,
                tau_fit_min=tau, tau_fit_se=tau_se, decay_asymptote=c_inf, decay_r2=r2,
                window_s=float(tp[-1] - t_p), recovered_fraction=float(1 - d_end / amp) if amp else np.nan,
                end_offset=d_end, area_ratio_min=float(_trapz(np.abs(d), (tp - t_p) / 60)),
                pH_before=sigmoid_ph(F0), pH_peak=sigmoid_ph(F0 + amp), n_points=int(post.sum()))


def secondary_at_peak(tp_epoch, feat_df, od_df, t_p, t_peak_s, ptype, cfg):
    """Change of the other features / OD channels around the primary peak time and at their own extreme."""
    out = {}
    tt = feat_df.index.values.astype("datetime64[ns]").astype("int64") / 1e9
    pre = (tt >= t_p - 60) & (tt < t_p - 5)
    win = (tt >= t_p) & (tt <= t_p + cfg["peak_window_s"])
    near = (tt >= t_p + t_peak_s - 7.5) & (tt <= t_p + t_peak_s + 7.5)
    for f in OTHER_FEATURES:
        y = feat_df[f].values
        if pre.sum() < 6 or near.sum() == 0:
            continue
        base = np.median(y[pre])
        out[f"d_{f}"] = float(np.median(y[near]) - base)
        out[f"sigma_{f}"] = float(mad_sd(y[(tt >= t_p - 150) & (tt < t_p - 5)]))
    if od_df is not None and len(od_df):
        to = od_df.index.values.astype("datetime64[ns]").astype("int64") / 1e9
        pre_o = (to >= t_p - 60) & (to < t_p - 5)
        win_o = (to >= t_p) & (to <= t_p + cfg["peak_window_s"] + 120)
        for col in od_df.columns:
            if pre_o.sum() >= 6 and win_o.sum() >= 3:
                base = np.median(od_df[col].values[pre_o])
                dv = od_df[col].values[win_o] - base
                out[f"d_{col}"] = float(dv[np.argmax(np.abs(dv))])
                out[f"sigma_{col}"] = float(mad_sd(od_df[col].values[(to >= t_p - 150) & (to < t_p - 5)]))
    return out


# ------------------------------------------------------------------ main analysis
def analyse_run(run, args, cfg):
    data, files = read_run_csvs(run["path"])
    if data["as7341"] is None or data["dosing_events"] is None:
        raise ValueError(f"{run['name']}: missing as7341 or dosing_events csv")
    spec = to_cycles(data["as7341"], "band", "reading")
    feats = feature_series(spec)
    od = None
    if data["od_readings"] is not None:
        od = to_cycles(data["od_readings"], "angle", "od_reading")
        od.columns = [f"OD{int(c)}" for c in od.columns]
        od = od.loc[:, od.std() > 0]
    pulses, ev_all = parse_dosing(data["dosing_events"], args.acid_event, args.base_event, args.max_pulse_ml)
    if args.inspect:
        print(f"\n=== {run['name']}  dosing_events columns: {list(data['dosing_events'].columns)}")
        print(data["dosing_events"].head(8).to_string())
        print("events seen:", ev_all.event.value_counts().to_dict())
        return None, files
    t = feats.index.values.astype("datetime64[ns]").astype("int64") / 1e9
    y = feats[FEATURE].values
    rows = []
    for k, pr in pulses.reset_index(drop=True).iterrows():
        tp = pr["t"].to_datetime64().astype("datetime64[ns]").astype("int64") / 1e9
        tn = None
        if k + 1 < len(pulses):
            tn = pulses.iloc[k + 1]["t"].to_datetime64().astype("datetime64[ns]").astype("int64") / 1e9
        a = analyse_pulse(t, y, tp, tn, pr["type"], cfg)
        if a is None:
            print(f"  ! {run['name']} pulse {k + 1}: not enough data around it")
            continue
        plan = PLAN.get(run["cond"], [None] * 3)
        plan_v = plan[k] if k < len(plan) else None
        a.update(secondary_at_peak(t, feats, od, tp, a["t_peak_s"], pr["type"], cfg))
        a.update(run=run["name"], cond=run["cond"], run_no=run["run"], order=k + 1, type=pr["type"],
                 volume=round(float(pr["vol"]), 3), planned_volume=plan_v,
                 plan_match=(plan_v is not None and abs(plan_v - pr["vol"]) < 0.011),
                 t_pulse_utc=str(pr["t"]), t_pulse_ist=str(pr["t"] + IST), seconds_to_next_pulse=(tn - tp) if tn else np.nan)
        rows.append(a)
    return dict(run=run, feats=feats, od=od, pulses=pulses, rows=rows), files


def pooled_tables(P, cfg):
    out = {}
    for ptype, g in P.groupby("type"):
        g = g.copy()
        g["vol_level"] = g["volume"].round(2)
        agg = g.groupby("vol_level").agg(
            n=("amp", "size"), amp_mean=("amp", "mean"), amp_sd=("amp", "std"), dpH_mean=("dpH", "mean"), dpH_sd=("dpH", "std"),
            peak_ratio_mean=("F_peak", "mean"), t_peak_s_mean=("t_peak_s", "mean"),
            t_half_decay_min_mean=("t_half_decay_min", "mean"), t_half_decay_min_sd=("t_half_decay_min", "std"),
            tau_fit_min_median=("tau_fit_min", "median"), recovered_fraction_mean=("recovered_fraction", "mean"),
            area_mean=("area_ratio_min", "mean"), sigma0_mean=("sigma0", "mean")).reset_index()
        agg["gain_per_mL"] = agg["amp_mean"].abs() / agg["vol_level"]
        agg["amp_over_5sigma"] = agg["amp_mean"].abs() / (5 * agg["sigma0_mean"])
        out[ptype] = agg
    return out


def gain_models(g):
    """linear through origin and power-law exponent for one pulse type."""
    v, a = g["volume"].values, np.abs(g["amp"].values)
    slope = float(np.sum(v * a) / np.sum(v ** 2))
    pred = slope * v
    r2 = 1 - np.sum((a - pred) ** 2) / np.sum((a - a.mean()) ** 2) if len(a) > 2 else np.nan
    lr = stats.linregress(np.log(v), np.log(a)) if len(v) > 3 and np.all(a > 0) else None
    return dict(slope_per_mL=slope, r2_through_origin=float(r2),
                exponent=float(lr.slope) if lr else np.nan, exponent_se=float(lr.stderr) if lr else np.nan,
                exponent_p_vs_1=float(2 * (1 - stats.t.cdf(abs((lr.slope - 1) / lr.stderr), len(v) - 2))) if lr else np.nan)


def reproducibility(P):
    rows = []
    for (cond, order), g in P.groupby(["cond", "order"]):
        if g.run_no.nunique() >= 2:
            r1, r2 = g[g.run_no == g.run_no.min()].iloc[0], g[g.run_no == g.run_no.max()].iloc[0]
            rows.append(dict(cond=cond, order=order, volume=r1.volume, volume_run2=r2.volume, amp_run1=r1.amp, amp_run2=r2.amp,
                             diff=r2.amp - r1.amp, rel_diff_pct=100 * (r2.amp - r1.amp) / np.mean([r1.amp, r2.amp])))
    return pd.DataFrame(rows)


def carryover(P):
    out = {}
    for ptype, g in P.groupby("type"):
        if len(g) < 6:
            continue
        gain = np.abs(g.amp.values) / g.volume.values
        X = np.column_stack([np.ones(len(g)), g.F0.values - g.F0.mean()])
        beta, res, rk, sv = np.linalg.lstsq(X, gain, rcond=None)
        resid = gain - X @ beta
        s2 = np.sum(resid ** 2) / (len(gain) - 2)
        se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
        tstat = beta[1] / se
        out[ptype] = dict(gain_vs_F0_slope=float(beta[1]), se=float(se), p=float(2 * (1 - stats.t.cdf(abs(tstat), len(gain) - 2))),
                          note="per-mL gain regressed on the pre-pulse baseline level (sigmoid curvature / incomplete recovery)")
    return out


def stage3_suggestions(P, tables, args):
    out = {}
    for ptype, g in P.groupby("type"):
        tb = tables[ptype]
        lim = LIMITS["acid_floor"] if ptype == "acid" else LIMITS["base_ceiling"]
        ok = []
        for _, r in tb.iterrows():
            sub = g[g.volume.round(2) == r.vol_level]
            worst = sub.F_peak.min() if ptype == "acid" else sub.F_peak.max()
            ok.append(dict(volume=float(r.vol_level), worst_peak_ratio=float(worst),
                           inside_limit=bool(worst >= lim if ptype == "acid" else worst <= lim),
                           mean_amp=float(r.amp_mean), per_level_if_4_levels=float(abs(r.amp_mean) / 4),
                           level_step_over_5sigma=float(abs(r.amp_mean) / 4 / (5 * r.sigma0_mean))))
        usable = [o for o in ok if o["inside_limit"]]
        out[ptype] = dict(limit=lim, per_volume=ok, largest_volume_inside_limit=max([o["volume"] for o in usable]) if usable else None,
                          note="a level step (v_max/4) should move the ratio by >= 5 sigma; sigma here is the per-cycle baseline noise")
    return out


# ------------------------------------------------------------------ figures
def fig_overview(results, ptype, figdir):
    runs = [r for r in results if r["rows"] and r["rows"][0]["type"] == ptype]
    if not runs:
        return
    conds = sorted(set(r["run"]["cond"] for r in runs))
    fig, axes = plt.subplots(len(conds), 2, figsize=(16, 3.6 * len(conds)), squeeze=False)
    for r in runs:
        i, j = conds.index(r["run"]["cond"]), (r["run"]["run"] - 1) % 2
        ax = axes[i, j]
        f = r["feats"][FEATURE]
        pul = r["pulses"]
        t0, t1 = pul["t"].min() - pd.Timedelta(minutes=4), pul["t"].max() + pd.Timedelta(minutes=12)
        s = f.loc[t0:t1]
        ax.plot(s.index + IST, s.values, ".", ms=2.5, color="0.4")
        ax.plot(s.index + IST, s.rolling(6, center=True, min_periods=2).median(), color="k", lw=1)
        ax.axhline(LIMITS["acid_floor" if ptype == "acid" else "base_ceiling"], color="tab:red", ls="--", lw=0.8)
        for k, p in pul.reset_index(drop=True).iterrows():
            ax.axvline(p["t"] + IST, color=vcol(p["vol"]), lw=1.6)
            ax.text(p["t"] + IST, ax.get_ylim()[1], f" {p['vol']:.2f} mL", rotation=90, va="top", fontsize=8, color=vcol(p["vol"]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_title(f"{r['run']['name']}  ({r['run']['hhmm']} IST)")
        ax.set_ylabel(FEATURE)
    fig.suptitle(f"{ptype.capitalize()} pulses – ratio over time (red dashed = limit {('1.40' if ptype == 'acid' else '1.58')})",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(figdir / f"fig1_overview_{ptype}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_aligned(results, P, figdir, cfg):
    types = sorted(P["type"].unique())
    fig, axes = plt.subplots(1, len(types), figsize=(8.5 * len(types), 5.5), squeeze=False)
    for ax, ptype in zip(axes[0], types):
        curves = {}
        for r in results:
            if not r["rows"]:
                continue
            f = r["feats"][FEATURE]
            tt = f.index.values.astype("datetime64[ns]").astype("int64") / 1e9
            for row in r["rows"]:
                if row["type"] != ptype:
                    continue
                tp = pd.Timestamp(row["t_pulse_utc"]).to_datetime64().astype("datetime64[ns]").astype("int64") / 1e9
                m = (tt >= tp - 60) & (tt <= tp + min(cfg["max_window_s"], row["window_s"] + 5))
                x = (tt[m] - tp) / 60
                yv = f.values[m] - row["F0"]
                ax.plot(x, yv, color=vcol(row["volume"]), alpha=0.35, lw=0.9)
                curves.setdefault(round(row["volume"], 2), []).append((x, yv))
        grid = np.linspace(-1, 8, 181)
        for v, cs in sorted(curves.items()):
            interp = [np.interp(grid, x, yv, left=np.nan, right=np.nan) for x, yv in cs]
            mean = np.nanmean(interp, axis=0)
            ax.plot(grid, mean, color=vcol(v), lw=2.6, label=f"{v:.2f} mL (n={len(cs)})")
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(0, color="0.5", ls=":", lw=1)
        ax.set_xlim(-1, 8)
        ax.set_xlabel("minutes since pulse")
        ax.set_ylabel(f"{FEATURE} change vs pre-pulse baseline")
        ax.set_title(f"{ptype.capitalize()} pulses aligned (thin = each pulse, thick = mean)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "fig2_aligned_responses.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_gain(P, gm, figdir):
    types = sorted(P["type"].unique())
    fig, axes = plt.subplots(len(types), 3, figsize=(17, 4.8 * len(types)), squeeze=False)
    mk = {1: "o", 2: "s", 3: "^"}
    for i, ptype in enumerate(types):
        g = P[P["type"] == ptype]
        for col, (yc, lab) in enumerate([("amp", f"peak change in {FEATURE}"), ("dpH", "peak pH shift (ladder sigmoid)"),
                                         ("gain", f"{FEATURE} change per mL")]):
            ax = axes[i, col]
            for o, gg in g.groupby("order"):
                ax.scatter(gg.volume, gg[yc], marker=mk.get(o, "o"), s=55, c=[vcol(v) for v in gg.volume], edgecolor="k",
                           label=f"pulse #{o} in run")
            mean = g.groupby(g.volume.round(2))[yc].agg(["mean", "std"])
            ax.errorbar(mean.index, mean["mean"], yerr=mean["std"], fmt="k_", ms=18, capsize=4, lw=1.5)
            if col == 0:
                v = np.linspace(0, g.volume.max() * 1.05, 20)
                sg = -1 if ptype == "acid" else 1
                ax.plot(v, sg * gm[ptype]["slope_per_mL"] * v, "k--", lw=1,
                        label=f"line through 0: {gm[ptype]['slope_per_mL']:.2f}/mL (R²={gm[ptype]['r2_through_origin']:.2f})")
                ax.text(0.03, 0.05, f"log-log exponent {gm[ptype]['exponent']:.2f} ± {gm[ptype]['exponent_se']:.2f}\n(1 = linear, <1 = saturating)",
                        transform=ax.transAxes, fontsize=8)
            ax.set_xlabel("pulse volume (mL)")
            ax.set_ylabel(lab)
            ax.set_title(f"{ptype.capitalize()}: {lab}")
            if col == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "fig3_gain_curve.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_timing(P, figdir, args):
    types = sorted(P["type"].unique())
    fig, axes = plt.subplots(len(types), 3, figsize=(16, 4.6 * len(types)), squeeze=False)
    for i, ptype in enumerate(types):
        g = P[P["type"] == ptype]
        for col, (yc, lab, scale) in enumerate([("t_peak_s", "time to peak (s)", 1), ("t_half_decay_min", "half-recovery time (min)", 1),
                                                ("tau_fit_min", "decay time constant, exp fit (min)", 1)]):
            ax = axes[i, col]
            rng = np.random.default_rng(0)
            ax.scatter(g.volume + rng.normal(0, 0.003, len(g)), g[yc] * scale, c=[vcol(v) for v in g.volume], edgecolor="k", s=50)
            if col > 0:
                ax.axhspan(args.tau_lo * (np.log(2) if col == 1 else 1), args.tau_hi * (np.log(2) if col == 1 else 1),
                           color="green", alpha=0.12, label=f"V/Q-based expectation (τ {args.tau_lo}–{args.tau_hi} min)")
                ax.legend(fontsize=7)
            ax.set_xlabel("pulse volume (mL)")
            ax.set_ylabel(lab)
            ax.set_title(f"{ptype.capitalize()}: {lab}")
    fig.tight_layout()
    fig.savefig(figdir / "fig4_timing.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_repro(P, R, figdir):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    if len(R):
        ax.scatter(R.amp_run1, R.amp_run2, s=60, c=[vcol(v) for v in R.volume], edgecolor="k")
        lim = [min(R.amp_run1.min(), R.amp_run2.min()) * 1.1, max(R.amp_run1.max(), R.amp_run2.max()) * 1.1]
        lo, hi = min(lim), max(lim)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(f"Run 01 vs run 02 (median |diff| {np.median(np.abs(R.rel_diff_pct)):.0f}%)")
    ax.set_xlabel("peak change, run 01")
    ax.set_ylabel("peak change, run 02")
    ax = axes[1]
    for ptype, g in P.groupby("type"):
        ax.scatter(g.F0 - g.F0.mean(), np.abs(g.amp) / g.volume, label=ptype, s=50, edgecolor="k")
    ax.set_xlabel("pre-pulse baseline minus its mean")
    ax.set_ylabel("|change| per mL")
    ax.set_title("Carry-over: does a lower start change the gain?")
    ax.legend()
    ax = axes[2]
    for ptype, g in P.groupby("type"):
        m = g.groupby("order").apply(lambda d: (np.abs(d.amp) / d.volume).mean())
        ax.plot(m.index, m.values, "o-", label=ptype)
    ax.set_xlabel("pulse position in run (1st, 2nd, 3rd)")
    ax.set_ylabel("mean |change| per mL")
    ax.set_title("Order effect")
    ax.set_xticks([1, 2, 3])
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "fig5_reproducibility_and_carryover.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_features(P, figdir):
    feats = [FEATURE] + OTHER_FEATURES
    types = sorted(P["type"].unique())
    fig, axes = plt.subplots(1, len(types), figsize=(8 * len(types), 5), squeeze=False)
    for ax, ptype in zip(axes[0], types):
        g = P[P["type"] == ptype]
        vols = sorted(g.volume.round(2).unique())
        w = 0.8 / len(vols)
        for iv, v in enumerate(vols):
            sub = g[g.volume.round(2) == v]
            vals = []
            for f in feats:
                col = "amp" if f == FEATURE else f"d_{f}"
                sg = "sigma0" if f == FEATURE else f"sigma_{f}"
                vals.append(np.nanmean(np.abs(sub[col]) / sub[sg]) if col in sub else np.nan)
            ax.bar(np.arange(len(feats)) + iv * w - 0.4 + w / 2, vals, width=w, color=vcol(v), label=f"{v:.2f} mL")
        ax.set_xticks(range(len(feats)), feats)
        ax.set_ylabel("peak change / baseline noise (signal-to-noise)")
        ax.set_title(f"{ptype.capitalize()}: which feature reacts best?")
        ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "fig6_feature_snr.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_od(P, figdir):
    cols = [c for c in ("d_OD135", "d_OD45") if c in P.columns and P[c].notna().any()]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(7.5 * len(cols), 4.8), squeeze=False)
    for ax, c in zip(axes[0], cols):
        for ptype, g in P.groupby("type"):
            ax.scatter(g.volume, g[c], label=ptype, s=50, edgecolor="k")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("pulse volume (mL)")
        ax.set_ylabel(f"largest {c[2:]} change within 5 min of the pulse")
        ax.set_title(f"{c[2:]} response (scatter/bubble check)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "fig7_od_response.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder holding the stage2_<cond>_run<NN>_<hhmm>-<hhmm>_IST folders (or zips)")
    ap.add_argument("--out", default="results_stage2")
    ap.add_argument("--only", choices=["A", "B"], help="analyse only the runs labelled A* or B* (labels only, not reagent)")
    ap.add_argument("--only-type", choices=["acid", "base"], help="analyse only pulses of this logged type")
    ap.add_argument("--acid-event", default="add_media")
    ap.add_argument("--base-event", default="add_alt_media")
    ap.add_argument("--max-pulse-ml", type=float, default=1.0, help="ignore dosing events larger than this")
    ap.add_argument("--peak-window-s", type=float, default=180)
    ap.add_argument("--max-window-s", type=float, default=900, help="longest response window per pulse")
    ap.add_argument("--tau-lo", type=float, default=2.0, help="expected residence time range (min), for reference bands")
    ap.add_argument("--tau-hi", type=float, default=2.5)
    ap.add_argument("--inspect", action="store_true", help="print dosing_events columns/values of every run and exit")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()
    cfg = dict(peak_window_s=args.peak_window_s, max_window_s=args.max_window_s)

    runs = discover_runs(args.root)
    if args.only:
        runs = [r for r in runs if r["cond"].startswith(args.only)]
    if not runs:
        sys.exit(f"no folders/zips named stage2_<A1..B3>_run<NN>_<hhmm>-<hhmm>_IST found in {args.root}")
    print(f"found {len(runs)} runs: " + ", ".join(r["name"] for r in runs))

    out = Path(args.out)
    figdir, tabdir = out / "figures", out / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)
    results, manifest, log = [], [], []
    for r in runs:
        res, files = analyse_run(r, args, cfg)
        for f in files:
            manifest.append(dict(run=r["name"], file=str(f), sha256=sha256_file(f), bytes=f.stat().st_size))
        if res is None:
            continue
        results.append(res)
        print(f"  {r['name']}: {len(res['rows'])} pulses analysed")
    if args.inspect:
        return
    rows = [row for res in results for row in res["rows"]]
    if not rows:
        sys.exit("no pulses analysed – run with --inspect and check the dosing_events columns / event names")
    P = pd.DataFrame(rows)
    if args.only_type:
        P = P[P["type"] == args.only_type].copy()
        if P.empty:
            sys.exit(f"no {args.only_type} pulses found")
    P["dpH"] = P["pH_peak"] - P["pH_before"]
    P["gain"] = np.abs(P["amp"]) / P["volume"]
    P = P.sort_values(["type", "cond", "run_no", "order"]).reset_index(drop=True)
    P.to_csv(tabdir / "pulses_table.csv", index=False)
    pd.DataFrame(manifest).to_csv(out / "inputs_manifest.csv", index=False)
    tables = pooled_tables(P, cfg)
    gm = {pt: gain_models(P[P["type"] == pt]) for pt in P["type"].unique()}
    for pt, tb in tables.items():
        tb.to_csv(tabdir / f"summary_by_volume_{pt}.csv", index=False)
    R = reproducibility(P)
    R.to_csv(tabdir / "reproducibility_run1_vs_run2.csv", index=False)
    co = carryover(P)
    s3 = stage3_suggestions(P, tables, args)
    mism = P[~P.plan_match]

    bycond = P.groupby("cond").agg(type=("type", lambda x: "/".join(sorted(set(x)))), n=("amp", "size"),
                                   direction_ok=("direction_ok", "mean"), mean_amp=("amp", "mean")).reset_index()
    bycond.to_csv(tabdir / "type_by_condition.csv", index=False)
    bad = bycond[bycond.direction_ok < 0.5]
    summary = dict(script_version=SCRIPT_VERSION, n_runs=len(results), n_pulses=int(len(P)), feature=FEATURE,
                   type_by_condition=bycond.to_dict("records"), gain_models=gm, carryover=co, stage3_suggestions=s3,
                   plan_mismatches=mism[["run", "order", "volume", "planned_volume"]].to_dict("records"),
                   reproducibility_median_abs_rel_diff_pct=float(np.median(np.abs(R.rel_diff_pct))) if len(R) else None,
                   direction_not_as_expected=int((~P.direction_ok).sum()),
                   od135_flag_pulses=int((P["d_OD135"].abs() > 5 * P["sigma_OD135"]).sum()) if "d_OD135" in P else None,
                   inputs=dict(root=str(args.root), manifest="inputs_manifest.csv"),
                   made_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), python=platform.python_version(),
                   script_sha256=sha256_file(__file__))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    L = ["STAGE 2 PULSE ANALYSIS", f"{len(results)} runs, {len(P)} pulses, feature {FEATURE}", "",
         "Pulse type per condition (from the logged dosing events) and whether the ratio moved the expected way:",
         bycond.round(4).to_string(index=False), ""]
    if len(bad):
        L += ["!! For " + ", ".join(bad.cond) + " most pulses moved the ratio OPPOSITE to their logged type.",
              "   The reagent bottles may be connected to the other pump: rerun with --acid-event add_alt_media --base-event add_media", ""]
    for pt, tb in tables.items():
        L += [f"--- {pt.upper()} pulses ---", tb.round(4).to_string(index=False), "",
              f"gain: {gm[pt]['slope_per_mL']:.3f} ratio/mL through origin (R² {gm[pt]['r2_through_origin']:.2f}); "
              f"log-log exponent {gm[pt]['exponent']:.2f} ± {gm[pt]['exponent_se']:.2f} (p vs 1 = {gm[pt]['exponent_p_vs_1']:.2f})", ""]
        if pt in co:
            L.append(f"carry-over: per-mL gain changes {co[pt]['gain_vs_F0_slope']:+.3f} per unit baseline offset (p = {co[pt]['p']:.2f})")
        s = s3[pt]
        L.append(f"Stage 3: largest tested volume inside the {s['limit']} limit = {s['largest_volume_inside_limit']}")
        for o in s["per_volume"]:
            L.append(f"   {o['volume']:.2f} mL: worst peak {o['worst_peak_ratio']:.3f} "
                     f"({'inside limit' if o['inside_limit'] else 'OUTSIDE limit'}); one level step (v/4) = {o['per_level_if_4_levels']:.4f} "
                     f"-> {'OK' if o['level_step_over_5sigma'] >= 1 else 'too small'} "
                     f"({o['level_step_over_5sigma']:.1f} x the 5-sigma noise requirement)")
        L.append("")
    if len(R):
        L.append(f"run 01 vs run 02: median |relative difference| {np.median(np.abs(R.rel_diff_pct)):.1f}% over {len(R)} pairs")
    if len(mism):
        L.append(f"! {len(mism)} pulses differ from the planned volumes – see pulses_table.csv (plan_match)")
    if (~P.direction_ok).any():
        L.append(f"! {(~P.direction_ok).sum()} pulses moved the ratio opposite to the expected direction")
    (out / "summary.txt").write_text("\n".join(L))
    print("\n".join(L))

    print("\nmaking figures ...")
    for pt in P["type"].unique():
        fig_overview(results, pt, figdir)
    fig_aligned(results, P, figdir, cfg)
    fig_gain(P, gm, figdir)
    fig_timing(P, figdir, args)
    fig_repro(P, R, figdir)
    fig_features(P, figdir)
    fig_od(P, figdir)

    if not args.no_zip:
        zp = out.with_suffix(".zip")
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(out.parent))
        print(f"\nresults zipped: {zp}")
    print(f"done. outputs in {out}/")


if __name__ == "__main__":
    main()