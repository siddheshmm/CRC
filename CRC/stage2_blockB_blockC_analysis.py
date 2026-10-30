#!/usr/bin/env python3
"""
stage2_blockB_blockC_analysis.py — Comprehensive Mathematical Audit
of Stage 2 Block B (Overshoot / Plateau Test) and Block C (Paired Pulses & Superposition Error)
for the Chemical Reservoir Computer (Methyl Orange, Pioreactor CSTR).

Audited & Revised:
  - Noise-adaptive pre-pulse baseline filtering via MAD SD (rejects drift)
  - 3-point rolling median and 15s window around peak (avoids chasing single-point noise spikes)
  - Sigmoid plateau anchoring against calibration limits (ya=1.3586, yb=1.6198)
  - Superposition decomposition via moving-block bootstrapping & AR(1) surrogate noise reference:
      * Hypothesis 1: Plain sum (a1=1, a2=1) fits within noise? -> Linearity test
      * Hypothesis 2: Linear once pulse gains a1, a2 are free? -> Reagent dilution / pump delivery drift
      * Hypothesis 3: Residual exceeds noise even with free gains? -> True physical chemical interaction
  - Automatic default path discovery for CRC workspace

Outputs:
  - Figures: figB_overshoot_overview.png, figC_paired_pulses.png, figC_superposition_summary.png
  - Tables: blockB_pulses.csv, blockB_by_volume.csv, blockC_pairs.csv
  - Summaries: summary.txt, summary.json, inputs_manifest.csv
"""

import argparse
import hashlib
import json
import platform
import sys
import tempfile
import zipfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=RuntimeWarning)
SCRIPT_VERSION = "2.0_audited"

# ------------------------------------------------------------------ CONFIG
CALIB = dict(ya=1.3586, yb=1.6198, pka=3.741)      # ladder sigmoid of 555/480: positions of the two plateaus
LIMITS = dict(acid_floor=1.40, base_ceiling=1.58)  # usable range chosen in the protocol
PRE_S = 120            # baseline window before a pulse (s)
PEAK_S = 180           # search window for the peak (s)
DECAY_MAX_S = 900      # longest response window per pulse (s)
HEADROOM_TARGET = 0.80 # Stage 3 rule: largest pulse should use about this fraction of the room
ALPHA = 0.01           # significance level for the superposition classification
N_SIM, N_BOOT, BLOCK = 2000, 400, 12   # noise simulations, bootstrap resamples, bootstrap block (samples)
MATCH_ML = 0.011       # volumes closer than this count as the same level

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "legend.frameon": False
})
RNG = np.random.default_rng(12345)


# ------------------------------------------------------------------ loading
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def list_runs(root, pattern):
    return sorted(p for p in Path(root).glob(pattern) if p.is_dir() or p.suffix.lower() == ".zip")


def read_run(path):
    path = Path(path)
    base = path
    if path.suffix.lower() == ".zip":
        base = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(path) as z:
            z.extractall(base)
    out, files = {}, []
    for kind in ("as7341", "dosing_events"):
        frames = []
        for f in sorted(base.rglob("*.csv")):
            if kind in f.name.lower() or kind in f.parent.name.lower():
                frames.append(pd.read_csv(f))
                files.append(f)
        out[kind] = pd.concat(frames, ignore_index=True) if frames else None
    return out, files


def utc_naive(s):
    return pd.to_datetime(s, utc=True).dt.tz_localize(None)


def to_cycles(df):
    df = df.copy()
    df["t"] = utc_naive(df["timestamp"])
    df = df.sort_values("t")
    cyc = (df["t"].diff() > pd.Timedelta("1s")).cumsum()
    w = df.assign(c=cyc).pivot_table(index="c", columns="band", values="reading", aggfunc="mean")
    w.index = pd.DatetimeIndex(df.assign(c=cyc).groupby("c")["t"].first().values)
    return w.dropna()


def feature_series(spec, feature):
    a, b = (int(x) for x in feature.split("/"))
    return spec[a] / spec[b]


def parse_dosing(df, acid_event, base_event, merge_s, max_ml):
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    tcol = low.get("timestamp") or next((c for c in cols if "time" in c.lower()), None)
    ecol = low.get("event") or next((c for c in cols if "event" in c.lower() and "source" not in c.lower()), None)
    vcol = next((c for c in cols if "volume" in c.lower()), None)
    if not (tcol and ecol and vcol):
        raise ValueError(f"dosing_events needs time/event/volume columns, found {cols}")
    ev = pd.DataFrame({"t": utc_naive(df[tcol]), "event": df[ecol].astype(str).str.lower(),
                       "vol": pd.to_numeric(df[vcol], errors="coerce")}).dropna().sort_values("t")
    ev = ev[ev.event.isin([acid_event, base_event]) & (ev.vol.abs() > 0) & (ev.vol.abs() <= max_ml)]
    rows = []
    for _, r in ev.iterrows():
        typ = "acid" if r["event"] == acid_event else "base"
        if rows and rows[-1]["type"] == typ and (r["t"] - rows[-1]["t_last"]).total_seconds() < merge_s:
            rows[-1]["vol"] += abs(r["vol"])
            rows[-1]["t_last"] = r["t"]
            rows[-1]["n_events"] += 1
        else:
            rows.append(dict(t=r["t"], t_last=r["t"], vol=abs(r["vol"]), type=typ, n_events=1))
    return pd.DataFrame(rows)


def epoch(ts):
    return pd.Timestamp(ts).to_datetime64().astype("datetime64[ns]").astype("int64") / 1e9


# ------------------------------------------------------------------ single-pulse metrics
def mad_sd(x):
    x = np.asarray(x, float)
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def baseline(t, y, tp):
    m = (t >= tp - PRE_S) & (t < tp - 5)
    if m.sum() < 8:
        return None
    return float(np.median(y[m])), float(mad_sd(y[m])), int(m.sum()), y[m] - np.median(y[m])


def exp_fit(tt_min, dd):
    f = lambda t, a, c, tau: c + (a - c) * np.exp(-t / tau)
    if len(tt_min) < 15:
        return np.nan, np.nan, np.nan, np.nan
    try:
        p, cov = curve_fit(f, tt_min, dd, p0=[dd[0], 0.0, 2.0], bounds=([-1, -1, 0.1], [1, 1, 60]), maxfev=20000)
        r2 = 1 - np.sum((dd - f(tt_min, *p)) ** 2) / np.sum((dd - dd.mean()) ** 2)
        return p[2], float(np.sqrt(cov[2, 2])), p[1], r2
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def pulse_metrics(t, y, tp, t_next, ptype, feature_room):
    b = baseline(t, y, tp)
    if b is None:
        return None
    F0, sig, n_pre, _ = b
    sgn = -1.0 if ptype == "acid" else 1.0
    t_end = tp + DECAY_MAX_S if t_next is None else min(tp + DECAY_MAX_S, t_next - 5)
    post = (t >= tp) & (t <= t_end)
    if post.sum() < 8:
        return None
    tp_, yp = t[post], y[post]
    ys = pd.Series(yp).rolling(3, center=True, min_periods=1).median().values
    d = ys - F0
    win = (tp_ - tp) <= PEAK_S
    idx = int(np.argmin(d[win])) if sgn < 0 else int(np.argmax(d[win]))
    amp = float(d[idx])
    near = np.abs(tp_ - tp_[idx]) <= 15
    amp_win = float(np.mean(yp[near]) - F0)
    tt = (tp_[idx:] - tp_[idx]) / 60
    tau, tau_se, c_inf, r2 = exp_fit(tt, (yp - F0)[idx:])
    h = np.where(np.abs(d[idx:]) <= 0.5 * abs(amp))[0]
    last = tp_ >= tp_[-1] - 60
    plateau = CALIB["ya"] if ptype == "acid" else CALIB["yb"]
    room = abs(plateau - F0)
    limit_room = (F0 - LIMITS["acid_floor"]) if ptype == "acid" else (LIMITS["base_ceiling"] - F0)
    return dict(F0=F0, sigma0=sig, n_pre=n_pre, amp=amp, amp_win=amp_win, snr=abs(amp) / sig if sig else np.nan,
                F_peak=F0 + amp, t_peak_s=float(tp_[idx] - tp), t_half_decay_min=float(tt[h[0]]) if len(h) else np.nan,
                tau_fit_min=tau, tau_fit_se=tau_se, decay_r2=r2, window_s=float(tp_[-1] - tp),
                end_offset=float(np.median(yp[last]) - F0), room_to_plateau=room, room_to_limit=limit_room,
                fraction_of_plateau_room=abs(amp) / room if room else np.nan,
                reached_plateau=bool(abs(F0 + amp - plateau) < 0.02), direction_ok=bool(np.sign(amp) == sgn))


# ------------------------------------------------------------------ Block B
def analyse_block_b(runs, args, manifest):
    rows, series = [], {}
    for r in runs:
        data, files = read_run(r)
        manifest += [dict(run=r.name, file=str(f), sha256=sha256_file(f)) for f in files]
        if data["as7341"] is None or data["dosing_events"] is None:
            print(f"  ! {r.name}: missing csv, skipped")
            continue
        s = feature_series(to_cycles(data["as7341"]), args.feature)
        t, y = s.index.values.astype("datetime64[ns]").astype("int64") / 1e9, s.values
        pul = parse_dosing(data["dosing_events"], args.acid_event, args.base_event, args.merge_s, args.max_ml).reset_index(drop=True)
        if pul.empty:
            print(f"  ! {r.name}: no pulses found in dosing_events")
            continue
        for k, p in pul.iterrows():
            tp = epoch(p["t"])
            tn = epoch(pul.iloc[k + 1]["t"]) if k + 1 < len(pul) else None
            m = pulse_metrics(t, y, tp, tn, p["type"], None)
            if m is None:
                continue
            m.update(run=r.name, pulse_no=k + 1, type=p["type"], volume=round(float(p["vol"]), 3), n_events=int(p["n_events"]),
                     t_pulse_utc=str(p["t"]))
            rows.append(m)
            mm = (t >= tp - PRE_S) & (t <= tp + (DECAY_MAX_S if tn is None else min(DECAY_MAX_S, tn - tp)))
            series[(r.name, k + 1)] = dict(t=t[mm] - tp, y=y[mm], F0=m["F0"], type=p["type"], vol=m["volume"])
    return pd.DataFrame(rows), series


def ref_gain_model(ref, ptype):
    g = ref[ref["type"] == ptype]
    if len(g) < 4 or g.volume.nunique() < 2:
        return None
    lr = stats.linregress(g.volume, g.amp.abs())
    return dict(gain=lr.slope, intercept=lr.intercept, n=len(g), v_max=g.volume.max(),
                amp_at_vmax=g[g.volume == g.volume.max()].amp.abs().mean(), r2=lr.rvalue ** 2)


def block_b_summary(B, ref, L):
    out = {}
    agg = B.groupby(["type", "volume"]).agg(n=("amp", "size"), amp_mean=("amp", "mean"), amp_sd=("amp", "std"),
                                            amp_win_mean=("amp_win", "mean"), F0=("F0", "mean"), sigma0=("sigma0", "mean"),
                                            frac_room=("fraction_of_plateau_room", "mean"),
                                            tau_fit_min=("tau_fit_min", "median"), t_half_min=("t_half_decay_min", "median"),
                                            end_offset=("end_offset", "mean")).reset_index()
    for _, r in agg.iterrows():
        typ = r["type"]
        sub = B[(B.type == typ) & (B.volume == r.volume)]
        rep = (abs(sub.amp.max() - sub.amp.min()) / abs(sub.amp.mean()) * 100) if len(sub) == 2 else np.nan
        sat = bool(sub.reached_plateau.any())
        limit_room = float(sub.room_to_limit.mean())
        target = HEADROOM_TARGET * limit_room
        k = target / abs(r.amp_mean) if r.amp_mean else np.nan
        d = dict(type=typ, volume=float(r.volume), n=int(r.n), amp_mean=float(r.amp_mean), amp_sd=float(r.amp_sd) if r.n > 1 else None,
                 replicate_diff_pct=None if np.isnan(rep) else float(rep), F0=float(r.F0), sigma0=float(r.sigma0),
                 fraction_of_room_to_plateau_used=float(r.frac_room), reached_plateau=sat,
                 room_to_limit=limit_room, target_amp_for_stage3=float(target), strength_multiplier_needed_at_same_volume=float(k),
                 washout_tau_min=float(r.tau_fit_min), baseline_offset_after=float(r.end_offset), snr=float(abs(r.amp_mean) / r.sigma0))
        L.append(f"{typ.upper()} {r.volume:.2f} mL (n={int(r.n)}): peak change {r.amp_mean:+.4f}"
                 + (f" (sd {r.amp_sd:.4f}, replicate difference {rep:.0f}%)" if r.n > 1 else "")
                 + f"; uses {100 * r.frac_room:.0f}% of the room to the {typ} plateau "
                 f"({'plateau reached' if sat else 'plateau NOT reached'}); baseline offset afterwards {r.end_offset:+.4f}; "
                 f"washout tau {r.tau_fit_min:.1f} min; signal/noise {abs(r.amp_mean) / r.sigma0:.1f}")
        need = (f"the reagent would need to be about x{k:.1f} stronger at this volume" if k >= 1.05 else
                f"this volume already exceeds it (about x{1 / k:.1f} more than needed)" if k <= 0.95 else "this volume is about right")
        L.append(f"   Stage 3 target ({int(HEADROOM_TARGET * 100)}% of the room to the limit) is a change of {target:.3f}; this pulse gives "
                 f"{abs(r.amp_mean):.4f}, so {need}")
        if ref is not None:
            m = ref_gain_model(ref, typ)
            if m:
                pred = m["gain"] * (r.volume - (-m["intercept"] / m["gain"] if m["gain"] else 0))
                ratio = abs(r.amp_mean) / pred if pred else np.nan
                mono = abs(r.amp_mean) < 0.8 * m["amp_at_vmax"] and r.volume > m["v_max"]
                d.update(blockA_amp_at_largest_volume=float(m["amp_at_vmax"]), blockA_largest_volume=float(m["v_max"]),
                         blockA_line_prediction=float(pred), measured_over_prediction=float(ratio), not_monotonic_with_blockA=bool(mono))
                L.append(f"   vs Block A ({m['n']} pulses up to {m['v_max']:.2f} mL, straight-line fit R2 {m['r2']:.2f}): Block A gave "
                         f"{m['amp_at_vmax']:.4f} at {m['v_max']:.2f} mL and its line predicts {pred:.4f} here; measured/predicted = {ratio:.2f}"
                         + ("  -> SMALLER than a smaller Block A pulse: reagent strength or pump delivery changed between sessions" if mono else ""))
        out[f"{typ}_{r.volume:.2f}"] = d
    return agg, out


# ------------------------------------------------------------------ Block C: templates
def aligned_curve(t, y, tp, t_next, step=5.0):
    b = baseline(t, y, tp)
    if b is None:
        return None
    F0 = b[0]
    t_end = tp + DECAY_MAX_S if t_next is None else min(tp + DECAY_MAX_S, t_next - 5)
    m = (t >= tp) & (t <= t_end)
    if m.sum() < 8:
        return None
    grid = np.arange(0, t_end - tp + 1e-9, step)
    return grid, np.interp(grid, t[m] - tp, y[m] - F0), b[1]


def build_templates(curves):
    groups = {}
    for typ, vol, grid, y, sig in curves:
        groups.setdefault((typ, round(vol / MATCH_ML) * MATCH_ML), []).append((grid, y, sig, vol))
    tmpl = {}
    for key, lst in groups.items():
        L = max(len(g) for g, _, _, _ in lst)
        M = np.full((len(lst), L), np.nan)
        for i, (g, y, _, _) in enumerate(lst):
            M[i, :len(y)] = y
        mean = np.nanmean(M, axis=0)
        n = np.sum(~np.isnan(M), axis=0)
        grid = np.arange(L) * 5.0
        tmpl[key] = dict(grid=grid, y=mean, n=n, sigma=float(np.mean([s for _, _, s, _ in lst])),
                         vol=float(np.mean([v for _, _, _, v in lst])), n_curves=len(lst))
    return tmpl


def extend_template(tm, until_s):
    grid, y = tm["grid"], tm["y"]
    if grid[-1] >= until_s:
        return tm
    ipk = int(np.nanargmax(np.abs(y)))
    tt, yy = grid[ipk:], y[ipk:]
    ok = np.isfinite(yy)
    try:
        p, _ = curve_fit(lambda t, a, tau: a * np.exp(-(t - tt[0]) / tau), tt[ok], yy[ok], p0=[yy[ok][0], 150.0],
                         bounds=([-1, 20], [1, 3000]))
        g2 = np.arange(grid[-1] + 5.0, until_s + 5.0, 5.0)
        y2 = p[0] * np.exp(-(g2 - tt[0]) / p[1])
        n2 = np.full(len(g2), max(1, int(tm["n"][-1])))
        return dict(tm, grid=np.concatenate([grid, g2]), y=np.concatenate([y, y2]), n=np.concatenate([tm["n"], n2]), extended_tau_s=float(p[1]))
    except Exception:
        g2 = np.arange(grid[-1] + 5.0, until_s + 5.0, 5.0)
        return dict(tm, grid=np.concatenate([grid, g2]), y=np.concatenate([y, np.zeros(len(g2))]),
                    n=np.concatenate([tm["n"], np.full(len(g2), 1)]), extended_tau_s=None)


def template_value(tm, tt):
    out = np.zeros_like(tt, dtype=float)
    m = tt >= 0
    out[m] = np.interp(tt[m], tm["grid"], np.nan_to_num(tm["y"]))
    return out


def find_template(tmpl, typ, vol):
    best = None
    for (t, lvl), tm in tmpl.items():
        if t == typ and abs(tm["vol"] - vol) <= MATCH_ML:
            if best is None or abs(tm["vol"] - vol) < abs(best["vol"] - vol):
                best = tm
    return best


# ------------------------------------------------------------------ Block C: superposition test
def ar1_null(n, sigma, rho, stat_fn, nsim=N_SIM):
    vals = np.empty(nsim)
    innov = np.sqrt(max(1e-9, 1 - rho ** 2))
    for i in range(nsim):
        e = np.empty(n)
        e[0] = RNG.normal()
        z = RNG.normal(size=n)
        for k in range(1, n):
            e[k] = rho * e[k - 1] + innov * z[k]
        vals[i] = stat_fn(e * sigma)
    return vals


def moving_block_bootstrap(fit, resid, X, y, nboot=N_BOOT, block=BLOCK):
    n = len(y)
    nb = int(np.ceil(n / block))
    betas = []
    for _ in range(nboot):
        starts = RNG.integers(0, max(1, n - block), nb)
        r = np.concatenate([resid[s:s + block] for s in starts])[:n]
        betas.append(np.linalg.lstsq(X, fit + r, rcond=None)[0])
    return np.std(betas, axis=0, ddof=1)


def analyse_pair(t, y, p1, p2, tmpl, rho, n_eval_cap=None):
    t1, t2 = epoch(p1["t"]), epoch(p2["t"])
    b = baseline(t, y, t1)
    if b is None:
        return None
    F0, sig_pre, _, _ = b
    dy = np.diff(y)
    sig = float(1.4826 * np.median(np.abs(dy - np.median(dy))) / np.sqrt(2 * (1 - rho)))
    tm1, tm2 = find_template(tmpl, p1["type"], p1["vol"]), find_template(tmpl, p2["type"], p2["vol"])
    if tm1 is None or tm2 is None:
        return dict(error=f"no single-pulse template for {p1['type']} {p1['vol']:.2f} mL or {p2['type']} {p2['vol']:.2f} mL")
    m = (t >= t1 - 60)
    tt, yy = t[m] - t1, y[m] - F0
    tm1, tm2 = extend_template(tm1, tt.max() + 10), extend_template(tm2, tt.max() + 10)
    T1, T2 = template_value(tm1, tt), template_value(tm2, tt - (t2 - t1))
    n1, n2 = max(1, int(np.min(tm1["n"][:60]))), max(1, int(np.min(tm2["n"][:60])))
    sig_e = sig * np.sqrt(1 + 1 / n1 + 1 / n2)
    e0 = yy - (T1 + T2)
    X = np.column_stack([T1, T2, np.ones_like(T1)])
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    fit = X @ beta
    e1 = yy - fit
    se = moving_block_bootstrap(fit, e1, X, yy)
    n = len(yy)
    stat_peak, stat_rmse = (lambda e: np.max(np.abs(e))), (lambda e: np.sqrt(np.mean(e ** 2)))
    nul_peak = ar1_null(n, sig_e, rho, stat_peak)
    nul_rmse = ar1_null(n, sig_e, rho, stat_rmse)
    obs = dict(peak0=float(stat_peak(e0)), rmse0=float(stat_rmse(e0)), peak1=float(stat_peak(e1)), rmse1=float(stat_rmse(e1)))
    p_peak0 = (np.sum(nul_peak >= obs["peak0"]) + 1) / (N_SIM + 1)
    p_rmse0 = (np.sum(nul_rmse >= obs["rmse0"]) + 1) / (N_SIM + 1)
    p_peak1 = (np.sum(nul_peak >= obs["peak1"]) + 1) / (N_SIM + 1)
    p_rmse1 = (np.sum(nul_rmse >= obs["rmse1"]) + 1) / (N_SIM + 1)
    if min(p_peak0, p_rmse0) > ALPHA:
        label = "linear (plain sum fits within noise)"
    elif min(p_peak1, p_rmse1) > ALPHA:
        label = "gain change only (linear once each pulse has its own gain)"
    else:
        label = "interaction beyond gain scaling"
    return dict(F0=F0, sigma0=sig, sigma_pre_window=float(sig_pre), sigma_eff=float(sig_e), n_eval=int(n), dt_s=float(t2 - t1), a1=float(beta[0]), a2=float(beta[1]),
                a1_se=float(se[0]), a2_se=float(se[1]), offset=float(beta[2]), meas_peak=float(np.max(np.abs(yy))),
                sum_peak=float(np.max(np.abs(T1 + T2))), p_peak_plain=float(p_peak0), p_rmse_plain=float(p_rmse0),
                p_peak_gain_fit=float(p_peak1), p_rmse_gain_fit=float(p_rmse1), null_peak_95=float(np.percentile(nul_peak, 95)),
                null_rmse_95=float(np.percentile(nul_rmse, 95)), template_curves=f"{tm1['n_curves']}+{tm2['n_curves']}",
                template_extended=bool(tm1.get("extended_tau_s") is not None or tm2.get("extended_tau_s") is not None),
                classification=label, **obs, _tt=tt, _y=yy, _T1=T1, _T2=T2, _fit=fit)


def pooled_rho(runs_baselines):
    r = []
    for b in runs_baselines:
        if len(b) >= 12:
            b = b - b.mean()
            r.append(np.sum(b[1:] * b[:-1]) / np.sum(b * b))
    return float(np.clip(np.median(r), 0.0, 0.9)) if r else 0.3


# ------------------------------------------------------------------ figures
def fig_block_b(B, series, ref, figdir):
    types = [t for t in ("acid", "base") if (B.type == t).any()]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for j, typ in enumerate(types):
        ax = axes[0, j]
        for (run, k), s in series.items():
            if s["type"] == typ:
                ax.plot(s["t"] / 60, s["y"], lw=1.5, label=f"{run} ({s['vol']:.2f} mL)")
        ax.axhline(CALIB["ya"] if typ == "acid" else CALIB["yb"], ls="--", color="crimson",
                   label=f"{typ} plateau ({CALIB['ya'] if typ == 'acid' else CALIB['yb']:.4f})")
        ax.axhline(LIMITS["acid_floor"] if typ == "acid" else LIMITS["base_ceiling"], ls=":", color="navy",
                   label=f"Stage 3 limit ({LIMITS['acid_floor'] if typ == 'acid' else LIMITS['base_ceiling']:.2f})")
        ax.set_title(f"{typ.upper()} overshoot (0.50 mL single pulse)")
        ax.set_xlabel("Time from pulse (min)")
        ax.set_ylabel("555/480 ratio")
        ax.legend(fontsize=8, loc="best")

        ax2 = axes[1, j]
        sub = B[B.type == typ]
        ax2.scatter(sub.volume, sub.amp.abs(), color="purple", s=70, zorder=3, label="Block B (0.50 mL)")
        if ref is not None:
            m = ref_gain_model(ref, typ)
            if m:
                r_sub = ref[ref.type == typ]
                ax2.scatter(r_sub.volume, r_sub.amp.abs(), color="gray", alpha=0.6, label="Block A reference")
                vv = np.linspace(0, max(sub.volume.max(), r_sub.volume.max()) * 1.05, 50)
                ax2.plot(vv, np.maximum(0, m["gain"] * vv + m["intercept"]), "gray", ls="--", label=f"Block A fit (slope {m['gain']:.3f})")
        ax2.set_title(f"{typ.upper()} amplitude vs volume")
        ax2.set_xlabel("Pulse volume (mL)")
        ax2.set_ylabel("|Peak change| (555/480)")
        ax2.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(figdir / "figB_overshoot_overview.png")
    plt.close(fig)


def fig_block_c(C, figdir):
    n = len(C)
    if not n:
        return
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.2 * n), squeeze=False)
    for i, c in enumerate(C):
        ax = axes[i, 0]
        tt = c["_tt"]
        ax.plot(tt, c["_y"], "k-", lw=1.5, label="Measured response")
        ax.plot(tt, c["_T1"] + c["_T2"], "b--", lw=1.2, label=f"Plain sum (peak {c['sum_peak']:.4f})")
        ax.plot(tt, c["_fit"], "g:", lw=1.2, label=f"Fitted gains (a1={c['a1']:.2f}, a2={c['a2']:.2f})")
        ax.set_title(f"{c['name']}  –  {c['label']}\n{c['classification']}", fontsize=10)
        ax.set_xlabel("Time from 1st pulse (s)")
        ax.set_ylabel("Δ(555/480)")
        ax.legend(fontsize=7.5, loc="best")

        ax2 = axes[i, 1]
        e0 = c["_y"] - (c["_T1"] + c["_T2"])
        e1 = c["_y"] - c["_fit"]
        ax2.plot(tt, e0, "b--", lw=1.0, label=f"Plain-sum error (RMS {c['rmse0']:.4f}, p={c['p_rmse_plain']:.3f})")
        ax2.plot(tt, e1, "g:", lw=1.0, label=f"Gain-fit error (RMS {c['rmse1']:.4f}, p={c['p_rmse_gain_fit']:.3f})")
        ax2.axhline(0, color="k", lw=0.5)
        ax2.axhspan(-c["null_rmse_95"], c["null_rmse_95"], color="gray", alpha=0.2, label=f"95% noise RMS (±{c['null_rmse_95']:.4f})")
        ax2.set_title(f"Residual errors vs noise band", fontsize=10)
        ax2.set_xlabel("Time from 1st pulse (s)")
        ax2.set_ylabel("Error")
        ax2.legend(fontsize=7.5, loc="best")
    plt.tight_layout()
    plt.savefig(figdir / "figC_paired_pulses.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [c["name"].split("_")[2] + " (" + c["classification"].split("(")[0].strip() + ")" for c in C]
    x = np.arange(len(C))
    w = 0.35
    ax.bar(x - w / 2, [c["rmse0"] for c in C], width=w, label="Plain-sum RMS", color="royalblue")
    ax.bar(x + w / 2, [c["rmse1"] for c in C], width=w, label="Fitted-gains RMS", color="forestgreen")
    ax.scatter(x, [c["null_rmse_95"] for c in C], color="red", zorder=4, label="95% noise ceiling")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("RMS residual")
    ax.set_title("Block C – Residual error vs noise ceiling for each pair")
    ax.legend()
    plt.tight_layout()
    plt.savefig(figdir / "figC_superposition_summary.png")
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", nargs="?", default="CRC/data/stage02", help="default: CRC/data/stage02")
    ap.add_argument("--out", default="CRC/results/stage 02 - blocks B & C", help="output directory")
    ap.add_argument("--b-glob", default="S2P2*", help="pattern of the Block B run folders inside data_dir")
    ap.add_argument("--c-glob", default="S2P3*", help="pattern of the Block C run folders")
    ap.add_argument("--singles-glob", help="single-pulse runs of the SAME session (templates for Block C)")
    ap.add_argument("--ref-pulses-csv", default="CRC/results/stage 02 - block A/tables/pulses_table.csv",
                    help="Block A pulses_table.csv (default: CRC/results/stage 02 - block A/tables/pulses_table.csv)")
    ap.add_argument("--ref-data-dir", default="CRC/data/stage02",
                    help="folder with Block A run folders (default: CRC/data/stage02)")
    ap.add_argument("--ref-glob", default="stage2_*", help="pattern of Block A run folders inside --ref-data-dir")
    ap.add_argument("--feature", default="555/480")
    ap.add_argument("--acid-event", default="add_media")
    ap.add_argument("--base-event", default="add_alt_media")
    ap.add_argument("--merge-s", type=float, default=15.0)
    ap.add_argument("--max-ml", type=float, default=5.0)
    args = ap.parse_args()

    out = Path(args.out)
    figdir, tabdir = out / "figures", out / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)
    manifest, L = [], ["STAGE 2 – BLOCK B (OVERSHOOT) AND BLOCK C (PAIRED PULSES) [AUDITED REPORT]", ""]
    ref = pd.read_csv(args.ref_pulses_csv) if (args.ref_pulses_csv and Path(args.ref_pulses_csv).exists()) else None

    # ---------------- Block B
    b_runs = list_runs(args.data_dir, args.b_glob)
    B, series, summaryB = pd.DataFrame(), {}, {}
    if b_runs:
        print(f"Block B: {len(b_runs)} runs")
        B, series = analyse_block_b(b_runs, args, manifest)
        if len(B):
            B.to_csv(tabdir / "blockB_pulses.csv", index=False)
            L += ["BLOCK B – overshoot", "-" * 70]
            agg, summaryB = block_b_summary(B, ref, L)
            agg.to_csv(tabdir / "blockB_by_volume.csv", index=False)
            bad = B[~B.direction_ok]
            if len(bad):
                L.append(f"! {len(bad)} pulses moved the ratio opposite to their logged type: {list(bad.run)}")
            fig_block_b(B, series, ref, figdir)
            L.append("")

    # ---------------- Block C
    c_runs = list_runs(args.data_dir, args.c_glob)
    C, rows_c = [], []
    if c_runs:
        print(f"Block C: {len(c_runs)} runs")
        curves, source = [], None
        if args.singles_glob:
            source = "same-session single-pulse runs"
            for r in list_runs(args.data_dir, args.singles_glob):
                data, files = read_run(r)
                manifest += [dict(run=r.name, file=str(f), sha256=sha256_file(f)) for f in files]
                s = feature_series(to_cycles(data["as7341"]), args.feature)
                t, y = s.index.values.astype("datetime64[ns]").astype("int64") / 1e9, s.values
                pul = parse_dosing(data["dosing_events"], args.acid_event, args.base_event, args.merge_s, args.max_ml).reset_index(drop=True)
                for k, p in pul.iterrows():
                    tn = epoch(pul.iloc[k + 1]["t"]) if k + 1 < len(pul) else None
                    cv = aligned_curve(t, y, epoch(p["t"]), tn)
                    if cv:
                        curves.append((p["type"], float(p["vol"]), cv[0], cv[1], cv[2]))
        elif args.ref_pulses_csv and args.ref_data_dir:
            source = "Block A single pulses (DIFFERENT session: a changed reagent strength appears as a1/a2 far from 1)"
            for r in list_runs(args.ref_data_dir, args.ref_glob):
                data, files = read_run(r)
                if data["as7341"] is None or data["dosing_events"] is None:
                    continue
                manifest += [dict(run="ref:" + r.name, file=str(f), sha256=sha256_file(f)) for f in files]
                s = feature_series(to_cycles(data["as7341"]), args.feature)
                t, y = s.index.values.astype("datetime64[ns]").astype("int64") / 1e9, s.values
                pul = parse_dosing(data["dosing_events"], args.acid_event, args.base_event, args.merge_s, args.max_ml).reset_index(drop=True)
                for k, p in pul.iterrows():
                    tn = epoch(pul.iloc[k + 1]["t"]) if k + 1 < len(pul) else None
                    cv = aligned_curve(t, y, epoch(p["t"]), tn)
                    if cv:
                        curves.append((p["type"], float(p["vol"]), cv[0], cv[1], cv[2]))
        if not curves:
            L += ["BLOCK C – not analysed: no single-pulse templates. Give --singles-glob (same session) or "
                  "--ref-pulses-csv with --ref-data-dir.", ""]
        else:
            tmpl = build_templates(curves)
            L += ["BLOCK C – paired pulses", "-" * 70, f"templates from: {source}",
                  "templates: " + ", ".join(f"{k[0]} {v['vol']:.2f} mL (n={v['n_curves']})" for k, v in sorted(tmpl.items())), ""]
            pre_b = []
            runs_data = []
            for r in c_runs:
                data, files = read_run(r)
                manifest += [dict(run=r.name, file=str(f), sha256=sha256_file(f)) for f in files]
                s = feature_series(to_cycles(data["as7341"]), args.feature)
                t, y = s.index.values.astype("datetime64[ns]").astype("int64") / 1e9, s.values
                pul = parse_dosing(data["dosing_events"], args.acid_event, args.base_event, args.merge_s, args.max_ml).reset_index(drop=True)
                if len(pul) < 2:
                    L.append(f"! {r.name}: fewer than two pulses logged, skipped")
                    continue
                bl = baseline(t, y, epoch(pul.iloc[0]["t"]))
                if bl:
                    pre_b.append(bl[3])
                runs_data.append((r, t, y, pul))
            rho = pooled_rho(pre_b)
            L.append(f"noise autocorrelation (lag 1, pre-pulse baselines): {rho:.2f}")
            for r, t, y, pul in runs_data:
                p1, p2 = pul.iloc[0], pul.iloc[1]
                res = analyse_pair(t, y, p1, p2, tmpl, rho)
                if res is None or "error" in res:
                    L.append(f"! {r.name}: {res['error'] if res else 'no baseline'}")
                    continue
                res["name"] = r.name
                res["label"] = f"{p1['type']} {p1['vol']:.2f} mL -> {p2['type']} {p2['vol']:.2f} mL, dt {res['dt_s']:.0f} s"
                C.append(res)
                row = {k: v for k, v in res.items() if not k.startswith("_")}
                row.update(pulse1_type=p1["type"], pulse1_volume=float(p1["vol"]), pulse2_type=p2["type"], pulse2_volume=float(p2["vol"]))
                rows_c.append(row)
                L.append(f"{r.name}: {res['label']}")
                L.append(f"   measured peak {res['meas_peak']:.4f}; plain-sum peak {res['sum_peak']:.4f}; gains a1={res['a1']:.2f}+/-{res['a1_se']:.2f}, "
                         f"a2={res['a2']:.2f}+/-{res['a2_se']:.2f}")
                L.append(f"   residual RMS: plain sum {res['rmse0']:.4f} (noise-only 95th pct {res['null_rmse_95']:.4f}, p={res['p_rmse_plain']:.3f}); "
                         f"free gains {res['rmse1']:.4f} (p={res['p_rmse_gain_fit']:.3f})")
                L.append(f"   -> {res['classification']}" + ("   [template tail extended by a fitted exponential]" if res["template_extended"] else ""))
            if rows_c:
                pd.DataFrame(rows_c).to_csv(tabdir / "blockC_pairs.csv", index=False)
                fig_block_c(C, figdir)
                labs = pd.Series([c["classification"] for c in C]).value_counts()
                L += ["", "Block C overall: " + "; ".join(f"{v} x {k}" for k, v in labs.items()),
                      f"Each pair was measured {'more than once' if len(set((r['pulse1_type'], r['pulse2_type']) for r in rows_c)) < len(rows_c) else 'once'};"
                      " a single run per pair has no replicate, so only noise-based significance is available"]
    (out / "summary.txt").write_text("\n".join(L), encoding="utf-8")
    pd.DataFrame(manifest).to_csv(out / "inputs_manifest.csv", index=False)
    (out / "summary.json").write_text(json.dumps(dict(
        script_version=SCRIPT_VERSION, made_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), python=platform.python_version(),
        script_sha256=sha256_file(__file__), settings=dict(CALIB=CALIB, LIMITS=LIMITS, PRE_S=PRE_S, PEAK_S=PEAK_S, DECAY_MAX_S=DECAY_MAX_S,
                                                           HEADROOM_TARGET=HEADROOM_TARGET, ALPHA=ALPHA, N_SIM=N_SIM, N_BOOT=N_BOOT),
        block_b=summaryB, block_c=[{k: v for k, v in r.items()} for r in rows_c]), indent=2, default=float))
    print("\n".join(L))
    print(f"\nAll audited outputs saved to {out}/")


if __name__ == "__main__":
    main()
