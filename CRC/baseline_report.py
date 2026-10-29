#!/usr/bin/env python3
"""
baseline_report.py – charts + numbers for the carrier check (stage 0) and the flow baseline (stage 1)
====================================================================================================

Input  : a sealed run folder (or .zip) written by pio_ratio_monitor.py            (--monitor)
         optionally the Pioreactor export zip for the same period (as7341 + od)   (--export)
Output : <out>/fig_*.png and baseline_report.json

Commands
--------
  stage0   carrier check (one sealed monitor run)
             python baseline_report.py stage0 20261026T134345Z_stage0_carrier_check_carrier_1L.zip --out report_stage0
  step     flow-start step response: tau (residence time) from the relaxation, monitor vs Pioreactor export
             python baseline_report.py step 20261026T173230Z_stage1_baseline_baseline.zip \
                    --export stage1_baseline_20261026_2255-2315_IST.zip --out report_step
  stage1   long flow baseline: in-window time, drift (± CI), noise, Allan deviation, 10 h projection, gates, Q and tau
             python baseline_report.py stage1 20261027T050917Z_stage1_baseline_1hr.zip --q 41 39 --q-minutes 10 \
                    --volume 10 --tau-fit 1.99 --prev 20261026T173230Z_stage1_baseline_baseline.zip --out report_stage1

What the stage 1 analysis does
  * finds the moment the flow started (step in the ratio) and fits  y(t) = y_inf + (y0 - y_inf)·exp(-(t - t_start)/tau)
    -> tau is the vial's residence time (= V/Q for a well-mixed tank) and y_inf the steady carrier value
  * tests for slow drift after the transient (slope ± SE for several window lengths) and says how long the baseline
    would stay inside the target window IF a given slope persisted
  * checks which bands moved, the OD channels, and that the monitor and the Pioreactor database agree

Times: UTC (monitor receive time / Pioreactor `timestamp`).  IST = UTC + 5:30.
Requires: numpy, pandas, scipy, matplotlib
"""
import argparse
import json
import math
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from scipy.optimize import curve_fit

BANDS = [415, 445, 480, 515, 555, 590, 630, 680]
BAND_COLORS = {415: "#5a0fc8", 445: "#2330ff", 480: "#00a8d6", 515: "#2ca02c",
               555: "#9acd00", 590: "#ff9900", 630: "#e31a1c", 680: "#7a0019"}
# 555/480 vs pH from the 2026-10-22 ladder (9 points, probe-calibrated pH): y = ya + (yb-ya)/(1+10^(pKa-pH))
CALIB = dict(ya=1.3586, yb=1.6198, pka=3.741)

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11,
                     "axes.titleweight": "bold", "legend.frameon": False})


# ------------------------------------------------------------------ loading
def _find(root, name):
    hits = sorted(Path(root).rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def load_monitor(path):
    path = Path(path)
    if path.suffix.lower() == ".zip":
        tmp = tempfile.mkdtemp()
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp)
        path = Path(tmp)
    rd = pd.read_csv(_find(path, "readings.csv"))
    rd["t"] = pd.to_datetime(rd["t_utc"]).dt.tz_localize(None)
    an = json.loads(_find(path, "analysis.json").read_text())
    md = json.loads(_find(path, "metadata.json").read_text())
    ev_path = _find(path, "events.csv")
    ev = pd.read_csv(ev_path)
    if len(ev):
        ev["t"] = pd.to_datetime(ev["t_utc"]).dt.tz_localize(None)
    return rd, an, md, ev


def load_export(path):
    path = Path(path)
    frames = {"as7341": [], "od_readings": []}
    with zipfile.ZipFile(path) as z:
        for n in sorted(z.namelist()):
            for k in frames:
                if n.lower().endswith(".csv") and k in n.lower():
                    frames[k].append(pd.read_csv(z.open(n)))
    a, o = pd.concat(frames["as7341"]), pd.concat(frames["od_readings"])
    a["t"] = pd.to_datetime(a["timestamp"], utc=True).dt.tz_localize(None)
    a = a.sort_values("t")
    cyc = (a["t"].diff() > pd.Timedelta("1s")).cumsum()
    w = a.assign(c=cyc).pivot_table(index="c", columns="band", values="reading")
    w.index = pd.DatetimeIndex(a.assign(c=cyc).groupby("c")["t"].first().values)
    w = w.dropna()
    w["r"] = w[555] / w[480]
    o["t"] = pd.to_datetime(o["timestamp"], utc=True).dt.tz_localize(None)
    od = o.pivot_table(index="t", columns="angle", values="od_reading")
    return w, od


# ------------------------------------------------------------------ helpers
def sigmoid_ph(y):
    r = (y - CALIB["ya"]) / (CALIB["yb"] - y)
    return CALIB["pka"] + np.log10(r) if r > 0 else np.nan


def sigmoid_ratio(ph):
    return CALIB["ya"] + (CALIB["yb"] - CALIB["ya"]) / (1 + 10 ** (CALIB["pka"] - ph))


def slope_pct10(t_min, y):
    lr = stats.linregress(t_min, y)
    k = 10 / np.mean(y) * 100
    return lr.slope * k, lr.stderr * k, lr.pvalue


def allan_dev(y, dt, taus):
    out = []
    for tau in taus:
        m = int(round(tau / dt))
        if m < 1 or 2 * m > len(y):
            out.append(np.nan)
            continue
        n = len(y) // m
        avg = y[: n * m].reshape(n, m).mean(axis=1)
        out.append(np.sqrt(0.5 * np.mean(np.diff(avg) ** 2)) if n > 1 else np.nan)
    return np.array(out)


def exp_step(t, ts, y0, yinf, tau):
    return np.where(t < ts, y0, yinf + (y0 - yinf) * np.exp(-(t - ts) / tau))


def fit_flow_step(x, y):
    """Fit a flat start, then an exponential approach to y_inf.  x in minutes."""
    ys = pd.Series(y).rolling(6, center=True, min_periods=1).mean().values
    pre, post = np.median(y[: max(8, len(y) // 5)]), np.median(y[-max(8, len(y) // 5):])
    half = (pre + post) / 2
    idx = np.argmax(ys < half) if pre > post else np.argmax(ys > half)
    p0 = [max(x[idx] - 1.4, x[0] + 0.5), pre, post, 2.5]
    popt, pcov = curve_fit(exp_step, x, y, p0=p0, maxfev=20000)
    return popt, np.sqrt(np.diag(pcov))


# ------------------------------------------------------------------ stage 0 figure
def fig_stage0(rd, an, md, out):
    feat = an["feature"]
    col = "r_" + feat
    lo, hi = an["target_window"]
    t = rd["t"]
    x = (t - t.iloc[0]).dt.total_seconds().values / 60
    y = rd[col].values
    w0 = pd.Timestamp(an["window_start_utc"]).tz_localize(None)
    m = (t >= w0).values
    st = an["features"][feat]
    fig, ax = plt.subplots(2, 3, figsize=(19, 10))
    # a) time series
    a = ax[0, 0]
    a.axhspan(lo, hi, color="green", alpha=0.13, label=f"target {lo}–{hi}")
    a.axvspan(x[m][0], x[m][-1], color="tab:blue", alpha=0.12, label="analysis window")
    a.fill_between(x[m], st["median"] - st["sd"], st["median"] + st["sd"], color="tab:blue", alpha=0.25, label="± 1 SD")
    a.plot(x, y, "o-", ms=2.5, lw=0.8, color="k", label=feat)
    a.axhline(st["median"], color="tab:blue", ls="--", lw=1.2, label=f"median {st['median']:.4f}")
    a.set_xlabel("minutes since first cycle")
    a.set_ylabel(feat)
    a.set_title(f"Carrier check – {feat} over time")
    a.legend(fontsize=8, loc="best")
    # b) bands % change
    b = ax[0, 1]
    base = rd[rd["t_epoch"] <= rd["t_epoch"].iloc[0] + 60][[f"b{k}" for k in BANDS]].mean()
    for k in BANDS:
        b.plot(x, (rd[f"b{k}"] / base[f"b{k}"] - 1) * 100, lw=1, color=BAND_COLORS[k], label=f"{k} nm")
    b.plot(x, (rd[col] / rd[col].iloc[:12].mean() - 1) * 100, lw=2.2, color="k", label=feat)
    b.set_xlabel("minutes since first cycle")
    b.set_ylabel("% change vs first minute")
    b.set_title("Which bands move?")
    b.legend(ncol=3, fontsize=7)
    # c) histogram
    c = ax[0, 2]
    c.hist(y[m], bins=14, color="tab:blue", alpha=0.6, density=True)
    g = np.linspace(y[m].min() - 0.002, y[m].max() + 0.002, 200)
    c.plot(g, stats.norm.pdf(g, y[m].mean(), y[m].std(ddof=1)), "k-", lw=1.4, label="normal fit")
    c.axvline(st["median"], color="tab:blue", ls="--")
    c.set_xlabel(feat)
    c.set_ylabel("density")
    c.set_title(f"Noise in the analysis window (SD {st['sd']:.4f}, CV {100 * st['sd'] / st['mean']:.3f}%)")
    c.legend(fontsize=8)
    # d) headroom on the calibrated curve
    d = ax[1, 0]
    ph = np.linspace(2.4, 5.4, 300)
    d.plot(ph, sigmoid_ratio(ph), "k-", lw=2)
    d.axhline(CALIB["ya"], color="tab:red", ls=":", lw=1)
    d.axhline(CALIB["yb"], color="gold", ls=":", lw=1.5)
    d.text(2.45, CALIB["ya"] + 0.004, "acid plateau", color="tab:red", fontsize=8)
    d.text(2.45, CALIB["yb"] - 0.012, "neutral plateau", color="goldenrod", fontsize=8)
    d.axhspan(lo, hi, color="green", alpha=0.15)
    med = st["median"]
    p_med = sigmoid_ph(med)
    d.plot([p_med], [med], "o", ms=11, color="tab:blue", zorder=5)
    d.annotate(f"baseline {med:.3f}\n(pH ≈ {p_med:.2f})", (p_med, med), xytext=(14, -38), textcoords="offset points", fontsize=9)
    d.annotate("", xy=(p_med, CALIB["ya"]), xytext=(p_med, med), arrowprops=dict(arrowstyle="<->", color="tab:red"))
    d.annotate("", xy=(p_med + 0.18, CALIB["yb"]), xytext=(p_med + 0.18, med), arrowprops=dict(arrowstyle="<->", color="goldenrod"))
    d.text(p_med + 0.04, (med + CALIB["ya"]) / 2, f"acid room\n{med - CALIB['ya']:.3f}", fontsize=8, color="tab:red")
    d.text(p_med + 0.25, (med + CALIB["yb"]) / 2, f"base room\n{CALIB['yb'] - med:.3f}", fontsize=8, color="goldenrod")
    d.axhline(1.58, color="grey", ls="--", lw=0.8)
    d.text(4.75, 1.583, "ratio 1.58 ≈ pH 4.5:\nbase pulses clip", fontsize=8, color="grey")
    d.set_xlabel("pH (from the 2026-10-22 ladder calibration)")
    d.set_ylabel(feat)
    d.set_title("Headroom: why the baseline should sit mid-curve")
    # e) Allan deviation
    e = ax[1, 1]
    dt = float(np.median(np.diff(rd["t_epoch"])))
    taus = np.array([dt * 2 ** k for k in range(0, 8)])
    ad = allan_dev(y, dt, taus)
    e.loglog(taus, ad, "o-", color="tab:blue", label="measured")
    ok = np.isfinite(ad)
    if ok.any():
        e.loglog(taus[ok], ad[ok][0] * np.sqrt(taus[ok][0] / taus[ok]), "k--", lw=1, label="white noise (1/√τ)")
    e.set_xlabel("averaging time τ (s)")
    e.set_ylabel("Allan deviation of the ratio")
    e.set_title("Noise vs averaging (flat/rising = drift)")
    e.legend(fontsize=8)
    # f) summary text
    f = ax[1, 2]
    f.axis("off")
    dc = an["drift_check"]
    lines = [f"Verdict: {an['verdict']}  ({an.get('verdict_text', '')})", "",
             f"Run:  {md['stage']} / {md['label']}", f"Note: {md.get('notes', '')}", "",
             f"{feat} median  {st['median']:.4f}", f"SD  {st['sd']:.4f}   CV {100 * st['sd'] / st['mean']:.3f}%",
             f"window  {an['window_seconds']:.0f} s,  n = {an['n_cycles_window']} cycles",
             f"drift over last {dc['window_seconds']:.0f} s:  {dc['drift_pct_per_10min']:+.2f} ± {dc['se']:.2f} %/10 min (p = {dc['p']:.2f})",
             f"target window {lo}–{hi}", f"implied pH ≈ {sigmoid_ph(st['median']):.2f} (ladder sigmoid)", "",
             f"cycles {an['n_cycles_total']},  dropped {an['n_incomplete_cycles_dropped']}",
             f"started {md['started_utc']}"]
    f.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9.5)
    fig.suptitle("Stage 0 – carrier baseline check", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "fig_stage0_carrier_check.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dict(median=st["median"], sd=st["sd"], implied_pH=float(sigmoid_ph(st["median"])),
                drift_pct_per_10min=dc["drift_pct_per_10min"], drift_se=dc["se"], verdict=an["verdict"])


# ------------------------------------------------------------------ stage 1 figures
def fig_stage1(rd, an, md, ev, w, od, out):
    lo, hi = an["target_window"]
    t0 = w.index[0]
    x = (w.index - t0).total_seconds().values / 60
    y = w["r"].values
    popt, perr = fit_flow_step(x, y)
    ts, y0, yinf, tau = popt
    resid = y - exp_step(x, *popt)
    flow_start = t0 + pd.Timedelta(minutes=float(ts))
    res = dict(flow_start_utc=str(flow_start), y0=float(y0), y_inf=float(yinf), y_inf_se=float(perr[2]),
               tau_min=float(tau), tau_se=float(perr[3]), step_size=float(yinf - y0),
               implied_Q_mL_per_min_for_10mL=float(10 / tau), residual_sd=float(resid.std(ddof=4)))
    # monitor vs database agreement
    mon = rd[["t", "r_" + an["feature"]]].rename(columns={"r_" + an["feature"]: "r_mon"}).sort_values("t")
    db = w[["r"]].reset_index()
    db.columns = ["td", "r_db"]
    mm = pd.merge_asof(mon, db.sort_values("td"), left_on="t", right_on="td", direction="nearest",
                       tolerance=pd.Timedelta("3s")).dropna()
    res["monitor_vs_db"] = dict(n=int(len(mm)), mean_diff=float((mm.r_mon - mm.r_db).mean()),
                                sd_diff=float((mm.r_mon - mm.r_db).std()))
    # ---- fig 1: relaxation
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.6), gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
    a = ax[0]
    a.axhspan(lo, hi, color="green", alpha=0.12, label=f"target {lo}–{hi}")
    a.plot(w.index, y, ".", ms=3, color="0.35", label="Pioreactor export (5 s)")
    a.plot(mon["t"], mon["r_mon"], ".", ms=3, color="tab:orange", alpha=0.7, label="monitor (live MQTT)")
    gt = np.linspace(x[0], x[-1], 600)
    a.plot(t0 + pd.to_timedelta(gt, unit="m"), exp_step(gt, *popt), "r-", lw=2,
           label=f"exp fit: τ = {tau:.2f} ± {perr[3]:.2f} min\nasymptote {yinf:.4f}")
    a.axvline(flow_start, color="tab:blue", ls="--", lw=1)
    a.text(flow_start, hi - 0.0005, " flow start", color="tab:blue", fontsize=8, va="top")
    if len(ev):
        for _, r in ev.iterrows():
            a.axvline(r["t"], color="tab:red", lw=0.8)
            a.text(r["t"], lo + 0.0005, " " + str(r["text"])[:25], rotation=90, fontsize=7, color="tab:red")
    a.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    a.set_xlabel("UTC time (IST = UTC + 5:30)")
    a.set_ylabel(an["feature"])
    a.set_title("Flow start and relaxation to the carrier value")
    a.legend(fontsize=8)
    b = ax[1]
    b.axhline(0, color="k", lw=0.8)
    b.plot(x, resid * 1000, ".", ms=3, color="0.4")
    b.set_xlabel("minutes since export start")
    b.set_ylabel("residual of the fit (×10⁻³)")
    b.set_title("Fit residuals (structure = slower component)")
    c = ax[2]
    post = x > ts
    z = (y[post] - yinf) / (y0 - yinf)
    keep = z > 0.004
    c.semilogy(x[post][keep] - ts, z[keep], ".", ms=4, color="0.4", label="data")
    c.semilogy(gt[gt > ts] - ts, np.exp(-(gt[gt > ts] - ts) / tau), "r-", label=f"single exponential, τ = {tau:.2f} min")
    c.set_xlabel("minutes since flow start")
    c.set_ylabel("remaining distance to asymptote (fraction)")
    c.set_title("Straight line = one mixing time constant")
    c.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_stage1_relaxation.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # ---- fig 2: drift checks
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    a = ax[0, 0]
    # drift tests use the export plus any monitor cycles recorded after the export ended
    tail = mon[mon["t"] > w.index[-1]]
    x_all = np.concatenate([x, (tail["t"] - t0).dt.total_seconds().values / 60])
    y_all = np.concatenate([y, tail["r_mon"].values])
    res["drift_series_end_utc"] = str(t0 + pd.Timedelta(minutes=float(x_all[-1])))
    t_end = x_all[-1]
    lens = [4, 6, 8, 10, 12, 14]
    rows = []
    for L in lens:
        m = x_all >= t_end - L
        if m.sum() > 10 and (x_all[m][0] > ts + 3 * tau):
            s, se, p = slope_pct10(x_all[m], y_all[m])
            rows.append((L, s, se, p))
    if rows:
        L_, s_, se_, p_ = (np.array(v) for v in zip(*rows))
        a.errorbar(L_, s_, yerr=1.96 * se_, fmt="o", color="tab:blue", capsize=4, label="slope ± 95% CI")
        for lv in (0.2, 0.05):
            a.axhline(-lv, color="red", ls="--", lw=0.8)
            a.axhline(lv, color="red", ls="--", lw=0.8)
        a.axhline(0, color="k", lw=0.8)
        a.text(L_.min(), 0.21, "±0.2 %/10 min (monitor threshold)", color="red", fontsize=8)
        a.text(L_.min(), 0.06, "±0.05 (needed for a 10 h run)", color="red", fontsize=8)
    a.set_xlabel("length of the final window used (min; windows start after 3τ)")
    a.set_ylabel("drift (% per 10 min)")
    a.set_title("Is there residual drift after the transient?")
    a.legend(fontsize=8)
    res["late_drift"] = [dict(window_min=float(r[0]), pct_per_10min=float(r[1]), se=float(r[2]), p=float(r[3])) for r in rows]
    # exit times
    exits = {}
    for s in (-0.17, -0.1, -0.05, -0.02):
        per_min = abs(s) / 100 * yinf / 10
        exits[f"{s:+.2f} %/10min"] = float((yinf - lo) / per_min)
    res["minutes_to_reach_lower_edge_if_slope_persists"] = exits
    b = ax[0, 1]
    b.axis("off")
    txt = ["If a slope persisted, minutes until the baseline", f"({yinf:.3f}) reaches the lower edge {lo}:", ""]
    txt += [f"  {k:>16s}  ->  {v:6.0f} min  ({v / 60:.1f} h)" for k, v in exits.items()]
    txt += ["", f"Single-exponential fit: τ = {tau:.2f} ± {perr[3]:.2f} min", f"asymptote {yinf:.4f} ± {perr[2]:.4f}",
            f"implied Q for a 10 mL vial ≈ {10 / tau:.1f} mL/min", "",
            "Headroom budget (rule of thumb):", "  drift allowed over a 10 h run ≈ ±0.04 of the ratio",
            "  = about 0.05 %/10 min;  Stage 2 block A (2.4 h): ≈ 0.1 %/10 min"]
    b.text(0, 1, "\n".join(txt), va="top", family="monospace", fontsize=10)
    c = ax[1, 0]
    pre = w[(x > 1) & (x < ts - 0.3)].mean()
    post_ = w[x > ts + 6 * tau].mean()
    labs = [f"{k}" for k in BANDS] + [an["feature"]]
    vals = [(post_[k] / pre[k] - 1) * 100 for k in BANDS] + [(post_["r"] / pre["r"] - 1) * 100]
    cols = [BAND_COLORS[k] for k in BANDS] + ["k"]
    c.bar(labs, vals, color=cols)
    c.axhline(0, color="k", lw=0.8)
    c.set_ylabel("% change: after flow settles vs before flow")
    c.set_title("Which bands changed when carrier replaced the vial content?")
    res["band_change_pct"] = {l: float(v) for l, v in zip(labs, vals)}
    d = ax[1, 1]
    for ang, colr in ((45, "tab:blue"), (135, "tab:orange")):
        if ang in od.columns and od[ang].std() > 0:
            d.plot(od.index, od[ang], lw=1, color=colr, label=f"OD {ang}°")
    d.axvline(flow_start, color="tab:blue", ls="--", lw=1)
    d.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    d.set_ylabel("OD reading")
    d.set_xlabel("UTC time")
    d.set_title("OD channels (90° reads all zeros)")
    d.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_stage1_drift_and_checks.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return res


# ------------------------------------------------------------------ stage 1: long baseline
def binned(x_min, y, width=1.0):
    df = pd.DataFrame({"b": np.floor(x_min / width), "y": y}).groupby("b").agg(y=("y", "mean"))
    return (df.index.values + 0.5) * width, df["y"].values


def trend_stats(x_min, y):
    """slope in %/10 min with a standard error from 1-minute means (white-noise points would understate it)."""
    xb, yb = binned(x_min, y, 1.0)
    lr = stats.linregress(xb, yb)
    k = 10 / np.mean(yb) * 100
    return dict(slope_pct10=lr.slope * k, se_pct10=lr.stderr * k, p=lr.pvalue, n_bins=len(xb),
                slope_per_min=lr.slope, intercept=lr.intercept, x_mean=float(np.mean(xb)))


def longest_in_window(t_s, y, lo, hi, roll=12):
    med = pd.Series(y).rolling(roll, center=True, min_periods=roll // 2).median().values
    ok = (med >= lo) & (med <= hi)
    best, start = 0.0, None
    for i, v in enumerate(ok):
        if v and start is None:
            start = i
        if (not v or i == len(ok) - 1) and start is not None:
            end = i if v else i - 1
            best = max(best, (t_s[end] - t_s[start]) / 60)
            start = None
    return best


def hydraulics(q_vols, q_minutes, volume, tau_fit):
    q = np.array(q_vols, float) / q_minutes
    out = dict(q_runs_mL_per_min=q.tolist(), Q_mean=float(q.mean()), Q_sd=float(q.std(ddof=1)) if len(q) > 1 else None,
               volume_mL=volume, tau_from_V_over_Q_min=float(volume / q.mean()))
    if tau_fit:
        out.update(tau_step_fit_min=float(tau_fit), effective_mixed_volume_mL=float(q.mean() * tau_fit),
                   effective_volume_fraction=float(q.mean() * tau_fit / volume))
    taus = [out["tau_from_V_over_Q_min"]] + ([tau_fit] if tau_fit else [])
    ts_opt = [t / 3 * 60 for t in taus]
    out["step_s_tau_over_3"] = [float(v) for v in ts_opt]
    return out


def fig_stage1_long(rd, an, md, out, a):
    feat = an["feature"]
    lo, hi = an["target_window"]
    t = rd["t"]
    x = (t - t.iloc[0]).dt.total_seconds().values / 60
    ts_s = (t - t.iloc[0]).dt.total_seconds().values
    y = rd["r_" + feat].values
    k0 = x >= a.settle_min
    x, y, ts_s = x[k0], y[k0], ts_s[k0]
    tr = trend_stats(x, y)
    xe = x[-1]
    y_end = tr["intercept"] + tr["slope_per_min"] * xe
    se_min = tr["se_pct10"] / 100 * np.mean(y) / 10             # slope SE per minute
    res = dict(feature=feat, n_cycles=len(y), duration_min=float(x[-1] - x[0]), settle_min=a.settle_min,
               median_all=float(np.median(y)), first_5min_median=float(np.median(y[x < x[0] + 5])),
               last_5min_median=float(np.median(y[x > xe - 5])), noise_sd_median_of_5min_bins=None)
    # occupancy
    inw = (y >= lo) & (y <= hi)
    res["fraction_in_window"] = float(inw.mean())
    res["minutes_in_window_total"] = float(inw.sum() * np.median(np.diff(ts_s)) / 60)
    res["longest_continuous_in_window_min"] = float(longest_in_window(ts_s, y, lo, hi))
    # noise
    sd5 = pd.Series(y).groupby(np.floor(x / 5)).std().values
    res["noise_sd_median_of_5min_bins"] = float(np.median(sd5))
    res["noise_sd_max_5min_bin"] = float(np.max(sd5))
    # drift windows
    dw = []
    for L in (10, 20, 30, 45, x[-1] - x[0]):
        m = x >= xe - L
        if m.sum() > 20:
            s = trend_stats(x[m], y[m])
            dw.append(dict(window_min=float(min(L, x[-1] - x[0])), pct_per_10min=float(s["slope_pct10"]),
                           se=float(s["se_pct10"]), p=float(s["p"])))
    res["drift_windows"] = dw
    res["drift_full"] = dict(pct_per_10min=float(tr["slope_pct10"]), se=float(tr["se_pct10"]), p=float(tr["p"]))
    # curvature
    xb, yb = binned(x, y, 1.0)
    c2 = np.polyfit(xb - xb.mean(), yb, 2)
    resid1 = yb - np.polyval(np.polyfit(xb - xb.mean(), yb, 1), xb - xb.mean())
    resid2 = yb - np.polyval(c2, xb - xb.mean())
    F = (np.sum(resid1 ** 2) - np.sum(resid2 ** 2)) / (np.sum(resid2 ** 2) / (len(xb) - 3))
    res["curvature"] = dict(quadratic_coef_per_min2=float(c2[0]), F=float(F), p=float(1 - stats.f.cdf(F, 1, len(xb) - 3)))
    # projections
    per_min = tr["slope_per_min"]
    worst = per_min - 1.96 * se_min
    proj = dict(y_end_fit=float(y_end),
                minutes_to_lower_edge_central=float((y_end - lo) / abs(per_min)) if per_min < 0 else None,
                minutes_to_lower_edge_worst95=float((y_end - lo) / abs(worst)) if worst < 0 else None,
                ratio_after_10h_central=float(y_end + per_min * 600), ratio_after_10h_worst95=float(y_end + worst * 600),
                note="straight-line extrapolation of this run; drift may slow (see curvature / between-run drift)")
    rec = next((d for d in dw if abs(d["window_min"] - 20) < 1e-6), None)
    if rec:
        pm_r = rec["pct_per_10min"] / 100 * np.mean(y) / 10
        se_r = rec["se"] / 100 * np.mean(y) / 10
        worst_r = pm_r - 1.96 * se_r
        proj.update(recent20_pct_per_10min=rec["pct_per_10min"],
                    minutes_to_lower_edge_recent_central=float((y_end - lo) / abs(pm_r)) if pm_r < 0 else None,
                    minutes_to_lower_edge_recent_worst95=float((y_end - lo) / abs(worst_r)) if worst_r < 0 else None,
                    ratio_after_10h_recent_central=float(y_end + pm_r * 600))
    res["projection_from_end_of_run"] = proj
    # gates
    ci = 1.96 * tr["se_pct10"]
    gates = [
        ("in window >= 30 min continuous (1.47-1.51)", res["longest_continuous_in_window_min"] >= 30,
         f"{res['longest_continuous_in_window_min']:.1f} min; {100 * res['fraction_in_window']:.1f}% of readings"),
        ("|drift| <= 0.2 %/10 min (stage gate)", abs(tr["slope_pct10"]) <= 0.2 and abs(tr["slope_pct10"]) + ci <= 0.2,
         f"{tr['slope_pct10']:+.3f} ± {ci:.3f} (95% CI)"),
        ("|drift| <= 0.05 %/10 min (10 h run guideline)", abs(tr["slope_pct10"]) <= 0.05,
         f"{tr['slope_pct10']:+.3f}; CI {'crosses' if abs(tr['slope_pct10']) + ci > 0.05 else 'stays within'} the limit"),
        ("recent drift (last 20 min) <= 0.05 %/10 min (not accelerating)",
         bool(rec and abs(rec["pct_per_10min"]) <= 0.05),
         (f"{rec['pct_per_10min']:+.3f} ± {1.96 * rec['se']:.3f} (95% CI); curvature p = {res['curvature']['p']:.1g}" if rec else "n/a")),
        ("noise SD per cycle <= 0.003 (design sigma)", res["noise_sd_median_of_5min_bins"] <= 0.003,
         f"median {res['noise_sd_median_of_5min_bins']:.4f}, worst bin {res['noise_sd_max_5min_bin']:.4f}"),
    ]
    res["gates"] = [dict(criterion=g[0], passed=bool(g[1]), detail=g[2]) for g in gates]
    # hydraulics + planning
    hyd = None
    if a.q:
        hyd = hydraulics(a.q, a.q_minutes, a.volume, a.tau_fit)
        res["hydraulics"] = hyd
        Q = hyd["Q_mean"]
        res["carrier_budget_L"] = dict(main_run=float(Q * a.steps * a.step_s / 60 / 1000),
                                       steps=a.steps, step_s=a.step_s, stage2_blockA_48tau=float(48 * a.volume / 1000))
    # between-run drift
    if a.prev:
        prd, pan, pmd, _ = load_monitor(a.prev)
        pc = "r_" + pan["feature"]
        p_end = float(np.median(prd[pc].values[prd["t"] >= prd["t"].iloc[-1] - pd.Timedelta(minutes=5)]))
        c_start = float(np.median(y[x < x[0] + 5]))
        gap_min = (t.iloc[0] - prd["t"].iloc[-1]).total_seconds() / 60
        res["between_runs"] = dict(prev_run=pmd["stage"] + "/" + pmd["label"], prev_end_median=p_end,
                                   this_start_median=c_start, gap_hours=gap_min / 60, change=c_start - p_end,
                                   avg_pct_per_10min=float((c_start - p_end) / p_end * 100 / (gap_min / 10)),
                                   note="only meaningful if the same carrier bottle and vial setup were used")
    # ---------------- figure
    fig, ax = plt.subplots(2, 3, figsize=(20, 10.5))
    A = ax[0, 0]
    A.axhspan(lo, hi, color="green", alpha=0.13, label=f"target {lo}–{hi}")
    A.plot(x, y, ".", ms=2.5, color="0.5", label="cycles (5 s)")
    A.plot(x, pd.Series(y).rolling(60, center=True, min_periods=20).median(), color="k", lw=1.5, label="5-min rolling median")
    g = np.linspace(x[0], x[-1], 100)
    xb, yb = binned(x, y, 1.0)
    sxx = np.sum((xb - xb.mean()) ** 2)
    s_res = np.sqrt(np.sum((yb - (tr["intercept"] + per_min * xb)) ** 2) / (len(xb) - 2))
    band = 1.96 * s_res * np.sqrt(1 / len(xb) + (g - xb.mean()) ** 2 / sxx)
    A.fill_between(g, tr["intercept"] + per_min * g - band, tr["intercept"] + per_min * g + band, color="tab:red", alpha=0.25)
    A.plot(g, tr["intercept"] + per_min * g, color="tab:red", lw=1.5,
           label=f"linear trend {tr['slope_pct10']:+.3f} ± {ci:.3f} %/10 min")
    A.set_xlabel(f"minutes since {t.iloc[0]:%Y-%m-%d %H:%M:%S} UTC")
    A.set_ylabel(feat)
    A.set_title("One-hour flow baseline")
    A.legend(fontsize=8, loc="best")
    B = ax[0, 1]
    xs = np.arange(len(sd5)) * 5 + 2.5
    B.bar(xs, sd5 * 1000, width=4.2, color="tab:blue", alpha=0.7)
    B.axhline(3, color="red", ls="--", lw=0.8)
    B.text(xs[0], 3.05, "design sigma limit 0.003", color="red", fontsize=8)
    B.set_xlabel("minutes")
    B.set_ylabel("SD within 5-min bin (×10⁻³)")
    B.set_title("Noise over time")
    C = ax[0, 2]
    dt = float(np.median(np.diff(ts_s)))
    taus = np.array([dt * 2 ** k for k in range(0, 10)])
    ad = allan_dev(y, dt, taus)
    C.loglog(taus / 60, ad, "o-", color="tab:blue", label="measured")
    ok = np.isfinite(ad)
    C.loglog(taus[ok] / 60, ad[ok][0] * np.sqrt(taus[ok][0] / taus[ok]), "k--", lw=1, label="white noise (1/√τ)")
    C.set_xlabel("averaging time τ (min)")
    C.set_ylabel("Allan deviation")
    C.set_title("Where drift takes over from noise")
    C.legend(fontsize=8)
    D = ax[1, 0]
    if dw:
        Ls = [d["window_min"] for d in dw]
        D.errorbar(range(len(dw)), [d["pct_per_10min"] for d in dw], yerr=[1.96 * d["se"] for d in dw], fmt="o",
                   capsize=4, color="tab:blue")
        D.set_xticks(range(len(dw)), [f"last {L:.0f} min" for L in Ls], rotation=20)
    for lv in (0.2, 0.05):
        D.axhline(lv, color="red", ls="--", lw=0.8)
        D.axhline(-lv, color="red", ls="--", lw=0.8)
    D.axhline(0, color="k", lw=0.8)
    D.set_ylabel("drift (% per 10 min, ±95% CI)")
    D.set_title("Drift vs window length (red = 0.2 and 0.05)")
    E = ax[1, 1]
    hh = np.linspace(0, 10, 60)
    E.axhspan(lo, hi, color="green", alpha=0.13, label="target 1.47–1.51")
    E.axhspan(1.45, 1.54, color="green", alpha=0.07, label="workable 1.45–1.54")
    E.plot(hh, y_end + per_min * hh * 60, color="tab:red", label=f"full-run slope {tr['slope_pct10']:+.3f} %/10 min")
    E.fill_between(hh, y_end + worst * hh * 60, y_end + (per_min + 1.96 * se_min) * hh * 60, color="tab:red", alpha=0.2)
    if rec:
        E.plot(hh, y_end + pm_r * hh * 60, color="tab:orange", label=f"last-20-min slope {rec['pct_per_10min']:+.3f} %/10 min")
        E.fill_between(hh, y_end + worst_r * hh * 60, y_end + (pm_r + 1.96 * se_r) * hh * 60, color="tab:orange", alpha=0.2)
    E.set_ylim(1.42, 1.55)
    E.set_xlabel("hours after the end of this run")
    E.set_ylabel(feat)
    E.set_title("If this drift persisted (pessimistic)")
    E.legend(fontsize=8)
    F_ = ax[1, 2]
    F_.axis("off")
    lines = [f"Run: {md['stage']} / {md['label']}", f"{res['n_cycles']} cycles, {res['duration_min']:.0f} min", "",
             f"{feat} median {res['median_all']:.4f}   first 5 min {res['first_5min_median']:.4f}",
             f"last 5 min {res['last_5min_median']:.4f}", ""]
    for gname, gp, gd in gates:
        lines += [f"[{'PASS' if gp else 'FAIL'}] {gname}", f"       {gd}"]
    lines += ["", f"curvature p = {res['curvature']['p']:.2f}  (drift {'is slowing/accelerating' if res['curvature']['p'] < 0.05 else 'is ~constant'})"]
    if hyd:
        lines += ["", f"Q = {hyd['Q_mean']:.2f} mL/min   V/Q = {hyd['tau_from_V_over_Q_min']:.2f} min"]
        if "tau_step_fit_min" in hyd:
            lines += [f"step-fit tau = {hyd['tau_step_fit_min']:.2f} min -> V_eff = {hyd['effective_mixed_volume_mL']:.1f} mL"]
        lines += [f"step time tau/3 = {', '.join(f'{v:.0f} s' for v in hyd['step_s_tau_over_3'])}"]
    if "between_runs" in res:
        b = res["between_runs"]
        lines += ["", f"vs previous run ({b['gap_hours']:.1f} h earlier): {b['change']:+.4f}  ({b['avg_pct_per_10min']:+.3f} %/10 min avg)"]
    F_.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9)
    fig.suptitle("Stage 1 – long flow baseline", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "fig_stage1_long_baseline.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # plain-text gate summary (kept with the evidence)
    txt = [f"STAGE 1 GATE SUMMARY  ({md['stage']}/{md['label']}, started {md['started_utc']})", ""]
    txt += [f"[{'PASS' if g['passed'] else 'FAIL'}] {g['criterion']}  -  {g['detail']}" for g in res["gates"]]
    (out / "stage1_gate_summary.txt").write_text("\n".join(txt) + "\n")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p0 = sub.add_parser("stage0", help="carrier check (monitor run)")
    p0.add_argument("monitor")
    p0.add_argument("--out", default="report_stage0")
    p1 = sub.add_parser("step", help="flow-start step response (monitor run + Pioreactor export)")
    p1.add_argument("monitor")
    p1.add_argument("--export", required=True)
    p1.add_argument("--out", default="report_step")
    p2 = sub.add_parser("stage1", help="long flow baseline (monitor run)")
    p2.add_argument("monitor")
    p2.add_argument("--out", default="report_stage1")
    p2.add_argument("--q", type=float, nargs="+", help="effluent volumes (mL) collected in --q-minutes, e.g. 41 39")
    p2.add_argument("--q-minutes", type=float, default=10)
    p2.add_argument("--volume", type=float, default=10, help="vial working volume in mL")
    p2.add_argument("--tau-fit", type=float, help="tau (min) from the step-response fit, to estimate the mixed volume")
    p2.add_argument("--prev", help="earlier monitor run (same carrier/setup) for the between-run comparison")
    p2.add_argument("--settle-min", type=float, default=0, help="ignore the first N minutes of the run")
    p2.add_argument("--steps", type=int, default=600)
    p2.add_argument("--step-s", type=float, default=45)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rd, an, md, ev = load_monitor(a.monitor)
    if a.cmd == "stage0":
        report = {"run": md["stage"] + "/" + md["label"], "stage0": fig_stage0(rd, an, md, out)}
    elif a.cmd == "step":
        w, od = load_export(a.export)
        report = {"run": md["stage"] + "/" + md["label"], "step_response": fig_stage1(rd, an, md, ev, w, od, out)}
    else:
        report = {"run": md["stage"] + "/" + md["label"], "stage1_long": fig_stage1_long(rd, an, md, out, a)}
    (out / "baseline_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
