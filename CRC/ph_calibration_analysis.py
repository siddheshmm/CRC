#!/usr/bin/env python3
"""
pH-indicator calibration analysis for the Pioreactor (AS7341 spectrometer + OD channels)
=======================================================================================

Usage
-----
    python ph_calibration_analysis.py exports_20261021_1700-1821_IST.zip
    python ph_calibration_analysis.py path/to/extracted_folder --out results --window-min 3
    python ph_calibration_analysis.py data.zip --mode centered      # fixed centred window
    python ph_calibration_analysis.py data.zip --mode manual        # uses MANUAL_WINDOWS below

Input : the Pioreactor export zip (or the extracted folder) containing
          as7341_spectrum_readings*.csv  and  od_readings*.csv
Output: <out>/figures/*.png   <out>/tables/*.csv   <out>/summary.txt

Timestamps
----------
All windows below are in *Nepal time* = the CSV column `timestamp_localtime`
(UTC+5:45).  The Pioreactor UI shows IST (UTC+5:30), i.e. 15 min earlier.

Requires: numpy, pandas, scipy, matplotlib   (no seaborn / statsmodels needed)
"""
import argparse
import sys
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =====================================================================================
# 1. CONFIGURATION  (edit this block)
# =====================================================================================
DATE = "2026-10-21"

# Recording windows are the Nepal-time (timestamp_localtime) windows from your table.
# ph_lo / ph_hi = pH-strip reading range; the strip midpoint is used as the x-value
# and half the range (minimum 0.5) as its uncertainty.
SAMPLES = {
    "A": dict(name="Distilled water",               short="Dist. water",    ph_lo=5,  ph_hi=6,  start="17:20:00", end="17:30:00"),
    "B": dict(name="Diluted vinegar",               short="Dil. vinegar",   ph_lo=3,  ph_hi=4,  start="17:55:00", end="18:05:00"),
    "C": dict(name="Diluted baking soda",           short="Dil. soda",      ph_lo=10, ph_hi=10, start="18:25:00", end="18:35:00"),
    "D": dict(name="Vinegar + baking soda",         short="Vinegar+soda",   ph_lo=9,  ph_hi=10, start="17:35:00", end="17:45:00"),
    "E": dict(name="Methyl orange 5 mL + water 5 mL", short="MO + water",   ph_lo=5,  ph_hi=6,  start="18:10:00", end="18:20:00"),
}

# --- analysis-window selection -------------------------------------------------------
WINDOW_MIN = 3.0            # length of the analysis window (minutes)  -> 2-3 min is fine
WINDOW_MODE = "auto_stable"  # "auto_stable" : most stable WINDOW_MIN inside each recording
                             # "centered"    : WINDOW_MIN centred in each recording
                             # "manual"      : use MANUAL_WINDOWS (falls back to auto_stable)
TRIM_START_S = 60           # ignore first N s of each recording when searching (handling/settling)
TRIM_END_S = 15             # ignore last N s
MANUAL_WINDOWS = {          # only used with mode "manual", Nepal time
    # "A": ("17:23:00", "17:26:00"),
}

# --- processing / calibration --------------------------------------------------------
REMOVE_OUTLIERS = True      # drop points > MAD_THRESH robust-sigmas from the window median
MAD_THRESH = 5.0
REF_BAND = 680              # band used to normalise away intensity changes (no MO absorption)
REF_SAMPLES = ("A", "C")    # reference samples for the relative-absorbance spectra
PRIMARY_FEATURE = "555/445"  # chosen a priori: green/blue ratio (MO acid form absorbs green,
                             # base form absorbs blue). Set to None to use the best-ranked one.
CAL_SAMPLES = ["A", "B", "C", "D", "E"]   # samples used in the calibration fits
STRIP_MIN_HALF_WIDTH = 0.5   # pH-strip uncertainty is at least +/- this

BANDS = [415, 445, 480, 515, 555, 590, 630, 680]
BAND_COLORS = {415: "#5a0fc8", 445: "#2330ff", 480: "#00a8d6", 515: "#2ca02c",
               555: "#9acd00", 590: "#ff9900", 630: "#e31a1c", 680: "#7a0019"}
SAMPLE_COLORS = {"A": "#7f7f7f", "B": "#d62728", "C": "#1f77b4", "D": "#9467bd", "E": "#ff7f0e"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})

# =====================================================================================
# 2. LOADING
# =====================================================================================
def read_csv_from(source, pattern):
    """Read every CSV whose path contains `pattern` from a zip or a folder."""
    source = Path(source)
    frames = []
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as z:
            for n in sorted(z.namelist()):
                if n.lower().endswith(".csv") and pattern in n.lower():
                    frames.append(pd.read_csv(z.open(n)))
    else:
        for f in sorted(source.rglob("*.csv")):
            if pattern in str(f).lower():
                frames.append(pd.read_csv(f))
    if not frames:
        raise FileNotFoundError(f"No CSV matching '{pattern}' found in {source}")
    return pd.concat(frames, ignore_index=True)


def to_wide(df, key, value):
    """Long format -> one row per measurement cycle (all channels of one cycle share a row)."""
    df = df.copy()
    df["t"] = pd.to_datetime(df["timestamp_localtime"])
    df = df.sort_values("t")
    df["cycle"] = (df["t"].diff() > pd.Timedelta("1s")).cumsum()
    w = df.pivot_table(index="cycle", columns=key, values=value, aggfunc="mean")
    tcyc = df.groupby("cycle")["t"].first()
    w.index = pd.DatetimeIndex(tcyc.loc[w.index].values, name="time")
    w.columns.name = None
    return w.dropna()


def load_data(source):
    spec_raw = read_csv_from(source, "as7341")
    od_raw = read_csv_from(source, "od_readings")
    # sanity check of the clock offset between UTC `timestamp` and `timestamp_localtime`
    off = (pd.to_datetime(spec_raw["timestamp_localtime"])
           - pd.to_datetime(spec_raw["timestamp"], utc=True).dt.tz_localize(None)).median()
    spec = to_wide(spec_raw, "band", "reading")
    spec = spec[[b for b in BANDS if b in spec.columns]]
    od = to_wide(od_raw, "angle", "od_reading")
    od.columns = [f"OD {int(c)}°" for c in od.columns]
    return spec, od, off


# =====================================================================================
# 3. WINDOW SELECTION, CLEANING, STATISTICS
# =====================================================================================
def ts(hms):
    return pd.Timestamp(f"{DATE} {hms}")


def pick_window(spec, sid, cfg):
    s, e = ts(cfg["start"]), ts(cfg["end"])
    dt = spec.index.to_series().diff().median().total_seconds()
    n = int(round(WINDOW_MIN * 60 / dt))
    half = pd.Timedelta(minutes=WINDOW_MIN / 2)
    if WINDOW_MODE == "manual" and sid in MANUAL_WINDOWS:
        return ts(MANUAL_WINDOWS[sid][0]), ts(MANUAL_WINDOWS[sid][1])
    if WINDOW_MODE == "centered":
        mid = s + (e - s) / 2
        return mid - half, mid + half
    seg = spec.loc[s + pd.Timedelta(seconds=TRIM_START_S): e - pd.Timedelta(seconds=TRIM_END_S)]
    if len(seg) < n:
        raise ValueError(f"Sample {sid}: recording too short for a {WINDOW_MIN}-min window")
    rel = seg / seg.mean()                       # relative noise per band
    score = rel.rolling(n).std().mean(axis=1)    # mean relative SD over all bands
    i = int(np.nanargmin(score.values))
    return seg.index[i - n + 1], seg.index[i]


def mad_filter(df, thr):
    out, removed = df.copy(), {}
    for c in df.columns:
        x = df[c]
        med = x.median()
        sig = 1.4826 * (x - med).abs().median()
        if sig > 0:
            bad = (x - med).abs() > thr * sig
            out.loc[bad, c] = np.nan
            removed[c] = int(bad.sum())
        else:
            removed[c] = 0
    return out, removed


def describe(df, sid):
    """Per-channel descriptive statistics inside one window."""
    tmin = (df.index - df.index[0]).total_seconds().values / 60.0
    span = tmin.max() - tmin.min()
    rows = []
    for c in df.columns:
        y = df[c]
        ok = y.notna().values
        yy = y[ok].values
        n = len(yy)
        mean, sd = yy.mean(), yy.std(ddof=1)
        sem = sd / np.sqrt(n)
        tc = stats.t.ppf(0.975, n - 1)
        if sd > 0:
            lr = stats.linregress(tmin[ok], yy)
            slope, p = lr.slope, lr.pvalue
        else:
            slope, p = 0.0, np.nan
        rows.append(dict(sample=sid, channel=c, n=n, mean=mean, median=np.median(yy), sd=sd, sem=sem,
                         cv_pct=100 * sd / mean if mean != 0 else np.nan,
                         min=yy.min(), max=yy.max(), range=yy.max() - yy.min(),
                         ci95_lo=mean - tc * sem, ci95_hi=mean + tc * sem,
                         drift_per_min=slope,
                         drift_pct_over_window=100 * slope * span / mean if mean != 0 else np.nan,
                         drift_p=p))
    return pd.DataFrame(rows)


def make_features(spec_w, od_w):
    """Pointwise (per-cycle) calibration features."""
    f = {}
    others = [b for b in BANDS if b != REF_BAND]
    for b in others:                                    # band / reference band
        f[f"{b}/{REF_BAND}"] = spec_w[b] / spec_w[REF_BAND]
    for i, b1 in enumerate(others):                     # longer / shorter wavelength ratios
        for b2 in others[i + 1:]:
            f[f"{b2}/{b1}"] = spec_w[b2] / spec_w[b1]
    f = pd.DataFrame(f)
    od = od_w.loc[:, od_w.std() > 0]                    # drop dead (all-zero) OD channels
    return f, od


def feature_group(name):
    if name.startswith("OD"):
        return "OD"
    return "band/ref" if name.endswith(f"/{REF_BAND}") else "band ratio"


# =====================================================================================
# 4. CALIBRATION FITTING
# =====================================================================================
def loo_ph(x, y):
    """Leave-one-out inverse prediction of pH from the signal."""
    pred = np.full(len(x), np.nan)
    for i in range(len(x)):
        m = np.arange(len(x)) != i
        if np.ptp(x[m]) == 0:
            continue
        sl, ic = np.polyfit(x[m], y[m], 1)
        if sl != 0:
            pred[i] = (y[i] - ic) / sl
    return pred


def linear_calibration(x, y, sigma_noise=None):
    n = len(x)
    lr = stats.linregress(x, y)
    yhat = lr.intercept + lr.slope * x
    resid = y - yhat
    dof = n - 2
    s = np.sqrt(np.sum(resid ** 2) / dof) if dof > 0 else np.nan
    pred = loo_ph(x, y)
    rho = stats.spearmanr(x, y)[0] if n > 2 else np.nan
    out = dict(n=n, slope=lr.slope, intercept=lr.intercept, slope_se=lr.stderr, r2=lr.rvalue ** 2,
               p_value=lr.pvalue, rmse_y=np.sqrt(np.mean(resid ** 2)), s_resid=s, spearman=rho,
               loo_rmse_pH=np.sqrt(np.nanmean((pred - x) ** 2)), loo_pred=pred, resid=resid, yhat=yhat)
    if sigma_noise is not None and lr.slope != 0:
        out["LOD_pH"] = 3.3 * sigma_noise / abs(lr.slope)
        out["LOQ_pH"] = 10 * sigma_noise / abs(lr.slope)
    return out


def bands_for_fit(x, fit, grid):
    n = len(x)
    xbar, sxx = x.mean(), np.sum((x - x.mean()) ** 2)
    t = stats.t.ppf(0.975, n - 2)
    s = fit["s_resid"]
    yg = fit["intercept"] + fit["slope"] * grid
    se_mean = s * np.sqrt(1 / n + (grid - xbar) ** 2 / sxx)
    se_pred = s * np.sqrt(1 + 1 / n + (grid - xbar) ** 2 / sxx)
    return yg, t * se_mean, t * se_pred


# =====================================================================================
# 5. PLOT HELPERS
# =====================================================================================
def savefig(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  saved", path.name)


def heatmap(ax, data, xlabels, ylabels, fmt="{:.3f}", cmap="viridis", center=None, label=""):
    data = np.asarray(data, float)
    if center is not None:
        vmax = np.nanmax(np.abs(data - center))
        vmax = vmax if vmax > 0 else 1
        norm = matplotlib.colors.Normalize(center - vmax, center + vmax)
    else:
        norm = matplotlib.colors.Normalize(np.nanmin(data), np.nanmax(data))
    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.grid(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            r, g, b, _ = im.cmap(im.norm(data[i, j]))
            ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label)
    return im


def ph_text(cfg):
    return f"{cfg['ph_lo']:g}–{cfg['ph_hi']:g}" if cfg["ph_lo"] != cfg["ph_hi"] else f"{cfg['ph_lo']:g}"


def sample_label(sid):
    return f"{sid}\n({SAMPLES[sid]['short']})\npH {ph_text(SAMPLES[sid])}"


# =====================================================================================
# 6. MAIN
# =====================================================================================
def main():
    global WINDOW_MIN, WINDOW_MODE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?", default="exports_20261021_1700-1821_IST.zip",
                    help="export .zip or extracted folder")
    ap.add_argument("--out", default="results_ph_calibration")
    ap.add_argument("--window-min", type=float, default=WINDOW_MIN)
    ap.add_argument("--mode", choices=["auto_stable", "centered", "manual"], default=WINDOW_MODE)
    args = ap.parse_args()
    WINDOW_MIN, WINDOW_MODE = args.window_min, args.mode

    out = Path(args.out)
    figdir, tabdir = out / "figures", out / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)
    log = []

    def say(msg=""):
        print(msg)
        log.append(msg)

    # ---- load ---------------------------------------------------------------------
    spec, od, offset = load_data(args.data)
    say(f"Loaded {len(spec)} AS7341 cycles and {len(od)} OD cycles  "
        f"({spec.index.min():%H:%M:%S} – {spec.index.max():%H:%M:%S} Nepal time)")
    say(f"Clock check: timestamp_localtime - UTC = {offset}  (expect 5:45:00 = Nepal time)")
    ids = list(SAMPLES.keys())

    # ---- windows ------------------------------------------------------------------
    win, spec_w, od_w, spec_full, od_full, removed_log = {}, {}, {}, {}, {}, []
    say(f"\nAnalysis windows (mode={WINDOW_MODE}, length={WINDOW_MIN:g} min):")
    for sid, cfg in SAMPLES.items():
        s, e = ts(cfg["start"]), ts(cfg["end"])
        a, b = pick_window(spec, sid, cfg)
        win[sid] = (a, b)
        spec_full[sid] = spec.loc[s:e]
        od_full[sid] = od.loc[s:e]
        sw, ow = spec.loc[a:b], od.loc[a:b]
        if REMOVE_OUTLIERS:
            sw, r1 = mad_filter(sw, MAD_THRESH)
            ow, r2 = mad_filter(ow, MAD_THRESH)
            nrem = sum(r1.values()) + sum(r2.values())
            removed_log.append((sid, nrem))
        spec_w[sid], od_w[sid] = sw, ow
        say(f"  {sid} {cfg['name']:<34s} recorded {cfg['start']}–{cfg['end']}  ->  "
            f"{a:%H:%M:%S}–{b:%H:%M:%S}  ({len(sw)} spectrometer cycles)")
    if REMOVE_OUTLIERS:
        say("  outliers removed (MAD filter): " + ", ".join(f"{s}:{n}" for s, n in removed_log))
    pd.DataFrame([dict(sample=k, name=SAMPLES[k]["name"], rec_start=SAMPLES[k]["start"],
                       rec_end=SAMPLES[k]["end"], win_start=f"{v[0]:%H:%M:%S}", win_end=f"{v[1]:%H:%M:%S}",
                       ph_lo=SAMPLES[k]["ph_lo"], ph_hi=SAMPLES[k]["ph_hi"])
                  for k, v in win.items()]).to_csv(tabdir / "analysis_windows.csv", index=False)

    # ---- descriptive statistics ---------------------------------------------------
    stat_frames = []
    for sid in ids:
        both = pd.concat([spec_w[sid].rename(columns=lambda c: f"{c} nm").reset_index(drop=True),
                          od_w[sid].reset_index(drop=True)], axis=1)
        both.index = pd.RangeIndex(len(both))
        # spectrometer and OD have separate time axes -> describe separately
        d1 = describe(spec_w[sid].rename(columns=lambda c: f"{c} nm"), sid)
        d2 = describe(od_w[sid], sid)
        stat_frames.append(pd.concat([d1, d2], ignore_index=True))
    stats_long = pd.concat(stat_frames, ignore_index=True)
    stats_long.to_csv(tabdir / "stats_per_sample_channel.csv", index=False)
    mean_tab = stats_long.pivot(index="sample", columns="channel", values="mean")
    sd_tab = stats_long.pivot(index="sample", columns="channel", values="sd")
    cv_tab = stats_long.pivot(index="sample", columns="channel", values="cv_pct")
    drift_tab = stats_long.pivot(index="sample", columns="channel", values="drift_pct_over_window")
    spec_cols = [f"{b} nm" for b in BANDS]
    od_cols = list(od.columns)
    for name, tab in [("mean", mean_tab), ("sd", sd_tab), ("cv_pct", cv_tab), ("drift_pct_over_window", drift_tab)]:
        tab.loc[ids, spec_cols + od_cols].to_csv(tabdir / f"{name}_by_sample.csv")

    say("\nMean ± SD per band (analysis window, raw units)")
    pretty = mean_tab.loc[ids, spec_cols].round(4).astype(str) + " ± " + sd_tab.loc[ids, spec_cols].round(5).astype(str)
    say(pretty.to_string())
    say("\nCV % per channel")
    say(cv_tab.loc[ids, spec_cols + od_cols].round(3).to_string())

    # QC: overall brightness vs the median sample
    total = mean_tab.loc[ids, spec_cols].sum(axis=1)
    rel_level = total / total.median()
    say("\nQC – overall brightness (sum of 8 bands) relative to median sample:")
    say("  " + ", ".join(f"{k}: {v:.2f}" for k, v in rel_level.items()))
    for k, v in rel_level.items():
        if abs(v - 1) > 0.10:
            say(f"  ! sample {k} is {100 * (v - 1):+.0f}% brighter/dimmer than the median sample – "
                f"check vial/position/LED setup; use ref-normalised features")
    dead = [c for c in od_cols if stats_long[stats_long.channel == c]["sd"].max() == 0]
    if dead:
        say(f"  ! OD channel(s) {dead} are constant (all zeros) in every window – no usable signal; excluded from fits")

    # ===============================================================================
    # FIGURES
    # ===============================================================================
    say("\nMaking figures ...")
    tfmt = mdates.DateFormatter("%H:%M")

    # --- Fig 01: all bands, raw, single chart ---------------------------------------
    fig, ax = plt.subplots(figsize=(15, 6.5))
    for b in BANDS:
        ax.plot(spec.index, spec[b], color=BAND_COLORS[b], lw=1.2, label=f"{b} nm")
    ytop = float(spec.max().max()) * 1.18
    ax.set_ylim(0, ytop)
    for sid, cfg in SAMPLES.items():
        s, e = ts(cfg["start"]), ts(cfg["end"])
        ax.axvspan(s, e, color=SAMPLE_COLORS[sid], alpha=0.10, lw=0)
        a, b = win[sid]
        ax.axvspan(a, b, facecolor=SAMPLE_COLORS[sid], alpha=0.30, hatch="///", edgecolor=SAMPLE_COLORS[sid], lw=0)
        ax.text(s + (e - s) / 2, ytop * 0.985, f"{sid}  {cfg['short']}\npH {ph_text(cfg)}", ha="center", va="top",
                fontsize=9, color=SAMPLE_COLORS[sid], fontweight="bold")
    ax.xaxis.set_major_formatter(tfmt)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 10)))
    ax.set_xlabel("Time (Nepal time, CSV timestamp_localtime)")
    ax.set_ylabel("AS7341 raw reading (a.u.)")
    ax.set_title("Raw AS7341 spectrometer – all 8 bands, full run")
    h, l = ax.get_legend_handles_labels()
    h += [Patch(facecolor="grey", alpha=0.15), Patch(facecolor="grey", alpha=0.4, hatch="///")]
    l += ["recording window", "analysis window"]
    ax.legend(h, l, loc="upper left", bbox_to_anchor=(1.01, 1), title="Band")
    savefig(fig, figdir / "fig01_raw_spectrometer_all_bands.png")

    # --- Fig 02: raw OD, all channels -------------------------------------------------
    fig, axes = plt.subplots(len(od_cols), 1, figsize=(15, 2.8 * len(od_cols)), sharex=True)
    for ax, c in zip(np.atleast_1d(axes), od_cols):
        ax.plot(od.index, od[c], lw=1, color="#333333")
        lo, hi = od[c].quantile(0.005), od[c].quantile(0.995)
        pad = (hi - lo) * 0.25 + 1e-3
        ax.set_ylim(lo - pad, hi + pad)
        for sid, cfg in SAMPLES.items():
            s, e = ts(cfg["start"]), ts(cfg["end"])
            ax.axvspan(s, e, color=SAMPLE_COLORS[sid], alpha=0.10, lw=0)
            a, b = win[sid]
            ax.axvspan(a, b, facecolor=SAMPLE_COLORS[sid], alpha=0.30, hatch="///", edgecolor=SAMPLE_COLORS[sid], lw=0)
        ax.set_ylabel(c)
        if od[c].std() == 0 or (od[c] == 0).mean() > 0.9:
            ax.text(0.5, 0.5, "≈ all zeros – no signal on this channel", transform=ax.transAxes,
                    ha="center", va="center", color="firebrick", fontsize=11)
    np.atleast_1d(axes)[0].set_title("Raw OD channels, full run (y-axis clipped to 0.5–99.5 percentile; sample-swap spikes are off-scale)")
    np.atleast_1d(axes)[-1].xaxis.set_major_formatter(tfmt)
    np.atleast_1d(axes)[-1].set_xlabel("Time (Nepal time)")
    savefig(fig, figdir / "fig02_raw_od_channels.png")

    # --- Fig 03: stability inside each recording (% deviation from analysis-window mean) ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False)
    axes = axes.ravel()
    for ax, sid in zip(axes, ids):
        sf = spec_full[sid]
        t0 = sf.index[0]
        mref = spec_w[sid].mean()
        allv = []
        for b in BANDS:
            y = (sf[b] / mref[b] - 1) * 100
            allv.append(y)
            ax.plot((sf.index - t0).total_seconds() / 60, y, color=BAND_COLORS[b], lw=1, label=f"{b} nm")
        allv = pd.concat(allv)
        lo, hi = allv.quantile(0.005), allv.quantile(0.995)
        pad = max((hi - lo) * 0.2, 0.05)
        ax.set_ylim(lo - pad, hi + pad)
        a, b = win[sid]
        ax.axvspan((a - t0).total_seconds() / 60, (b - t0).total_seconds() / 60, color=SAMPLE_COLORS[sid], alpha=0.25)
        ax.set_title(f"{sid} – {SAMPLES[sid]['name']} (pH {ph_text(SAMPLES[sid])})", color=SAMPLE_COLORS[sid])
        ax.set_xlabel("Minutes since recording start")
        ax.set_ylabel("% deviation from window mean")
    axes[-1].axis("off")
    h, l = axes[0].get_legend_handles_labels()
    h.append(Patch(facecolor="grey", alpha=0.3))
    l.append("analysis window")
    axes[-1].legend(h, l, loc="center", ncol=2, title="Band")
    fig.suptitle("Signal stability within each recording (shaded = analysis window)", fontweight="bold", y=1.0)
    fig.tight_layout()
    savefig(fig, figdir / "fig03_stability_within_recordings.png")

    # --- Fig 04: mean spectra (raw + ref-normalised) ----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for sid in ids:
        m = mean_tab.loc[sid, spec_cols].values.astype(float)
        sd = sd_tab.loc[sid, spec_cols].values.astype(float)
        axes[0].errorbar(BANDS, m, yerr=sd, marker="o", ms=5, capsize=3, lw=1.6, color=SAMPLE_COLORS[sid],
                         label=f"{sid} – {SAMPLES[sid]['short']} (pH {ph_text(SAMPLES[sid])})")
        n = spec_w[sid].div(spec_w[sid][REF_BAND], axis=0)
        axes[1].errorbar(BANDS, n.mean().values, yerr=n.std().values, marker="o", ms=5, capsize=3, lw=1.6,
                         color=SAMPLE_COLORS[sid], label=sid)
    axes[0].set_title("Mean spectrum per sample (± 1 SD of the analysis window)")
    axes[0].set_ylabel("AS7341 reading (a.u.)")
    axes[1].set_title(f"Spectrum normalised to {REF_BAND} nm (removes overall intensity changes)")
    axes[1].set_ylabel(f"I(λ) / I({REF_BAND} nm)")
    for ax in axes:
        ax.set_xlabel("Band centre (nm)")
        ax.set_xticks(BANDS)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, figdir / "fig04_mean_spectra.png")

    # --- Fig 05: relative absorbance spectra ------------------------------------------
    refs = [r for r in REF_SAMPLES if r in ids]
    fig, axes = plt.subplots(1, len(refs), figsize=(7 * len(refs), 5), squeeze=False)
    norm_mean = {sid: spec_w[sid].div(spec_w[sid][REF_BAND], axis=0).mean() for sid in ids}
    for ax, ref in zip(axes[0], refs):
        for sid in ids:
            if sid == ref:
                continue
            A = -np.log10(norm_mean[sid] / norm_mean[ref])
            ax.plot(BANDS, A.values, marker="o", lw=1.8, color=SAMPLE_COLORS[sid],
                    label=f"{sid} – {SAMPLES[sid]['short']} (pH {ph_text(SAMPLES[sid])})")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(BANDS)
        ax.set_xlabel("Band centre (nm)")
        ax.set_ylabel("Apparent absorbance  −log10(T_rel)")
        ax.set_title(f"Relative absorbance vs {ref} ({SAMPLES[ref]['short']}), {REF_BAND} nm-normalised")
        ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, figdir / "fig05_relative_absorbance_spectra.png")

    # --- Fig 06: boxplots per band ----------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    for ax, b in zip(axes.ravel(), BANDS):
        data = [spec_w[s][b].dropna().values for s in ids]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, medianprops=dict(color="k"),
                        flierprops=dict(marker=".", ms=3))
        for patch, s in zip(bp["boxes"], ids):
            patch.set_facecolor(SAMPLE_COLORS[s])
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(ids) + 1), ids)
        ax.set_title(f"{b} nm")
        ax.set_ylabel("reading (a.u.)")
    fig.suptitle("Distribution of readings inside each analysis window", fontweight="bold")
    fig.tight_layout()
    savefig(fig, figdir / "fig06_boxplots_per_band.png")

    # --- Fig 07: variability (CV bars + drift heatmap) --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw=dict(width_ratios=[1.25, 1]))
    chans = spec_cols + [c for c in od_cols if c in cv_tab.columns and cv_tab[c].notna().any() and sd_tab[c].max() > 0]
    w = 0.8 / len(ids)
    for i, sid in enumerate(ids):
        axes[0].bar(np.arange(len(chans)) + i * w - 0.4 + w / 2, cv_tab.loc[sid, chans].values, width=w,
                    color=SAMPLE_COLORS[sid], label=sid)
    axes[0].set_xticks(range(len(chans)), chans, rotation=30)
    axes[0].set_ylabel("CV  (SD / mean, %)")
    axes[0].set_title("Within-window variability (coefficient of variation)")
    axes[0].legend(ncol=5, title="Sample")
    heatmap(axes[1], drift_tab.loc[ids, chans].values, chans, ids, fmt="{:+.2f}", cmap="RdBu_r", center=0,
            label="% change over window")
    axes[1].set_xticks(range(len(chans)), chans, rotation=30)
    axes[1].set_title("Linear drift across the analysis window (% of mean)")
    fig.tight_layout()
    savefig(fig, figdir / "fig07_variability_cv_and_drift.png")

    # --- Fig 08: mean-intensity heatmaps ------------------------------------------------
    ref = REF_SAMPLES[0] if REF_SAMPLES[0] in ids else ids[0]
    fig, axes = plt.subplots(1, 3, figsize=(20, 4.6))
    heatmap(axes[0], mean_tab.loc[ids, spec_cols].values, [str(b) for b in BANDS], ids, fmt="{:.3f}",
            cmap="viridis", label="reading (a.u.)")
    axes[0].set_title("Mean raw reading")
    raw_pc = (mean_tab.loc[ids, spec_cols].div(mean_tab.loc[ref, spec_cols], axis=1) - 1) * 100
    heatmap(axes[1], raw_pc.values, [str(b) for b in BANDS], ids, fmt="{:+.1f}", cmap="RdBu_r", center=0, label="%")
    axes[1].set_title(f"Raw change vs {ref}")
    nm = pd.DataFrame({s: norm_mean[s] for s in ids}).T
    nm_pc = (nm.div(nm.loc[ref], axis=1) - 1) * 100
    nm_pc[REF_BAND] = 0.0
    heatmap(axes[2], nm_pc.values, [str(b) for b in BANDS], ids, fmt="{:+.1f}", cmap="RdBu_r", center=0, label="%")
    axes[2].set_title(f"Change vs {ref} after {REF_BAND} nm normalisation")
    for ax in axes:
        ax.set_xlabel("Band (nm)")
    fig.tight_layout()
    savefig(fig, figdir / "fig08_mean_intensity_heatmaps.png")

    # --- Fig 09: OD summary -------------------------------------------------------------
    fig, axes = plt.subplots(1, len(od_cols), figsize=(5.2 * len(od_cols), 4.6))
    for ax, c in zip(np.atleast_1d(axes), od_cols):
        m, s = mean_tab.loc[ids, c].values, sd_tab.loc[ids, c].values
        ax.bar(range(len(ids)), m, yerr=s, capsize=4, color=[SAMPLE_COLORS[i] for i in ids], alpha=0.85)
        ax.set_xticks(range(len(ids)), [sample_label(i) for i in ids], fontsize=8)
        ax.set_title(c)
        ax.set_ylabel("OD reading")
        for i, (mm, ss) in enumerate(zip(m, s)):
            ax.text(i, mm + ss, f"{mm:.4f}", ha="center", va="bottom", fontsize=8)
        if np.all(m == 0):
            ax.text(0.5, 0.5, "all zeros –\nno signal", transform=ax.transAxes, ha="center", color="firebrick")
    fig.suptitle("OD channels: mean ± SD in the analysis window", fontweight="bold")
    fig.tight_layout()
    savefig(fig, figdir / "fig09_od_summary.png")

    # --- Fig 13: window-choice sensitivity ---------------------------------------------
    full_mean = pd.DataFrame({s: spec_full[s].mean() for s in ids}).T
    dif = (mean_tab.loc[ids, spec_cols].values / full_mean.loc[ids, BANDS].values - 1) * 100
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    heatmap(axes[0], dif, [str(b) for b in BANDS], ids, fmt="{:+.2f}", cmap="RdBu_r", center=0,
            label="% difference")
    axes[0].set_title("3-min analysis window vs full recording: difference in mean")
    axes[0].set_xlabel("Band (nm)")
    # primary feature: analysis window vs full recording
    num, den = [int(v) for v in (PRIMARY_FEATURE or "555/445").split("/")]
    fa = [(spec_w[s][num] / spec_w[s][den]) for s in ids]
    ff = [(spec_full[s][num] / spec_full[s][den]) for s in ids]
    x = np.arange(len(ids))
    axes[1].bar(x - 0.2, [v.mean() for v in fa], 0.4, yerr=[v.std() for v in fa], capsize=3, label="analysis window",
                color=[SAMPLE_COLORS[s] for s in ids])
    axes[1].bar(x + 0.2, [v.mean() for v in ff], 0.4, yerr=[v.std() for v in ff], capsize=3, label="full recording",
                color=[SAMPLE_COLORS[s] for s in ids], alpha=0.4, hatch="//")
    axes[1].set_xticks(x, ids)
    axes[1].set_ylim(min(v.mean() for v in fa) * 0.9, max(v.mean() for v in fa) * 1.05)
    axes[1].set_title(f"Ratio {num}/{den}: analysis window vs full recording")
    axes[1].legend()
    fig.tight_layout()
    savefig(fig, figdir / "fig13_window_choice_sensitivity.png")

    # ===============================================================================
    # CALIBRATION
    # ===============================================================================
    say("\nCalibration (linear, signal = a + b·pH; x = pH-strip midpoint)")
    cal = [s for s in CAL_SAMPLES if s in ids]
    xph = np.array([(SAMPLES[s]["ph_lo"] + SAMPLES[s]["ph_hi"]) / 2 for s in cal])
    xerr = np.array([max((SAMPLES[s]["ph_hi"] - SAMPLES[s]["ph_lo"]) / 2, STRIP_MIN_HALF_WIDTH) for s in cal])
    say(f"  samples: {cal}   pH x-values: {xph.tolist()}   (strip uncertainty ±{xerr.tolist()})")
    if len(cal) < 4:
        say("  ! fewer than 4 calibration points – fits are not meaningful")

    feat_rows, feat_series = [], {}
    for sid in ids:
        fsp, fod = make_features(spec_w[sid], od_w[sid])
        fall = {c: fsp[c].dropna() for c in fsp.columns}
        fall.update({c: fod[c].dropna() for c in fod.columns})
        feat_series[sid] = fall
        for k, v in fall.items():
            feat_rows.append(dict(sample=sid, feature=k, group=feature_group(k), mean=v.mean(), sd=v.std(ddof=1),
                                  cv_pct=100 * v.std(ddof=1) / v.mean()))
    feat_long = pd.DataFrame(feat_rows)
    feat_long.to_csv(tabdir / "features_by_sample.csv", index=False)
    fmean = feat_long.pivot(index="sample", columns="feature", values="mean")
    fsd = feat_long.pivot(index="sample", columns="feature", values="sd")

    rank_rows, fits = [], {}
    for k in fmean.columns:
        y = fmean.loc[cal, k].values.astype(float)
        sig = float(np.sqrt(np.mean(fsd.loc[cal, k].values ** 2)))   # pooled within-window SD
        fit = linear_calibration(xph, y, sig)
        fits[k] = fit
        rank_rows.append(dict(feature=k, group=feature_group(k), r2=fit["r2"], spearman=fit["spearman"],
                              slope=fit["slope"], intercept=fit["intercept"], slope_se=fit["slope_se"],
                              p_value=fit["p_value"], rmse_y=fit["rmse_y"], loo_rmse_pH=fit["loo_rmse_pH"],
                              noise_sd=sig, LOD_pH=fit.get("LOD_pH", np.nan), LOQ_pH=fit.get("LOQ_pH", np.nan)))
    rank = pd.DataFrame(rank_rows).sort_values("r2", ascending=False).reset_index(drop=True)
    rank.to_csv(tabdir / "calibration_feature_ranking.csv", index=False)
    say("\n  Top 8 features by R²  (exploratory: ~30 candidate features, n = %d  ->  expect optimistic R²)" % len(cal))
    say(rank.head(8)[["feature", "group", "r2", "spearman", "loo_rmse_pH", "p_value"]].round(4).to_string(index=False))

    prim = PRIMARY_FEATURE if PRIMARY_FEATURE in fits else rank.loc[0, "feature"]
    pf = fits[prim]
    yprim = fmean.loc[cal, prim].values.astype(float)
    ysd = fsd.loc[cal, prim].values.astype(float)
    say(f"\n  PRIMARY feature: {prim}")
    say(f"    signal = {pf['intercept']:.4f} + {pf['slope']:.4f}·pH   (slope SE {pf['slope_se']:.4f}, "
        f"R² = {pf['r2']:.3f}, p = {pf['p_value']:.3g}, n = {pf['n']})")
    say(f"    LOO-CV pH RMSE = {pf['loo_rmse_pH']:.2f} pH units;  noise-limited LOD ≈ {pf.get('LOD_pH', np.nan):.3f} pH, "
        f"LOQ ≈ {pf.get('LOQ_pH', np.nan):.3f} pH (instrument noise only; excludes prep/strip error)")
    pd.DataFrame(dict(sample=cal, strip_pH=xph, strip_pH_unc=xerr, signal_mean=yprim, signal_sd=ysd,
                      fit=pf["yhat"], residual=pf["resid"], loo_pH_pred=pf["loo_pred"],
                      pH_from_signal=(yprim - pf["intercept"]) / pf["slope"])
                 ).to_csv(tabdir / "calibration_primary_predictions.csv", index=False)

    # --- Fig 10: primary calibration ------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
    grid = np.linspace(min(xph.min(), 3) - 0.5, max(xph.max(), 10) + 0.5, 200)
    yg, ci, pi = bands_for_fit(xph, pf, grid)
    ax = axes[0]
    ax.fill_between(grid, yg - pi, yg + pi, color="grey", alpha=0.12, label="95% prediction band")
    ax.fill_between(grid, yg - ci, yg + ci, color="tab:blue", alpha=0.20, label="95% confidence band")
    ax.plot(grid, yg, color="tab:blue", lw=2, label="OLS fit")
    for s, xx, yy, ee, ss in zip(cal, xph, yprim, xerr, ysd):
        ax.errorbar(xx, yy, xerr=ee, yerr=ss, fmt="o", ms=9, color=SAMPLE_COLORS[s], ecolor="k", capsize=3, zorder=5)
        ax.annotate(s, (xx, yy), xytext=(7, 7), textcoords="offset points", fontweight="bold", color=SAMPLE_COLORS[s])
    ax.set_xlabel("pH (strip midpoint; bars = strip range, min ±0.5)")
    ax.set_ylabel(f"Signal: I({prim.split('/')[0]} nm)/I({prim.split('/')[1]} nm)" if "/" in prim else prim)
    ax.set_title(f"Calibration curve – {prim}")
    ax.text(0.03, 0.97, f"y = {pf['intercept']:.4f} + {pf['slope']:.4f}·pH\nR² = {pf['r2']:.3f}   p = {pf['p_value']:.3g}\n"
            f"RMSE = {pf['rmse_y']:.4f}   n = {pf['n']}\nLOO pH RMSE = {pf['loo_rmse_pH']:.2f}",
            transform=ax.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.legend(loc="lower right", fontsize=8)
    ax = axes[1]
    ax.axhline(0, color="k", lw=0.8)
    for s, xx, rr in zip(cal, xph, pf["resid"]):
        ax.scatter(xx, rr, s=90, color=SAMPLE_COLORS[s], zorder=5)
        ax.annotate(s, (xx, rr), xytext=(6, 6), textcoords="offset points", fontweight="bold", color=SAMPLE_COLORS[s])
    ax.set_xlabel("pH (strip)")
    ax.set_ylabel("Residual (signal − fit)")
    ax.set_title("Residuals")
    ax = axes[2]
    lim = [min(xph.min(), np.nanmin(pf["loo_pred"])) - 0.7, max(xph.max(), np.nanmax(pf["loo_pred"])) + 0.7]
    ax.plot(lim, lim, "k--", lw=1, label="1 : 1")
    ax.fill_between(lim, [l - 1 for l in lim], [l + 1 for l in lim], color="grey", alpha=0.12, label="± 1 pH")
    for s, xx, pp in zip(cal, xph, pf["loo_pred"]):
        ax.scatter(xx, pp, s=90, color=SAMPLE_COLORS[s], zorder=5)
        ax.annotate(s, (xx, pp), xytext=(6, 6), textcoords="offset points", fontweight="bold", color=SAMPLE_COLORS[s])
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("pH (strip)")
    ax.set_ylabel("pH predicted from signal (leave-one-out)")
    ax.set_title("Leave-one-out validation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, figdir / "fig10_calibration_primary_feature.png")

    # --- Fig 11: calibration small-multiples for every band/ref and OD ----------------------
    panel = [f"{b}/{REF_BAND}" for b in BANDS if b != REF_BAND] + [c for c in od_cols if c in fmean.columns]
    ncol = 4
    nrow = int(np.ceil(len(panel) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.8 * nrow))
    for ax, k in zip(axes.ravel(), panel):
        y = fmean.loc[cal, k].values.astype(float)
        f = fits[k]
        g = np.linspace(grid.min(), grid.max(), 100)
        yg2, ci2, _ = bands_for_fit(xph, f, g)
        ax.fill_between(g, yg2 - ci2, yg2 + ci2, alpha=0.2, color="tab:blue")
        ax.plot(g, yg2, color="tab:blue", lw=1.8)
        for s, xx, yy in zip(cal, xph, y):
            ax.scatter(xx, yy, s=55, color=SAMPLE_COLORS[s], zorder=5)
            ax.annotate(s, (xx, yy), xytext=(5, 4), textcoords="offset points", fontsize=8, color=SAMPLE_COLORS[s])
        ax.set_title(f"{k}   R² = {f['r2']:.2f}", fontsize=10)
        ax.set_xlabel("pH (strip)")
        ax.set_ylabel(k)
    for ax in axes.ravel()[len(panel):]:
        ax.axis("off")
    fig.suptitle(f"Calibration for every {REF_BAND} nm-normalised band and OD channel (line = OLS, band = 95% CI)",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, figdir / "fig11_calibration_all_bands_and_od.png")

    # --- Fig 12: feature ranking -------------------------------------------------------------
    top = rank.head(15).iloc[::-1]
    gcol = {"band/ref": "tab:blue", "band ratio": "tab:green", "OD": "tab:orange"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    axes[0].barh(top["feature"], top["r2"], color=[gcol[g] for g in top["group"]])
    axes[0].set_xlabel("R² (linear fit vs pH)")
    axes[0].set_title("Top 15 candidate features")
    axes[1].barh(top["feature"], top["loo_rmse_pH"], color=[gcol[g] for g in top["group"]])
    axes[1].set_xlabel("Leave-one-out pH RMSE (pH units)")
    axes[1].set_title("Cross-validated error")
    axes[1].legend(handles=[Patch(color=v, label=k) for k, v in gcol.items()], loc="lower right")
    for yl in axes[0].get_yticklabels():
        if yl.get_text() == prim:
            yl.set_fontweight("bold")
    fig.suptitle(f"Feature ranking (≈{len(rank)} features tried on n = {len(cal)} points: treat the best R² as optimistic)",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, figdir / "fig12_feature_ranking.png")

    # ---- sigmoid (Henderson–Hasselbalch) fit: only with enough distinct pH levels --------------
    ndistinct = len(set(xph.tolist()))
    if ndistinct >= 6:
        from scipy.optimize import curve_fit

        def sig(pH, lo, hi, pka, nH=1.0):
            return lo + (hi - lo) / (1 + 10 ** (nH * (pka - pH)))

        popt, _ = curve_fit(sig, xph, yprim, p0=[yprim.min(), yprim.max(), np.median(xph), 1.0], maxfev=20000)
        say(f"  sigmoid fit: lo={popt[0]:.3f} hi={popt[1]:.3f} pKa={popt[2]:.2f} nH={popt[3]:.2f}")
    else:
        say(f"\n  Sigmoid/Henderson–Hasselbalch fit skipped: only {ndistinct} distinct pH levels "
            f"(need ≥ 6 for a 4-parameter curve). Add pH 2–9 steps for a proper S-curve.")

    # ---- write summary -------------------------------------------------------------------------
    (out / "summary.txt").write_text("\n".join(log), encoding="utf-8")
    print(f"\nDone. Figures: {figdir}\n      Tables : {tabdir}\n      Summary: {out / 'summary.txt'}")


if __name__ == "__main__":
    sys.exit(main())
