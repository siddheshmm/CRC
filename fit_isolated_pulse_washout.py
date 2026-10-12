#!/usr/bin/env python3
"""
fit_isolated_pulse_washout.py

Protocol 4A/4C/4D analysis: fit per-pulse washout (and, for 4D, an onset delay) from
isolated-dye-pulse runs, compare across conditions (stirring settings, tubing lengths,
flow rates), and optionally regress tau against 1/Q (Protocol 4B: expect Q*tau ~ V).

Layout expected:
  <data-root>/<condition_label>/<block>/dosing_events/*.csv
  <data-root>/<condition_label>/<block>/as7341_spectrum_readings/*.csv

Each "block" is one balanced-order replicate set (matches Protocol 2B/4A structure): a
handful of well-separated pulses (default >=5 min apart) recorded continuously. Pulses are
detected automatically and fit independently; consecutive-pulse spacing must be long enough
that one pulse's tail does not still contain the previous pulse's transient (checked and
flagged, not assumed).

Two fit models:
  --delay-model off (default, for 4A/4C): c(t) = a*exp(-t/tau) + b, t measured from
      commanded pulse start.
  --delay-model on  (for 4D): c(t) = b for t<t0, a*exp(-(t-t0)/tau)+b for t>=t0. t0 found by
      a coarse grid + refit (profile fit), not assumed zero.

Usage:
  python fit_isolated_pulse_washout.py --data-root mechanism_data --outdir out_4a
  python fit_isolated_pulse_washout.py --data-root mechanism_data --outdir out_4b \
      --flow-map "Q3:3,Q6:6,Q9:9.3"
  python fit_isolated_pulse_washout.py --data-root mechanism_data --outdir out_4d --delay-model on
"""

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BANDS = [415, 445, 480, 515, 555, 590, 630, 680]
COND_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
FLAGS = []


def flag(msg):
    print(f"  [FLAG] {msg}")
    FLAGS.append(msg)


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
    if len(files) > 1:
        flag(f"{d}/{sub}: {len(files)} CSVs, using {os.path.basename(files[0])} only")
    return pd.read_csv(files[0])


def load_block(block_dir, args, tag):
    dose = read_first_csv(block_dir, "dosing_events")
    spec = read_first_csv(block_dir, "as7341_spectrum_readings")
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
                   volume_ml=("volume_change_ml", "sum")).reset_index(drop=True))
    if len(pulses) < 1:
        raise ValueError(f"{tag}: no pulses detected")
    gaps = pulses["start_time"].diff().dt.total_seconds()
    short = gaps[gaps < args.min_pulse_gap_sec]
    if len(short):
        flag(f"{tag}: {len(short)} pulse(s) spaced <{args.min_pulse_gap_sec:.0f}s apart "
             f"(min {short.min():.0f}s) — response windows may overlap the previous pulse's tail")

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
    if len(wide) < 20:
        raise ValueError(f"{tag}: only {len(wide)} complete scans")

    print(f"  {tag}: {len(pulses)} pulses, {len(wide)} scans, cadence {wide.index.to_series().diff().dt.total_seconds().median():.1f}s")
    return {"pulses": pulses, "wide": wide,
            "cadence": float(wide.index.to_series().diff().dt.total_seconds().median())}


# ---- fit models --------------------------------------------------------------------

def _exp_model(t, a, tau, b):
    return a * np.exp(-t / np.maximum(tau, 1e-6)) + b


def _delay_exp_model(t, a, tau, b, t0):
    out = np.full_like(t, b)
    m = t >= t0
    out[m] = a * np.exp(-(t[m] - t0) / np.maximum(tau, 1e-6)) + b
    return out


def fit_pulse_band(t, y, delay_model, max_t0_frac=0.3):
    """t in seconds from commanded pulse start, y = A(t). Returns dict or None."""
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) < 8:
        return None
    a0 = y[np.argmax(np.abs(y - y[-min(5, len(y)):].mean()))] - y[-min(5, len(y)):].mean()
    b0 = y[-min(5, len(y)):].mean()
    tau0 = max(10.0, (t[-1] - t[0]) / 4.0)
    if not delay_model:
        try:
            p, cov = curve_fit(_exp_model, t, y, p0=[a0, tau0, b0], maxfev=8000,
                               bounds=([-np.inf, 1.0, -np.inf], [np.inf, 1e5, np.inf]))
        except Exception:
            return None
        if p[1] >= 0.95 * 1e5:
            return None  # hit the upper tau bound: not a resolved decay, usually too-short a window
        pred = _exp_model(t, *p)
        sse = np.sum((y - pred) ** 2)
        r2 = 1 - sse / max(np.sum((y - y.mean()) ** 2), 1e-12)
        return {"a": p[0], "tau": p[1], "b": p[2], "t0": 0.0, "r2": r2,
                "n": len(t), "converged": True, "t": t, "y": y, "pred": pred}
    # delay model: grid over t0, refit at each, keep best SSE
    t0_grid = np.linspace(0, max_t0_frac * (t[-1] - t[0]), 25)
    best = None
    for t0g in t0_grid:
        try:
            p, cov = curve_fit(lambda tt, a, tau, b: _delay_exp_model(tt, a, tau, b, t0g),
                               t, y, p0=[a0, tau0, b0], maxfev=4000,
                               bounds=([-np.inf, 1.0, -np.inf], [np.inf, 1e5, np.inf]))
        except Exception:
            continue
        pred = _delay_exp_model(t, p[0], p[1], p[2], t0g)
        sse = np.sum((y - pred) ** 2)
        if best is None or sse < best["sse"]:
            best = {"a": p[0], "tau": p[1], "b": p[2], "t0": t0g, "sse": sse, "pred": pred}
    if best is None or best["tau"] >= 0.95 * 1e5:
        return None  # no fit converged, or hit the upper tau bound: not a resolved decay
    r2 = 1 - best["sse"] / max(np.sum((y - y.mean()) ** 2), 1e-12)
    return {"a": best["a"], "tau": best["tau"], "b": best["b"], "t0": best["t0"], "r2": r2,
            "n": len(t), "converged": True, "t": t, "y": y, "pred": best["pred"]}


def local_baseline(block, pulses, i, args):
    """Quiescent window right before pulse i: last baseline_window_min minutes (minus a guard)
    before this pulse's start, clipped so it never reaches back past the previous pulse's
    start (so it never straddles a still-decaying prior pulse). Returns (bmean, n_scans,
    window_start, window_end) or (None, n, ...) if too few scans."""
    wide = block["wide"]
    p = pulses.iloc[i]
    end = p["start_time"] - pd.Timedelta(seconds=args.baseline_guard_sec)
    beg = end - pd.Timedelta(minutes=args.baseline_window_min)
    floor = wide.index.min() if i == 0 else pulses.iloc[i - 1]["start_time"]
    beg = max(beg, floor)
    win = wide[(wide.index >= beg) & (wide.index < end)]
    if len(win) < 5:
        return None, len(win), beg, end
    return win.mean(), len(win), beg, end


def response_band(block, pulses, baselines, args):
    """Pick the band with the largest median |A| deviation across pulses, for the composite fit."""
    dev = []
    for i, p in pulses.iterrows():
        bmean = baselines[i]
        if bmean is None:
            continue
        win = block["wide"][(block["wide"].index >= p["start_time"]) &
                            (block["wide"].index <= p["start_time"] + pd.Timedelta(seconds=args.window_sec))]
        if len(win):
            A = -np.log10((win / bmean).clip(lower=1e-6))
            dev.append(A.abs().max())
    if not dev:
        return BANDS[3]
    return pd.concat(dev, axis=1).mean(axis=1).idxmax()


def fit_block(block, block_tag, args):
    pulses = block["pulses"].reset_index(drop=True)
    baselines, base_info = [], []
    for i in range(len(pulses)):
        bmean, n_scans, beg, end = local_baseline(block, pulses, i, args)
        baselines.append(bmean)
        want = args.baseline_window_min * 60
        have = (end - beg).total_seconds()
        if bmean is None:
            flag(f"{block_tag} pulse {i}: local baseline has only {n_scans} scan(s), skipped "
                 f"(window available: {have:.0f}s of {want:.0f}s requested)")
        elif have < 0.5 * want:
            flag(f"{block_tag} pulse {i}: local baseline window truncated to {have:.0f}s "
                 f"(of {want:.0f}s requested) - close to the previous pulse or recording start")
        base_info.append({"pulse": i, "bmean": bmean, "n_scans": n_scans,
                          "window_sec": have, "window_start": beg, "window_end": end})
    bm = pd.DataFrame([b["bmean"] for b in base_info if b["bmean"] is not None])
    if len(bm) > 1:
        drift = (bm.iloc[-1] / bm.iloc[0] - 1) * 100
        if drift.abs().max() > 2.0:
            flag(f"{block_tag}: local baseline level drifts {drift.abs().max():.1f}% from first "
                 f"to last usable pulse - check for incomplete washout between pulses (residual "
                 f"dye) or slow instrument drift, not just fit each pulse in isolation")

    rb = response_band(block, pulses, baselines, args)
    rows = []
    for i, p in pulses.iterrows():
        bmean = baselines[i]
        if bmean is None:
            continue
        t_end = p["start_time"] + pd.Timedelta(seconds=args.window_sec)
        if i + 1 < len(pulses):
            t_end = min(t_end, pulses["start_time"].iloc[i + 1] - pd.Timedelta(seconds=args.guard_sec))
        win = block["wide"][(block["wide"].index >= p["start_time"]) & (block["wide"].index < t_end)]
        if len(win) < 8:
            flag(f"{block_tag} pulse {i}: only {len(win)} scans in response window, skipped")
            continue
        A = -np.log10((win / bmean).clip(lower=1e-6))
        t = (win.index - p["start_time"]).total_seconds().values
        actual_dur = float(t.max() - t.min()) if len(t) else 0.0
        for b in BANDS:
            fit = fit_pulse_band(t, A[b].values, args.delay_model)
            if fit is None:
                if b == rb:
                    flag(f"{block_tag} pulse {i}, band {b} (response band): fit did not converge to a "
                         f"resolved decay (hit the tau bound) - likely too little post-pulse data "
                         f"({(t_end - p['start_time']).total_seconds():.0f}s window); excluded")
                continue
            tail_adequate = actual_dur >= 2.0 * fit["tau"]
            if not tail_adequate and b == rb:
                flag(f"{block_tag} pulse {i}, band {b} (response band): only {actual_dur:.0f}s of post-pulse "
                     f"data but fitted tau={fit['tau']:.0f}s (need >=2x tau to constrain the decay) - "
                     f"tau is unreliable here, excluded from condition summary")
            rows.append({"block": block_tag, "pulse": i, "band": b, "is_response_band": b == rb,
                        "volume_ml": p["volume_ml"], "window_n": fit["n"],
                        "baseline_window_sec": base_info[i]["window_sec"],
                        "window_sec": actual_dur, "tail_adequate": tail_adequate,
                        **{k: fit[k] for k in ["a", "tau", "b", "t0", "r2"]}})
    return pd.DataFrame(rows), rb, baselines


def main():
    ap = argparse.ArgumentParser(description="Protocol 4A/4C/4D isolated-pulse washout fitting")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--outdir", default="out_washout")
    ap.add_argument("--block-glob", default="*")
    ap.add_argument("--alt-source", default="specified_pump_dosing:alt_media")
    ap.add_argument("--gap-sec", type=float, default=5.0)
    ap.add_argument("--scan-gap-sec", type=float, default=2.0)
    ap.add_argument("--min-pulse-gap-sec", type=float, default=300.0,
                    help="expected minimum spacing between isolated pulses; shorter is flagged")
    ap.add_argument("--window-sec", type=float, default=600.0, help="max response window per pulse")
    ap.add_argument("--guard-sec", type=float, default=5.0, help="stop before the next pulse by this much")
    ap.add_argument("--baseline-window-min", type=float, default=10.0)
    ap.add_argument("--baseline-guard-sec", type=float, default=30.0)
    ap.add_argument("--delay-model", choices=["on", "off"], default="off")
    ap.add_argument("--flow-map", default=None,
                    help="condition_label:Q_mL_per_min,... to regress tau vs 1/Q (Protocol 4B)")
    args = ap.parse_args()
    args.delay_model = args.delay_model == "on"
    os.makedirs(args.outdir, exist_ok=True)

    conditions = sorted(d for d in glob.glob(os.path.join(args.data_root, "*")) if os.path.isdir(d))
    if not conditions:
        raise FileNotFoundError(f"No condition folders in {args.data_root}")

    all_rows, per_cond_blocks = [], {}
    for cdir in conditions:
        cname = os.path.basename(cdir)
        blocks = sorted(d for d in glob.glob(os.path.join(cdir, args.block_glob)) if os.path.isdir(d))
        if not blocks:
            flag(f"condition {cname}: no block folders matching '{args.block_glob}'")
            continue
        print(f"\nCondition {cname}: {len(blocks)} block(s)")
        per_cond_blocks[cname] = []
        for bdir in blocks:
            btag = f"{cname}/{os.path.basename(bdir)}"
            try:
                block = load_block(bdir, args, btag)
            except Exception as e:
                flag(f"{btag}: failed to load ({e})")
                continue
            df, rb, baselines = fit_block(block, btag, args)
            if len(df):
                df["condition"] = cname
                all_rows.append(df)
                per_cond_blocks[cname].append((btag, block, df, rb, baselines))

    if not all_rows:
        print("No fits produced. Check --data-root layout and --alt-source.")
        return
    fits = pd.concat(all_rows, ignore_index=True)
    fits.to_csv(os.path.join(args.outdir, "pulse_fits_all_bands.csv"), index=False)

    # ---- per-condition summary on the auto-selected response band, and on band median
    resp = fits[fits["is_response_band"] & fits["tail_adequate"]]
    n_excluded = (fits["is_response_band"] & ~fits["tail_adequate"]).sum()
    if n_excluded:
        print(f"\n{n_excluded} response-band fit(s) excluded from the summary for insufficient post-pulse "
              f"data (<2x fitted tau); see pulse_fits_all_bands.csv (tail_adequate=False) for the raw fits.")
    summ = (resp.groupby("condition")
            .agg(n_pulses=("tau", "size"), tau_mean=("tau", "mean"), tau_sd=("tau", "std"),
                 t0_mean=("t0", "mean"), t0_sd=("t0", "std"), r2_mean=("r2", "mean"),
                 r2_min=("r2", "min")).reset_index())
    allband = (fits.groupby("condition")
              .agg(tau_median_allbands=("tau", "median"), r2_median_allbands=("r2", "median")).reset_index())
    summ = summ.merge(allband, on="condition")
    summ.to_csv(os.path.join(args.outdir, "condition_summary.csv"), index=False)

    print("\n" + "=" * 78)
    print(" PER-CONDITION WASHOUT SUMMARY (response-band fit; see CSV for all 8 bands)")
    print("=" * 78)
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    low_r2 = summ[summ["r2_min"] < 0.8]
    if len(low_r2):
        flag(f"conditions with at least one band-fit R^2 < 0.8: {low_r2['condition'].tolist()} "
             f"(poor exponential fit — check for structured residuals, e.g. incomplete mixing)")

    # ---- figures: raw + fit per condition (one representative pulse each), tau by condition
    fig, axes = plt.subplots(1, len(per_cond_blocks), figsize=(5 * len(per_cond_blocks), 4.2),
                             dpi=130, squeeze=False)
    for ax, (cname, blocks) in zip(axes[0], per_cond_blocks.items()):
        if not blocks:
            continue
        btag, block, df, rb, baselines = blocks[0]
        p0 = block["pulses"].iloc[0]
        bmean0 = baselines[0]
        if bmean0 is None:
            continue
        win = block["wide"][(block["wide"].index >= p0["start_time"]) &
                            (block["wide"].index < p0["start_time"] + pd.Timedelta(seconds=args.window_sec))]
        A0 = -np.log10((win / bmean0).clip(lower=1e-6))
        t = (win.index - p0["start_time"]).total_seconds().values
        ax.plot(t, A0[rb].values, "ko", ms=3, alpha=0.6, label=f"data (band {rb})")
        row = df[(df["pulse"] == 0) & (df["band"] == rb)]
        if len(row):
            r = row.iloc[0]
            tt = np.linspace(0, t.max(), 200)
            pred = (_delay_exp_model(tt, r["a"], r["tau"], r["b"], r["t0"]) if args.delay_model
                   else _exp_model(tt, r["a"], r["tau"], r["b"]))
            ax.plot(tt, pred, "r-", lw=1.8, label=f"fit tau={r['tau']:.0f}s"
                    + (f" t0={r['t0']:.0f}s" if args.delay_model else "") + f" R2={r['r2']:.2f}")
        ax.set_title(cname, fontsize=10)
        ax.set_xlabel("s since dose")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("Fig 1: representative pulse fit per condition", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "01_example_fits.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(summ)), 5), dpi=130)
    x = np.arange(len(summ))
    ax.bar(x, summ["tau_mean"], yerr=summ["tau_sd"].fillna(0), capsize=4,
          color=[COND_COLORS[i % len(COND_COLORS)] for i in range(len(summ))])
    ax.set_xticks(x)
    ax.set_xticklabels(summ["condition"])
    ax.set_ylabel("tau, s (response-band fit, mean +/- SD over pulses)")
    ax.grid(alpha=0.3)
    ax.set_title("Fig 2: washout time constant by condition", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "02_tau_by_condition.png"))
    plt.close(fig)

    if args.delay_model:
        fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(summ)), 5), dpi=130)
        ax.bar(x, summ["t0_mean"], yerr=summ["t0_sd"].fillna(0), capsize=4,
              color=[COND_COLORS[i % len(COND_COLORS)] for i in range(len(summ))])
        ax.set_xticks(x)
        ax.set_xticklabels(summ["condition"])
        ax.set_ylabel("t0 (onset delay), s")
        ax.grid(alpha=0.3)
        ax.set_title("Fig 3: onset delay by condition (Protocol 4D)", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "03_t0_by_condition.png"))
        plt.close(fig)

    # ---- Protocol 4B: tau vs 1/Q
    if args.flow_map:
        fmap = {}
        for kv in args.flow_map.split(","):
            k, v = kv.split(":")
            fmap[k.strip()] = float(v)
        s2 = summ[summ["condition"].isin(fmap)].copy()
        if len(s2) < 2:
            flag("flow-map given but fewer than 2 matching conditions found; skipping tau-vs-1/Q regression")
        else:
            s2["Q"] = s2["condition"].map(fmap)
            s2["invQ"] = 1.0 / s2["Q"]
            coef = np.polyfit(s2["invQ"], s2["tau_mean"], 1)
            pred = np.polyval(coef, s2["invQ"])
            ss_res = np.sum((s2["tau_mean"] - pred) ** 2)
            ss_tot = np.sum((s2["tau_mean"] - s2["tau_mean"].mean()) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-12)
            print(f"\ntau vs 1/Q: tau ~ {coef[0]:.1f} * (1/Q) + {coef[1]:.1f}  (R^2={r2:.3f})")
            print(f"  slope ~ implied working volume V ~ {coef[0]:.2f} mL (compare to measured/nominal 10 mL)")
            s2.to_csv(os.path.join(args.outdir, "tau_vs_invQ.csv"), index=False)
            fig, ax = plt.subplots(figsize=(6, 5), dpi=130)
            ax.errorbar(s2["invQ"], s2["tau_mean"], yerr=s2["tau_sd"].fillna(0), fmt="o", ms=8, capsize=4)
            xx = np.linspace(0, s2["invQ"].max() * 1.15, 50)
            ax.plot(xx, np.polyval(coef, xx), "r--", label=f"fit: slope={coef[0]:.1f}, R2={r2:.2f}")
            for _, row in s2.iterrows():
                ax.annotate(row["condition"], (row["invQ"], row["tau_mean"]), fontsize=8,
                           xytext=(5, 5), textcoords="offset points")
            ax.set_xlabel("1/Q, min/mL")
            ax.set_ylabel("tau, s")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_title("Fig 4: tau vs 1/Q (Protocol 4B; expect slope ~ V*60)", fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(args.outdir, "04_tau_vs_invQ.png"))
            plt.close(fig)

    print("\nFLAGS:")
    for f in FLAGS:
        print("  - " + f)
    if not FLAGS:
        print("  none")
    print(f"\nOutputs in: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
