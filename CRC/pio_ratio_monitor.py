#!/usr/bin/env python3
"""
pio_ratio_monitor.py – live 555/480 check from Pioreactor MQTT, with evidence logging
=====================================================================================

Subscribes to the AS7341 readings your plugin publishes on MQTT, assembles the 8 bands into
cycles, computes 555/480 (and 555/445, 555/680, 630/555) live, and tells you whether the carrier
sits in the target window.  Everything is saved as a self-contained, hash-sealed run folder.

Quick start (run on the leader Pi or any PC that can reach the broker)
----------------------------------------------------------------------
  pip install paho-mqtt numpy matplotlib

  1) find the topic your plugin publishes on (prints topics + sample payloads for 30 s):
       python pio_ratio_monitor.py --host pioreactor.local --discover 30
  2) Stage 0 carrier check (5 min, final verdict from the last 2 min):
       python pio_ratio_monitor.py --stage stage0_carrier_check --label carrier_1L_batch1 \
              --duration 300 --analyze-last 120 --match <text from step 1> \
              --note "1 L carrier: 130 mL MO + 1.74 mL vinegar + 868 mL DI water, 10 mL vial"
  3) open-ended monitoring (Ctrl-C to stop; type a line + Enter to log an event, e.g. "acid 0.2 mL"):
       python pio_ratio_monitor.py --stage stage1_baseline --label baseline --match <text>

Offline replay of an existing export (to test the analysis, or to seal old data as evidence):
       python pio_ratio_monitor.py --replay exports.zip --replay-from "2026-10-22 14:53:00" \
              --replay-to "2026-10-22 14:58:00" --stage replay --label T6   (times = timestamp_localtime)

Run folder  runs/<UTC>_<stage>_<label>/
  raw_mqtt_log.jsonl   every matched MQTT message, verbatim, with receive time (the primary evidence)
  readings.csv         assembled cycles: 8 bands + ratios
  events.csv           timestamped operator events (typed during the run)
  analysis.json        statistics over the final window + PASS/FAIL verdict + thresholds used
  fig_ratio.png        ratio vs time with the target window
  metadata.json        who/when/what: argv, broker, topic, notes, script SHA-256, library versions
  SHA256SUMS.txt      hashes of all of the above; files are then made read-only

MQTT defaults come from the Pioreactor config.ini [mqtt] section (username/password/port 1883);
check yours if the connection is refused.  Times: receive time on the machine running this script (UTC in files).
"""
import argparse
import csv
import getpass
import hashlib
import json
import math
import os
import platform
import re
import socket
import sys
import threading
import time
import zipfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_VERSION = "1.0"
BANDS = [415, 445, 480, 515, 555, 590, 630, 680]
F_TO_BAND = {f"F{i + 1}": b for i, b in enumerate(BANDS)}      # AS7341 channel names
RATIO_SET = ["555/480", "555/445", "555/680", "630/555"]
BAND_RE = re.compile(r"(?<!\d)(415|445|480|515|555|590|630|680)(?!\d)")
F_RE = re.compile(r"(?<![A-Za-z0-9])F([1-8])(?!\d)")


# =====================================================================================
# message parsing  (formats are guessed so that it works with different plugins)
# =====================================================================================
def band_from_text(text):
    m = BAND_RE.search(str(text))
    if m:
        return int(m.group(1))
    m = F_RE.search(str(text))
    return F_TO_BAND[f"F{m.group(1)}"] if m else None


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    if isinstance(v, dict):                       # {"reading": x, ...}
        for k in ("reading", "value", "counts", "v"):
            if k in v:
                return _num(v[k])
    return None


def extract(obj, hint=""):
    """Return [(band, value), ...] from a decoded JSON object."""
    out = []
    if isinstance(obj, list):
        for it in obj:
            out += extract(it, hint)
    elif isinstance(obj, dict):
        band_key = next((k for k in ("band", "wavelength", "channel", "name") if k in obj), None)
        if band_key and any(k in obj for k in ("reading", "value", "counts")):
            b = band_from_text(obj[band_key]) if not isinstance(obj[band_key], (int, float)) else (
                int(obj[band_key]) if int(obj[band_key]) in BANDS else None)
            v = _num(obj)
            if b is not None and v is not None:
                out.append((b, v))
        else:
            for k, v in obj.items():
                b = band_from_text(k)
                if b is not None and _num(v) is not None:
                    out.append((b, _num(v)))
                elif isinstance(v, (dict, list)):
                    out += extract(v, f"{hint}/{k}")
    return out


def parse_message(topic, payload):
    try:
        obj = json.loads(payload)
    except Exception:
        obj = None
    if obj is None or isinstance(obj, (int, float)):
        v = _num(payload if obj is None else obj)
        b = band_from_text(topic.split("/")[-1]) or band_from_text(topic.split("/")[-2] if topic.count("/") > 1 else "")
        return [(b, v)] if (b is not None and v is not None) else []
    return extract(obj)


# =====================================================================================
# run state, evidence files, analysis
# =====================================================================================
def utc_iso(t):
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def feature_value(cycle, feat):
    a, b = (int(x) for x in feat.split("/"))
    return cycle[a] / cycle[b]


def classify(med, drift10, drift_p, sd, n, lo, hi, max_drift, max_sd, min_n=12):
    """Out-of-window medians are reported first (the direction tells you what to trim, even while the
    ratio is still moving).  Drift only counts when it is both larger than max_drift AND statistically
    real (p < 0.01); short windows have a noisy slope, so use >= 5 min of data for the drift test."""
    if n < min_n:
        return "COLLECTING"
    drifting = abs(drift10) > max_drift and drift_p < 0.01
    tail = " (still drifting)" if drifting else ""
    if med > hi:
        return "TOO HIGH (too yellow/basic) -> add vinegar in 0.2 mL steps per litre" + tail
    if med < lo:
        return "TOO LOW (too red/acidic) -> add ~1:150 ammonia 1-2 mL, or 10% MO-in-water" + tail
    if drifting or sd > max_sd:
        return f"SETTLING (drift {drift10:+.2f}%/10min, SD {sd:.4f})"
    return "IN WINDOW"


def window_stats(t, y):
    """median, mean, SD, robust SD, drift (%/10 min) with its standard error and p-value."""
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    n = len(y)
    nan = float("nan")
    if n < 4:
        return dict(n=n, median=nan, mean=nan, sd=nan, mad_sd=nan, drift_pct_per_10min=nan, drift_se=nan, drift_p=1.0)
    med = float(np.median(y))
    x = t - t.mean()
    sxx = float(np.sum(x ** 2))
    slope = float(np.sum(x * (y - y.mean())) / sxx) if sxx > 0 else 0.0
    res = y - y.mean() - slope * x
    se = math.sqrt(float(np.sum(res ** 2)) / (n - 2) / sxx) if sxx > 0 else nan
    p = math.erfc(abs(slope / se) / math.sqrt(2)) if se and se > 0 else 1.0
    k = 600 / med * 100
    return dict(n=n, median=med, mean=float(y.mean()), sd=float(y.std(ddof=1)),
                mad_sd=float(1.4826 * np.median(np.abs(y - med))),
                drift_pct_per_10min=slope * k, drift_se=se * k, drift_p=p)


class Run:
    def __init__(self, args, outdir, source_info):
        self.a = args
        self.dir = outdir
        self.t = []                                   # epoch of each cycle
        self.cycles = []                              # dict band->value
        self.pending, self.pending_t0 = {}, None
        self.dropped = 0
        self.n_msgs = 0
        self.events = []
        self.start = time.time()
        self.lock = threading.Lock()
        self.raw = open(outdir / "raw_mqtt_log.jsonl", "a", encoding="utf-8")
        self.csvf = open(outdir / "readings.csv", "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.csvf)
        self.w.writerow(["t_utc", "t_epoch"] + [f"b{b}" for b in BANDS] + [f"r_{r}" for r in RATIO_SET])
        self.evf = open(outdir / "events.csv", "w", newline="", encoding="utf-8")
        self.evw = csv.writer(self.evf)
        self.evw.writerow(["t_utc", "t_epoch", "text"])
        self.last_status = 0
        self.source_info = source_info

    # ---- ingest ----
    def log_raw(self, t, topic, payload, retain=False, extra=None):
        rec = {"t_utc": utc_iso(t), "t_epoch": round(t, 3), "topic": topic, "payload": payload, "retain": retain}
        if extra:
            rec.update(extra)
        with self.lock:
            self.raw.write(json.dumps(rec) + "\n")
            self.raw.flush()

    def add_values(self, values, t):
        with self.lock:
            for b, v in values:
                if self.pending and (b in self.pending or t - self.pending_t0 > 2.0):
                    self.dropped += 1                 # incomplete cycle: discard
                    self.pending, self.pending_t0 = {}, None
                if not self.pending:
                    self.pending_t0 = t
                self.pending[b] = v
                if len(self.pending) == len(BANDS):
                    self._finish_cycle(t)

    def _finish_cycle(self, t):
        c = dict(self.pending)
        self.pending, self.pending_t0 = {}, None
        ratios = [feature_value(c, r) for r in RATIO_SET]
        self.t.append(t)
        self.cycles.append(c)
        self.w.writerow([utc_iso(t), round(t, 3)] + [c[b] for b in BANDS] + [f"{r:.6f}" for r in ratios])
        self.csvf.flush()

    def add_event(self, text, t=None):
        t = t or time.time()
        with self.lock:
            self.events.append((t, text))
            self.evw.writerow([utc_iso(t), round(t, 3), text])
            self.evf.flush()
        print(f"  [event logged {utc_iso(t)}] {text}")

    # ---- live status ----
    def series(self, feat, last_s=None):
        with self.lock:
            t = np.array(self.t)
            y = np.array([feature_value(c, feat) for c in self.cycles])
        if last_s and len(t):
            m = t >= t[-1] - last_s
            t, y = t[m], y[m]
        return t, y

    def status_line(self):
        a = self.a
        t, y = self.series(a.feature, max(a.window_s, a.drift_window_s))
        stamp = datetime.now().strftime("%H:%M:%S")
        if len(t) < 4:
            return f"[{stamp}] cycles={len(self.t)}  -> WAITING FOR DATA"
        m = t >= t[-1] - a.window_s
        st = window_stats(t[m], y[m])
        dr = window_stats(t, y)
        verdict = classify(st["median"], dr["drift_pct_per_10min"], dr["drift_p"], st["sd"], st["n"],
                           a.target[0], a.target[1], a.max_drift, a.max_sd)
        return (f"[{stamp}] cycles={len(self.t)} {a.feature}: median={st['median']:.4f} SD={st['sd']:.4f} "
                f"(last {a.window_s:g}s, n={st['n']}) drift={dr['drift_pct_per_10min']:+.2f}±{dr['drift_se']:.2f}%/10min "
                f"(last {t[-1] - t[0]:.0f}s)  -> {verdict}")

    # ---- final analysis ----
    def analyse(self):
        a = self.a
        with self.lock:
            t = np.array(self.t)
            cyc = list(self.cycles)
        res = dict(feature=a.feature, target_window=list(a.target), thresholds=dict(
            max_drift_pct_per_10min=a.max_drift, max_sd=a.max_sd, analyze_last_s=a.analyze_last,
            drift_window_s=a.drift_window_s),
            n_cycles_total=len(t), n_incomplete_cycles_dropped=self.dropped, n_messages_matched=self.n_msgs)
        if len(t) < 3:
            res["verdict"] = "INSUFFICIENT_DATA"
            return res
        m = t >= t[-1] - a.analyze_last
        tt = t[m]
        res.update(window_start_utc=utc_iso(tt[0]), window_end_utc=utc_iso(tt[-1]),
                   window_seconds=float(tt[-1] - tt[0]), n_cycles_window=int(m.sum()))
        feats = {}
        for f in sorted(set([a.feature] + RATIO_SET)):
            y = np.array([feature_value(c, f) for c, keep in zip(cyc, m) if keep])
            feats[f] = window_stats(tt, y)
        res["features"] = feats
        bands = {}
        for b in BANDS:
            y = np.array([c[b] for c, keep in zip(cyc, m) if keep])
            s = window_stats(tt, y)
            s["cv_pct"] = 100 * s["sd"] / s["mean"]
            bands[str(b)] = s
        res["bands"] = bands
        p = feats[a.feature]
        md = t >= t[-1] - max(a.analyze_last, a.drift_window_s)
        yd = np.array([feature_value(c, a.feature) for c, keep in zip(cyc, md) if keep])
        dr = window_stats(t[md], yd)
        res["drift_check"] = dict(window_seconds=float(t[md][-1] - t[md][0]), n=dr["n"],
                                  drift_pct_per_10min=dr["drift_pct_per_10min"], se=dr["drift_se"], p=dr["drift_p"],
                                  note="verdict treats drift as real only if |drift| > max and p < 0.01")
        verdict = classify(p["median"], dr["drift_pct_per_10min"], dr["drift_p"], p["sd"], p["n"], a.target[0], a.target[1],
                           a.max_drift, a.max_sd)
        res["verdict_text"] = verdict
        res["verdict"] = ("PASS" if verdict == "IN WINDOW" else "FAIL_HIGH" if verdict.startswith("TOO HIGH") else
                          "FAIL_LOW" if verdict.startswith("TOO LOW") else "UNSTABLE" if verdict.startswith("SETTLING")
                          else "INSUFFICIENT_DATA")
        return res

    def plot(self, analysis):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return
        a = self.a
        t, y = self.series(a.feature)
        if len(t) < 3:
            return
        x = (t - t[0]) / 60
        fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        ax[0].axhspan(a.target[0], a.target[1], color="green", alpha=0.15, label=f"target {a.target[0]}–{a.target[1]}")
        ax[0].plot(x, y, lw=1, color="k", label=a.feature)
        if "window_start_utc" in analysis:
            w0 = (t[t >= t[-1] - a.analyze_last][0] - t[0]) / 60
            ax[0].axvspan(w0, x[-1], color="tab:blue", alpha=0.12, label="analysis window")
            med = analysis["features"][a.feature]["median"]
            ax[0].axhline(med, color="tab:blue", lw=1.2, ls="--", label=f"window median {med:.4f}")
        for te, txt in self.events:
            if te >= t[0]:
                ax[0].axvline((te - t[0]) / 60, color="tab:red", lw=0.8)
                ax[0].text((te - t[0]) / 60, ax[0].get_ylim()[1], txt[:30], rotation=90, va="top", fontsize=7, color="tab:red")
        ax[0].set_ylabel(a.feature)
        ax[0].legend(loc="best", fontsize=8)
        ax[0].set_title(f"{a.stage} / {a.label} – verdict: {analysis.get('verdict', '?')}")
        for b in BANDS:
            ax[1].plot(x, [c[b] for c in self.cycles], lw=0.9, label=f"{b} nm")
        ax[1].set_xlabel(f"minutes since first cycle ({utc_iso(t[0])})")
        ax[1].set_ylabel("raw band reading")
        ax[1].legend(ncol=4, fontsize=7)
        for axx in ax:
            axx.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.dir / "fig_ratio.png", dpi=170)
        plt.close(fig)

    def close_files(self):
        for f in (self.raw, self.csvf, self.evf):
            try:
                f.flush()
                os.fsync(f.fileno())
                f.close()
            except Exception:
                pass


def seal(run, analysis, args, argv):
    (run.dir / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    run.plot(analysis)
    try:
        import paho.mqtt
        paho_v = getattr(paho.mqtt, "__version__", "?")
    except Exception:
        paho_v = "not installed"
    meta = dict(
        script="pio_ratio_monitor.py", script_version=SCRIPT_VERSION,
        script_sha256=sha256_file(__file__), argv=argv, stage=args.stage, label=args.label, notes=args.note,
        extra=dict(kv.split("=", 1) for kv in args.meta if "=" in kv),
        started_utc=utc_iso(run.start), ended_utc=utc_iso(time.time()), host=socket.gethostname(),
        operator=getpass.getuser(), python=platform.python_version(), numpy=np.__version__, paho_mqtt=paho_v,
        source=run.source_info, n_events=len(run.events))
    (run.dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    files = sorted(p for p in run.dir.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt")
    (run.dir / "SHA256SUMS.txt").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files), encoding="utf-8")
    if not args.no_lock:
        for p in run.dir.iterdir():
            try:
                os.chmod(p, 0o444)
            except Exception:
                pass


# =====================================================================================
# live MQTT / replay / discover
# =====================================================================================
def make_client(args):
    import paho.mqtt.client as mqtt
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:                       # paho-mqtt 1.x
        client = mqtt.Client()
    if args.user:
        client.username_pw_set(args.user, args.password)
    return client


def topic_filter(args):
    return args.topic or (f"pioreactor/{args.unit}/+/#" if args.unit else "pioreactor/+/+/#")


def discover(args):
    seen = {}
    client = make_client(args)

    def on_connect(c, u, f, rc, *x):
        print(f"connected (rc={rc}); subscribing to {topic_filter(args)} for {args.discover:g} s ...")
        c.subscribe(topic_filter(args))

    def on_message(c, u, msg):
        parts = msg.topic.split("/")
        key = "/".join(parts[3:]) if len(parts) > 3 else msg.topic         # drop pioreactor/unit/experiment
        d = seen.setdefault(key, [0, ""])
        d[0] += 1
        d[1] = msg.payload.decode("utf-8", "replace")[:160]

    client.on_connect, client.on_message = on_connect, on_message
    client.connect(args.host, args.port, 60)
    client.loop_start()
    time.sleep(args.discover)
    client.loop_stop()
    print("\nTopics seen (after pioreactor/<unit>/<experiment>/), message count, last payload:")
    for k, (n, p) in sorted(seen.items()):
        flag = "   <-- parsed as AS7341" if parse_message(k, p) else ""
        print(f"  {k}  x{n}   {p!r}{flag}")
    if not seen:
        print("  nothing received: check --host/--port/--user/--password and that readings are running")
    print("\nUse --match <substring of the topic>, e.g. --match as7341")


def stdin_events(run, stop):
    if not sys.stdin or not sys.stdin.isatty():
        return

    def loop():
        while not stop.is_set():
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if line:
                run.add_event(line)
    threading.Thread(target=loop, daemon=True).start()


def run_live(args, run, stop):
    client = make_client(args)
    match = args.match.lower()

    def on_connect(c, u, f, rc, *x):
        print(f"connected (rc={rc}); subscribing to {topic_filter(args)} (match '{args.match}')")
        c.subscribe(topic_filter(args))

    def on_message(c, u, msg):
        topic = msg.topic
        if not args.log_all and match not in topic.lower():
            return
        t = time.time()
        payload = msg.payload.decode("utf-8", "replace")
        run.log_raw(t, topic, payload, bool(msg.retain))
        if msg.retain or match not in topic.lower():
            return
        run.n_msgs += 1
        vals = parse_message(topic, payload)
        if vals:
            run.add_values(vals, t)

    client.on_connect, client.on_message = on_connect, on_message
    client.connect(args.host, args.port, 60)
    client.loop_start()
    end = time.time() + args.duration if args.duration else None
    try:
        while not stop.is_set() and (end is None or time.time() < end):
            time.sleep(0.5)
            if time.time() - run.last_status >= args.status_every:
                run.last_status = time.time()
                print(run.status_line())
    except KeyboardInterrupt:
        print("\nstopping (Ctrl-C) ...")
    finally:
        client.loop_stop()
        client.disconnect()


def run_replay(args, run):
    import pandas as pd
    src = Path(args.replay)
    frames = []
    if src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as z:
            for n in sorted(z.namelist()):
                if n.lower().endswith(".csv") and "as7341" in n.lower():
                    frames.append(pd.read_csv(z.open(n)))
    else:
        frames.append(pd.read_csv(src))
    df = pd.concat(frames, ignore_index=True)
    df["tl"] = pd.to_datetime(df["timestamp_localtime"])
    if args.replay_from:
        df = df[df.tl >= pd.Timestamp(args.replay_from)]
    if args.replay_to:
        df = df[df.tl <= pd.Timestamp(args.replay_to)]
    df = df.sort_values("tl")
    df["t"] = pd.to_datetime(df["timestamp"], utc=True).map(lambda x: x.timestamp())
    print(f"replaying {len(df)} rows from {src.name}")
    for r in df.itertuples():
        run.log_raw(float(r.t), f"replay/{src.name}/as7341", json.dumps({"band": int(r.band), "reading": float(r.reading)}),
                    extra={"timestamp_localtime": str(r.tl)})
        run.n_msgs += 1
        run.add_values([(int(r.band), float(r.reading))], float(r.t))
    run.start = float(df.t.iloc[0]) if len(df) else time.time()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("MQTT")
    g.add_argument("--host", default=os.environ.get("PIO_MQTT_HOST", "pioreactor.local"))
    g.add_argument("--port", type=int, default=1883)
    g.add_argument("--user", default="pioreactor")
    g.add_argument("--password", default="raspberry")
    g.add_argument("--unit", help="restrict to one Pioreactor unit, e.g. pioreactor02")
    g.add_argument("--topic", help="full subscription filter (default pioreactor/+/+/# or per --unit)")
    g.add_argument("--match", default="as7341", help="only use topics containing this text (case-insensitive)")
    g.add_argument("--discover", type=float, metavar="SECONDS", help="list topics/payloads and exit")
    g.add_argument("--log-all", action="store_true", help="also store every unmatched message in the raw log")
    r = ap.add_argument_group("run")
    r.add_argument("--stage", default="adhoc")
    r.add_argument("--label", default="run")
    r.add_argument("--note", default="", help="free text stored in metadata (recipe, batch, vial volume ...)")
    r.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE")
    r.add_argument("--duration", type=float, default=0, help="seconds to record (0 = until Ctrl-C)")
    r.add_argument("--outdir", default="runs")
    r.add_argument("--no-lock", action="store_true", help="do not make the sealed files read-only")
    c = ap.add_argument_group("check")
    c.add_argument("--feature", default="555/480", help="ratio a/b of band centres, e.g. 555/480")
    c.add_argument("--target", type=float, nargs=2, default=[1.47, 1.51], metavar=("LO", "HI"))
    c.add_argument("--window-s", type=float, default=120, help="rolling window for the live status")
    c.add_argument("--analyze-last", type=float, default=120, help="seconds at the end used for the final verdict")
    c.add_argument("--drift-window-s", type=float, default=300, help="seconds of data used for the drift test (>= 300 advised)")
    c.add_argument("--max-drift", type=float, default=0.2, help="max |drift| in %% per 10 min to call it stable")
    c.add_argument("--max-sd", type=float, default=0.002, help="max SD of the ratio to call it stable")
    c.add_argument("--status-every", type=float, default=10)
    p = ap.add_argument_group("replay")
    p.add_argument("--replay", help="Pioreactor export .zip or as7341 CSV instead of live MQTT")
    p.add_argument("--replay-from", help="timestamp_localtime lower bound")
    p.add_argument("--replay-to", help="timestamp_localtime upper bound")
    args = ap.parse_args()

    if args.discover:
        return discover(args)
    a, b = (int(x) for x in args.feature.split("/"))
    if a not in BANDS or b not in BANDS:
        sys.exit(f"--feature must use bands from {BANDS}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = lambda s: re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    outdir = Path(args.outdir) / f"{stamp}_{safe(args.stage)}_{safe(args.label)}"
    outdir.mkdir(parents=True, exist_ok=False)
    if args.replay:
        source = dict(mode="replay", file=str(args.replay), file_sha256=sha256_file(args.replay),
                      replay_from=args.replay_from, replay_to=args.replay_to)
    else:
        source = dict(mode="live", broker=f"{args.host}:{args.port}", topic_filter=topic_filter(args), match=args.match)
    run = Run(args, outdir, source)
    stop = threading.Event()
    print(f"run folder: {outdir}")
    if args.replay:
        run_replay(args, run)
    else:
        stdin_events(run, stop)
        print("recording; type a note + Enter to log an event; Ctrl-C to stop")
        run_live(args, run, stop)
    stop.set()
    run.close_files()
    analysis = run.analyse()
    seal(run, analysis, args, sys.argv)
    print("\n" + "=" * 70)
    if analysis.get("verdict") == "INSUFFICIENT_DATA":
        print(f"INSUFFICIENT DATA ({len(run.t)} complete cycles) – check --match / topic with --discover")
    else:
        f = analysis["features"][args.feature]
        d = analysis["drift_check"]
        print(f"{args.feature}: median {f['median']:.4f}  mean {f['mean']:.4f}  SD {f['sd']:.4f}  "
              f"(n={f['n']}, last {analysis['window_seconds']:.0f} s)")
        print(f"drift over last {d['window_seconds']:.0f} s: {d['drift_pct_per_10min']:+.2f} ± {d['se']:.2f} %/10min (p={d['p']:.2f})")
        print(f"target {args.target[0]}–{args.target[1]}  ->  {analysis['verdict']}  ({analysis['verdict_text']})")
    print(f"evidence sealed in {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
