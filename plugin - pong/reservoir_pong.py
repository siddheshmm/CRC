# -*- coding: utf-8 -*-
"""
reservoir_pong.py — v0.1.0
──────────────────────────
Closed-loop "lane pong" on the dye reservoir (Pioreactor background job).

What it does, every `duration` minutes (one "slot"):
  1. A seeded ball says which lane it is in -> that lane is encoded as an alt_media dye dose
     (lane i -> levels[i] mL).  Default ball = "wander": a hidden target lane that moves now and
     then, with the ball jittering +-1 lane around it.  The best paddle spot is a moving average of
     recent lanes - what the leaky dye state computes - while a paddle that copies the current
     lane (no memory) is knocked off by every jitter.  The paddle is scored against the hidden aim.
  2. The last complete 8-band AS7341 scan BEFORE the dose ("pre") and the first one
     `observe_delay` seconds after the dose starts ("post") are read from the Pioreactor
     SQLite database and turned into absorbance A = -log10(I / I_baseline).
  3. A frozen ridge readout (from fit_pong_readout.py; it declares which of pre/post it
     uses) maps that vector to the lane the ball is AIMED at in the NEXT slot.  That is the paddle position.  Nothing else decides it:
     the controller keeps no history of past lanes; any memory must come from the dye.
  4. Next slot the paddle is scored against the lane the ball is aimed at (hit / miss).
  Results are published as job settings (MQTT) for pong_display.py and logged to SQLite.

Without --model it runs in COLLECT mode: same dosing/observation/logging, no prediction.
Use that (or specified_pump_dosing with a --make-sequence ball) to gather training data.

Like specified_pump_dosing it inherits BackgroundJobContrib, touches ONLY the alt_media
pump, and leaves your continuous media/waste pumps alone.

Run:
    pio run reservoir_pong --model pong_readout.json --max-cycles 250 --duration 0.545
Start from a CLEAN vial: the baseline is the mean of the first --baseline-min minutes.
"""
from __future__ import annotations

import json
import random
import re
import sqlite3
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any

import click
import numpy as np

from pioreactor.actions.pump import add_alt_media
from pioreactor.background_jobs.base import BackgroundJobContrib
from pioreactor.cli.run import run
from pioreactor.config import config
from pioreactor.utils import sqlite_worker
from pioreactor.utils.timing import current_utc_timestamp
from pioreactor.whoami import get_assigned_experiment_name
from pioreactor.whoami import get_latest_experiment_name
from pioreactor.whoami import get_unit_name


__plugin_name__ = "Reservoir Pong"
__plugin_summary__ = "Closed-loop lane pong: dye reservoir memory + frozen ridge readout drives the paddle"
__plugin_version__ = "0.1.0"
__plugin_author__ = "Siddhesh"
__plugin_homepage__ = "https://pioreactor.com"

BANDS = [415, 445, 480, 515, 555, 590, 630, 680]


# ── ball (identical copy lives in fit_pong_readout.py / pong_display.py) ──────

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


# ── small helpers ─────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    """Naive UTC 'now' (same clock domain as the DB timestamps)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _parse(ts: Any) -> datetime:
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).replace(tzinfo=None)


def _parse_levels(raw: Any) -> list[float]:
    vals = [float(t) for t in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", str(raw))]
    if len(vals) < 2:
        raise ValueError("levels needs at least 2 values, e.g. '0.1,0.3,0.5'")
    return vals


class ReservoirPong(BackgroundJobContrib):
    job_name = "reservoir_pong"

    published_settings = {
        "phase": {"datatype": "string", "settable": False, "unit": None},
        "slot": {"datatype": "integer", "settable": False, "unit": None},
        "ball_lane": {"datatype": "integer", "settable": False, "unit": None},
        "predicted_next_lane": {"datatype": "integer", "settable": False, "unit": None},
        "hits": {"datatype": "integer", "settable": False, "unit": None},
        "scored": {"datatype": "integer", "settable": False, "unit": None},
        "last_slot": {"datatype": "string", "settable": False, "unit": None},
    }

    def __init__(
        self,
        unit: str,
        experiment: str,
        levels: str = "0.1,0.3,0.5",
        duration: float | str = 0.5,
        max_cycles: int | str = 250,
        seed: int | str = 7,
        ball_kind: str = "wander",
        hold: float | str = 20.0,
        jitter: float | str = 0.6,
        observe_delay_s: float | str = 5.0,
        baseline_min: float | str = 10.0,
        model_path: str = "",
        spectrum_table: str = "as7341_spectrum_readings",
        obs_timeout_s: float | str = 20.0,
        scan_gap_s: float | str = 2.0,
        poll_s: float | str = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(unit=unit, experiment=experiment, plugin_name="reservoir_pong", **kwargs)

        self.levels = _parse_levels(levels)
        self.n_lanes = len(self.levels)
        self.interval_s = float(duration) * 60.0
        self.max_cycles = int(float(max_cycles))
        if self.max_cycles <= 0:
            raise ValueError("max_cycles must be > 0")
        self.baseline_s = float(baseline_min) * 60.0
        self.obs_timeout_s = float(obs_timeout_s)
        self.scan_gap_s = float(scan_gap_s)   # max gap between band rows of ONE scan
        self.poll_s = float(poll_s)           # how often to re-check the DB while waiting for a scan
        self.observe_delay_s = float(observe_delay_s)
        if not re.fullmatch(r"[A-Za-z0-9_]+", str(spectrum_table)):
            raise ValueError("spectrum_table must be a plain table name")
        self.spectrum_table = str(spectrum_table)
        self.db_path = config.get("storage", "database")

        self.model = self._load_model(model_path)
        if self.model is not None and abs(float(self.model["observe_delay_s"]) - self.observe_delay_s) > 1e-6:
            self.logger.warning(
                f"observe_delay {self.observe_delay_s}s overridden by the model's "
                f"{self.model['observe_delay_s']}s (readout was fitted at that delay)."
            )
            self.observe_delay_s = float(self.model["observe_delay_s"])

        if self.interval_s < self.observe_delay_s + 10.0:
            self.logger.warning(
                f"interval {self.interval_s:.1f}s is short for observe_delay {self.observe_delay_s}s + DB latency."
            )

        self.ball_cfg = {"kind": str(ball_kind), "hold": float(hold), "jitter": float(jitter)}
        if self.model is not None and self.model.get("ball") and {
            k: self.model["ball"].get(k) for k in self.ball_cfg} != self.ball_cfg:
            self.logger.warning(f"Ball settings {self.ball_cfg} differ from the ones the readout was trained on: "
                                f"{self.model['ball']}. Accuracy will not mean what you think.")
        self.ball, self.aim = make_ball(self.max_cycles + 1, self.n_lanes, int(float(seed)), **self.ball_cfg)
        self.baseline: np.ndarray | None = None
        self._pending_pred: int | None = None
        self._halt = threading.Event()
        self._thread: threading.Thread | None = None
        self._t_start = _utcnow()

        # published settings (assigning publishes them)
        self.phase = "init"
        self.slot = 0
        self.ball_lane = -1
        self.predicted_next_lane = -1
        self.hits = 0
        self.scored = 0
        self.last_slot = ""

        self._db = sqlite_worker.Sqlite3Worker(self.db_path)
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS reservoir_pong_records (
                experiment      TEXT NOT NULL,
                pioreactor_unit TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                slot            INTEGER NOT NULL,
                lane            INTEGER NOT NULL,
                aim             INTEGER,
                volume_ml       REAL NOT NULL,
                dose_start      TEXT NOT NULL,
                obs_time        TEXT,
                features_json   TEXT,
                y_raw           REAL,
                pred_next_lane  INTEGER NOT NULL,
                hit_prev        INTEGER NOT NULL
            )
            """
        )
        mode = "PLAY (frozen readout)" if self.model else "COLLECT (no readout)"
        self.logger.info(
            f"Reservoir Pong init | {mode} | levels={self.levels} mL | {self.max_cycles} slots "
            f"every {self.interval_s:.1f}s | observe +{self.observe_delay_s}s"
        )

    # ── model ────────────────────────────────────────────────────────────────

    def _load_model(self, model_path: str) -> dict | None:
        if not str(model_path).strip():
            return None
        with open(model_path, "r") as f:
            m = json.load(f)
        need = {"feature_set", "observe_delay_s", "mx", "sx", "w", "b", "n_lanes"}
        missing = need.difference(m)
        if missing:
            raise ValueError(f"{model_path} is missing keys {sorted(missing)}")
        if int(m["n_lanes"]) != self.n_lanes:
            raise ValueError(f"model has {m['n_lanes']} lanes but levels gives {self.n_lanes}")
        m["mx"] = np.asarray(m["mx"], float)
        m["sx"] = np.asarray(m["sx"], float)
        m["w"] = np.asarray(m["w"], float)
        return m

    def _predict(self, x: np.ndarray) -> tuple[float, int]:
        m = self.model
        y = float(((x - m["mx"]) / m["sx"]) @ m["w"] + float(m["b"]))
        return y, int(np.clip(np.rint(y), 0, self.n_lanes - 1))

    # ── spectrum access (reads the table the spectrometer job writes) ────────

    def _check_spectrum_table(self) -> bool:
        con = sqlite3.connect(self.db_path, timeout=5.0)
        try:
            cols = {r[1] for r in con.execute(f"PRAGMA table_info({self.spectrum_table})")}
            need = {"experiment", "pioreactor_unit", "timestamp", "band", "reading"}
            if need.issubset(cols):
                return True
            like = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND (name LIKE '%7341%' OR name LIKE '%spectr%')"
            )]
            self.logger.error(
                f"Table '{self.spectrum_table}' missing or lacks columns {sorted(need)}. "
                f"Spectrum-like tables found: {like}. Pass --spectrum-table."
            )
            return False
        finally:
            con.close()

    def _query_rows(self, since: datetime) -> list[tuple]:
        con = sqlite3.connect(self.db_path, timeout=5.0)
        try:
            return con.execute(
                f"SELECT timestamp, band, reading FROM {self.spectrum_table} "
                "WHERE experiment = ? AND pioreactor_unit = ? AND timestamp >= ? ORDER BY timestamp",
                (self.experiment, self.unit, _iso(since)),
            ).fetchall()
        finally:
            con.close()

    @staticmethod
    def _group_scans(rows: list[tuple], gap_s: float = 2.0) -> list[tuple[datetime, np.ndarray]]:
        """Group band rows into complete 8-band scans -> [(completion_time, intensities[8])]."""
        scans: list[tuple[datetime, np.ndarray]] = []
        cur: dict[int, list[float]] = {}
        t_last: datetime | None = None
        for ts, band, reading in rows:
            t = _parse(ts)
            if t_last is not None and (t - t_last).total_seconds() > gap_s:
                if all(b in cur for b in BANDS):
                    scans.append((t_last, np.array([np.mean(cur[b]) for b in BANDS])))
                cur = {}
            cur.setdefault(int(band), []).append(float(reading))
            t_last = t
        if t_last is not None and all(b in cur for b in BANDS):
            scans.append((t_last, np.array([np.mean(cur[b]) for b in BANDS])))
        return scans

    def _scans_between(self, t0: datetime, t1: datetime) -> list[tuple[datetime, np.ndarray]]:
        rows = self._query_rows(t0 - timedelta(seconds=3))
        return [(t, v) for t, v in self._group_scans(rows, self.scan_gap_s) if t0 <= t <= t1]

    def _last_scan_before(self, t: datetime) -> np.ndarray | None:
        rows = self._query_rows(t - timedelta(seconds=15))
        prior = [v for tt, v in self._group_scans(rows, self.scan_gap_s) if tt < t]
        return prior[-1] if prior else None

    def _scan_at_or_after(self, t_target: datetime) -> tuple[datetime, np.ndarray] | None:
        """First complete scan whose completion time >= t_target (same rule as the offline fit)."""
        self._halt.wait(max(0.0, (t_target - _utcnow()).total_seconds()))
        deadline = time.monotonic() + self.obs_timeout_s
        while time.monotonic() < deadline and not self._halt.is_set():
            for t, v in self._group_scans(self._query_rows(t_target - timedelta(seconds=3)), self.scan_gap_s):
                if t >= t_target:
                    return t, v
            self._halt.wait(self.poll_s)
        return None

    def _absorbance(self, intensities: np.ndarray) -> np.ndarray:
        return -np.log10(np.clip(intensities / self.baseline, 1e-6, None))

    # ── lifecycle ────────────────────────────────────────────────────────────

    def on_ready(self) -> None:
        super().on_ready()
        self._t_start = _utcnow()
        self._thread = threading.Thread(target=self._run, name="reservoir_pong", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            if not self._check_spectrum_table():
                return
            self._measure_baseline()
            if self.baseline is None or self._halt.is_set():
                return
            self.phase = "running"
            t_next = time.monotonic()
            for n in range(self.max_cycles):
                if self._halt.is_set():
                    break
                self.slot = n
                self._cycle(n)
                t_next += self.interval_s
                wait = t_next - time.monotonic()
                if wait <= 0:
                    self.logger.warning(f"Slot {n} overran its {self.interval_s:.1f}s window by {-wait:.1f}s.")
                    t_next = time.monotonic()
                elif self._halt.wait(wait):
                    break
            self.phase = "done"
            if self.scored:
                self.logger.info(f"Finished: {self.hits}/{self.scored} hits ({100.0 * self.hits / self.scored:.1f}%).")
        except Exception as e:  # keep the job from dying silently in a thread
            self.logger.error(f"reservoir_pong stopped: {e!r}\n{traceback.format_exc()}")
        finally:
            self._finish()

    def _finish(self) -> None:
        if self.state != self.DISCONNECTED:
            self.set_state(self.DISCONNECTED)

    def _measure_baseline(self) -> None:
        self.phase = "baseline"
        self.logger.info(f"Recording clean-vial baseline for {self.baseline_s:.0f}s (vial must be dye-free).")
        if self._halt.wait(self.baseline_s):
            return
        guard = timedelta(seconds=min(20.0, 0.1 * self.baseline_s))
        scans = self._scans_between(self._t_start + guard, _utcnow() - guard)
        if len(scans) < 10:
            self.logger.error(f"Only {len(scans)} complete baseline scans found - is the spectrometer job running?")
            return
        arr = np.array([v for _, v in scans])
        self.baseline = arr.mean(axis=0)
        third = max(1, len(arr) // 3)
        drift = (arr[-third:].mean(0) / arr[:third].mean(0) - 1.0) * 100.0
        cv = arr.std(axis=0) / arr.mean(axis=0) * 100.0
        sig = cv * np.sqrt(2.0 / third)               # drift you would see from scan noise alone
        noise = float(np.median(self._absorbance(arr).std(axis=0)))
        self.logger.info(f"Baseline from {len(arr)} scans; A-noise ~{noise:.4f}; max drift {np.abs(drift).max():.2f}%.")
        if np.any(np.abs(drift) > np.maximum(1.0, 4.0 * sig)):
            self.logger.warning("Baseline is still drifting beyond scan noise - vial may not be clean/settled; results will be off.")

    def _fire(self, volume_ml: float) -> bool:
        try:
            add_alt_media(
                ml=volume_ml,
                unit=self.unit,
                experiment=self.experiment,
                source_of_event=f"{self.job_name}:alt_media",
                mqtt_client=self.pub_client,
                logger=self.logger,
            )
            return True
        except Exception as e:
            self.logger.error(f"Dose of {volume_ml} mL failed: {e}")
            return False

    # ── one slot ─────────────────────────────────────────────────────────────

    def _cycle(self, n: int) -> None:
        lane = self.ball[n]
        vol = self.levels[lane]

        # score where the paddle was put last slot against the lane the ball is aimed at now
        hit = -1
        if self._pending_pred is not None:
            hit = int(self._pending_pred == self.aim[n])
            self.scored += 1
            self.hits += hit

        fs = str(self.model["feature_set"]).split("+") if self.model is not None else []
        t_dose = _utcnow()
        pre = self._last_scan_before(t_dose)          # newest state before this dose (always logged)
        self._fire(vol)

        obs = self._scan_at_or_after(t_dose + timedelta(seconds=self.observe_delay_s))
        a_pre = None if pre is None else self._absorbance(pre)
        a_post = None if obs is None else self._absorbance(obs[1])
        obs_time = "" if obs is None else _iso(obs[0])
        if obs is None:
            self.logger.warning(f"Slot {n}: no scan found within {self.obs_timeout_s}s of the observation time.")

        y_raw, pred = None, None
        feats = {"pre": None if a_pre is None else [round(float(v), 6) for v in a_pre],
                 "post": None if a_post is None else [round(float(v), 6) for v in a_post]}
        if self.model is not None:
            parts = [a_pre if "pre" in fs else None, a_post if "post" in fs else None]
            wanted = [p for p, name in zip(parts, ("pre", "post")) if name in fs]
            if any(p is None for p in wanted):
                self.logger.warning(f"Slot {n}: missing scan for features {fs}; no prediction this slot.")
            else:
                y_raw, pred = self._predict(np.concatenate(wanted))

        self._pending_pred = pred
        self.ball_lane = lane
        self.predicted_next_lane = -1 if pred is None else pred
        payload = {
            "slot": n, "lane": lane, "aim": self.aim[n], "volume_ml": vol, "pred_next": self.predicted_next_lane,
            "y": None if y_raw is None else round(y_raw, 4), "hit_prev": hit,
            "hits": self.hits, "scored": self.scored, "n_lanes": self.n_lanes,
            "mode": "play" if self.model else "collect",
        }
        self.last_slot = json.dumps(payload)
        self.logger.info(
            f"Slot {n}: ball lane {lane} ({vol} mL) -> paddle {self.predicted_next_lane}"
            + (f" | prev {'HIT' if hit else 'miss'} ({self.hits}/{self.scored})" if hit >= 0 else "")
        )
        self._save_record(n, lane, vol, t_dose, obs_time, feats, y_raw, pred, hit)

    def _save_record(self, n, lane, vol, t_dose, obs_time, feats, y_raw, pred, hit) -> None:
        try:
            self._db.execute(
                """
                INSERT INTO reservoir_pong_records
                (experiment, pioreactor_unit, timestamp, slot, lane, aim, volume_ml, dose_start, obs_time,
                 features_json, y_raw, pred_next_lane, hit_prev)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.experiment, self.unit, current_utc_timestamp(), n, lane, int(self.aim[n]), vol, _iso(t_dose), obs_time,
                    json.dumps(feats),
                    None if y_raw is None else float(y_raw),
                    -1 if pred is None else int(pred), int(hit),
                ),
            )
        except Exception as e:
            self.logger.debug(f"Could not save record: {e}")

    def on_sleeping(self) -> None:
        self.logger.info("Reservoir Pong paused: stopping the loop (it does not resume; restart the job).")
        self._halt.set()
        super().on_sleeping()

    def on_disconnected(self) -> None:
        self._halt.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=5.0)
        try:
            self._db.close()
        except Exception:
            pass
        super().on_disconnected()


# ── CLI ───────────────────────────────────────────────────────────────────────

@run.command(name="reservoir_pong", help=__plugin_summary__)
@click.option("--levels", default="0.1,0.3,0.5", type=str, help="Dose in mL for each lane, lowest lane first")
@click.option("--duration", default=0.5, type=float, help="Minutes between dose STARTS")
@click.option("--max-cycles", default=250, type=int, help="Number of slots to play")
@click.option("--seed", default=7, type=int, help="Ball seed - use a DIFFERENT seed from the training run")
@click.option("--ball", "ball_kind", default="wander", type=click.Choice(["wander", "zigzag"]))
@click.option("--hold", default=20.0, type=float, help="wander: mean slots before the hidden target lane moves")
@click.option("--jitter", default=0.6, type=float, help="wander: chance the ball is knocked one lane off target")
@click.option("--observe-delay", default=5.0, type=float, help="Seconds after dose start to read the scan")
@click.option("--baseline-min", default=10.0, type=float, help="Clean-vial baseline minutes before slot 0")
@click.option("--model", "model_path", default="", type=str, help="Frozen readout JSON; empty = collect mode")
@click.option("--spectrum-table", default="as7341_spectrum_readings", type=str)
@click.option("--unit", default=None, type=str)
@click.option("--experiment", default=None, type=str)
def click_reservoir_pong(levels, duration, max_cycles, seed, ball_kind, hold, jitter, observe_delay, baseline_min,
                         model_path, spectrum_table, unit, experiment) -> None:
    """Run the dye-reservoir lane-pong loop as an independent background job."""
    unit = unit or get_unit_name()
    experiment = experiment or get_assigned_experiment_name(unit) or get_latest_experiment_name()
    with ReservoirPong(
        unit=unit, experiment=experiment, levels=levels, duration=duration, max_cycles=max_cycles,
        seed=seed, ball_kind=ball_kind, hold=hold, jitter=jitter, observe_delay_s=observe_delay, baseline_min=baseline_min,
        model_path=model_path, spectrum_table=spectrum_table,
    ) as job:
        job.block_until_disconnected()
