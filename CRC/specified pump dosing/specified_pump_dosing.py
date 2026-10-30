# -*- coding: utf-8 -*-
"""
specified_pump_dosing.py — v1.2.0
──────────────────────────────────
Flexible single-pump and bipolar sequence dosing plugin for the Pioreactor.

Allows running a sequence of doses using:
  - 'media'     (Nutrient / Media pump)
  - 'alt_media' (Alternative Media / Salt / Inducer pump)
  - 'waste'     (Effluent / Waste pump)
  - 'bipolar'   (Dual-pump mode: positive doses -> media, negative doses -> alt_media)

Key Features:
  1. Complete Pump Isolation: Only actuates designated dosing pumps. Continuous media & waste pumps
     are NEVER touched or stopped upon exit.
  2. Background Job Architecture (Disconnect-Proof): Inherits from BackgroundJobContrib.
     Runs independently in Pioreactor's background daemon. Closing terminal, SSH timeouts, or
     laptop sleeping will NOT interrupt execution.
  3. Bipolar Dual-Pump Mode: Supports alternating between acid (media) and base (alt_media) within
     a single job, eliminating timer drift and MQTT collisions.
  4. Flexible Input: Accepts comma-separated strings or direct sequence files (.csv or .txt).
  5. Clean Shutdown: Cancels only its own cycle timer and closes its own DB handle.
"""
from __future__ import annotations

import csv
import re
import time
from pathlib import Path
from threading import Timer
from typing import Any

import click

from pioreactor.actions.pump import add_alt_media
from pioreactor.actions.pump import add_media
from pioreactor.actions.pump import remove_waste
from pioreactor.background_jobs.base import BackgroundJobContrib
from pioreactor.cli.run import run
from pioreactor.config import config
from pioreactor.utils import sqlite_worker
from pioreactor.utils.timing import current_utc_timestamp
from pioreactor.whoami import get_assigned_experiment_name
from pioreactor.whoami import get_latest_experiment_name
from pioreactor.whoami import get_unit_name


__plugin_name__ = "Specified Pump Dosing"
__plugin_summary__ = "Execute a sequence of doses using any specified pump (including bipolar) without touching other pumps"
__plugin_version__ = "1.2.0"
__plugin_author__ = "Siddhesh"
__plugin_homepage__ = "https://pioreactor.com"


class SpecifiedPumpDosing(BackgroundJobContrib):
    job_name = "specified_pump_dosing"

    published_settings = {
        "pump": {
            "datatype": "string",
            "settable": True,
            "unit": None,
        },
        "duration": {
            "datatype": "float",
            "settable": True,
            "unit": "min",
        },
        "volume_sequence": {
            "datatype": "string",
            "settable": True,
            "unit": "mL",
        },
        "default_volume_ml": {
            "datatype": "float",
            "settable": True,
            "unit": "mL",
        },
        "max_cycles": {
            "datatype": "integer",
            "settable": True,
            "unit": "cycles",
        },
        "current_dose_ml": {
            "datatype": "float",
            "settable": False,
            "unit": "mL",
        },
        "current_cycle": {
            "datatype": "integer",
            "settable": False,
            "unit": None,
        },
    }

    def __init__(
        self,
        unit: str,
        experiment: str,
        pump: str = "media",
        duration: float | str = 20.0,
        volume_sequence: str = "",
        sequence_file: str | None = None,
        default_volume_ml: float | str = 0.10,
        max_cycles: int | str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            unit=unit,
            experiment=experiment,
            plugin_name="specified_pump_dosing",
            **kwargs,
        )

        self.pump = str(pump).strip().lower()
        if self.pump not in ("media", "alt_media", "waste", "bipolar"):
            self.logger.warning(
                f"Unknown pump '{self.pump}', falling back to 'media'. "
                "Valid options: 'media', 'alt_media', 'waste', 'bipolar'."
            )
            self.pump = "media"

        try:
            self.duration = float(duration)
        except (ValueError, TypeError):
            self.duration = 20.0
        if self.duration <= 0:
            raise ValueError("duration must be greater than 0")

        self.sequence_file = str(sequence_file).strip() if sequence_file else ""
        self.volume_sequence = str(volume_sequence).strip() if volume_sequence is not None else ""
        try:
            self.default_volume_ml = float(default_volume_ml)
        except (ValueError, TypeError):
            self.default_volume_ml = 0.10

        # Parse sequence from file or string
        self.sequence = self._load_sequence()

        # Parse max_cycles
        try:
            if max_cycles is not None and str(max_cycles).strip() != "":
                self.max_cycles = int(float(str(max_cycles).strip()))
            elif self.sequence:
                self.max_cycles = len(self.sequence)
            else:
                self.max_cycles = 0
        except (ValueError, TypeError):
            self.max_cycles = len(self.sequence) if self.sequence else 0

        self.current_cycle = 0
        self.current_dose_ml = self._calculate_dose_volume(0)
        self._timer: Timer | None = None

        # Dedicated SQLite table for logging doses
        db_path = config.get("storage", "database")
        self._db = sqlite_worker.Sqlite3Worker(db_path)
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS specified_pump_dosing_records (
                experiment       TEXT NOT NULL,
                pioreactor_unit  TEXT NOT NULL,
                timestamp        TEXT NOT NULL,
                cycle            INTEGER NOT NULL,
                pump             TEXT NOT NULL,
                volume_ml        REAL NOT NULL
            )
            """
        )

        seq_info = f"{len(self.sequence)} steps" if self.sequence else f"fixed {self.default_volume_ml} mL"
        cycle_info = f"max_cycles={self.max_cycles}" if self.max_cycles > 0 else "max_cycles=unlimited"
        self.logger.info(
            f"Specified Pump Dosing v{__plugin_version__} ready | Target Pump='{self.pump}' | "
            f"Sequence: {seq_info} | Interval: {self.duration} min | {cycle_info}"
        )

    def _load_sequence(self) -> list[float]:
        """Loads dose sequence from sequence_file or volume_sequence (supports CSV, TXT, or string)."""
        target_path = None
        if self.sequence_file:
            target_path = Path(self.sequence_file)
        elif self.volume_sequence:
            p = Path(self.volume_sequence.strip().strip("'").strip('"'))
            if p.exists() and p.is_file():
                target_path = p

        if target_path and target_path.is_file():
            if target_path.suffix.lower() == ".csv":
                try:
                    with open(target_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        seq = []
                        for row in reader:
                            vol = float(row.get("volume_mL") or row.get("vol") or row.get("volume") or 0.0)
                            reagent = str(row.get("acid_or_base") or row.get("reagent") or "").lower()
                            pump = str(row.get("pump") or "").lower()
                            lvl = float(row.get("level") or 0.0)

                            if self.pump == "bipolar":
                                # Base = negative, Acid = positive
                                if reagent == "base" or pump == "add_alt_media" or lvl < 0:
                                    seq.append(-abs(vol))
                                elif reagent == "acid" or pump == "add_media" or lvl > 0:
                                    seq.append(abs(vol))
                                else:
                                    seq.append(0.0)
                            else:
                                seq.append(abs(vol))

                        self.logger.info(f"Loaded {len(seq)} steps from CSV file: {target_path.name}")
                        return seq
                except Exception as e:
                    self.logger.warning(f"Error parsing CSV sequence file {target_path.name}: {e}")
            else:
                return self._parse_sequence_string(target_path.read_text(encoding="utf-8"))

        return self._parse_sequence_string(self.volume_sequence)

    def _parse_sequence_string(self, raw_str: str) -> list[float]:
        """Extract floating point numbers from formatted strings, line-breaks, or commas."""
        if not raw_str:
            return []
        try:
            cleaned = str(raw_str).replace("\n", " ").replace("\r", " ").replace(";", ",")
            tokens = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", cleaned)
            if self.pump == "bipolar":
                # In bipolar mode, preserve sign: positive = media (acid), negative = alt_media (base)
                return [float(t) for t in tokens]
            else:
                return [max(0.0, float(t)) for t in tokens]
        except Exception as e:
            self.logger.warning(f"Could not parse sequence string: {e}")
            return []

    def _calculate_dose_volume(self, cycle_idx: int | None = None) -> float:
        """Get the dose volume for the current or specified cycle index."""
        c = self.current_cycle if cycle_idx is None else cycle_idx
        if self.sequence:
            return round(self.sequence[c % len(self.sequence)], 4)
        return round(self.default_volume_ml, 4)

    def _save_record(self, volume_ml: float, pump: str | None = None) -> None:
        """Record dosing event into local SQLite database."""
        target_pump = pump or self.pump
        try:
            self._db.execute(
                """
                INSERT INTO specified_pump_dosing_records
                (experiment, pioreactor_unit, timestamp, cycle, pump, volume_ml)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    self.experiment,
                    self.unit,
                    current_utc_timestamp(),
                    self.current_cycle,
                    target_pump,
                    volume_ml,
                ),
            )
        except Exception as e:
            self.logger.debug(f"Could not save dosing record to DB: {e}")

    def on_ready(self) -> None:
        super().on_ready()
        self.logger.info(f"Specified Pump Dosing ready. Initiating cycle 0 on pump '{self.pump}'...")
        self._run_step()

    def _run_step(self) -> None:
        if self.state != self.READY:
            return

        # Check if max_cycles has already been reached
        if self.max_cycles > 0 and self.current_cycle >= self.max_cycles:
            self.logger.info(
                f"Completed all {self.max_cycles} cycle(s). Stopping job without touching other pumps."
            )
            self._stop_timer()
            self.set_state(self.DISCONNECTED)
            return

        dose_val = self._calculate_dose_volume()
        cycle_str = f"{self.current_cycle}" + (f"/{self.max_cycles}" if self.max_cycles > 0 else "")

        # Resolve pump and actual volume to dose
        if self.pump == "bipolar":
            if dose_val > 0:
                target_pump = "media"
                act_vol = dose_val
            elif dose_val < 0:
                target_pump = "alt_media"
                act_vol = abs(dose_val)
            else:
                target_pump = "none"
                act_vol = 0.0
        else:
            target_pump = self.pump
            act_vol = abs(dose_val)

        self.current_dose_ml = act_vol
        self._save_record(act_vol, pump=target_pump)

        # Actuate designated pump
        if target_pump != "none" and act_vol > 0:
            self.logger.info(f"Cycle {cycle_str} | Dosing {act_vol:.4f} mL on '{target_pump}' pump.")
            self._fire_pump(target_pump, act_vol)
        else:
            self.logger.info(f"Cycle {cycle_str} | Idle step (0.0 mL). No dosing actuated.")

        self.current_cycle += 1

        # Check if max_cycles reached after completing current dose
        if self.max_cycles > 0 and self.current_cycle >= self.max_cycles:
            self.logger.info(
                f"Finished all {self.max_cycles} cycle(s). Shutting down."
            )
            self._stop_timer()
            self.set_state(self.DISCONNECTED)
            return

        # Schedule next dosing cycle
        duration_s = self.duration * 60.0
        self.logger.info(f"Next cycle in {self.duration} min ({duration_s:.0f}s).")
        self._timer = Timer(duration_s, self._run_step)
        self._timer.daemon = True
        self._timer.start()

    def _fire_pump(self, pump_name: str, volume_ml: float) -> bool:
        """Actuates ONLY the designated pump. Leaves continuous flow pumps completely untouched."""
        kwargs = dict(
            unit=self.unit,
            experiment=self.experiment,
            source_of_event=f"{self.job_name}:{pump_name}",
            mqtt_client=self.pub_client,
            logger=self.logger,
        )

        try:
            if pump_name == "media":
                add_media(ml=volume_ml, **kwargs)
            elif pump_name == "alt_media":
                add_alt_media(ml=volume_ml, **kwargs)
            elif pump_name == "waste":
                remove_waste(ml=volume_ml, **kwargs)
            return True
        except Exception as e:
            self.logger.error(f"Failed to dose {volume_ml} mL on pump '{pump_name}': {e}")
            return False

    def _stop_timer(self) -> None:
        """Cancel active cycle timer."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def on_sleeping(self) -> None:
        self.logger.info(f"Specified Pump Dosing paused/sleeping for pump '{self.pump}'.")
        self._stop_timer()
        super().on_sleeping()

    def on_disconnected(self) -> None:
        self.logger.info(f"Specified Pump Dosing disconnected. Only pump '{self.pump}' was managed.")
        self._stop_timer()
        try:
            self._db.close()
        except Exception:
            pass
        super().on_disconnected()


# ── Standalone CLI Command ───────────────────────────────────────────────────

@run.command(name="specified_pump_dosing", help=__plugin_summary__)
@click.option(
    "--pump",
    default="media",
    type=click.Choice(["media", "alt_media", "waste", "bipolar"], case_sensitive=False),
    help="Target pump: 'media', 'alt_media', 'waste', or 'bipolar'",
)
@click.option(
    "--duration",
    default=20.0,
    type=float,
    help="Minutes between dosing steps (schedule interval)",
)
@click.option(
    "--volume-sequence",
    default="",
    type=str,
    help="Comma-separated volume sequence in mL (e.g. '0.01, -0.01')",
)
@click.option(
    "--sequence-file",
    default=None,
    type=click.Path(exists=True),
    help="Path to CSV or TXT file containing dose sequence",
)
@click.option(
    "--default-volume-ml",
    default=0.10,
    type=float,
    help="Fixed volume in mL if volume_sequence is empty",
)
@click.option(
    "--max-cycles",
    default=None,
    type=int,
    help="Number of cycles to run (defaults to sequence length if provided)",
)
@click.option("--unit", default=None, type=str, help="Target Pioreactor unit name")
@click.option("--experiment", default=None, type=str, help="Experiment name")
def click_specified_pump_dosing(
    pump: str,
    duration: float,
    volume_sequence: str,
    sequence_file: str | None,
    default_volume_ml: float,
    max_cycles: int | None,
    unit: str | None,
    experiment: str | None,
) -> None:
    """Run specified pump dosing as an independent background job from CLI."""
    unit = unit or get_unit_name()
    experiment = experiment or get_assigned_experiment_name(unit) or get_latest_experiment_name()

    with SpecifiedPumpDosing(
        unit=unit,
        experiment=experiment,
        pump=pump,
        duration=duration,
        volume_sequence=volume_sequence,
        sequence_file=sequence_file,
        default_volume_ml=default_volume_ml,
        max_cycles=max_cycles,
    ) as job:
        job.block_until_disconnected()
