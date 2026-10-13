# Reservoir lane pong (dye reservoir + frozen ridge readout)

A closed loop on the Pioreactor dye reservoir. Each slot (~33 s) a seeded **ball** picks a lane; the lane is
dosed as dye (lane 0/1/2 → 0.1/0.3/0.5 mL). A **frozen linear readout** of the 8-band AS7341 scan guesses the
lane the ball is **aimed at** next slot and moves the **paddle** there. The controller keeps no history of
lanes, so any benefit over a memoryless paddle has to come from dye still in the vial.

The ball ("wander"): a hidden aim lane that moves now and then (mean 20 slots); each slot the ball sits on
the aim, but with probability 0.6 it is knocked one lane off. The best guess for the aim is a moving average
of recent lanes, which is what the leaky dye state computes. A paddle that copies the last ball lane is the
memoryless comparison; `pong_display.py` scores it live next to the reservoir.

> Why not a bouncing (zigzag) ball? Its next position needs a *difference* of past inputs. One leaky state only
> supplies a smoothed sum, so even noise-free the readout sits within 0.01 of a rounding boundary. It is still
> available (`--ball zigzag`) for comparison. Synthetic checks: ideal single-state ceiling ≈ 0.79 vs 0.58 for
> copy-last-lane at the default settings (hold 20, jitter 0.6).

## Files
| file | where it runs | what it does |
|---|---|---|
| `reservoir_pong.py` + `.yaml` | on the Pioreactor (`~/.pioreactor/plugins/`, same place as `specified_pump_dosing`) | the job: dose, read scans from SQLite, apply readout, publish, log |
| `fit_pong_readout.py` | laptop or Pi | makes training sequences; fits/freezes the readout from normal exports; prints honest CV numbers |
| `pong_display.py` | laptop | live window / text view over MQTT, or `--sim` with no hardware |
| `local_trial.py` | laptop | whole chain with no Pioreactor: simulated vial -> real fit -> frozen readout -> game replay |

## Try it on your own computer first (no Pioreactor, no pio, no MQTT)
```
pip install numpy pandas pygame          # pygame optional (use --text without it)
python local_trial.py                    # simulated dye vial -> fit_pong_readout.py -> play -> game window
python local_trial.py --text             # same, plain text
python local_trial.py --noise 0.012      # noisier sensor: watch the advantage over the memoryless paddle shrink
python pong_display.py --sim             # just the game with a toy reservoir (no fitting)
```
Keep the files in one folder. `local_trial.py` simulates three training runs with different ball seeds, runs the real
`fit_pong_readout.py` on them (it prints the cross-validated numbers), simulates one play run with an unseen ball, applies the
frozen readout slot by slot and replays it in the display, printing reservoir vs copy-last-lane accuracy first. On WSL2 the
window needs WSLg; otherwise use `--text` or run it with Windows Python. The physics is one leaky state (tau 5.2 slots) plus
noise, so it checks the pipeline; it does not predict your real vial. For a realistic noise setting, match your own measured
lag-1 memory (see the `fit_capacity_multiseq.py` output).

## Workflow
Start every run from a **clean, dye-free vial** with the spectrometer job running. Keep `--duration` identical
in training and play (0.545 min ≈ the 32.7 s slots of your earlier runs).

1. **Collect training data (ball only, no readout).** Use a different seed per run, three runs or more:
   ```
   pio run reservoir_pong --max-cycles 250 --duration 0.545 --seed 3
   pio run reservoir_pong --max-cycles 250 --duration 0.545 --seed 4
   pio run reservoir_pong --max-cycles 250 --duration 0.545 --seed 5
   ```
   (Alternative: `python fit_pong_readout.py --make-sequence 250 --levels 0.1,0.3,0.5 --seed 3` prints a
   `--volume-sequence` for `specified_pump_dosing`; then use `--alt-source specified_pump_dosing:alt_media` below.)
2. **Export** `dosing_events` and `as7341_spectrum_readings` for each run into one folder per run.
3. **Fit and freeze** (it regenerates each run's ball from its seed and refuses to continue if the doses differ):
   ```
   python fit_pong_readout.py --runs run_s3 run_s4 run_s5 --seeds 3 4 5 \
       --levels 0.1,0.3,0.5 --alt-source reservoir_pong:alt_media --out pong_readout.json
   ```
   Read the blocked-CV block. Only continue if accuracy clearly beats the memoryless lines and the p-value is
   small. The "ideal leaky average" line is the ceiling for one leaky state; expect to land below it.
4. **Play** with an unseen seed, unchanged settings, no refitting afterwards:
   ```
   pio run reservoir_pong --model pong_readout.json --seed 9 --max-cycles 250 --duration 0.545
   python pong_display.py --host <leader-address> --unit <unit-name> --experiment "<experiment name>"
   ```
5. **Judge** with the job's final log line (hits/scored) against the copy-last-lane baseline on the same slots;
   per-slot rows are in the `reservoir_pong_records` table (lane, aim, prediction, hit, features).

## Things to check on the real unit (these were stubbed in my tests)
* The spectrum table is assumed to be `as7341_spectrum_readings` with columns `experiment, pioreactor_unit,
  timestamp, band, reading`. If not, the job logs the spectrum-like tables it found; pass `--spectrum-table`.
* Doses from this plugin carry `source_of_event = reservoir_pong:alt_media` (use `--alt-source` in analysis).
* Published settings are what `pong_display.py` listens to: `pioreactor/<unit>/<experiment>/reservoir_pong/last_slot`.
* Pause (sleeping) stops the loop for good; restart the job instead.
* Expect lower accuracy than on synthetic data: real scans carry pump transients and slower drifts.
