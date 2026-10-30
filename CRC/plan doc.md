# Chemical reservoir benchmarking protocol (methyl orange, Pioreactor)

Draft v1 · 23 Oct 2026 · flow-through methyl orange reservoir, acid/base pulse inputs, spectrometer readout

## 1. Setup and what the calibration gives us

- **Vial:** constant working volume V = 10 mL. Continuous inflow from the Arduino pump (60% duty), continuous efflux through the waste pump. Residence time τ = V / Q, with Q measured in Stage 1.
- **Inputs:** acid pulses from add-media, base pulses from alt-media. Positive input = acid, negative = base.
- **Readout:** AS7341 bands (5 s cadence). Primary feature: ratio 555/480. Backups: 555/445, 555/680, 630/555 (all passed the ladder test).
- **Calibration, in one line:** across the vinegar ladder the 555/480 ratio runs from 1.386 (T1, acid plateau, probe pH about 2.6) to 1.601 (T8 and T0, neutral plateau, pH 4.5 and above), with the sigmoid centre near pH 3.7 (ratio about 1.49). Methyl orange is flat above about pH 4.5, so a base ladder would only re-measure the plateau. It is not needed.
- **What the calibration is for:** choosing the operating point and the pulse sizes. The reservoir readout itself is trained directly on spectra, so it does not depend on the pH numbers.
- **Limits to remember:** probe slope was 75% of nominal and the best feature was picked from 30 candidates on 9 points, so quote pH values as roughly ±0.2.

**Operating window.** Keep every pulse response inside 555/480 = 1.40 to 1.58. Outside that the indicator saturates and information is lost.

**Noise floor.** Within-window noise SD of 555/480 is about 0.0013. Two input levels are distinguishable only if their responses differ by at least 5 SD (about 0.007).

## 2. Schedule at a glance

| Day | Stages | Time |
| --- | --- | --- |
| 1 | 0 prep, 1 hydraulics and baseline, 2 pulse-response map, 3 parameter choice | about 8 h |
| 2 | 4 main benchmark run | about 10 h plus 1 h baseline |
| 3 | 5 echo-state run, 6 hold-out run (optional) | about 6 to 9 h |

Times assume τ about 3 min. If τ is longer, everything that waits in multiples of τ stretches proportionally.

## 3. Stage 0: preparation (about 1.5 h)

1. **Lock the optics.** Use the same Pioreactor, vial, vial position, stirring RPM, LED channel and intensity, and AS7341 settings (integration time, gain) as on the ladder day. The calibration only transfers if these match. Write the settings down.
2. **Prepare the carrier (4 L, enough for the main run).** This is the T5/T6 recipe scaled up, at about 1/500 vinegar: 3.48 L water + about 7 mL vinegar + 520 mL methyl orange stock (13% stock by volume, the same as 1.5 mL MO added to the 10 mL left in each ladder tube). Mix well in a covered bottle. If your vinegar is not 5%, adjust the vinegar volume so the carrier lands in step 4.
3. **Prepare the pulse stocks.**
   - Acid stock: start with neat vinegar.
   - Base stock: start with household ammonia diluted to about 1:150 in water (brand strength varies, so the titration in step 5 sets the final dilution).
4. **Bench-check the carrier (hold-out check of the calibration).** Put 10 mL of carrier in the vial, record 5 min, and take the median 555/480 of the last 2 min. It should be 1.47 to 1.51. If it is above 1.51, add a little vinegar; if it is below 1.47, dilute with water or a drop of base stock. Repeat until it is in range. This is a free test of whether the ladder calibration reproduces on a new day and a new preparation.
5. **Bench-titrate the pulse stocks (static vial, fresh 10 mL of carrier each time).**
   - Add 0.2 mL of acid stock, wait 1 min, record the ratio. Target: 1.40 to 1.42.
   - Fresh carrier, add 0.2 mL of base stock, wait 1 min, record the ratio. Target: 1.56 to 1.58.
   - If a response overshoots the target, dilute that stock and repeat. If it undershoots, use a stronger stock. Write down the final stock concentrations. These are upper bounds, because the flowing system dilutes the peak.
6. **Prime the lines.** Fill the carrier, acid and base lines completely. Check nothing dribbles or siphons with the pumps stopped for 10 min.

**Gate to continue:** carrier baseline in 1.47 to 1.51, and both stocks reach their targets with 0.2 mL.

## 4. Stage 1: hydraulics and baseline (about 1.5 h)

1. Fill the vial to the 10 mL level with carrier. Start stirring and the spectrometer logging.
2. Start the carrier pump (60% duty) and the waste pump. Let it run 10 min and confirm the level stays constant.
3. **Measure Q.** Collect effluent in a graduated cylinder for 10 min. Q = volume collected / 10 min. Then **τ = 10 mL / Q**. Repeat once and average.
4. **Baseline.** Let the system run for at least 3τ + 20 min with no pulses. Accept it when the 555/480 drift is under 0.2% per 10 min and the noise SD is 0.002 or less. Record the baseline value F0 and noise SD sigma.
5. **Choose the step length.** Use T\_s ≈ τ/3, rounded to a multiple of 15 s and at least 30 s:

| τ | T\_s |
| --- | --- |
| 2 min | 40 s |
| 3 min | 60 s |
| 4.5 min | 90 s |

**Gate to continue:** stable baseline and a measured τ. Send me Q and τ and I will fix the sequence length and run times.

## 5. Stage 2: pulse-response map (about 4.5 h at τ about 3 min)

Purpose: measure how big and how long each pulse response is, whether it saturates, and how strongly pulses interact.

Set the trial order with a random shuffle and write it down before starting. Wait 4τ between independent trials. The pulse volumes below are starting guesses; after the first few trials you may scale them.

**Block A: single pulses (12 trials, about 2.4 h)**

| Condition | Volume |
| --- | --- |
| A1, A2, A3 | 0.05, 0.10, 0.20 mL acid |
| B1, B2, B3 | 0.05, 0.10, 0.20 mL base |

Two repetitions of each, in shuffled order.

For each trial:

1. Confirm the baseline is back within 0.003 of F0.
2. Fire the pulse at a noted time t\_p.
3. Record until the feature is back within 0.003 of F0, and at least 4τ.
4. Save: peak change dF, time to peak, and the decay time constant (fit an exponential to the last 70% of the decay).

**Block B: overshoot / plateau test (2 trials, about 0.5 h)**

- 0.5 mL acid and 0.5 mL base, one trial each, waiting 5τ afterwards.
- Check that the response saturates near the plateaus (about 1.39 and about 1.60) instead of continuing to grow, and that the base side is flat after the peak. This is the check that ammonia does not add a spectral artifact of its own.

**Block C: paired pulses (6 trials, about 1.5 h)**

- Pairs: acid then acid, acid then base, base then acid, with the second pulse one step (T\_s) after the first. Use the middle volume (0.10 mL) for both.
- Two repetitions of each, 5τ between trials.
- Superposition error = (measured response to the pair) minus (sum of the two single responses). A large error is a good sign: it means the reservoir is nonlinear, which is what the nonlinear benchmarks need.

**Outputs of Stage 2 (a table plus three plots):**

- Gain curve (dF vs volume, acid and base separately).
- Decay time constant vs V/Q.
- Acid/base asymmetry.
- Superposition error vs pair type.

## 6. Stage 3: choose the run parameters (about 30 min, no experiment)

Apply these rules in order:

1. **v\_max:** the largest volume whose peak stays inside 1.40 to 1.58, and whose dF is about 80% of the overshoot-test dF (so it is not saturated). Use the smaller of the acid and base v\_max for both sides unless the asymmetry is large; if it is, scale each side so that equal levels give equal peak dF.
2. **Level size:** dv = v\_max / 4, giving 9 levels from −4 to +4. Check that one level step changes the response by at least 0.007 (5 sigma). If not, use 7 levels (−3 to +3).
3. **Decay check:** if the measured decay constant is within 20% of τ, the hydraulics behave as a simple stirred tank. If it is much longer, mixing is slow (check stirring and the inlet position).
4. **Step length:** keep T\_s from Stage 1 unless the decay constant differs strongly from τ; then use T\_s = (measured decay constant) / 3.

## 7. Stage 4: main benchmark run (about 10 h)

**Sequence.**

- Generate with a fixed seed (for example 42) before starting and save it as a CSV with columns: step, level, volume\_mL, acid\_or\_base.
- Levels are independent and uniform over {−4 ... +4}. Level 0 means no pulse. Positive means acid, negative means base.
- N = 600 steps: steps 0 to 29 are washout, 30 to 429 are training (400), 430 to 599 are test (170). The split is contiguous; never shuffle it.

**Procedure.**

1. Re-run the baseline check (Stage 1, step 4) and write down F0 and sigma.
2. Record 20 min of baseline with no pulses.
3. Note the start time t0 in UTC and Nepal time. Step k starts at t0 + k × T\_s.
4. At the start of each step, fire the pulse for that step's level (nothing for level 0).
5. Keep spectrometer logging on throughout. Do not touch the setup.
6. After step 599, keep recording 20 min with no pulses (end-of-run baseline).
7. Once a day, note room temperature; if possible keep it constant. Note carrier level in the bottle at the start and end.

**Data to save:** the spectrometer export for the whole run, the sequence CSV, the t0 timestamp, the baseline F0 at start and end, Q measured before and after, and the settings from Stage 0, step 1.

**If something fails mid-run:** a short pause (under 5 steps) is tolerable and counts as a gap; note the step numbers. A longer pause means restarting with a fresh 30-step washout and treating the two halves as separate segments.

## 8. Stage 5: echo-state (consistency) run (about 4 to 5 h)

Purpose: show that the reservoir state depends on the input history and not on where it started.

1. **Run ESP-A:** pre-load the vial with a 0.5 mL acid slug (add it and start the sequence immediately). Run the first 120 steps of the Stage 4 sequence.
2. Return to baseline (5τ after the run ends).
3. **Run ESP-B:** same, but pre-load with a 0.5 mL base slug.
4. The first 120 steps of the main run (which started from baseline) are the third trajectory.

**Analysis:**

- Distance between trajectories: d(t) = |F\_A(t) − F\_B(t)| in units of sigma.
- Pass if d(t) falls to about 2 sigma or less within 3τ and stays there.
- Consistency: after the washout, the R² between the ESP-A, ESP-B and main-run feature series should be at least 0.95.
- Also report the decay of d(t) as an exponential time constant and compare it with τ.

## 9. Stage 6: hold-out run (optional, about 3.5 h)

- New seed (for example 43), 200 steps (30 washout, 170 test), same pulse levels and step length.
- Apply the readout weights trained in Stage 4 without refitting.
- If the test score is close to the Stage 4 test score, the readout is stable over hours and days. If it drops sharply, drift is limiting and the baseline checks show how much.

## 10. Analysis plan

**Features per step.** Take the readings at taps 10, 20, 30, ... T\_s seconds after the pulse. For each tap use the ratios 555/480, 555/445 and 630/555. This gives about 18 features per step (6 taps × 3 ratios). Add a bias term and a linear time term (to absorb slow drift).

**Readout.** Ridge regression. Choose the regularisation by blocked 5-fold cross-validation on the training segment only. Never tune on the test segment.

**Benchmarks and what to report.**

| Benchmark | Definition | Report |
| --- | --- | --- |
| Memory capacity (MC) | r² between the readout and u(t−k) for k = 1 to 15; MC = sum over k | MC\_k curve, total MC, noise floor (about features / N\_test), shuffled-input null |
| Effective dimension | participation ratio of the feature covariance eigenvalues | the practical ceiling for MC |
| Delayed parity / XOR | sign(u(t)) × sign(u(t−1)), delays 1 to 3 | accuracy vs 50% chance |
| NARMA-2 | standard recursion with input mapped to \[0, 0.5\] | NMSE |
| NARMA-10 | same, order 10 | NMSE (expected to be poor; a stress test) |
| Degree-2 information capacity | products of Legendre polynomials of u(t−i), u(t−j) | capacity, compared with degree 1 |

**Baselines (these decide whether the reservoir adds anything).**

1. Linear regression on the raw delayed inputs u(t) ... u(t−15).
2. The same readout on a zero-input run (carrier only).
3. The same readout with the input sequence circularly shifted (a null for chance fits).

The reservoir only counts as a win on a task if it beats baseline 1 on the test segment with a confidence interval that excludes zero (block bootstrap over the test steps).

**Expected ballpark.** One indicator has about two independent colour states, so expect an effective dimension of 2 to 4 and a total MC of 2 to 4. A low MC is a finding about the chemistry, not a failed experiment.

## 11. Decision tree

- **Baseline will not settle (drift above 0.2% per 10 min):** check stirring, temperature, bubbles in the optical path, and whether the carrier bottle is well mixed.
- **Pulse response saturates at the smallest volume:** dilute the stocks.
- **Pulse response too small to separate levels (below 0.007 per level):** use stronger stocks, or fewer levels.
- **Decay much slower than τ:** poor mixing; check the stir bar and the inlet and outlet positions.
- **Acid and base responses very different:** rescale volumes so equal levels give equal peak dF.
- **Echo-state distance does not decay:** the system has not forgotten its start; check for a leak or a stagnant zone in the vial.
- **MC near zero:** the step is too long compared with τ (memory has decayed between steps). Reduce T\_s or reduce Q to raise τ.
- **NARMA results no better than the linear baseline:** the nonlinearity is too weak; try larger pulses (still inside 1.40 to 1.58) or add a second indicator later.

## 12. File naming

Use one folder per stage: `stage1_baseline`, `stage2_pulse_map`, `stage4_main`, `stage5_esp`, `stage6_holdout`. In each, keep the spectrometer export zip, the sequence CSV and a short text note with t0 (UTC and Nepal), Q, τ, T\_s, pulse volumes and stock concentrations.
