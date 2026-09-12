"""
titration_reservoir.py  (Phase: acid-base driven reservoir, phenolphthalein +
methyl orange readout)

Physical picture: a buffered vial in the Pio, periodically dosed with small
pulses of acid (e.g. acetic acid / vinegar) and base (e.g. sodium
bicarbonate solution) by the peristaltic pumps. No self-sustained
oscillation is expected or needed here -- the PUMP SCHEDULE is the drive
signal, and the vial's pH response (nonlinear, buffered, with some memory
from incomplete neutralization between doses) is the reservoir dynamics.
Readout is via two indicators read across AS7341's 8 spectral bands.

Chemistry (real, standard weak-acid/buffer equilibria):
  R1 (fast, reversible): HA <-> H+ + A-     [acetic acid dissociation]
        Ka_acetic = 1.8e-5 (real literature value)
  R2 (effectively irreversible): H+ + HCO3- -> H2CO3 -> (H2O + CO2 escapes)
        Because CO2 escapes as gas from an open vial, this reaction is
        pulled to completion rather than sitting at an equilibrium --
        modeled as a fast one-way sink for H+, consuming bicarbonate.

Dosing: implemented as discrete pulses (batch integration between pulses,
then an instantaneous concentration jump at each dose time) -- this matches
how a real peristaltic pump actually adds reagent, rather than smearing it
into a continuous CSTR inflow.

Indicator model: pH -> fractional "colored form" via the Henderson-
Hasselbalch relation (standard, not invented): fraction_high_pH_form =
1 / (1 + 10^(pKa - pH)). Each indicator's colored/uncolored forms are then
given an approximate absorbance spectrum across AS7341's 8 channel
wavelengths (415-680 nm), sourced from published visible-spectrum
absorption peaks:
  - Phenolphthalein: pKa ~9.4; colorless below ~8.2, pink above ~10
    (pink form absorption peak ~552 nm)
  - Methyl orange: pKa ~3.7; red below ~3.1, yellow above ~4.4
    (red form peak ~506 nm, yellow form peak ~464 nm)
These per-band absorbance NUMBERS are approximate (order-of-magnitude,
literature-informed peak positions) -- they are NOT a substitute for
measuring your actual indicators on your actual AS7341 once reagents
arrive. Treat this readout model as a planning tool, not ground truth.
"""

import numpy as np
from crn_simulator import Reaction, ChemicalReactionNetwork


# ---- CRN: weak acid equilibrium + irreversible base neutralization --------
def build_titration_crn(Ka_acetic=1.8e-5, Ka1_carbonic=4.3e-7, k1_relax=1e4, k2_relax=1e3):
    """
    k1_relax, k2_relax: forward rate constants for the two fast acid-base
    equilibria (R1, R2). Both are treated as REVERSIBLE with a real
    equilibrium constant (k_f/k_r = Ka), not a one-way sink -- an earlier
    version modeled bicarbonate neutralization as irreversible, which let
    it drain residual H+ indefinitely with no equilibrium ceiling and
    produced an unphysical pH~11.5 spike even with no acid present yet.
    Real sodium bicarbonate solutions are mildly basic (~pH 8-9), not that
    extreme -- making R2 a genuine equilibrium (like R1) fixes this by
    giving the system a real thermodynamic floor instead of unlimited
    proton consumption.
    """
    k1_r = k1_relax
    k1_f = Ka_acetic * k1_relax
    k2_r = k2_relax
    k2_f = Ka1_carbonic * k2_relax
    return ChemicalReactionNetwork(
        species=["HA", "Aion", "H", "HCO3", "H2CO3"],
        reactions=[
            Reaction("R1_acid_equilibrium", reactants={"HA": 1},
                     products={"H": 1, "Aion": 1}, k_f=k1_f, k_r=k1_r),
            Reaction("R2_bicarbonate_equilibrium", reactants={"H": 1, "HCO3": 1},
                     products={"H2CO3": 1}, k_f=k2_f, k_r=k2_r),
        ],
        dilution_rate=0.0,   # closed batch vial between doses -- no CSTR flow
    )


# ---- Pulse dosing: batch-integrate between doses, jump concentration at each
def simulate_with_dosing(crn, y0: dict, dose_schedule, t_end, dt_report=1.0):
    """
    dose_schedule: list of (time, species, delta_concentration) tuples,
    sorted by time. Each represents an instantaneous pump pulse adding
    `delta_concentration` to that species (approximating a small volume of
    concentrated stock solution injected into the well-mixed vial).
    Returns (t, conc_dict) stitched across all segments.
    """
    dose_schedule = sorted(dose_schedule, key=lambda d: d[0])
    t_all, conc_all = [], {sp: [] for sp in crn.species}
    y_current = dict(y0)
    t_cursor = 0.0

    boundaries = [d[0] for d in dose_schedule] + [t_end]
    dose_idx = 0
    for boundary in boundaries:
        if boundary <= t_cursor:
            continue
        n_pts = max(2, int((boundary - t_cursor) / dt_report))
        t_seg, conc_seg = crn.simulate(y_current, (t_cursor, boundary), n_points=n_pts)
        t_all.append(t_seg)
        for sp in crn.species:
            conc_all[sp].append(conc_seg[sp])
        y_current = {sp: conc_seg[sp][-1] for sp in crn.species}
        t_cursor = boundary
        # apply any doses scheduled exactly at this boundary
        while dose_idx < len(dose_schedule) and dose_schedule[dose_idx][0] == boundary:
            _, sp, delta = dose_schedule[dose_idx]
            y_current[sp] = max(0.0, y_current[sp] + delta)
            dose_idx += 1

    t_full = np.concatenate(t_all)
    conc_full = {sp: np.concatenate(conc_all[sp]) for sp in crn.species}
    return t_full, conc_full


# ---- Indicator color model --------------------------------------------------
def henderson_hasselbalch_fraction(H_conc, pKa):
    """Fraction of indicator in its HIGH-pH (deprotonated) form."""
    pH = -np.log10(np.clip(H_conc, 1e-14, None))
    return 1.0 / (1.0 + 10 ** (pKa - pH))


# Approximate AS7341-band absorbance contributions per indicator colored
# form. Bands (nm): 415, 445, 480, 515, 555, 590, 630, 680
AS7341_BANDS_NM = [415, 445, 480, 515, 555, 590, 630, 680]

# Rough Gaussian-peak absorbance profiles (peak position, width, height) --
# literature-informed peak wavelengths, but width/height are order-of-
# magnitude estimates, NOT measured values. Flagged in the docstring above.
def _gaussian_peak(wavelengths, peak, width, height):
    return height * np.exp(-0.5 * ((np.array(wavelengths) - peak) / width) ** 2)

PHENOLPHTHALEIN_PINK_SPECTRUM = _gaussian_peak(AS7341_BANDS_NM, peak=552, width=35, height=1.0)
METHYL_ORANGE_RED_SPECTRUM    = _gaussian_peak(AS7341_BANDS_NM, peak=506, width=30, height=0.8)
METHYL_ORANGE_YELLOW_SPECTRUM = _gaussian_peak(AS7341_BANDS_NM, peak=464, width=30, height=0.6)


def predicted_band_absorbance(H_conc, phen_total_conc, mo_total_conc,
                               pKa_phen=9.4, pKa_mo=3.7):
    """
    Returns an (n_timepoints, 8) array of predicted relative absorbance per
    AS7341 band, from the combined phenolphthalein + methyl orange color
    state at each timepoint's pH.
    """
    phen_pink_frac = henderson_hasselbalch_fraction(H_conc, pKa_phen)  # 0=colorless,1=pink
    mo_yellow_frac = henderson_hasselbalch_fraction(H_conc, pKa_mo)    # 0=red,1=yellow
    mo_red_frac = 1.0 - mo_yellow_frac

    n = len(H_conc)
    absorbance = np.zeros((n, len(AS7341_BANDS_NM)))
    for i in range(n):
        absorbance[i] = (
            phen_total_conc * phen_pink_frac[i] * PHENOLPHTHALEIN_PINK_SPECTRUM
            + mo_total_conc * mo_red_frac[i] * METHYL_ORANGE_RED_SPECTRUM
            + mo_total_conc * mo_yellow_frac[i] * METHYL_ORANGE_YELLOW_SPECTRUM
        )
    return absorbance


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # buffer capacity increased 20x (0.001 -> 0.02) so the first acid dose
    # doesn't immediately exhaust it and strand the system in a low-pH
    # corner -- lets the sawtooth ride around a more useful working range
    # instead of an offset acidic plateau
    crn = build_titration_crn()

    dose_schedule = []
    t = 5.0
    rng = np.random.default_rng(42)
    while t < 200:
        if (t // 5) % 2 == 0:
            dose_schedule.append((t, "HA", 0.002 * rng.uniform(0.5, 1.5)))
        else:
            dose_schedule.append((t, "HCO3", 0.0022 * rng.uniform(0.5, 1.5)))
        t += 5.0

    y0 = {"HA": 0.0, "Aion": 0.0, "H": 1e-7, "HCO3": 0.01, "H2CO3": 0.0}
    t_trace, conc = simulate_with_dosing(crn, y0, dose_schedule, t_end=200, dt_report=0.5)

    pH_trace = -np.log10(np.clip(conc["H"], 1e-14, None))
    print(f"pH range over run: [{pH_trace.min():.2f}, {pH_trace.max():.2f}]")

    # indicator concentrations scaled to match the real optically-detectable
    # range we validated empirically with actual food dyes (~0.01-0.4
    # relative concentration gave clear AS7341 signal in the earlier
    # experiments) -- the original 0.0005/0.0003 values were ~30x too small
    # and produced band signals that rounded to display as zero
    phen_conc, mo_conc = 0.02, 0.015
    band_signal = predicted_band_absorbance(conc["H"], phen_total_conc=phen_conc, mo_total_conc=mo_conc)

    print(f"\nPredicted AS7341-band absorbance range per band (with realistic "
          f"indicator concentrations phen={phen_conc}, mo={mo_conc}):")
    for i, b in enumerate(AS7341_BANDS_NM):
        col = band_signal[:, i]
        print(f"  {b}nm: [{col.min():.4f}, {col.max():.4f}]")

    # --- Plot pH and band signal over time ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t_trace, pH_trace, color="tab:purple")
    axes[0].set_ylabel("pH")
    axes[0].set_title("Titration reservoir: pH driven by alternating acid/base pulse dosing")
    axes[1].plot(t_trace, band_signal)
    axes[1].set_ylabel("predicted AS7341 band absorbance")
    axes[1].set_xlabel("time")
    axes[1].legend([f"{b}nm" for b in AS7341_BANDS_NM], ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig("titration_reservoir_trace.png", dpi=120)
    plt.close(fig)
    print("\nSaved trace plot -> titration_reservoir_trace.png")

    # --- Closure test: can a linear readout recover pH from the 8-band
    # signal alone? Same ridge-regression readout method used throughout
    # (ChemComp-style y(t) = sum alpha_i * c_i(t)), now applied to our own
    # simulated system instead of a paper's benchmark. ---
    from crn_simulator import fit_readout, rmse
    n = len(t_trace)
    train_mask = np.arange(n) < n // 2
    test_mask = ~train_mask

    model = fit_readout(band_signal[train_mask], pH_trace[train_mask], alpha=1e-6)
    pred = model.predict(band_signal)
    train_rmse = rmse(pH_trace[train_mask], pred[train_mask])
    test_rmse = rmse(pH_trace[test_mask], pred[test_mask])
    print(f"\n--- Readout closure test: recovering pH from 8-band AS7341 signal ---")
    print(f"train RMSE (pH units): {train_rmse:.4f}")
    print(f"test RMSE (pH units):  {test_rmse:.4f}")