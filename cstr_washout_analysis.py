"""
cstr_washout_analysis.py

Audited Analysis of Continuous-Flow CSTR Dye Washout Experiments.
Supports arbitrary datasets (e.g. data/15th sept, data/14th sept) with full auto-detection:
1. Mathematical consistency: Normalised comparison uses WashoutFit.normalised_observations()
   defined as (y(t) - baseline(t)) / amplitude, directly matching exp(-t/tau).
2. Unified baseline reference: Peak incremental absorbance (Delta A_peak) and residual
   excess (Delta A_post) are defined with respect to the immediate pre-dose baseline,
   with fractional recovery = (1 - Delta A_post / Delta A_peak) * 100%.
3. Algorithmic statistical recovery test: Labels are dynamically generated via two-sample
   Welch's t-test and Z-score (relative to pre-dose SD) along with relaxation duration
   (t_span vs t_95%), eliminating hard-coded labels.
4. Synchronized peak-aligned frames: Peaks are detected starting strictly at logged injection
   time, and spectral profiles in Plot 3 use the synchronized single-acquisition frame at t_peak.
   Decay windows start strictly at t_peak (t_elapsed = 0).
5. Multi-pulse and multi-channel fitting: Evaluates all pulses and key absorbing
   channels (480, 515, 445, 555 nm) using cstr_washout_fit_tools.fit_washout with TRF bounds
   and parameter covariance estimation.

Outputs 5 high-resolution individual plots:
1. plot1_raw_spectral_transmission_trace.png
2. plot2_raw_transmission_drop_and_absorbance.png
3. plot3_raw_spectral_absorbance_profile.png
4. plot4_raw_washout_kinetics_cstr_model.png
5. plot5_raw_baseline_recovery_summary.png
"""

import os
import sys
import glob
import shutil
import json
import math
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from cstr_washout_fit_tools import fit_washout, hydraulic_comparison, WashoutFit

AS7341_BANDS = [415, 445, 480, 515, 555, 590, 630, 680]
BAND_COLORS = {
    415: "#6a0dad",  # violet
    445: "#0055d4",  # blue
    480: "#00a2e8",  # cyan
    515: "#008a00",  # green (peak red dye absorption)
    555: "#8cb800",  # lime/yellow-green
    590: "#e67e22",  # orange
    630: "#e74c3c",  # red
    680: "#880e4f",  # deep red / NIR
}


def load_and_preprocess_raw_data(spec_path: str, dose_path: str):
    """Load spectrometer and dosing logs, convert timestamps to IST (UTC+05:30)."""
    # 1. Spectrometer readings
    spec = pd.read_csv(spec_path)
    spec["ts_ist"] = pd.to_datetime(spec["timestamp"]).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    spec["ts_sec"] = spec["ts_ist"].dt.round("1s")

    wide = spec.pivot_table(index="ts_sec", columns="band", values="reading", aggfunc="mean")
    wide.columns = [int(c) for c in wide.columns]
    wide = wide.sort_index()

    # Filter out optical bubble / chamber disturbances
    bubbles = (wide[515] < 0.15) | (wide[445] < 0.12) | (wide[630] < 0.15)
    wide_clean = wide.copy()
    if bubbles.any():
        wide_clean[bubbles] = np.nan
        wide_clean = wide_clean.interpolate(method="time")

    # 2. Dosing events
    dose = pd.read_csv(dose_path)
    dose["ts_ist"] = pd.to_datetime(dose["timestamp"]).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    media = dose[dose["event"] == "add_media"]
    waste = dose[dose["event"] == "remove_waste"]
    alt = dose[dose["event"] == "add_alt_media"].copy().sort_values("ts_ist")

    # Cluster discrete alt-media injection pulses (time difference > 15s)
    alt["time_diff"] = alt["ts_ist"].diff().dt.total_seconds()
    alt["event_id"] = (alt["time_diff"] > 15.0).cumsum()

    alt_events = []
    for ev_id, group in alt.groupby("event_id"):
        t_event = group["ts_ist"].iloc[0]
        total_vol = group["volume_change_ml"].sum()
        alt_events.append({
            "event_id": ev_id,
            "time": t_event,
            "volume_mL": round(total_vol, 2),
            "n_pulses": len(group)
        })
    alt_events_df = pd.DataFrame(alt_events)

    return wide, wide_clean, alt_events_df, media, waste


def analyze_raw_response(wide: pd.DataFrame, wide_clean: pd.DataFrame,
                         alt_events_df: pd.DataFrame,
                         media: pd.DataFrame, waste: pd.DataFrame,
                         data_label: str = "Experiment",
                         output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    art_dir = r"C:\Users\SIDDHESH\.gemini\antigravity-cli\brain\699f08a6-3530-4968-8e7b-b79a8ab841a1"
    os.makedirs(art_dir, exist_ok=True)

    print("=" * 78)
    print(f"AUDITED CSTR DYE WASHOUT ANALYSIS: {data_label.upper()}")
    print("=" * 78)

    t_m_start, t_m_end = media["ts_ist"].min(), media["ts_ist"].max()
    dur_media_min = (t_m_end - t_m_start).total_seconds() / 60.0
    flow_media = media["volume_change_ml"].sum() / dur_media_min
    flow_waste = waste["volume_change_ml"].sum() / dur_media_min

    print(f"\n1. FLOW & DOSING SUMMARY:")
    print(f"   Clear water inflow rate: {flow_media:.2f} mL/min ({media['volume_change_ml'].sum():.1f} mL total)")
    print(f"   Waste outflow rate:       {flow_waste:.2f} mL/min ({waste['volume_change_ml'].sum():.1f} mL total)")
    print(f"   Dosing events (IST timestamps matching UI):")
    for i, row in alt_events_df.iterrows():
        print(f"     - Dose {i+1}: {row['time'].strftime('%H:%M:%S')} IST -> {row['volume_mL']:.2f} mL red dye ({row['n_pulses']} pulses)")

    # Dynamically locate the clean water blank reference period before Dose 1
    t_first = alt_events_df["time"].iloc[0]
    b_start = max(wide.index.min() + pd.Timedelta(seconds=120), t_first - pd.Timedelta(minutes=10))
    b_end = t_first - pd.Timedelta(seconds=45)
    blank_ref = wide_clean.loc[b_start:b_end].mean()
    blank_std = wide_clean.loc[b_start:b_end].std()

    print(f"\n   Clean water blank reference ({b_start.strftime('%H:%M:%S')} to {b_end.strftime('%H:%M:%S')} IST):")
    print(f"     T_515 = {blank_ref[515]:.5f} +/- {blank_std[515]:.5f}")

    # Optical absorbance A_b(t) = -log10(T_b(t) / Blank_b)
    absorbance = -np.log10(wide_clean / blank_ref)
    dom_band = 515  # Dominant absorption wavelength for red food dye

    # Dynamically build dose configs
    dose_info = []
    n_events = len(alt_events_df)
    for i, row in alt_events_df.iterrows():
        t_dose = row["time"]
        vol = row["volume_mL"]
        if i + 1 < n_events:
            decay_end = alt_events_df["time"].iloc[i + 1] - pd.Timedelta(seconds=10)
            delta_to_next = (alt_events_df["time"].iloc[i + 1] - t_dose).total_seconds()
        else:
            decay_end = wide.index.max()
            delta_to_next = (decay_end - t_dose).total_seconds()

        # Adaptive window sizing based on inter-pulse interval
        pre_span_s = min(75.0, max(20.0, delta_to_next * 0.20))
        pre_s = t_dose - pd.Timedelta(seconds=pre_span_s + 5)
        pre_e = t_dose - pd.Timedelta(seconds=5)

        post_span_s = min(75.0, max(20.0, delta_to_next * 0.20))
        post_s = decay_end - pd.Timedelta(seconds=post_span_s)
        post_e = decay_end

        dose_info.append({
            "name": f"Dose {i+1} ({vol:.1f} mL)",
            "volume_mL": vol,
            "t_dose": t_dose,
            "pre_window": (pre_s, pre_e),
            "post_window": (post_s, post_e),
            "decay_end": decay_end,
        })

    # =========================================================================
    # 2. ALGORITHMIC PEAK DETECTION & STATISTICAL RECOVERY TESTING
    # =========================================================================
    print("\n2. ALGORITHMIC PEAK DETECTION & STATISTICAL RECOVERY ANALYSIS:")
    pulse_results = []

    for info in dose_info:
        t_dose = info["t_dose"]
        # Peak search strictly begins at injection timestamp over a 45s window
        search_slice = absorbance.loc[t_dose : t_dose + pd.Timedelta(seconds=45), dom_band]
        t_peak = search_slice.idxmax()
        A_peak = float(search_slice.loc[t_peak])
        peak_delay_s = (t_peak - t_dose).total_seconds()

        # Single-acquisition frame snapshot across all 8 bands at t_peak
        peak_spectrum_row = absorbance.loc[t_peak]

        # Pre-dose baseline stats
        pre_s, pre_e = info["pre_window"]
        pre_samples = absorbance.loc[pre_s:pre_e, dom_band]
        mu_pre = float(pre_samples.mean())
        std_pre = float(pre_samples.std(ddof=1))
        n_pre = len(pre_samples)

        pre_trans_samples = wide_clean.loc[pre_s:pre_e, dom_band]
        mu_pre_trans = float(pre_trans_samples.mean())
        std_pre_trans = float(pre_trans_samples.std(ddof=1))

        # Post-washout stats
        post_s, post_e = info["post_window"]
        post_samples = absorbance.loc[post_s:post_e, dom_band]
        mu_post = float(post_samples.mean())
        std_post = float(post_samples.std(ddof=1))
        n_post = len(post_samples)

        post_trans_samples = wide_clean.loc[post_s:post_e, dom_band]
        mu_post_trans = float(post_trans_samples.mean())
        std_post_trans = float(post_trans_samples.std(ddof=1))

        # Consistent incremental metrics relative to pre-dose baseline
        delta_peak = A_peak - mu_pre
        delta_post = mu_post - mu_pre
        recovery_pct = (1.0 - (delta_post / delta_peak)) * 100.0 if delta_peak > 0 else 100.0

        # Welch's two-sample t-test (unequal variances)
        t_stat, p_val = stats.ttest_ind(post_samples, pre_samples, equal_var=False)
        z_score = abs(delta_post) / std_pre if std_pre > 0 else 0.0

        # Washout fit from t_peak to decay_end (t_elapsed starts strictly at 0.0)
        decay_seg = absorbance.loc[t_peak:info["decay_end"], dom_band].dropna()
        t_el = (decay_seg.index - t_peak).total_seconds().values
        fit = fit_washout(t_el, decay_seg.values)
        s = fit.summary()

        # Algorithmic status determination:
        t_span = t_el[-1]
        t95 = s["t95_of_fitted_excess_s"]
        if t_span < t95:
            status_str = "Incomplete (t_span < t_95%)"
        elif p_val > 0.05 or z_score <= 2.0:
            status_str = f"Recovered (Z = {z_score:.1f} SD, p = {p_val:.2f})"
        else:
            status_str = f"Residual offset (Delta={delta_post:+.4f} AU)"

        res = {
            "info": info,
            "t_peak": t_peak,
            "peak_delay_s": peak_delay_s,
            "A_peak": A_peak,
            "peak_spectrum": peak_spectrum_row,
            "mu_pre": mu_pre,
            "std_pre": std_pre,
            "mu_pre_trans": mu_pre_trans,
            "std_pre_trans": std_pre_trans,
            "mu_post": mu_post,
            "std_post": std_post,
            "mu_post_trans": mu_post_trans,
            "std_post_trans": std_post_trans,
            "delta_peak": delta_peak,
            "delta_post": delta_post,
            "recovery_pct": recovery_pct,
            "welch_t": float(t_stat),
            "p_val": float(p_val),
            "z_score": float(z_score),
            "fit": fit,
            "fit_summary": s,
            "status_str": status_str,
        }
        pulse_results.append(res)

        print(f"\n   [{info['name']}]")
        print(f"     Logged dose time:    {t_dose.strftime('%H:%M:%S')} IST")
        print(f"     Algorithmic peak:    {t_peak.strftime('%H:%M:%S')} IST (+{peak_delay_s:.1f}s after injection)")
        print(f"     Peak absorbance:     A_515 = {A_peak:.5f} AU (Delta_peak = {delta_peak:.5f} AU)")
        print(f"     Pre-dose baseline:   {mu_pre:.5f} +/- {std_pre:.5f} AU (T = {mu_pre_trans:.5f})")
        print(f"     Post-washout level:  {mu_post:.5f} +/- {std_post:.5f} AU (T = {mu_post_trans:.5f})")
        print(f"     Residual Delta_post: {delta_post:+.5f} AU | Recovery: {recovery_pct:.1f}%")
        print(f"     Welch t-test:        t = {t_stat:.2f}, p-value = {p_val:.4f} (Z = {z_score:.2f} SD)")
        print(f"     Fitted Washout Tau:  tau = {s['tau_s']:.1f} +/- {s['tau_standard_error_iid_approx_s']:.1f}s (t95 = {t95:.1f}s, R^2 = {s['r_squared_descriptive']:.3f})")
        print(f"     Algorithmic Status:  {status_str}")

    # =========================================================================
    # 3. MULTI-CHANNEL FITS ACROSS ABSORBING WAVELENGTHS (DOSE 2)
    # =========================================================================
    fit_dose_idx = 1 if len(pulse_results) > 1 else 0
    ref_res = pulse_results[fit_dose_idx]
    ref_peak = ref_res["t_peak"]
    ref_end = ref_res["info"]["decay_end"]

    print(f"\n3. MULTI-CHANNEL FIT COMPARISON ({ref_res['info']['name']}):")
    channels_to_fit = [480, 515, 445, 555]
    channel_fits = {}
    for ch in channels_to_fit:
        seg = absorbance.loc[ref_peak:ref_end, ch].dropna()
        t_el = (seg.index - ref_peak).total_seconds().values
        c_fit = fit_washout(t_el, seg.values)
        cs = c_fit.summary()
        channel_fits[ch] = c_fit
        print(f"   Band {ch} nm: tau = {cs['tau_s']:.1f} +/- {cs['tau_standard_error_iid_approx_s']:.1f} s | Amp = {cs['amplitude']:.4f} AU | R^2 = {cs['r_squared_descriptive']:.3f} | RMSE = {cs['rmse']:.5f}")

    # Hydraulic comparison
    hyd = hydraulic_comparison(volume_ml=10.0, flow_ml_min=flow_media, fitted_tau_s=ref_res["fit_summary"]["tau_s"])
    print(f"\n4. HYDRAULIC COMPARISON & RECONCILIATION:")
    print(f"   Assumed vial volume:          {hyd['assumed_or_measured_volume_ml']:.1f} mL")
    print(f"   Independently measured flow:  {hyd['independently_supplied_flow_ml_min']:.2f} mL/min")
    print(f"   Ideal hydraulic tau (10 mL):  {hyd['ideal_hydraulic_tau_s']:.1f} s ({hyd['ideal_hydraulic_tau_s']/60:.2f} min)")
    print(f"   Fitted optical tau (515 nm):  {hyd['fitted_optical_tau_s']:.1f} s ({hyd['fitted_optical_tau_s']/60:.2f} min)")
    print(f"   Optical-to-ideal tau ratio:   {hyd['optical_to_ideal_tau_ratio']:.3f}x")
    print(f"   Hydraulic-equivalent volume:  {hyd['hydraulic_equivalent_volume_not_measured_ml']:.2f} mL (unmeasured)")
    print(f"   Flow required for 10 mL:      {hyd['flow_required_to_reconcile_assumed_volume_ml_min']:.2f} mL/min")

    def save_and_copy(fig, filename):
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        art_path = os.path.join(art_dir, filename)
        shutil.copy(path, art_path)
        print(f" -> Saved: {filename}")

    # =========================================================================
    # PLOT 1: Full Multi-Band Spectral Trace (Raw Transmission)
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    for b in AS7341_BANDS:
        ax1.plot(wide.index, wide[b], label=f"{b} nm", color=BAND_COLORS[b], lw=1.8)

    y_ann = wide[630].max() * 0.99
    for i, res in enumerate(pulse_results):
        t_ev = res["info"]["t_dose"]
        ax1.axvline(t_ev, color="darkred", ls="--", lw=1.6, alpha=0.85)
        y_offset = -12 if (i % 2 == 0) else -28
        ax1.annotate(f"{res['info']['name']}\n{t_ev.strftime('%H:%M:%S')}",
                     xy=(t_ev, y_ann), xytext=(4, y_offset),
                     textcoords="offset points", color="darkred",
                     fontweight="bold", fontsize=8.0)

    ax1.set_ylabel("Raw AS7341 Transmission Reading", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Local Time (IST, HH:MM)", fontsize=11, fontweight="bold")
    ax1.set_title(f"1. Full Spectral Trace: Raw Transmission ({data_label})",
                  fontsize=13, fontweight="bold", pad=12)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.legend(loc="lower left", ncol=8, fontsize=9.0, framealpha=0.95)
    ax1.grid(True, alpha=0.3, ls=":")
    save_and_copy(fig1, "plot1_raw_spectral_transmission_trace.png")

    # =========================================================================
    # PLOT 2: Raw Absorbance Dynamics Across Wavelengths Over Time
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(14.5, 6))
    plot_start = pulse_results[0]["info"]["t_dose"] - pd.Timedelta(minutes=3)
    sub_abs = absorbance.loc[plot_start:]
    for b in [445, 480, 515, 555, 630]:
        ax2.plot(sub_abs.index, sub_abs[b], label=f"{b} nm", color=BAND_COLORS[b], lw=2.0)

    max_A = max(r["A_peak"] for r in pulse_results)
    for i, res in enumerate(pulse_results):
        t_ev = res["info"]["t_dose"]
        t_pk = res["t_peak"]
        ax2.axvline(t_ev, color="darkred", ls="--", lw=1.3, alpha=0.7)
        ax2.plot(t_pk, res["A_peak"], "o", color=BAND_COLORS[515], ms=7, mec="black", mew=1.2)
        y_txt_offset = 12 if (i % 2 == 0) else 26
        ax2.annotate(
            f"{res['info']['name']}\n{t_ev.strftime('%H:%M:%S')}\n(Peak: +{res['peak_delay_s']:.0f}s)",
            xy=(t_pk, res["A_peak"]),
            xytext=(6, y_txt_offset),
            textcoords="offset points",
            color="darkred",
            fontweight="bold",
            fontsize=8.0,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="darkred", alpha=0.90, lw=0.8),
        )

    ax2.axhline(0, color="black", ls=":", lw=1.2, alpha=0.7)
    ax2.set_ylabel(r"Optical Absorbance $-\log_{10}(T / T_{\mathrm{blank}})$ [AU]", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Local Time (IST, HH:MM)", fontsize=11, fontweight="bold")
    ax2.set_title(f"2. Raw Optical Absorbance Dynamics with Algorithmic Peak Markers ({data_label})",
                  fontsize=13, fontweight="bold", pad=12)
    ax2.set_ylim(-0.015, max_A * 1.35)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.legend(loc="upper left", ncol=5, fontsize=9.5, framealpha=0.95)
    ax2.grid(True, alpha=0.3, ls=":")
    save_and_copy(fig2, "plot2_raw_transmission_drop_and_absorbance.png")

    # =========================================================================
    # PLOT 3: Synchronized Single-Acquisition Frame Peak Absorption Profile
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(11.5, 6))
    cmap_spec = plt.cm.plasma(np.linspace(0.12, 0.88, len(pulse_results)))

    # Pre-dose reference spectrum
    pre_spec = absorbance.loc[dose_info[0]["pre_window"][0]:dose_info[0]["pre_window"][1]].mean()
    ax3.plot(AS7341_BANDS, [pre_spec[b] for b in AS7341_BANDS], "k--", lw=1.8,
             label="Pre-Dose Baseline (Clear Water, A ~ 0)")

    for i, res in enumerate(pulse_results):
        snapshot = res["peak_spectrum"]
        vals = [snapshot[b] for b in AS7341_BANDS]
        col = cmap_spec[i]
        ax3.plot(AS7341_BANDS, vals, "o-",
                 label=f"{res['info']['name']} (Frame @ {res['t_peak'].strftime('%H:%M:%S')})",
                 color=col, lw=2.2, ms=6.5)

    # Post-washout spectrum for Dose 2
    if len(pulse_results) > 1:
        post2_spec = absorbance.loc[dose_info[1]["post_window"][0]:dose_info[1]["post_window"][1]].mean()
        ax3.plot(AS7341_BANDS, [post2_spec[b] for b in AS7341_BANDS], "s:", color="#3182bd", lw=2.0, ms=6,
                 label=f"Post-Dose 2 Washout")

    ax3.axvspan(470, 530, color="green", alpha=0.10, label="Cyan-Green Peak Absorption Band (480-515 nm)")
    ax3.set_xlabel("Wavelength (nm)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Optical Absorbance [AU]", fontsize=11, fontweight="bold")
    ax3.set_title(f"3. Synchronized Single-Frame Spectral Profiles ({data_label})",
                  fontsize=13, fontweight="bold", pad=12)
    ax3.set_xticks(AS7341_BANDS)
    ax3.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    ax3.grid(True, alpha=0.3, ls=":")
    save_and_copy(fig3, "plot3_raw_spectral_absorbance_profile.png")

    # =========================================================================
    # PLOT 4: Multi-Pulse & Multi-Channel Washout Kinetics (cstr_washout_fit_tools)
    # =========================================================================
    fig4, (ax4_pulse, ax4_chan) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Multi-Pulse Comparison at 515 nm
    pulse_cmap = plt.cm.tab10(np.linspace(0, 1, max(10, len(pulse_results))))
    max_span = max(float(res["fit"].time_s[-1]) for res in pulse_results)
    for i, res in enumerate(pulse_results):
        fit_obj = res["fit"]
        norm_obs = fit_obj.normalised_observations()
        s_obj = res["fit_summary"]
        col = pulse_cmap[i]
        ax4_pulse.plot(fit_obj.time_s, norm_obs, "o", color=col, alpha=0.45, ms=3.5,
                       label=f"{res['info']['name']} obs")
        t_line = np.linspace(0, fit_obj.time_s[-1], 200)
        ax4_pulse.plot(t_line, np.exp(-t_line / fit_obj.tau_s), color=col, lw=2.0,
                       label=f"Fit {i+1}: $\\tau={fit_obj.tau_s:.0f}s$ ($R^2={s_obj['r_squared_descriptive']:.2f}$)")

    t_eval = np.linspace(0, max_span * 1.05, 250)
    ax4_pulse.plot(t_eval, np.exp(-t_eval / hyd["ideal_hydraulic_tau_s"]), "k:", lw=2.2,
                   label=f"Ideal Hydraulic (10mL): $\\tau={hyd['ideal_hydraulic_tau_s']:.0f}s$")
    ax4_pulse.axhline(0.05, color="gray", ls="--", lw=1.2, label=r"$5\%$ Excess Threshold ($t_{95\%}$)")
    ax4_pulse.axhline(0.00, color="black", ls="-", lw=0.8, alpha=0.4)
    ax4_pulse.set_xlabel("Elapsed Time from Detected Peak (s)", fontsize=10.5, fontweight="bold")
    ax4_pulse.set_ylabel(r"Normalised Signal $\frac{y(t) - \mathrm{baseline}(t)}{\mathrm{amplitude}}$",
                         fontsize=10.5, fontweight="bold")
    ax4_pulse.set_title("A. Multi-Pulse Washout Kinetics (515 nm)", fontsize=12, fontweight="bold")
    ax4_pulse.set_xlim(-5, max_span * 1.05)
    ax4_pulse.legend(loc="upper right", fontsize=7.8 if len(pulse_results) > 4 else 8.5, framealpha=0.92)
    ax4_pulse.grid(True, alpha=0.3, ls=":")

    # Panel B: Multi-Channel Comparison for Reference Dose
    chan_colors = {480: "#00a2e8", 515: "#008a00", 445: "#0055d4", 555: "#8cb800"}
    for ch in channels_to_fit:
        c_fit = channel_fits[ch]
        c_s = c_fit.summary()
        c_norm = c_fit.normalised_observations()
        ax4_chan.plot(c_fit.time_s, c_norm, "o", color=chan_colors[ch], alpha=0.45, ms=3.5,
                      label=f"{ch} nm obs")
        t_line = np.linspace(0, c_fit.time_s[-1], 200)
        ax4_chan.plot(t_line, np.exp(-t_line / c_fit.tau_s), color=chan_colors[ch], lw=2.0,
                      label=f"{ch} nm fit: $\\tau={c_fit.tau_s:.0f} \\pm {c_s['tau_standard_error_iid_approx_s']:.1f}$s")

    ax4_chan.plot(t_eval, np.exp(-t_eval / hyd["ideal_hydraulic_tau_s"]), "k:", lw=2.2,
                  label=f"Ideal Hydraulic: $\\tau={hyd['ideal_hydraulic_tau_s']:.0f}s$")
    ax4_chan.axhline(0.05, color="gray", ls="--", lw=1.2)
    ax4_chan.axhline(0.00, color="black", ls="-", lw=0.8, alpha=0.4)
    ax4_chan.set_xlabel("Elapsed Time from Detected Peak (s)", fontsize=10.5, fontweight="bold")
    ax4_chan.set_ylabel(r"Normalised Signal $\frac{y(t) - \mathrm{baseline}(t)}{\mathrm{amplitude}}$",
                         fontsize=10.5, fontweight="bold")
    ax4_chan.set_title(f"B. Multi-Channel Washout ({ref_res['info']['name']})", fontsize=12, fontweight="bold")
    ax4_chan.set_xlim(-5, max_span * 1.05)
    ax4_chan.legend(loc="upper right", fontsize=8.2, framealpha=0.92)
    ax4_chan.grid(True, alpha=0.3, ls=":")

    fig4.suptitle(f"4. Audited Normalised Washout Kinetics vs CSTR Model ({data_label})",
                  fontsize=13, fontweight="bold", y=0.98)
    save_and_copy(fig4, "plot4_raw_washout_kinetics_cstr_model.png")

    # =========================================================================
    # PLOT 5: Raw Baseline Stability & Algorithmic Statistical Recovery
    # =========================================================================
    fig5, (ax5_bar, ax5_tab) = plt.subplots(2, 1, figsize=(14, 8.5), height_ratios=[1.0, 1.0])

    stages = ["Initial\nBlank"]
    vals = [blank_ref[515]]
    errs = [blank_std[515]]
    colors = ["#2b5c8f"]

    for i, res in enumerate(pulse_results):
        stages.append(f"Pre-Dose {i+1}\n(Baseline)")
        vals.append(res["mu_pre_trans"])
        errs.append(res["std_pre_trans"])
        colors.append("#4575b4")

        rec_label = "Recovered" if "Recovered" in res["status_str"] else "Residual"
        stages.append(f"Post-Dose {i+1}\n({rec_label})")
        vals.append(res["mu_post_trans"])
        errs.append(res["std_post_trans"])
        colors.append("#74add1" if "Recovered" in res["status_str"] else "#f46d43")

    bars = ax5_bar.bar(range(len(stages)), vals, yerr=errs, capsize=3.0, color=colors,
                       edgecolor="black", width=0.55, alpha=0.88)
    ax5_bar.axhline(blank_ref[515], color="darkblue", ls="--", lw=1.5,
                    label=f"Initial Blank Reference ({blank_ref[515]:.5f})")

    min_val = min(vals) - 0.005
    max_val = max(vals) + 0.004
    ax5_bar.set_ylim(min_val, max_val)
    ax5_bar.set_xticks(range(len(stages)))
    ax5_bar.set_xticklabels(stages, rotation=35, ha="right", fontsize=8.0, fontweight="bold")
    ax5_bar.set_ylabel("Raw Transmission Reading (515 nm)", fontsize=10.5, fontweight="bold")
    ax5_bar.set_title(f"5. Transmission Baseline Stability & Statistical Recovery Across Cycles ({data_label})",
                      fontsize=13, fontweight="bold", pad=12)
    ax5_bar.legend(loc="lower left", fontsize=9.0)
    ax5_bar.grid(True, alpha=0.3, ls=":", axis="y")

    for bar, val, err in zip(bars, vals, errs):
        h = bar.get_height()
        ax5_bar.text(bar.get_x() + bar.get_width() / 2.0, h + err + 0.0003, f"{val:.4f}",
                     ha="center", va="bottom", fontsize=7.2 if len(pulse_results) > 4 else 8.0, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.85))

    # Table subplot generated dynamically via statistical hypothesis test
    ax5_tab.axis("off")
    table_data = [
        ["Event / Cycle", "Vol", "ΔA_peak", "Residual ΔA", "Recovery", "Statistical Welch Test", "Algorithmic Status"],
        ["Initial Water Blank", "0.0 mL", "0.0000 AU", "0.0000 AU", "100.0%", "Reference baseline", "Clean Water Reference"],
    ]
    for res in pulse_results:
        t_s = res["welch_t"]
        p_v = res["p_val"]
        z_s = res["z_score"]
        stat_test_str = f"t = {t_s:+.2f}, p = {p_v:.2f} (Z = {z_s:.1f}σ)"
        table_data.append([
            res["info"]["name"],
            f"{res['info']['volume_mL']:.1f} mL",
            f"{res['delta_peak']:.4f} AU",
            f"{res['delta_post']:+.4f} AU",
            f"{res['recovery_pct']:.1f}%",
            stat_test_str,
            res["status_str"],
        ])

    tab = ax5_tab.table(cellText=table_data, loc="center", cellLoc="center",
                        colWidths=[0.16, 0.08, 0.12, 0.13, 0.10, 0.19, 0.22])
    tab.auto_set_font_size(False)
    tab.set_fontsize(7.8 if len(pulse_results) > 4 else 8.5)
    tab.scale(1.0, 1.35 if len(pulse_results) > 4 else 1.6)
    for (r, c), cell in tab.get_celld().items():
        if r == 0:
            cell.set_facecolor("#204a87")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 1:
            cell.set_facecolor("#f0f4f8")

    save_and_copy(fig5, "plot5_raw_baseline_recovery_summary.png")

    # =========================================================================
    # PLOT 6: Reservoir Memory & Short-Term Memory Capacity Analysis
    # =========================================================================
    if len(pulse_results) >= 4:
        from sklearn.linear_model import Ridge, LinearRegression

        fig6, (ax6_a, ax6_b) = plt.subplots(1, 2, figsize=(14.5, 6))

        X_peaks = []
        u_curr = []
        u_prev = []
        peaks_515 = []
        pre_515 = []
        cycle_names = []

        for i, res in enumerate(pulse_results):
            X_peaks.append(res["peak_spectrum"].values)
            u_curr.append(res["info"]["volume_mL"])
            u_prev.append(pulse_results[i-1]["info"]["volume_mL"] if i > 0 else 0.0)
            peaks_515.append(res["A_peak"])
            pre_515.append(res["mu_pre"])
            cycle_names.append(f"Dose {i+1}")

        X_peaks = np.array(X_peaks)
        u_curr = np.array(u_curr)
        u_prev = np.array(u_prev)
        peaks_515 = np.array(peaks_515)
        pre_515 = np.array(pre_515)

        # Panel A: Physical Demonstration of Memory Superposition
        scatter = ax6_a.scatter(u_curr, peaks_515, c=u_prev, cmap="plasma", s=130, edgecolor="black", lw=1.5, zorder=5)
        cbar = fig6.colorbar(scatter, ax=ax6_a)
        cbar.set_label("Previous Dose Volume $u(t - \\Delta t)$ [mL]", fontsize=10, fontweight="bold")

        for i in range(len(u_curr)):
            x_off = -52 if u_curr[i] >= 1.15 else 7
            y_off = -8 if (i == 7) else 6
            ax6_a.annotate(f"Dose {i+1}\n({u_curr[i]:.1f}mL)", (u_curr[i], peaks_515[i]),
                           xytext=(x_off, y_off), textcoords="offset points", fontsize=8.0, fontweight="bold")

        ax6_a.set_xlim(0.2, 1.35)
        ax6_a.set_ylim(0.015, 0.053)

        lin_mem = LinearRegression().fit(np.column_stack([u_curr, u_prev]), peaks_515)
        alpha_val = lin_mem.coef_[0]
        beta_val = lin_mem.coef_[1]
        r2_superpos = lin_mem.score(np.column_stack([u_curr, u_prev]), peaks_515)

        ax6_a.set_xlabel("Current Injected Dose Volume $u(t)$ [mL]", fontsize=10.5, fontweight="bold")
        ax6_a.set_ylabel("Peak Optical Absorbance $A_{\\mathrm{peak}}$ (515 nm) [AU]", fontsize=10.5, fontweight="bold")
        ax6_a.set_title(f"A. Fading Memory Superposition ($R^2 = {r2_superpos:.3f}$, Retention $\\beta/\\alpha = {beta_val/alpha_val*100:.1f}\\%$)",
                        fontsize=11.5, fontweight="bold")
        ax6_a.grid(True, alpha=0.3, ls=":")

        # Panel B: Reservoir Linear Readout
        model_u0 = Ridge(alpha=1e-3).fit(X_peaks, u_curr)
        pred_u0 = model_u0.predict(X_peaks)
        r2_u0 = np.corrcoef(u_curr, pred_u0)[0, 1]**2

        model_u1 = Ridge(alpha=1e-3).fit(X_peaks[1:], u_prev[1:])
        pred_u1 = model_u1.predict(X_peaks[1:])
        r2_u1 = np.corrcoef(u_prev[1:], pred_u1)[0, 1]**2
        total_mc = r2_u0 + r2_u1

        x_indices = np.arange(len(u_curr))
        ax6_b.plot(x_indices, u_curr, "ko-", lw=2.0, ms=6, label="Ground Truth $u(t)$")
        ax6_b.plot(x_indices, pred_u0, "s--", color="#0055d4", lw=1.8, ms=6,
                   label=f"Readout $\\hat{{u}}(t)$ [$k=0$, $R^2={r2_u0:.3f}$]")

        ax6_b.plot(x_indices[1:], u_prev[1:], "d-.", color="gray", lw=1.5, ms=5, alpha=0.7, label="Ground Truth $u(t - \\Delta t)$")
        ax6_b.plot(x_indices[1:], pred_u1, "^-", color="#e74c3c", lw=1.8, ms=6,
                   label=f"Readout $\\hat{{u}}(t - \\Delta t)$ [$k=1$ Memory, $R^2={r2_u1:.3f}$]")

        ax6_b.set_xticks(x_indices)
        ax6_b.set_xticklabels(cycle_names, rotation=35, ha="right", fontsize=8.5, fontweight="bold")
        ax6_b.set_xlabel("Experiment Cycle / Injection", fontsize=10.5, fontweight="bold")
        ax6_b.set_ylabel("Dose Volume [mL]", fontsize=10.5, fontweight="bold")
        ax6_b.set_title(f"B. Reservoir Readout & Memory Capacity ($MC = {total_mc:.3f}$)", fontsize=11.5, fontweight="bold")
        ax6_b.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
        ax6_b.grid(True, alpha=0.3, ls=":")

        fig6.suptitle(f"6. Reservoir Fading Memory & Short-Term Memory Capacity ({data_label})",
                      fontsize=13, fontweight="bold", y=0.98)
        save_and_copy(fig6, "plot6_reservoir_memory_capacity_analysis.png")

        # Individual Figure for Panel B: Reservoir Linear Readout & Memory Capacity (clean without callout box)
        fig6b, ax_ind = plt.subplots(figsize=(10, 6.2))
        ax_ind.plot(x_indices, u_curr, "ko-", lw=2.2, ms=7, label="Ground Truth Input $u(t)$")
        ax_ind.plot(x_indices, pred_u0, "s--", color="#0055d4", lw=2.0, ms=7,
                   label=f"Readout $\\hat{{u}}(t)$ [$k=0$, $R^2={r2_u0:.3f}$]")
        ax_ind.plot(x_indices[1:], u_prev[1:], "d-.", color="gray", lw=1.6, ms=6, alpha=0.75,
                   label="Ground Truth Delayed $u(t - \\Delta t)$")
        ax_ind.plot(x_indices[1:], pred_u1, "^-", color="#e74c3c", lw=2.0, ms=7,
                   label=f"Readout $\\hat{{u}}(t - \\Delta t)$ [$k=1$ Memory, $R^2={r2_u1:.3f}$]")

        ax_ind.set_xticks(x_indices)
        ax_ind.set_xticklabels(cycle_names, rotation=35, ha="right", fontsize=9.5, fontweight="bold")
        ax_ind.set_xlabel("Experiment Cycle / Injection", fontsize=11, fontweight="bold")
        ax_ind.set_ylabel("Dose Volume [mL]", fontsize=11, fontweight="bold")
        ax_ind.set_title(f"Reservoir Linear Readout & Short-Term Memory Capacity ($MC = {total_mc:.3f}$, {data_label})",
                        fontsize=13, fontweight="bold", pad=12)
        ax_ind.set_ylim(0.2, 1.35)
        ax_ind.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
        ax_ind.grid(True, alpha=0.3, ls=":")

        save_and_copy(fig6b, "plot6b_reservoir_linear_readout_memory.png")

    print(f"\nAll audited individual plots for {data_label} generated successfully.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audited CSTR Washout Analysis")
    parser.add_argument("data_dir", nargs="?", default="data/16th sept", help="Path to experiment data folder")
    parser.add_argument("--output-dir", "-o", default=None, help="Directory to save output plots and reports")
    args = parser.parse_args()

    data_dir = args.data_dir
    spec_files = glob.glob(os.path.join(data_dir, "as7341_spectrum_readings*.csv"))
    dose_files = glob.glob(os.path.join(data_dir, "dosing_events*.csv"))

    if not spec_files or not dose_files:
        print(f"Error: Missing CSV files in {data_dir}")
        sys.exit(1)

    spec_file = spec_files[0]
    dose_file = dose_files[0]
    label = os.path.basename(os.path.normpath(data_dir))

    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join("outputs", f"{label}_memory_experiment" if "16th" in label else f"{label}_output")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading data from: {data_dir} ({label})")
    print(f"Saving outputs to: {out_dir}")
    wide, wide_clean, alt_events_df, media, waste = load_and_preprocess_raw_data(spec_file, dose_file)
    analyze_raw_response(wide, wide_clean, alt_events_df, media, waste, data_label=label, output_dir=out_dir)

