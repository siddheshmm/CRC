"""Auditable washout fitting helpers for cstr_washout_analysis.py.

This is a replacement for the fitting/normalisation block, NOT a complete
reanalysis of the user's experiment. No experimental CSVs were available when
this module was tested. Running this file executes explicitly synthetic tests.

Model: y(t) = amplitude * exp(-t/tau) + offset [+ slope * t].
The offset/drift terms describe the measured optical baseline. They are not
proof of a dye-free state. A fitted optical tau is a hydraulic residence time
only if mixing, flow, volume and concentration-to-signal assumptions hold.

Dependencies: numpy, scipy. Optional plotting in the integration example needs
matplotlib. The helpers do not smooth, interpolate, delete outliers, or modify
input observations. Statistical standard errors assume the fitted error model;
temporally correlated residuals require block-bootstrap or replicate analyses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import math
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import curve_fit


@dataclass
class WashoutFit:
    amplitude: float
    tau_s: float
    offset: float
    slope_per_s: float
    covariance: NDArray[np.float64]
    time_s: NDArray[np.float64]
    observed: NDArray[np.float64]
    prediction: NDArray[np.float64]
    residuals: NDArray[np.float64]
    model: str
    warnings: tuple[str, ...]

    def baseline(self, elapsed_s: ArrayLike) -> NDArray[np.float64]:
        """Fitted optical background, not independently measured clear water."""
        t = np.asarray(elapsed_s, dtype=float)
        return self.offset + self.slope_per_s * t

    def predict(self, elapsed_s: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(elapsed_s, dtype=float)
        return self.amplitude * np.exp(-t / self.tau_s) + self.baseline(t)

    def normalised_observations(self) -> NDArray[np.float64]:
        """Observed signal minus fitted background, divided by fitted amplitude.

        Compare directly with exp(-t/tau). The first noisy observation need not
        equal 1. This deliberately avoids using a noisy single-point divisor.
        Values are not clipped: negative values expose noise/model mismatch.
        """
        return (self.observed - self.baseline(self.time_s)) / self.amplitude

    def summary(self) -> dict[str, Any]:
        ssr = float(np.sum(self.residuals**2))
        sst = float(np.sum((self.observed - self.observed.mean())**2))
        stderr = np.sqrt(np.maximum(np.diag(self.covariance), 0.0))
        r = self.residuals
        if len(r) > 3 and np.std(r[:-1]) > 0 and np.std(r[1:]) > 0:
            lag1 = float(np.corrcoef(r[:-1], r[1:])[0, 1])
        else:
            lag1 = None
        return {
            'model': self.model,
            'amplitude': self.amplitude,
            'tau_s': self.tau_s,
            'tau_standard_error_iid_approx_s': float(stderr[1]),
            'offset': self.offset,
            'slope_per_s': self.slope_per_s,
            'half_life_s': self.tau_s * math.log(2),
            't95_of_fitted_excess_s': self.tau_s * math.log(20),
            't99_of_fitted_excess_s': self.tau_s * math.log(100),
            'recorded_fit_span_s': float(self.time_s[-1]),
            'n_observations': int(len(self.observed)),
            'rmse': float(np.sqrt(ssr / len(self.observed))),
            'r_squared_descriptive': 1.0 - ssr / sst if sst > 0 else None,
            'residual_lag1_correlation': lag1,
            'warnings': list(self.warnings),
        }


def _constant(t, amplitude, tau, offset):
    return amplitude * np.exp(-t / tau) + offset


def _linear(t, amplitude, tau, offset, slope):
    return amplitude * np.exp(-t / tau) + offset + slope * t


def fit_washout(
    elapsed_s: ArrayLike,
    signal: ArrayLike,
    *,
    linear_drift: bool = False,
    sigma: ArrayLike | None = None,
    tau_bounds_s: tuple[float, float] = (0.1, 36000.0),
) -> WashoutFit:
    """Fit a post-injection, post-mixing decay window with positive amplitude/tau.

    elapsed_s must start at zero and increase strictly. The caller must choose
    the window using actual dose completion and mixing information; this helper
    does not label an arbitrary first point as a measured peak.

    signal may be apparent absorbance or calibrated concentration. For physical
    concentration interpretation, independently verify the observation model.
    sigma, when provided, contains positive per-observation standard deviations
    and is treated as absolute; otherwise covariance uses residual variance.
    A linear baseline is a sensitivity model, not permission to remove real tails.
    """
    t = np.asarray(elapsed_s, dtype=np.float64)
    y = np.asarray(signal, dtype=np.float64)
    npar = 4 if linear_drift else 3
    if t.ndim != 1 or y.ndim != 1 or t.shape != y.shape:
        raise ValueError('Time and signal must be equal-length 1D arrays.')
    if len(t) < max(12, npar + 3):
        raise ValueError('At least 12 observations are required by this helper.')
    if not np.isfinite(t).all() or not np.isfinite(y).all():
        raise ValueError('Non-finite observations: inspect/flag them explicitly before fitting.')
    if not np.isclose(t[0], 0.0, atol=1e-9) or not np.all(np.diff(t) > 0):
        raise ValueError('Elapsed time must start at zero and increase strictly.')
    lo, hi = map(float, tau_bounds_s)
    if not (0 < lo < hi and np.isfinite(hi)):
        raise ValueError('tau_bounds_s must contain finite positive increasing bounds.')
    s = None if sigma is None else np.asarray(sigma, dtype=np.float64)
    if s is not None and (s.shape != y.shape or not np.isfinite(s).all() or np.any(s <= 0)):
        raise ValueError('sigma must match signal and contain finite positive values.')
    spread = float(np.ptp(y))
    if spread <= np.finfo(float).eps * max(1.0, float(np.max(np.abs(y)))):
        raise ValueError('Signal is effectively constant; decay time is not identifiable.')

    tail_n = max(3, len(y) // 5)
    offset0 = float(np.median(y[-tail_n:]))
    slope0 = 0.0
    if linear_drift:
        slope0, offset0 = np.polyfit(t[-tail_n:], y[-tail_n:], 1)
    amplitude0 = max(float(np.median(y[:3]) - offset0), spread / 2.0, 1e-12)
    func = _linear if linear_drift else _constant
    lower = [0.0, lo, -np.inf] + ([-np.inf] if linear_drift else [])
    upper = [np.inf, hi, np.inf] + ([np.inf] if linear_drift else [])
    best = None
    # Multiple starting timescales reduce dependence on a single initial guess.
    for tau0 in (t[-1] / 8, t[-1] / 3, t[-1]):
        tau0 = float(np.clip(tau0, lo * 1.01, hi * 0.99))
        p0 = [amplitude0, tau0, offset0] + ([slope0] if linear_drift else [])
        try:
            params, cov = curve_fit(
                func, t, y, p0=p0, bounds=(lower, upper),
                sigma=s, absolute_sigma=s is not None,
                method='trf', x_scale='jac', max_nfev=20000,
                ftol=1e-12, xtol=1e-12, gtol=1e-12,
            )
            residuals = y - func(t, *params)
            loss = float(np.sum((residuals if s is None else residuals / s)**2))
            if best is None or loss < best[0]:
                best = (loss, params, cov, residuals)
        except (RuntimeError, ValueError, FloatingPointError):
            continue
    if best is None:
        raise RuntimeError('No fit converged. Review data, windows and model assumptions.')
    _, params, covariance, residuals = best
    amplitude, tau, offset = map(float, params[:3])
    if amplitude <= np.finfo(float).eps * max(1.0, spread):
        raise ValueError('Fitted positive decay amplitude is negligible; do not interpret tau.')
    slope = float(params[3]) if linear_drift else 0.0
    flags = []
    if tau < 1.02 * lo or tau > 0.98 * hi:
        flags.append('Tau is near a fit bound; it may not be identifiable.')
    if t[-1] < tau * math.log(20):
        flags.append('Modelled 95% relaxation lies beyond the recorded fit window.')
    if not np.isfinite(covariance).all():
        flags.append('Non-finite parameter covariance; uncertainty is not reliable.')
    elif np.sqrt(max(0.0, covariance[1, 1])) > 0.5 * tau:
        flags.append('Large approximate tau uncertainty; the fit is weakly constrained.')
    flags.append('Offset/drift are fitted backgrounds, not independently confirmed dye-free baselines.')
    return WashoutFit(amplitude, tau, offset, slope, covariance, t.copy(), y.copy(),
                      func(t, *params), residuals, 'exponential + linear baseline' if linear_drift
                      else 'exponential + constant baseline', tuple(flags))


def hydraulic_comparison(volume_ml: float, flow_ml_min: float, fitted_tau_s: float) -> dict[str, float]:
    """Compare *independently supplied* volume/flow with an optical decay fit.

    No volume is inferred from cumulative commanded waste removal, and the
    hydraulic-equivalent volume is explicitly NOT a measurement of vial volume.
    """
    if not all(math.isfinite(x) and x > 0 for x in (volume_ml, flow_ml_min, fitted_tau_s)):
        raise ValueError('Volume, flow and fitted tau must be finite and positive.')
    tau_h = 60 * volume_ml / flow_ml_min
    return {
        'assumed_or_measured_volume_ml': volume_ml,
        'independently_supplied_flow_ml_min': flow_ml_min,
        'ideal_hydraulic_tau_s': tau_h,
        'ideal_hydraulic_t95_s': tau_h * math.log(20),
        'fitted_optical_tau_s': fitted_tau_s,
        'optical_to_ideal_tau_ratio': fitted_tau_s / tau_h,
        'hydraulic_equivalent_volume_not_measured_ml': flow_ml_min * fitted_tau_s / 60,
        'flow_required_to_reconcile_assumed_volume_ml_min': 60 * volume_ml / fitted_tau_s,
    }


def self_test() -> dict[str, Any]:
    """Synthetic numerical tests only. These are NOT experimental results."""
    t = np.arange(0.0, 601.0, 5.0)
    y = 0.04 * np.exp(-t / 108.0) - 0.004
    constant = fit_washout(t, y)
    assert abs(constant.tau_s - 108.0) < 1e-4
    assert np.max(np.abs(constant.normalised_observations() - np.exp(-t / 108.0))) < 1e-7
    shifted = fit_washout(t, y + 0.013)
    assert abs(shifted.tau_s - constant.tau_s) < 1e-4
    with_drift = fit_washout(t, y - 0.000003 * t, linear_drift=True)
    assert abs(with_drift.tau_s - 108.0) < 1e-3
    assert abs(with_drift.slope_per_s + 0.000003) < 1e-9
    rng = np.random.default_rng(42)
    noisy = fit_washout(t, y + rng.normal(0.0, 0.00015, len(t)))
    assert abs(noisy.tau_s - 108.0) < 5.0
    for bad_t, bad_y in [(t[::-1], y), (t, np.ones(len(t))), (t, np.where(t == 5, np.nan, y))]:
        try:
            fit_washout(bad_t, bad_y)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid input was not rejected.')
    hydraulic = hydraulic_comparison(10.0, 9.30, 107.6)
    assert abs(hydraulic['ideal_hydraulic_tau_s'] - 64.51612903225806) < 1e-10
    return {
        'status': 'PASS',
        'data_type': 'SYNTHETIC; no user experimental CSVs analysed',
        'checks': ['known exponential plus non-zero offset', 'correct normalisation',
                   'invariance to constant reference shift', 'linear baseline sensitivity model',
                   'seeded noisy signal', 'invalid input rejection', 'hydraulic arithmetic'],
        'synthetic_noisy_tau_s': noisy.tau_s,
        'reported_number_arithmetic_only': hydraulic,
    }


if __name__ == '__main__':
    print(json.dumps(self_test(), indent=2))
