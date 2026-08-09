"""Split conformal prediction and the coverage/sharpness evaluation.

Two conformal variants, chosen by what the arm can emit:

* **CQR** (Romano, Patterson & Candès 2019) for arms with quantiles. Score on the
  calibration set is `max(q_lo(x) - y, y - q_hi(x))`, and the interval becomes
  `[q_lo - Q, q_hi + Q]`. This preserves whatever input-dependent adaptivity the
  quantile model had, so a heteroscedastic arm keeps narrow intervals where it was
  already confident.
* **Absolute-residual conformal** for point-only arms. Score is `|y - yhat|`, and the
  interval is `yhat ± Q`. Necessarily a *constant* width for every input, which is
  exactly why we also report regime-conditioned coverage: a constant-width interval
  cannot adapt to the near-failure regime even when it is marginally valid.

Both use the finite-sample corrected quantile `ceil((n+1)(1-alpha)) / n`, which is what
makes the marginal-coverage guarantee hold at finite n rather than only asymptotically.

Validity requires the calibration and evaluation scores to be exchangeable. That is why
the calibration split is constructed identically to the eval split (D-004) and why no
base model ever sees calibration data during fitting (D-007).
"""

from __future__ import annotations

import numpy as np

from . import config as C, metrics


def conformal_quantile(scores: np.ndarray, nominal: float) -> float:
    """Finite-sample conformal quantile of calibration scores."""
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    k = int(np.ceil((n + 1) * nominal))
    if k > n:
        # Not enough calibration points to certify this level; the guarantee would
        # require a score beyond the largest observed one.
        return float("inf")
    return float(s[k - 1])


def conformal_feasible_levels(n_calib: int) -> dict[float, bool]:
    return {
        lv: int(np.ceil((n_calib + 1) * lv)) <= n_calib for lv in C.NOMINAL_LEVELS
    }


def apply_conformal(
    calib_pred, eval_pred, nominal: float, clamp_lo: float = 0.0
) -> dict:
    """Calibrate on `calib_pred`, return conformal bounds for `eval_pred`.

    Lower bounds are clamped at 0 because RUL cannot be negative; that cannot reduce
    coverage (no truth is ever below it) and is applied identically to every arm. The
    upper bound is deliberately NOT clamped at the Variant B cap, even though truth is
    <= 125 there, so that the point-mass discussion stays interpretable.
    """
    if calib_pred.has_intervals and eval_pred.has_intervals:
        levels = np.asarray(calib_pred.levels)
        c_lo, c_hi = metrics.interval_bounds(levels, calib_pred.quantiles, nominal)
        e_lo, e_hi = metrics.interval_bounds(levels, eval_pred.quantiles, nominal)
        scores = np.maximum(c_lo - calib_pred.y_true, calib_pred.y_true - c_hi)
        q = conformal_quantile(scores, nominal)
        lo, hi, method = e_lo - q, e_hi + q, "CQR"
    else:
        scores = np.abs(calib_pred.y_true - calib_pred.pred)
        q = conformal_quantile(scores, nominal)
        lo, hi, method = eval_pred.pred - q, eval_pred.pred + q, "abs-residual"

    lo = np.maximum(lo, clamp_lo)
    hi = np.maximum(hi, lo)  # keep intervals non-degenerate after clamping
    return {
        "lo": lo, "hi": hi, "q": q, "method": method,
        "n_calib": int(len(calib_pred.y_true)),
        "feasible": np.isfinite(q),
    }


def evaluate_cell(pred, lo, hi, nominal: float, with_ci: bool = True) -> dict:
    """Coverage + sharpness for one (arm, variant, level), overall and per RUL bin."""
    out = {"nominal": nominal}
    cw = metrics.coverage_and_width(pred.y_true, lo, hi)
    out["overall"] = {k: v for k, v in cw.items() if not k.startswith("_")}
    if with_ci:
        out["overall"]["coverage_ci"] = metrics.coverage_ci(
            pred.unit, pred.y_true, lo, hi
        )
        out["overall"]["width_ci"] = metrics.width_ci(pred.unit, lo, hi)
    out["overall"]["n_units"] = int(len(np.unique(pred.unit)))

    # Regime conditioning is on the TRUE UNCAPPED RUL, which is the physically
    # meaningful health regime, and is therefore identical across target variants.
    out["bins"] = {}
    for name, mask in metrics.rul_bin_masks(pred.rul_true_uncapped).items():
        if mask.sum() == 0:
            continue
        sub = metrics.coverage_and_width(pred.y_true[mask], lo[mask], hi[mask])
        cell = {k: v for k, v in sub.items() if not k.startswith("_")}
        cell["n_units"] = int(len(np.unique(pred.unit[mask])))
        if with_ci:
            cell["coverage_ci"] = metrics.coverage_ci(
                pred.unit[mask], pred.y_true[mask], lo[mask], hi[mask]
            )
        out["bins"][name] = cell
    return out


def marginal_bounds(pred, nominal: float, clamp_lo: float = 0.0):
    """The arm's own (unconformalised) central interval at `nominal`."""
    if not pred.has_intervals:
        return None, None
    lo, hi = metrics.interval_bounds(np.asarray(pred.levels), pred.quantiles, nominal)
    lo = np.maximum(lo, clamp_lo)
    return lo, np.maximum(hi, lo)


def pinball(pred) -> dict | None:
    if not pred.has_intervals:
        return None
    return metrics.pinball_loss(pred.y_true, pred.quantiles, np.asarray(pred.levels))
