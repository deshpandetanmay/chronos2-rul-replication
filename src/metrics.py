"""Point-accuracy and interval metrics, plus the by-unit clustered bootstrap.

The bootstrap resamples *units*, not windows. Truncation points inside a unit share
a trajectory and are strongly correlated, so a naive binomial or window-level
interval understates uncertainty on every coverage number we report.
"""

from __future__ import annotations

import numpy as np

from . import config as C

# Saxena et al. (2008) asymmetric prognostics score constants. Late predictions
# (predicted RUL greater than truth, i.e. claiming more life than there is) decay
# with the smaller constant and are therefore penalised harder.
SCORE_EARLY_TAU = 13.0
SCORE_LATE_TAU = 10.0


def prognostics_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-window asymmetric score. Lower is better; 0 is perfect."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return np.where(d < 0, np.expm1(-d / SCORE_EARLY_TAU), np.expm1(d / SCORE_LATE_TAU))


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE, MAE, prognostics score, and the direction of the error skew."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    err = y_pred - y_true  # >0 = late (over-predicts remaining life) = dangerous
    s = prognostics_score(y_true, y_pred)
    return {
        "n": int(len(y_true)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "score_total": float(np.sum(s)),
        "score_mean": float(np.mean(s)),
        "bias_mean_signed_error": float(np.mean(err)),
        "bias_median_signed_error": float(np.median(err)),
        "frac_late": float(np.mean(err > 0)),
        "mean_late_error": float(err[err > 0].mean()) if (err > 0).any() else 0.0,
        "mean_early_error": float(err[err < 0].mean()) if (err < 0).any() else 0.0,
    }


def pinball_loss(y_true: np.ndarray, q_pred: np.ndarray, levels: np.ndarray) -> dict:
    """Pinball (quantile) loss per level and averaged across levels.

    `q_pred` is (n, n_levels) aligned with `levels`.
    """
    y = np.asarray(y_true, float)[:, None]
    q = np.asarray(q_pred, float)
    tau = np.asarray(levels, float)[None, :]
    diff = y - q
    losses = np.maximum(tau * diff, (tau - 1.0) * diff)  # (n, n_levels)
    per_level = losses.mean(axis=0)
    return {
        "pinball_mean": float(per_level.mean()),
        "pinball_per_level": {float(t): float(v) for t, v in zip(levels, per_level)},
    }


def interval_bounds(levels: np.ndarray, q_pred: np.ndarray, nominal: float) -> tuple:
    """Extract the central interval at `nominal` from a quantile matrix."""
    lo_t, hi_t = (1.0 - nominal) / 2.0, (1.0 + nominal) / 2.0
    i_lo = int(np.argmin(np.abs(np.asarray(levels) - lo_t)))
    i_hi = int(np.argmin(np.abs(np.asarray(levels) - hi_t)))
    if not np.isclose(levels[i_lo], lo_t) or not np.isclose(levels[i_hi], hi_t):
        raise ValueError(
            f"quantile levels {list(levels)} do not contain {lo_t} and {hi_t} "
            f"needed for a central {nominal:.0%} interval"
        )
    return q_pred[:, i_lo], q_pred[:, i_hi]


def coverage_and_width(y_true, lo, hi) -> dict:
    y = np.asarray(y_true, float)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    covered = (y >= lo) & (y <= hi)
    w = hi - lo
    return {
        "coverage": float(covered.mean()),
        "width_mean": float(w.mean()),
        "width_median": float(np.median(w)),
        "n": int(len(y)),
        "_covered": covered,
    }


def enforce_monotone_quantiles(q_pred: np.ndarray) -> tuple[np.ndarray, int]:
    """Sort quantiles within each row, returning the count of rows that crossed.

    Independently fitted quantile regressors are not guaranteed to be ordered.
    Sorting is the standard repair and is order-preserving where no crossing
    occurred, so it is a no-op on well-behaved rows. The crossing count is
    reported rather than hidden, because a high rate is itself a finding about the
    arm's interval quality.
    """
    q = np.asarray(q_pred, float)
    crossed = int(np.sum(np.any(np.diff(q, axis=1) < 0, axis=1)))
    return np.sort(q, axis=1), crossed


def clustered_bootstrap(
    units: np.ndarray,
    statistic,
    n_resamples: int = C.BOOTSTRAP_RESAMPLES,
    seed: int = C.SEEDS["bootstrap"],
    alpha: float = 0.05,
) -> dict:
    """Percentile CI for `statistic` under resampling of whole units.

    `statistic(mask_indices) -> float` is evaluated on the row indices selected by
    each unit-level resample. Units are drawn with replacement; all rows belonging
    to a drawn unit are included, and a unit drawn twice contributes its rows
    twice.
    """
    units = np.asarray(units)
    uniq = np.unique(units)
    rows_by_unit = {u: np.flatnonzero(units == u) for u in uniq}
    rng = np.random.default_rng(seed)

    vals = np.empty(n_resamples, float)
    for b in range(n_resamples):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([rows_by_unit[u] for u in drawn])
        vals[b] = statistic(rows)

    finite = vals[np.isfinite(vals)]
    return {
        "point": float(statistic(np.arange(len(units)))),
        "ci_lo": float(np.percentile(finite, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(finite, 100 * (1 - alpha / 2))),
        "boot_mean": float(finite.mean()),
        "boot_sd": float(finite.std(ddof=1)),
        "n_resamples": int(len(finite)),
        "n_units": int(len(uniq)),
        "alpha": alpha,
        "method": "percentile bootstrap over units (clustered)",
    }


def coverage_ci(units, y_true, lo, hi, **kw) -> dict:
    """Clustered CI on empirical coverage."""
    y = np.asarray(y_true, float)
    covered = ((y >= np.asarray(lo, float)) & (y <= np.asarray(hi, float))).astype(float)
    out = clustered_bootstrap(units, lambda rows: covered[rows].mean(), **kw)
    out["naive_binomial_se"] = float(
        np.sqrt(max(out["point"] * (1 - out["point"]), 1e-12) / len(covered))
    )
    return out


def width_ci(units, lo, hi, **kw) -> dict:
    w = np.asarray(hi, float) - np.asarray(lo, float)
    return clustered_bootstrap(units, lambda rows: w[rows].mean(), **kw)


def metric_ci(units, y_true, y_pred, which: str = "rmse", **kw) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if which == "rmse":
        fn = lambda r: float(np.sqrt(np.mean((y_pred[r] - y_true[r]) ** 2)))
    elif which == "mae":
        fn = lambda r: float(np.mean(np.abs(y_pred[r] - y_true[r])))
    elif which == "score_mean":
        s = prognostics_score(y_true, y_pred)
        fn = lambda r: float(np.mean(s[r]))
    else:
        raise ValueError(f"unknown metric {which!r}")
    return clustered_bootstrap(units, fn, **kw)


def rul_bin_masks(rul_true: np.ndarray) -> dict[str, np.ndarray]:
    """Regime masks keyed by bin name, built from the *true uncapped* RUL."""
    r = np.asarray(rul_true, float)
    out = {}
    for name, lo, hi in C.RUL_BINS:
        out[name] = (r >= lo) if not np.isfinite(hi) else ((r >= lo) & (r < hi))
    return out
