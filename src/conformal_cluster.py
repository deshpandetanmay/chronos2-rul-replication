"""M7: which conformal variant each arm gets, and a cluster-aware sensitivity run.

Two things the review asked for.

1. State the variant per arm. Arms that emit quantiles get **CQR** (Romano et al. 2019):
   score max(q_lo - y, y - q_hi), interval [q_lo - Q, q_hi + Q], which preserves
   input-dependent width. Point-only arms get **absolute-residual** conformal, score
   |y - yhat|, interval yhat +- Q, which is necessarily constant width. This matters for
   reading the regime result: constant width makes "inverts the conditional" nearly
   automatic, whereas for a CQR arm the same finding is a statement about the model.

2. Attempt the obvious remedy for the clustering violation rather than only noting it.
   Exchangeability holds at the engine level, not the window level, so the effective
   calibration sample is 20 engines and not 300 windows. The cheapest cluster-respecting
   calibration is one window per calibration engine, drawn at random: 20 exchangeable
   scores instead of 300 correlated ones. That trades a much smaller calibration set for
   an assumption that actually holds, and the comparison shows which cost dominates.

Run: uv run python -m src.conformal_cluster
"""

from __future__ import annotations

import json
import sys

import numpy as np

from . import calibration as cal, config as C, metrics, preds
from .phase1 import hr

ARMS = ["trivial", "lgbm_summary", "lgbm_raw", "tsfm", "tsfm_q",
        "control_randproj", "control_randproj_q"]
N_DRAWS = 200  # random one-per-engine calibration subsets, to average out the draw


def one_per_unit(pred, rng):
    """Row indices selecting exactly one window per calibration engine."""
    idx = []
    for u in np.unique(pred.unit):
        rows = np.flatnonzero(pred.unit == u)
        idx.append(int(rng.choice(rows)))
    return np.array(sorted(idx))


class Subset:
    """A row-subset view of a Prediction, enough for apply_conformal."""

    def __init__(self, p, rows):
        self.y_true = p.y_true[rows]
        self.pred = p.pred[rows]
        self.quantiles = None if p.quantiles is None else p.quantiles[rows]
        self.levels = p.levels
        self.has_intervals = p.quantiles is not None
        self.unit = p.unit[rows]


def main() -> int:
    L = C.LOOKBACK
    report = {"n_draws": N_DRAWS, "cells": {}}
    hr("M7: CONFORMAL VARIANT PER ARM, AND CLUSTER-AWARE CALIBRATION")

    print("  variant used per arm (chosen by what the arm can emit):")
    for arm in ARMS:
        p = preds.load(f"{arm}__A__L{L}__eval")
        print(f"    {arm:<22} {'CQR (adaptive width)' if p.has_intervals else 'absolute-residual (constant width)'}")

    print(f"\n  Sensitivity at nominal 90%: full 300-window calibration set vs "
          f"{N_DRAWS} random")
    print("  one-window-per-engine subsets (20 exchangeable scores).\n")
    print(f"  {'arm':<22}{'v':>2}{'cov(300)':>10}{'width':>8}"
          f"{'cov(20)':>10}{'sd':>7}{'width':>8}{'sd':>7}")
    for variant in ("A", "B"):
        for arm in ARMS:
            ev = preds.load(f"{arm}__{variant}__L{L}__eval")
            ca = preds.load(f"{arm}__{variant}__L{L}__calib")
            full = cal.apply_conformal(ca, ev, 0.90)
            cw_full = metrics.coverage_and_width(ev.y_true, full["lo"], full["hi"])

            rng = np.random.default_rng(C.SEEDS["bootstrap"] + 3)
            covs, wids, feas = [], [], 0
            for _ in range(N_DRAWS):
                rows = one_per_unit(ca, rng)
                sub = Subset(ca, rows)
                cf = cal.apply_conformal(sub, ev, 0.90)
                if not cf["feasible"]:
                    continue
                feas += 1
                r = metrics.coverage_and_width(ev.y_true, cf["lo"], cf["hi"])
                covs.append(r["coverage"]); wids.append(r["width_mean"])
            covs, wids = np.array(covs), np.array(wids)
            report["cells"][f"{arm}__{variant}"] = {
                "method": full["method"],
                "full_coverage": cw_full["coverage"],
                "full_width_mean": cw_full["width_mean"],
                "full_q": full["q"],
                "cluster_n_feasible_draws": feas,
                "cluster_coverage_mean": float(covs.mean()) if feas else None,
                "cluster_coverage_sd": float(covs.std(ddof=1)) if feas > 1 else None,
                "cluster_width_mean": float(wids.mean()) if feas else None,
                "cluster_width_sd": float(wids.std(ddof=1)) if feas > 1 else None,
            }
            if feas:
                print(f"  {arm:<22}{variant:>2}{100*cw_full['coverage']:>10.1f}"
                      f"{cw_full['width_mean']:>8.0f}{100*covs.mean():>10.1f}"
                      f"{100*covs.std(ddof=1):>7.1f}{wids.mean():>8.0f}"
                      f"{wids.std(ddof=1):>7.0f}")
            else:
                print(f"  {arm:<22}{variant:>2}{100*cw_full['coverage']:>10.1f}"
                      f"{cw_full['width_mean']:>8.0f}{'infeasible':>10}")

    n_needed = int(np.ceil(21 * 0.90))
    print(f"\n  With 20 calibration scores the 90% conformal quantile needs the "
          f"{n_needed}th of 20 order statistics, so the level is attainable but the "
          f"quantile is estimated from 20 points: coverage is unbiased in expectation "
          f"and far noisier, which the sd column shows directly.")
    (C.RESULTS / "conformal_cluster.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n")
    print("\nwrote results/conformal_cluster.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
