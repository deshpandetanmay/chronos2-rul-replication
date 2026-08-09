"""Phase 2: baselines. Run BEFORE the foundation model, deliberately (brief §6).

Run: uv run python -m src.phase2
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from . import baselines, config as C, dataset, manifest, metrics, preds
from .ompguard import assert_single_omp_runtime
from .phase1 import hr
from .seeding import seed_everything


def main() -> int:
    # LightGBM-only process; torch here would segfault mid-fit (src/ompguard.py).
    assert_single_omp_runtime("phase2 / baselines")
    seed_everything()
    report: dict = {"phase": 2, "lookbacks": {}}

    for L, with_q, tag in (
        (C.LOOKBACK, True, "PRIMARY"),
        (C.LOOKBACK_SECONDARY, False, "SECONDARY (point accuracy only, D-009)"),
    ):
        hr(f"BASELINES AT L={L}  [{tag}]")
        t0 = time.perf_counter()
        b = dataset.build(L, verbose=True)
        print(f"  splits: {b.splits.sizes}   features: {b.n_features} "
              f"{b.feature_cols}")
        print(f"  quantile levels: {list(C.QUANTILE_LEVELS)}" if with_q
              else "  quantiles: not emitted at this look-back")
        print()
        rep = baselines.run_all(b, with_quantiles=with_q, log=print)
        rep["wall_seconds"] = time.perf_counter() - t0
        rep["bundle"] = b.describe()
        report["lookbacks"][str(L)] = rep
        print(f"\n  wall clock: {rep['wall_seconds']:.1f}s")

    # ------------------------------------------------------------ comparison
    hr("POINT-ACCURACY COMPARISON (eval split, 3 baselines x 2 variants)")
    arms = ["trivial", "lgbm_summary", "lgbm_raw"]
    for L in (C.LOOKBACK, C.LOOKBACK_SECONDARY):
        rep = report["lookbacks"][str(L)]
        print(f"\n  L={L}")
        print(f"    {'arm':<14}{'variant':>8}{'RMSE':>9}{'MAE':>9}"
              f"{'score/n':>11}{'bias':>9}{'%late':>8}")
        for arm in arms:
            for v in ("A", "B"):
                m = rep["arms"][arm][v]
                print(f"    {arm:<14}{v:>8}{m['rmse']:>9.2f}{m['mae']:>9.2f}"
                      f"{m['score_mean']:>11.2f}{m['bias_mean_signed_error']:>+9.2f}"
                      f"{100*m['frac_late']:>7.1f}%")

    # ------------------------------------------------------------ CIs at primary L
    hr("CLUSTERED (BY-UNIT) CONFIDENCE INTERVALS, L=%d, eval split" % C.LOOKBACK)
    print(f"  {C.BOOTSTRAP_RESAMPLES} resamples of the 30 eval units, percentile method\n")
    ci_out = {}
    print(f"    {'arm':<14}{'v':>3}{'RMSE [95% CI]':>26}{'MAE [95% CI]':>26}")
    for arm in arms:
        for v in ("A", "B"):
            p = preds.load(f"{arm}__{v}__L{C.LOOKBACK}__eval")
            r = metrics.metric_ci(p.unit, p.y_true, p.pred, "rmse")
            m = metrics.metric_ci(p.unit, p.y_true, p.pred, "mae")
            ci_out[f"{arm}__{v}"] = {"rmse": r, "mae": m}
            rs = "%.2f [%.2f, %.2f]" % (r["point"], r["ci_lo"], r["ci_hi"])
            ms = "%.2f [%.2f, %.2f]" % (m["point"], m["ci_lo"], m["ci_hi"])
            print(f"    {arm:<14}{v:>3}{rs:>26}{ms:>26}")
    report["clustered_ci_primary"] = ci_out

    # Show how much the clustering matters for a coverage-style quantity.
    hr("WHY CLUSTERING MATTERS (nominal 90%% interval, L=%d)" % C.LOOKBACK)
    print(f"    {'arm':<14}{'v':>3}{'coverage':>10}{'clustered 95% CI':>22}"
          f"{'naive binom SE':>16}{'width_mean':>12}")
    cov_out = {}
    for arm in arms:
        for v in ("A", "B"):
            p = preds.load(f"{arm}__{v}__L{C.LOOKBACK}__eval")
            lo, hi = metrics.interval_bounds(np.array(p.levels), p.quantiles, 0.90)
            c = metrics.coverage_ci(p.unit, p.y_true, lo, hi)
            w = metrics.width_ci(p.unit, lo, hi)
            cov_out[f"{arm}__{v}"] = {"coverage90": c, "width90": w}
            half = (c["ci_hi"] - c["ci_lo"]) / 2
            cs = "[%.3f, %.3f]" % (c["ci_lo"], c["ci_hi"])
            ratio = half / max(c["naive_binomial_se"], 1e-9)
            print(f"    {arm:<14}{v:>3}{c['point']:>10.3f}{cs:>22}"
                  f"{c['naive_binomial_se']:>16.4f}{w['point']:>12.2f}"
                  f"   (clustered half-width {half:.3f} = {ratio:.1f}x naive SE)")
    report["coverage90_preview"] = cov_out
    print("\n  Phase 5 reports all five nominal levels; the 90% row above is a preview to")
    print("  show the clustered CI is materially wider than a naive binomial interval.")

    (C.RESULTS / "phase2_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    manifest.merge_into("phase2", report)
    print(f"\nwrote results/phase2_report.json and "
          f"{len(preds.available())} prediction files under results/preds/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
