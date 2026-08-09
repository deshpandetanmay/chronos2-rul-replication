"""C1 adjudication: does the frozen-TSFM arm beat the baselines on point accuracy?

Reads only saved prediction files, so it imports neither torch nor LightGBM and can
run in the same process as anything (src/ompguard.py).

Comparisons are **paired and clustered**: each bootstrap resample draws eval units,
and both arms are scored on exactly the same resampled rows, so the CI is on the
*difference* rather than on two independent metrics. Two overlapping marginal CIs do
not establish a tie, and two non-overlapping ones are not needed for a difference to
be real -- only the paired interval answers the question.

Run: uv run python -m src.compare_c1
"""

from __future__ import annotations

import json
import sys

import numpy as np

from . import config as C, metrics, preds
from .phase1 import hr

PRIMARY_ARMS = ["trivial", "lgbm_summary", "lgbm_raw", "tsfm", "tsfm_q",
                f"tsfm_abl_{C.REDUCTION_ABLATION}", "control_randproj"]
SECONDARY_ARMS = ["trivial", "lgbm_summary", "lgbm_raw", "tsfm", "control_randproj"]
REFERENCE = "lgbm_summary"  # the brief's designated most-important baseline

# Phase 4 controls, compared against the intact TSFM arm rather than the baseline:
# the question they answer is "how much of the TSFM arm's performance is attributable
# to pretraining?", so `tsfm` is the right reference.
CONTROL_ARMS = ["control_shufflabel", "control_chanscramble",
                "control_randproj", "control_randproj_q"]


def _load(arm: str, variant: str, lookback: int):
    return preds.load(f"{arm}__{variant}__L{lookback}__eval")


def assert_aligned(a, b) -> None:
    """Two arms must be scored on identical (unit, t) rows in identical order."""
    if not (np.array_equal(a.unit, b.unit) and np.array_equal(a.t, b.t)):
        raise AssertionError(
            f"{a.arm} and {b.arm} are not row-aligned; pairing would be invalid"
        )
    if not np.allclose(a.y_true, b.y_true):
        raise AssertionError(f"{a.arm} and {b.arm} disagree on y_true")


def paired_diff(a, b, which: str = "rmse", **kw) -> dict:
    """Clustered CI on metric(a) - metric(b), scored on the same resampled units."""
    assert_aligned(a, b)
    yt = a.y_true

    def m(pred, rows):
        e = pred[rows] - yt[rows]
        if which == "rmse":
            return float(np.sqrt(np.mean(e**2)))
        if which == "mae":
            return float(np.mean(np.abs(e)))
        raise ValueError(which)

    out = metrics.clustered_bootstrap(
        a.unit, lambda rows: m(a.pred, rows) - m(b.pred, rows), **kw
    )
    out["excludes_zero"] = bool(out["ci_lo"] > 0 or out["ci_hi"] < 0)
    out["direction"] = ("a_worse" if out["point"] > 0 else "a_better")
    return out


def main() -> int:
    report: dict = {"reference_baseline": REFERENCE, "primary": {}, "secondary": {}}

    for label, L, arms in (
        ("PRIMARY", C.LOOKBACK, PRIMARY_ARMS),
        ("SECONDARY", C.LOOKBACK_SECONDARY, SECONDARY_ARMS),
    ):
        hr(f"C1 -- POINT ACCURACY, L={L} [{label}], eval split (30 units, 450 windows)")
        print(f"  Clustered bootstrap: {C.BOOTSTRAP_RESAMPLES} resamples of eval units\n")
        print(f"  {'arm':<18}{'v':>2}{'RMSE [95% CI]':>25}{'MAE [95% CI]':>25}"
              f"{'score/n':>10}{'bias':>8}{'/trivial':>10}")
        cell = {}
        for variant in ("A", "B"):
            triv_rmse = None
            for arm in arms:
                try:
                    p = _load(arm, variant, L)
                except FileNotFoundError:
                    continue
                r = metrics.metric_ci(p.unit, p.y_true, p.pred, "rmse")
                m = metrics.metric_ci(p.unit, p.y_true, p.pred, "mae")
                pm = p.point_metrics()
                if arm == "trivial":
                    triv_rmse = r["point"]
                ratio = r["point"] / triv_rmse if triv_rmse else float("nan")
                cell[f"{arm}__{variant}"] = {
                    "rmse_ci": r, "mae_ci": m, "point_metrics": pm,
                    "rmse_over_trivial": ratio,
                }
                rs = "%.2f [%.2f, %.2f]" % (r["point"], r["ci_lo"], r["ci_hi"])
                ms = "%.2f [%.2f, %.2f]" % (m["point"], m["ci_lo"], m["ci_hi"])
                print(f"  {arm:<18}{variant:>2}{rs:>25}{ms:>25}"
                      f"{pm['score_mean']:>10.2f}{pm['bias_mean_signed_error']:>+8.2f}"
                      f"{ratio:>10.3f}")
            print()
        report["primary" if label == "PRIMARY" else "secondary"] = cell

        # ---------------------------------------------------------- paired tests
        print(f"  PAIRED DIFFERENCES vs {REFERENCE} (positive = TSFM arm is WORSE)")
        print(f"  {'comparison':<34}{'v':>2}{'dRMSE [95% CI]':>26}{'verdict':>26}")
        pairs = {}
        for variant in ("A", "B"):
            base = _load(REFERENCE, variant, L)
            for arm in arms:
                if arm in (REFERENCE, "trivial"):
                    continue
                try:
                    p = _load(arm, variant, L)
                except FileNotFoundError:
                    continue
                d = paired_diff(p, base, "rmse")
                pairs[f"{arm}_minus_{REFERENCE}__{variant}"] = d
                ds = "%+.2f [%+.2f, %+.2f]" % (d["point"], d["ci_lo"], d["ci_hi"])
                if not d["excludes_zero"]:
                    verdict = "no significant difference"
                elif d["point"] > 0:
                    verdict = "first arm significantly WORSE"
                else:
                    verdict = "first arm significantly better"
                print(f"  {arm + ' - ' + REFERENCE:<34}{variant:>2}{ds:>26}{verdict:>26}")
            # Does the TSFM arm at least beat the trivial marginal?
            triv = _load("trivial", variant, L)
            d = paired_diff(_load("tsfm", variant, L), triv, "rmse")
            pairs[f"tsfm_minus_trivial__{variant}"] = d
            ds = "%+.2f [%+.2f, %+.2f]" % (d["point"], d["ci_lo"], d["ci_hi"])
            v = ("beats trivial" if d["excludes_zero"] and d["point"] < 0
                 else "does NOT beat trivial")
            print(f"  {'tsfm - trivial':<34}{variant:>2}{ds:>26}{v:>26}")
        report[("primary" if label == "PRIMARY" else "secondary") + "_paired"] = pairs
        print()

    # -------------------------------------------------------------- controls
    hr("CONTROLS vs THE INTACT TSFM ARM (L=%d) -- attribution, not contamination testing"
       % C.LOOKBACK)
    print("  Positive dRMSE = the control is WORSE than the intact TSFM arm, i.e. the")
    print("  thing removed was contributing. Negative or zero = it was not.\n")
    print(f"  {'control':<24}{'v':>2}{'RMSE':>9}{'dRMSE vs tsfm [95% CI]':>30}{'reading':>34}")
    ctl = {}
    for variant in ("A", "B"):
        base = _load("tsfm", variant, C.LOOKBACK)
        triv = metrics.point_metrics(_load("trivial", variant, C.LOOKBACK).y_true,
                                    _load("trivial", variant, C.LOOKBACK).pred)["rmse"]
        for arm in CONTROL_ARMS:
            try:
                p = _load(arm, variant, C.LOOKBACK)
            except FileNotFoundError:
                continue
            d = paired_diff(p, base, "rmse")
            r = metrics.point_metrics(p.y_true, p.pred)["rmse"]
            ctl[f"{arm}__{variant}"] = {"rmse": r, "diff_vs_tsfm": d,
                                        "rmse_over_trivial": r / triv}
            ds = "%+.2f [%+.2f, %+.2f]" % (d["point"], d["ci_lo"], d["ci_hi"])
            if arm == "control_shufflabel":
                read = ("collapsed (>= trivial)" if r >= triv
                        else "DID NOT COLLAPSE -- leak")
            elif not d["excludes_zero"]:
                read = "indistinguishable from tsfm"
            elif d["point"] > 0:
                read = "worse than tsfm"
            else:
                read = "BETTER than tsfm"
            print(f"  {arm:<24}{variant:>2}{r:>9.2f}{ds:>30}{read:>34}")
    report["controls_vs_tsfm"] = ctl
    print()

    # -------------------------------------------------------------- cross-L note
    hr("CROSS-LOOK-BACK READING (see D-009: absolute error is NOT comparable)")
    print("  The L=30 and L=80 eval sets are different (mean RUL 88.7 vs 63.3; 26.4% vs")
    print("  10.2% at the cap), because a longer window forces later truncation points.")
    print("  Only the within-L ranking, and each arm's error relative to the trivial")
    print("  marginal on its own eval set, are comparable across L.\n")
    print(f"  {'arm':<18}{'v':>2}{'RMSE/trivial @L30':>20}{'RMSE/trivial @L80':>20}")
    for variant in ("A", "B"):
        for arm in SECONDARY_ARMS:
            a = report["primary"].get(f"{arm}__{variant}", {}).get("rmse_over_trivial")
            b = report["secondary"].get(f"{arm}__{variant}", {}).get("rmse_over_trivial")
            if a is None or b is None:
                continue
            print(f"  {arm:<18}{variant:>2}{a:>20.3f}{b:>20.3f}")
    print()

    (C.RESULTS / "c1_comparison.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    print("wrote results/c1_comparison.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
