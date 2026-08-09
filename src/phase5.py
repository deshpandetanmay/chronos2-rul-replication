"""Phase 5: calibration analysis. The project's actual contribution.

For every (arm, target variant) cell, at five nominal levels, before and after split
conformal, overall and per RUL regime:
  coverage (with by-unit clustered bootstrap CIs), mean/median interval width,
  pinball loss, point accuracy, error skew.

Emits `results/metrics.csv` in long format: one row per
(arm, variant, lookback, split, conformal, nominal level, RUL bin, metric).

The `conformal` column takes "point" for rows that are not about an interval at all,
"marginal" for the arm's own intervals and "conformal" for post-conformal intervals.
"point" rather than "n/a" deliberately: pandas.read_csv maps "n/a" to NaN by default,
which silently dropped every point-accuracy row when the CSV was read back.

Run: uv run python -m src.phase5
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from . import calibration as cal, config as C, metrics, preds
from .dataset import OFFICIAL_SPLIT
from .phase1 import hr

# Arm display order and grouping. Point-only arms have no interval before conformal;
# that is reported as such rather than silently omitted.
ARMS = [
    ("trivial", "baseline"),
    ("lgbm_summary", "baseline"),
    ("lgbm_raw", "baseline"),
    ("tsfm", "tsfm"),
    ("tsfm_q", "tsfm"),
    (f"tsfm_abl_{C.REDUCTION_ABLATION}", "tsfm"),
    ("control_randproj", "control"),
    ("control_randproj_q", "control"),
    ("control_chanscramble", "control"),
    ("control_shufflabel", "control"),
]


def _rows_for(pred, calib, arm, variant, group, split) -> list[dict]:
    """All long-format metric rows for one arm x variant x split."""
    rows: list[dict] = []
    base = dict(arm=arm, arm_group=group, variant=variant,
                lookback=pred.lookback, split=split)

    # ---------------------------------------------------------- point accuracy
    pm = pred.point_metrics()
    with_ci = split == "eval"
    for k in ("rmse", "mae", "score_mean", "score_total",
              "bias_mean_signed_error", "bias_median_signed_error", "frac_late",
              "mean_late_error", "mean_early_error"):
        r = dict(base, conformal="point", nominal=np.nan, rul_bin="all",
                 metric=k, value=pm[k], ci_lo=np.nan, ci_hi=np.nan,
                 n_windows=pm["n"], n_units=len(np.unique(pred.unit)))
        if with_ci and k in ("rmse", "mae", "score_mean"):
            ci = metrics.metric_ci(pred.unit, pred.y_true, pred.pred, k)
            r["ci_lo"], r["ci_hi"] = ci["ci_lo"], ci["ci_hi"]
        rows.append(r)

    # Point accuracy per RUL regime.
    for bname, mask in metrics.rul_bin_masks(pred.rul_true_uncapped).items():
        if mask.sum() == 0:
            continue
        bpm = metrics.point_metrics(pred.y_true[mask], pred.pred[mask])
        for k in ("rmse", "mae", "score_mean", "bias_mean_signed_error"):
            rows.append(dict(base, conformal="point", nominal=np.nan, rul_bin=bname,
                             metric=k, value=bpm[k], ci_lo=np.nan, ci_hi=np.nan,
                             n_windows=bpm["n"],
                             n_units=len(np.unique(pred.unit[mask]))))

    # ---------------------------------------------------------- pinball
    pb = cal.pinball(pred)
    if pb:
        rows.append(dict(base, conformal="marginal", nominal=np.nan, rul_bin="all",
                         metric="pinball_mean", value=pb["pinball_mean"],
                         ci_lo=np.nan, ci_hi=np.nan, n_windows=len(pred.y_true),
                         n_units=len(np.unique(pred.unit))))

    # ---------------------------------------------------------- intervals
    for nominal in C.NOMINAL_LEVELS:
        variants = []
        m_lo, m_hi = cal.marginal_bounds(pred, nominal)
        if m_lo is not None:
            variants.append(("marginal", m_lo, m_hi, np.nan))
        if calib is not None:
            cf = cal.apply_conformal(calib, pred, nominal)
            if cf["feasible"]:
                variants.append(("conformal", cf["lo"], cf["hi"], cf["q"]))
        for tag, lo, hi, q in variants:
            ev = cal.evaluate_cell(pred, lo, hi, nominal, with_ci=with_ci)
            o = ev["overall"]
            for metric, key in (("coverage", "coverage"),
                                ("width_mean", "width_mean"),
                                ("width_median", "width_median")):
                r = dict(base, conformal=tag, nominal=nominal, rul_bin="all",
                         metric=metric, value=o[key],
                         ci_lo=np.nan, ci_hi=np.nan,
                         n_windows=o["n"], n_units=o["n_units"])
                if with_ci and metric == "coverage":
                    r["ci_lo"], r["ci_hi"] = o["coverage_ci"]["ci_lo"], o["coverage_ci"]["ci_hi"]
                if with_ci and metric == "width_mean":
                    r["ci_lo"], r["ci_hi"] = o["width_ci"]["ci_lo"], o["width_ci"]["ci_hi"]
                rows.append(r)
            if np.isfinite(q):
                rows.append(dict(base, conformal=tag, nominal=nominal, rul_bin="all",
                                 metric="conformal_q", value=q, ci_lo=np.nan,
                                 ci_hi=np.nan, n_windows=o["n"], n_units=o["n_units"]))
            for bname, cell in ev["bins"].items():
                for metric, key in (("coverage", "coverage"),
                                    ("width_mean", "width_mean"),
                                    ("width_median", "width_median")):
                    r = dict(base, conformal=tag, nominal=nominal, rul_bin=bname,
                             metric=metric, value=cell[key], ci_lo=np.nan,
                             ci_hi=np.nan, n_windows=cell["n"],
                             n_units=cell["n_units"])
                    if with_ci and metric == "coverage":
                        r["ci_lo"] = cell["coverage_ci"]["ci_lo"]
                        r["ci_hi"] = cell["coverage_ci"]["ci_hi"]
                    rows.append(r)
    return rows


def main() -> int:
    L = C.LOOKBACK
    hr("PHASE 5: CALIBRATION ANALYSIS")
    n_calib = len(preds.load(f"tsfm__A__L{L}__calib").y_true)
    feas = cal.conformal_feasible_levels(n_calib)
    print(f"  calibration split: {n_calib} windows / 20 units")
    print(f"  finite-sample conformal quantile index ceil((n+1)*level):")
    for lv, ok in feas.items():
        k = int(np.ceil((n_calib + 1) * lv))
        print(f"    level {lv:.2f} -> order statistic {k} of {n_calib}"
              f"  {'feasible' if ok else 'NOT FEASIBLE'}")
    if not all(feas.values()):
        print("  !! some levels are not certifiable with this calibration size")

    all_rows: list[dict] = []
    available = set(preds.available())
    for split in ("eval", OFFICIAL_SPLIT):
        for arm, group in ARMS:
            for variant in ("A", "B"):
                key = f"{arm}__{variant}__L{L}__{split}"
                if key not in available:
                    continue
                pred = preds.load(key)
                ck = f"{arm}__{variant}__L{L}__calib"
                calib = preds.load(ck) if ck in available else None
                all_rows += _rows_for(pred, calib, arm, variant, group, split)
    # L=80 secondary: point accuracy only.
    for arm in C.SECONDARY_ARMS:
        for variant in ("A", "B"):
            key = f"{arm}__{variant}__L{C.LOOKBACK_SECONDARY}__eval"
            if key not in available:
                continue
            p = preds.load(key)
            pm = p.point_metrics()
            for k in ("rmse", "mae", "score_mean", "bias_mean_signed_error"):
                ci = (metrics.metric_ci(p.unit, p.y_true, p.pred, k)
                      if k in ("rmse", "mae", "score_mean") else None)
                all_rows.append(dict(
                    arm=arm, arm_group="secondary", variant=variant,
                    lookback=C.LOOKBACK_SECONDARY, split="eval", conformal="point",
                    nominal=np.nan, rul_bin="all", metric=k, value=pm[k],
                    ci_lo=ci["ci_lo"] if ci else np.nan,
                    ci_hi=ci["ci_hi"] if ci else np.nan,
                    n_windows=pm["n"], n_units=len(np.unique(p.unit))))

    df = pd.DataFrame(all_rows)
    df.to_csv(C.RESULTS / "metrics.csv", index=False)
    print(f"\n  wrote results/metrics.csv: {len(df):,} rows, "
          f"{df['arm'].nunique()} arms, columns {list(df.columns)}")

    # ------------------------------------------------------------------ tables
    ev = df[(df.split == "eval") & (df.lookback == L)]

    hr("C2: MARGINAL COVERAGE (the arm's OWN intervals, before conformal)")
    print("  Point-only arms are absent by construction: an MSE-trained point head")
    print("  emits no uncertainty at all. That is itself the finding for those arms.\n")
    _coverage_table(ev, "marginal")

    hr("C3: COVERAGE AFTER SPLIT CONFORMAL (calibrated on 20 held-out units)")
    _coverage_table(ev, "conformal")

    hr("C2/C3: REGIME-CONDITIONED COVERAGE AT NOMINAL 90%")
    _regime_table(ev)

    hr("PINBALL LOSS (mean over 11 quantile levels, lower better)")
    pb = ev[(ev.metric == "pinball_mean")]
    for variant in ("A", "B"):
        sub = pb[pb.variant == variant].sort_values("value")
        print(f"  variant {variant}:  " +
              "   ".join(f"{r.arm}={r.value:.3f}" for r in sub.itertuples()))

    hr("SECONDARY: OFFICIAL TEST SET (test_FD001 + RUL_FD001, one point per unit)")
    print("  Reported separately and never pooled with the primary eval split. With one")
    print("  labelled point per unit, coverage here is far too noisy to support C2.\n")
    off = df[(df.split == OFFICIAL_SPLIT) & (df.metric == "rmse") &
             (df.rul_bin == "all")]
    print(f"  {'arm':<24}{'A RMSE':>10}{'B RMSE':>10}")
    for arm, _ in ARMS:
        a = off[(off.arm == arm) & (off.variant == "A")]["value"]
        b = off[(off.arm == arm) & (off.variant == "B")]["value"]
        if len(a) and len(b):
            print(f"  {arm:<24}{a.iloc[0]:>10.2f}{b.iloc[0]:>10.2f}")

    hr("CONFORMAL COST: width inflation to buy nominal coverage")
    _conformal_cost(ev)

    from . import figures
    figures.make_all(df)
    return 0


def _coverage_table(ev: pd.DataFrame, tag: str) -> None:
    sub = ev[(ev.conformal == tag) & (ev.rul_bin == "all")]
    arms = [a for a, _ in ARMS if a in set(sub.arm)]
    for variant in ("A", "B"):
        print(f"  --- variant {variant} " + "-" * 56)
        print(f"  {'arm':<24}" + "".join(f"{int(100*lv):>8}%" for lv in C.NOMINAL_LEVELS))
        for arm in arms:
            cov, wid = [], []
            for lv in C.NOMINAL_LEVELS:
                c = sub[(sub.arm == arm) & (sub.variant == variant) &
                        (sub.nominal == lv) & (sub.metric == "coverage")]["value"]
                w = sub[(sub.arm == arm) & (sub.variant == variant) &
                        (sub.nominal == lv) & (sub.metric == "width_mean")]["value"]
                cov.append(c.iloc[0] if len(c) else np.nan)
                wid.append(w.iloc[0] if len(w) else np.nan)
            print(f"  {arm:<24}" + "".join(f"{100*c:>8.1f}" for c in cov)
                  + "   coverage %")
            print(f"  {'':<24}" + "".join(f"{w:>8.1f}" for w in wid)
                  + "   mean width")
        print()


def _regime_table(ev: pd.DataFrame) -> None:
    print("  Bins are on TRUE UNCAPPED RUL, so they are the same physical regime in")
    print("  both target variants. A marginally valid interval can still fail badly in")
    print("  the near-failure regime, which is the only regime a maintenance decision")
    print("  is actually made in.\n")
    bins = [b for b, _, _ in C.RUL_BINS]
    for tag in ("marginal", "conformal"):
        sub = ev[(ev.conformal == tag) & (ev.nominal == 0.90) &
                 (ev.metric == "coverage")]
        if sub.empty:
            continue
        print(f"  --- {tag} intervals, nominal 90% " + "-" * 40)
        print(f"  {'arm':<24}{'v':>2}" + "".join(f"{b:>10}" for b in bins))
        for arm, _ in ARMS:
            for variant in ("A", "B"):
                vals = []
                for b in bins:
                    x = sub[(sub.arm == arm) & (sub.variant == variant) &
                            (sub.rul_bin == b)]["value"]
                    vals.append(x.iloc[0] if len(x) else np.nan)
                if all(np.isnan(v) for v in vals):
                    continue
                print(f"  {arm:<24}{variant:>2}"
                      + "".join(f"{100*v:>10.1f}" if not np.isnan(v) else f"{'-':>10}"
                                for v in vals))
        print()


def _conformal_cost(ev: pd.DataFrame) -> None:
    print(f"  {'arm':<24}{'v':>2}{'marg cov':>10}{'conf cov':>10}"
          f"{'marg width':>12}{'conf width':>12}{'inflation':>11}")
    sub = ev[(ev.nominal == 0.90) & (ev.rul_bin == "all")]
    for arm, _ in ARMS:
        for variant in ("A", "B"):
            def g(tag, metric):
                x = sub[(sub.arm == arm) & (sub.variant == variant) &
                        (sub.conformal == tag) & (sub.metric == metric)]["value"]
                return x.iloc[0] if len(x) else np.nan
            mc, cc = g("marginal", "coverage"), g("conformal", "coverage")
            mw, cw = g("marginal", "width_mean"), g("conformal", "width_mean")
            if np.isnan(cc):
                continue
            infl = cw / mw if mw and not np.isnan(mw) else np.nan
            f = lambda v, s: (f"{v:{s}}" if not np.isnan(v) else f"{'n/a':>{s.split('.')[0].lstrip('>')}}")
            print(f"  {arm:<24}{variant:>2}"
                  f"{(f'{100*mc:.1f}' if not np.isnan(mc) else 'none'):>10}"
                  f"{100*cc:>10.1f}"
                  f"{(f'{mw:.1f}' if not np.isnan(mw) else 'none'):>12}"
                  f"{cw:>12.1f}"
                  f"{(f'{infl:.2f}x' if not np.isnan(infl) else '-'):>11}")


if __name__ == "__main__":
    sys.exit(main())
