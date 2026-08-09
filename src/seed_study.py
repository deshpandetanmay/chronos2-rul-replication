"""M1/M2: head-seed replication, and the mechanism behind the quantile-loss anomaly.

Why this exists. The by-unit clustered bootstrap quantifies *evaluation-set*
uncertainty. It says nothing about *training-seed* variance in a 417,857-parameter head
trained on 8,993 windows with dropout. C1 rests on a paired difference of about +5 RMSE,
which is not far from zero, so the seed distribution has to be measured rather than
assumed away.

What it produces:

* per-arm RMSE over `N_SEEDS` head seeds (mean, sd, min, max);
* a **combined** paired interval that resamples evaluation units *and* draws an
  independent training seed for each arm on every bootstrap replicate, so the reported
  interval carries both variance sources instead of only one;
* a diagnostic for why switching MSE -> pinball moves the random-projection arm far more
  than the Chronos-2 arm: the fraction of evaluation predictions pinned at exactly zero
  by the head's final ReLU. A point head whose output unit has died contributes no
  gradient and underfits, and an 11-output pinball head has eleven chances to stay alive.

torch-only process (no LightGBM) -- see src/ompguard.py.

Run: uv run python -m src.seed_study
"""

from __future__ import annotations

import json
import sys

import numpy as np

from . import config as C, dataset, embed, head, metrics, preds
from .ompguard import assert_single_omp_runtime
from .phase1 import hr
from .features import random_projection
from .seeding import seed_everything

N_SEEDS = 5
# Derived from the configured base seed so the set is reproducible and declared.
SEEDS = [C.SEEDS["head_init"] + 1000 * i for i in range(N_SEEDS)]

# arm -> (feature source, head mode)
ARMS = {
    "tsfm": ("chronos", "point"),
    "tsfm_q": ("chronos", "quantile"),
    "control_randproj": ("randproj", "point"),
    "control_randproj_q": ("randproj", "quantile"),
}


def zero_fraction(pred: np.ndarray) -> float:
    """Share of predictions pinned at exactly 0 by the head's final ReLU."""
    return float(np.mean(np.asarray(pred) == 0.0))


def combined_paired_ci(
    units, y_true, preds_a: list[np.ndarray], preds_b: list[np.ndarray],
    n_resamples: int = C.BOOTSTRAP_RESAMPLES, seed: int = C.SEEDS["bootstrap"],
) -> dict:
    """Paired RMSE difference resampling units AND drawing a seed per arm per replicate.

    `preds_a`/`preds_b` hold one prediction vector per training seed. Each replicate
    draws units with replacement and, independently, one seed from each arm's list. The
    resulting interval reflects evaluation-set variance and training-seed variance
    together, which is what a claim about "arm A beats arm B" actually needs.
    """
    units = np.asarray(units)
    uniq = np.unique(units)
    rows_by_unit = {u: np.flatnonzero(units == u) for u in uniq}
    y = np.asarray(y_true, float)
    rng = np.random.default_rng(seed)

    def rmse(p, rows):
        e = p[rows] - y[rows]
        return float(np.sqrt(np.mean(e**2)))

    vals = np.empty(n_resamples)
    for b in range(n_resamples):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([rows_by_unit[u] for u in drawn])
        pa = preds_a[rng.integers(len(preds_a))]
        pb = preds_b[rng.integers(len(preds_b))]
        vals[b] = rmse(pa, rows) - rmse(pb, rows)

    point = float(np.mean([rmse(a, np.arange(len(y))) for a in preds_a])
                  - np.mean([rmse(b, np.arange(len(y))) for b in preds_b]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {
        "point_mean_over_seeds": point,
        "ci_lo": float(lo), "ci_hi": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_resamples": n_resamples,
        "n_seeds_a": len(preds_a), "n_seeds_b": len(preds_b),
        "method": "paired bootstrap over units x independent seed draw per arm",
    }


def main() -> int:
    assert_single_omp_runtime("seed_study")
    seed_everything()
    L = C.LOOKBACK
    report: dict = {"n_seeds": N_SEEDS, "seeds": SEEDS, "lookback": L, "arms": {}}

    hr("M1/M2: HEAD-SEED REPLICATION (%d seeds)" % N_SEEDS)
    b = dataset.build(L)
    feats = {"chronos": {}, "randproj": {}}
    for split in ("train", "eval"):
        path = embed.cache_path(L, split, "main")
        if not path.exists():
            raise SystemExit(f"missing {path}; run phase3 first")
        feats["chronos"][split] = embed.load_cache(path)[C.REDUCTION_PRIMARY]
    d_out = embed.reduced_dim(b.n_features)
    for split in ("train", "eval"):
        feats["randproj"][split] = random_projection(
            b.windows(split), d_out, C.SEEDS["random_projection_control"]
        )
    print(f"  feature blocks: chronos {feats['chronos']['train'].shape}, "
          f"randproj {feats['randproj']['train'].shape}")
    print(f"  seeds: {SEEDS}\n")

    store: dict = {}
    print(f"  {'arm':<22}{'v':>2}{'mean RMSE':>11}{'sd':>7}{'min':>8}{'max':>8}"
          f"{'spread':>9}{'pred==0':>9}")
    for variant in ("A", "B"):
        y_tr = b.target("train", variant)
        y_ev = b.target("eval", variant)
        for arm, (src, mode) in ARMS.items():
            rmses, plist, zeros = [], [], []
            for sd in SEEDS:
                net, scaler, _ = head.train_head(
                    feats[src]["train"], y_tr, mode=mode, seed=sd
                )
                pt, _ = head.predict_head(net, scaler, feats[src]["eval"], mode)
                rmses.append(metrics.point_metrics(y_ev, pt)["rmse"])
                plist.append(np.asarray(pt, float))
                zeros.append(zero_fraction(pt))
            store[f"{arm}__{variant}"] = plist
            r = np.array(rmses)
            report["arms"][f"{arm}__{variant}"] = {
                "rmse_per_seed": [float(x) for x in r],
                "mean": float(r.mean()), "sd": float(r.std(ddof=1)),
                "min": float(r.min()), "max": float(r.max()),
                "spread": float(r.max() - r.min()),
                "zero_pred_fraction_per_seed": zeros,
                "zero_pred_fraction_mean": float(np.mean(zeros)),
                "mode": mode, "features": src,
            }
            print(f"  {arm:<22}{variant:>2}{r.mean():>11.2f}{r.std(ddof=1):>7.2f}"
                  f"{r.min():>8.2f}{r.max():>8.2f}{r.max()-r.min():>9.2f}"
                  f"{100*np.mean(zeros):>8.1f}%")

    # ------------------------------------------------------------- M2 mechanism
    hr("M2: WHY DOES PINBALL LOSS HELP THE RANDOM PROJECTION AND NOT THE BACKBONE?")
    for variant in ("A", "B"):
        for src, point_arm, q_arm in (("chronos", "tsfm", "tsfm_q"),
                                      ("randproj", "control_randproj",
                                       "control_randproj_q")):
            a = report["arms"][f"{point_arm}__{variant}"]
            q = report["arms"][f"{q_arm}__{variant}"]
            gain = a["mean"] - q["mean"]
            print(f"  {src:<9} variant {variant}: MSE head {a['mean']:6.2f} "
                  f"(sd {a['sd']:.2f}, {100*a['zero_pred_fraction_mean']:5.1f}% preds==0)"
                  f"  ->  pinball head {q['mean']:6.2f} "
                  f"(sd {q['sd']:.2f}, {100*q['zero_pred_fraction_mean']:5.1f}% preds==0)"
                  f"   gain {gain:+.2f}")
            report.setdefault("m2", {})[f"{src}__{variant}"] = {
                "mse_mean": a["mean"], "pinball_mean": q["mean"], "gain": gain,
                "mse_zero_frac": a["zero_pred_fraction_mean"],
                "pinball_zero_frac": q["zero_pred_fraction_mean"],
            }

    # ------------------------------------------------------- combined intervals
    hr("C1 UNDER COMBINED UNIT + SEED UNCERTAINTY")
    print("  Each bootstrap replicate resamples eval units AND draws one training seed")
    print("  per arm, so the interval carries both variance sources.\n")
    print(f"  {'comparison':<40}{'v':>2}{'dRMSE [95% CI]':>28}{'verdict':>26}")
    comb = {}
    for variant in ("A", "B"):
        y_ev = b.target("eval", variant)
        lg = preds.load(f"lgbm_summary__{variant}__L{L}__eval")
        # LightGBM point predictions are deterministic given its seed; its seed
        # sensitivity is measured separately in seed_study_lgbm (a LightGBM-only
        # process). Here it enters as a single-seed reference.
        for arm in ("tsfm", "tsfm_q", "control_randproj", "control_randproj_q"):
            d = combined_paired_ci(lg.unit, y_ev, store[f"{arm}__{variant}"], [lg.pred])
            comb[f"{arm}_minus_lgbm_summary__{variant}"] = d
            ds = "%+.2f [%+.2f, %+.2f]" % (d["point_mean_over_seeds"], d["ci_lo"], d["ci_hi"])
            v = ("no significant difference" if not d["excludes_zero"]
                 else ("arm WORSE" if d["point_mean_over_seeds"] > 0 else "arm BETTER"))
            print(f"  {arm + ' - lgbm_summary':<40}{variant:>2}{ds:>28}{v:>26}")
        # randproj vs tsfm, the attribution question, now seed-aware on both sides
        for a_arm, b_arm in (("control_randproj", "tsfm"),
                             ("control_randproj_q", "tsfm_q")):
            d = combined_paired_ci(lg.unit, y_ev, store[f"{a_arm}__{variant}"],
                                   store[f"{b_arm}__{variant}"])
            comb[f"{a_arm}_minus_{b_arm}__{variant}"] = d
            ds = "%+.2f [%+.2f, %+.2f]" % (d["point_mean_over_seeds"], d["ci_lo"], d["ci_hi"])
            v = ("indistinguishable" if not d["excludes_zero"]
                 else ("control WORSE" if d["point_mean_over_seeds"] > 0
                       else "control BETTER"))
            print(f"  {a_arm + ' - ' + b_arm:<40}{variant:>2}{ds:>28}{v:>26}")
    report["combined_paired"] = comb

    (C.RESULTS / "seed_study.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n")
    print("\nwrote results/seed_study.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
