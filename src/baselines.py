"""Phase 2 baselines: trivial marginal, window-summary LightGBM, raw-window LightGBM.

All three consume `dataset.Bundle` windows, so they see exactly the preprocessing
and exactly the windows the TSFM arm will see. Hyperparameters are fixed in
`config.LGBM_PARAMS`; nothing here searches.

Every arm needs a *calibration-split* prediction as well as an eval prediction,
because Phase 5's split-conformal step scores on the calibration units. Base models
are fitted on the training split only and never see calibration labels during
fitting -- they only predict on them.
"""

from __future__ import annotations

# Imported at module scope, not lazily inside the fit helpers: LightGBM must be the
# only OpenMP-linked library in this process (see src/ompguard.py). Importing it
# here makes any accidental co-import of torch fail fast at the phase guard rather
# than segfault mid-fit.
import lightgbm as lgb

import time

import numpy as np

from . import config as C, metrics, preds
from .dataset import OFFICIAL_SPLIT, Bundle


# ------------------------------------------------------------------ features
# Definitions live in features.py so the ridge probe can reuse them without importing
# LightGBM; re-exported here for readability of the arms below.
from .features import (  # noqa: E402
    SUMMARY_STATS, raw_features, raw_feature_names, summary_features,
    summary_feature_names,
)


# ------------------------------------------------------------------ arms


def fit_trivial(bundle: Bundle, variant: str) -> dict:
    """Empirical marginal of the training-split target, ignoring all features.

    The point estimate is the training median, which is the natural point summary
    of the predicted distribution and is consistent with the empirical quantiles
    used for its intervals. The training *mean* would score marginally better on
    RMSE, so we record that too rather than let the trivial arm look weaker than
    it is by an arbitrary choice.
    """
    y_tr = bundle.target("train", variant)
    q = np.quantile(y_tr, C.QUANTILE_LEVELS)
    return {
        "quantiles": q,
        "point": float(np.median(y_tr)),
        "train_mean": float(np.mean(y_tr)),
        "n_train": int(len(y_tr)),
    }


def _lgbm(objective: str, seed: int, **extra):
    return lgb.LGBMRegressor(
        objective=objective, random_state=seed, **C.LGBM_PARAMS, **extra
    )


def fit_lgbm(
    X_tr: np.ndarray, y_tr: np.ndarray, with_quantiles: bool, seed: int
) -> dict:
    """One L2 model for the point estimate, plus one model per quantile level."""
    models = {"point": _lgbm("regression", seed).fit(X_tr, y_tr)}
    if with_quantiles:
        for lv in C.QUANTILE_LEVELS:
            models[lv] = _lgbm("quantile", seed, alpha=float(lv)).fit(X_tr, y_tr)
    return models


def predict_lgbm(models: dict, X: np.ndarray, with_quantiles: bool):
    point = models["point"].predict(X)
    if not with_quantiles:
        return point, None
    q = np.column_stack([models[lv].predict(X) for lv in C.QUANTILE_LEVELS])
    return point, q


# ------------------------------------------------------------------ driver

FEATURISERS = {
    "lgbm_summary": (summary_features, summary_feature_names),
    "lgbm_raw": (raw_features, raw_feature_names),
}


def run_all(
    bundle: Bundle, variants=("A", "B"), with_quantiles: bool = True, log=print
) -> dict:
    """Fit and predict all three baselines for each variant.

    Returns a report dict; predictions are written via `preds.Prediction.save()`
    for both the eval and calib splits.
    """
    L = bundle.lookback
    report: dict = {"lookback": L, "arms": {}, "lgbm_params": dict(C.LGBM_PARAMS)}

    emit_splits = ("eval", "calib", OFFICIAL_SPLIT)
    win = {s: bundle.windows(s) for s in ("train", "calib", "eval", OFFICIAL_SPLIT)}
    log(f"  windows materialised: " + " ".join(f"{s}{win[s].shape}" for s in win))

    feats: dict[str, dict[str, np.ndarray]] = {}
    for arm, (fn, namer) in FEATURISERS.items():
        t0 = time.perf_counter()
        feats[arm] = {s: fn(win[s]) for s in win}
        names = (
            namer(bundle.feature_cols)
            if arm == "lgbm_summary"
            else namer(bundle.feature_cols, L)
        )
        if feats[arm]["train"].shape[1] != len(names):
            raise ValueError(f"{arm}: {feats[arm]['train'].shape[1]} cols vs {len(names)} names")
        log(f"  {arm}: {feats[arm]['train'].shape[1]} features "
            f"({time.perf_counter()-t0:.1f}s)")
        report["arms"].setdefault(arm, {})["n_features"] = int(feats[arm]["train"].shape[1])
        report["arms"][arm]["feature_names_head"] = names[:8]

    for variant in variants:
        y = {s: bundle.target(s, variant) for s in
             ("train", "calib", "eval", OFFICIAL_SPLIT)}

        # -------------------------------------------------- trivial marginal
        t0 = time.perf_counter()
        triv = fit_trivial(bundle, variant)
        for split in emit_splits:
            n = len(y[split])
            p = preds.Prediction(
                arm="trivial", variant=variant, lookback=L, split=split,
                unit=bundle.units(split), t=bundle.idx[split].t,
                rul_true_uncapped=bundle.idx[split].rul,
                y_true=y[split],
                pred=np.full(n, triv["point"]),
                quantiles=np.tile(triv["quantiles"], (n, 1)) if with_quantiles else None,
                meta={
                    "point_rule": "training-split median",
                    "train_median": triv["point"],
                    "train_mean": triv["train_mean"],
                    "n_train": triv["n_train"],
                },
            )
            p.save()
            if split == "eval":
                pm = p.point_metrics()
                report["arms"].setdefault("trivial", {})[variant] = {
                    **pm,
                    "fit_seconds": time.perf_counter() - t0,
                    "train_median": triv["point"],
                    "train_mean": triv["train_mean"],
                    "rmse_if_mean_predictor": float(
                        np.sqrt(np.mean((triv["train_mean"] - y["eval"]) ** 2))
                    ),
                }
                log(f"  [{variant}] trivial      RMSE={pm['rmse']:7.2f} "
                    f"MAE={pm['mae']:7.2f} score/n={pm['score_mean']:9.2f} "
                    f"bias={pm['bias_mean_signed_error']:+7.2f}")

        # -------------------------------------------------- LightGBM arms
        for arm in FEATURISERS:
            t0 = time.perf_counter()
            models = fit_lgbm(
                feats[arm]["train"], y["train"], with_quantiles, C.SEEDS["lightgbm"]
            )
            fit_s = time.perf_counter() - t0
            for split in emit_splits:
                point, q = predict_lgbm(models, feats[arm][split], with_quantiles)
                p = preds.Prediction(
                    arm=arm, variant=variant, lookback=L, split=split,
                    unit=bundle.units(split), t=bundle.idx[split].t,
                    rul_true_uncapped=bundle.idx[split].rul,
                    y_true=y[split], pred=point, quantiles=q,
                    meta={"n_features": int(feats[arm][split].shape[1]),
                          "n_models": len(models), "fit_seconds": fit_s},
                )
                p.save()
                if split == "eval":
                    pm = p.point_metrics()
                    report["arms"][arm][variant] = {
                        **pm, "fit_seconds": fit_s,
                        "quantile_crossing_rate": p.meta.get("quantile_crossing_rate"),
                    }
                    log(f"  [{variant}] {arm:<12} RMSE={pm['rmse']:7.2f} "
                        f"MAE={pm['mae']:7.2f} score/n={pm['score_mean']:9.2f} "
                        f"bias={pm['bias_mean_signed_error']:+7.2f} "
                        f"({fit_s:.1f}s, {len(models)} models)")
    return report
