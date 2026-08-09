"""M6: our baseline under the standard FD001 protocol, for external comparability.

Every other number in this study uses our by-engine protocol: 50 training engines, with
20 reserved for conformal calibration and 30 held out, and 15 stratified truncation points
per evaluation engine. That protocol is right for C2/C3 but it is *not* the protocol the
published FD001 literature uses, so our numbers are not directly comparable to theirs.

This script runs the one configuration that is comparable, and nothing else:

  * train on **all 100** engines of train_FD001;
  * predict once per test engine, at its last observed cycle;
  * score against RUL_FD001.txt with the piecewise-linear cap at 125.

That is the standard FD001 test protocol. The resulting number exists purely so a reader
can situate our baseline against published work using a figure we measured ourselves,
rather than us quoting numbers from papers we could not verify from primary sources.

**No claim in the paper depends on this run.** It trains on engines that are held out
everywhere else, so it must never be mixed with any C1/C2/C3 result.

LightGBM-only process. Run: uv run python -m src.standard_protocol
"""

from __future__ import annotations

import json
import sys

import lightgbm as lgb  # first import in this process, before anything torch-linked
import numpy as np

from . import cmapss, config as C, features as F, metrics, preprocess, windows as W
from .ompguard import assert_single_omp_runtime
from .phase1 import hr


def main() -> int:
    assert_single_omp_runtime("standard_protocol")
    L = C.LOOKBACK
    hr("M6: STANDARD FD001 PROTOCOL (all 100 training engines) -- FOR COMPARABILITY ONLY")
    print("  Not used by any claim in the paper. Trains on engines held out elsewhere.\n")

    train_df = cmapss.load_train()
    cmapss.integrity_checks(train_df, "train_FD001")
    test_df, test_rul = cmapss.load_official_test()
    cmapss.integrity_checks(test_df, "test_FD001")

    # Preprocessing fitted on the whole training file, as the standard protocol allows.
    pre = preprocess.Preprocessor(preprocess.candidate_feature_columns()).fit(train_df)
    lengths = cmapss.trajectory_lengths(train_df)
    idx_tr, _ = W.dense_truncations(lengths, L)
    ex_tr = W.WindowExtractor(pre.transform_frame(train_df), pre.kept_cols, L)
    ex_tr.assert_no_lookahead(idx_tr)
    Xtr = F.summary_features(ex_tr.batch(idx_tr))

    tl = cmapss.trajectory_lengths(test_df)
    keep = [u for u in tl.index if tl.loc[u] >= L]
    ex_te = W.WindowExtractor(pre.transform_frame(test_df), pre.kept_cols, L)
    Xte = np.stack([ex_te.one(int(u), int(tl.loc[u])) for u in keep])
    Xte = F.summary_features(Xte)
    y_te = np.minimum(np.array([int(test_rul.loc[u]) for u in keep]), C.RUL_CAP)

    print(f"  train: {len(idx_tr):,} windows from {train_df['unit'].nunique()} engines "
          f"(all of them)")
    print(f"  test:  {len(keep)} engines, one window each at the last observed cycle "
          f"({len(tl) - len(keep)} dropped as shorter than L={L})")
    print(f"  target: piecewise-linear RUL capped at {C.RUL_CAP}\n")

    out = {}
    for variant, y_fn in (("B", lambda r: np.minimum(r, C.RUL_CAP)),):
        y_tr = y_fn(idx_tr.rul.astype(float))
        model = lgb.LGBMRegressor(objective="regression",
                                  random_state=C.SEEDS["lightgbm"], **C.LGBM_PARAMS)
        model.fit(Xtr, y_tr)
        pred = np.clip(model.predict(Xte), 0, C.RUL_CAP)
        pm = metrics.point_metrics(y_te, pred)
        # One test engine = one point, so units are singletons and the clustered
        # bootstrap here reduces to an ordinary bootstrap over engines.
        ci = metrics.metric_ci(np.arange(len(y_te)), y_te, pred, "rmse")
        out[variant] = {**pm, "rmse_ci_lo": ci["ci_lo"], "rmse_ci_hi": ci["ci_hi"],
                        "n_test_engines": len(keep), "n_train_windows": len(idx_tr),
                        "lookback": L, "cap": C.RUL_CAP,
                        "features": "7 per-channel window statistics",
                        "protocol": "standard FD001: all 100 train engines, "
                                    "one prediction per test engine at its last cycle"}
        print(f"  LightGBM window-summary, standard protocol, cap {C.RUL_CAP}:")
        print(f"    RMSE = {pm['rmse']:.2f} [{ci['ci_lo']:.2f}, {ci['ci_hi']:.2f}]"
              f"   MAE = {pm['mae']:.2f}   prognostics score/n = {pm['score_mean']:.2f}")
    (C.RESULTS / "standard_protocol.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\nwrote results/standard_protocol.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
