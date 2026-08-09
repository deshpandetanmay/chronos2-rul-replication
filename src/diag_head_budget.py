"""Diagnostic: is the negative C1 an artifact of the paper's head training budget?

NOT hyperparameter search and NOT used for any reported arm. The primary TSFM arm
stays at the paper's specified budget (m=32, 50 epochs). This script exists solely
to answer one question a reviewer will ask about a negative replication: was the
head underfit, so that the comparison was never fair?

The head's training loss is still falling monotonically at epoch 50, which is the
symptom that makes the question live.

Run: uv run python -m src.diag_head_budget
"""
from __future__ import annotations
import json, sys
import numpy as np
from . import config as C, dataset, embed, head, metrics, preds
from .ompguard import assert_single_omp_runtime
from .phase1 import hr
from .seeding import seed_everything

def main() -> int:
    assert_single_omp_runtime("diag_head_budget")
    seed_everything()
    hr("DIAGNOSTIC: head capacity / training budget vs the negative C1")
    b = dataset.build(C.LOOKBACK)
    feats = {}
    pipe = None
    for split in ("train", "eval"):
        path = embed.cache_path(C.LOOKBACK, split, "main")
        if not path.exists():
            raise SystemExit(f"missing embedding cache {path}; run phase3 first")
        feats[split] = embed.load_cache(path)[C.REDUCTION_PRIMARY]

    ref = preds.load(f"lgbm_summary__A__L{C.LOOKBACK}__eval")
    ref_rmse = metrics.point_metrics(ref.y_true, ref.pred)["rmse"]
    base = preds.load(f"tsfm__A__L{C.LOOKBACK}__eval")
    base_rmse = metrics.point_metrics(base.y_true, base.pred)["rmse"]
    print(f"  reference: lgbm_summary variant A RMSE = {ref_rmse:.2f}")
    print(f"  as-reported tsfm variant A (m={C.HEAD_HIDDEN}, {C.HEAD_EPOCHS} ep) "
          f"RMSE = {base_rmse:.2f}\n")
    print(f"  {'m':>6}{'epochs':>8}{'final loss':>12}{'eval RMSE':>11}"
          f"{'vs lgbm_summary':>17}{'train s':>9}")

    y_tr = b.target("train", "A"); y_ev = b.target("eval", "A")
    rows = []
    for m in (32, 256):
        for ep in (50, 200):
            C.HEAD_HIDDEN = m
            net, sc, hist = head.train_head(feats["train"], y_tr, "point",
                                           C.SEEDS["head_init"], epochs=ep)
            pt, _ = head.predict_head(net, sc, feats["eval"], "point")
            r = metrics.point_metrics(y_ev, pt)["rmse"]
            rows.append({"m": m, "epochs": ep, "final_loss": hist["loss_last"],
                         "eval_rmse": r, "delta_vs_ref": r - ref_rmse,
                         "params": hist["params"]["total"],
                         "train_seconds": hist["train_seconds"]})
            print(f"  {m:>6}{ep:>8}{hist['loss_last']:>12.2f}{r:>11.2f}"
                  f"{r-ref_rmse:>+17.2f}{hist['train_seconds']:>9.0f}")
    C.HEAD_HIDDEN = 32
    (C.RESULTS / "diag_head_budget.json").write_text(
        json.dumps({"reference_lgbm_summary_A_rmse": ref_rmse,
                    "reported_tsfm_A_rmse": base_rmse, "grid": rows}, indent=2) + "\n")
    print("\n  Positive delta = still worse than the window-summary LightGBM baseline.")
    print("  wrote results/diag_head_budget.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())
