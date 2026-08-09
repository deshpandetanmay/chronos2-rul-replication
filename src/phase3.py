"""Phase 3: frozen Chronos-2 embeddings arm. Produces the CHECKPOINT 2 evidence.

Run: uv run python -m src.phase3
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from . import config as C, dataset, embed, head, manifest, metrics, preds
from .dataset import OFFICIAL_SPLIT
from .ompguard import assert_single_omp_runtime
from .phase1 import hr
from .seeding import seed_everything

ARM = "tsfm"


def train_and_emit(
    arm: str, feats: dict, bundle, variant: str, mode: str, seed: int,
    lookback: int, with_quantiles: bool, extra_meta: dict | None = None, log=print,
) -> dict:
    """Train one head and write eval + calib predictions."""
    emit = ("eval", "calib") + ((OFFICIAL_SPLIT,) if feats.get(OFFICIAL_SPLIT) is not None else ())
    y = {s: bundle.target(s, variant) for s in ("train",) + emit}
    net, scaler, hist = head.train_head(
        feats["train"], y["train"], mode=mode, seed=seed, log=log
    )
    out = {"history": {k: v for k, v in hist.items() if k != "loss_trace"},
           "loss_trace_head": hist["loss_trace"][:3],
           "loss_trace_tail": hist["loss_trace"][-3:]}
    for split in emit:
        point, q = head.predict_head(net, scaler, feats[split], mode)
        p = preds.Prediction(
            arm=arm, variant=variant, lookback=lookback, split=split,
            unit=bundle.units(split), t=bundle.idx[split].t,
            rul_true_uncapped=bundle.idx[split].rul,
            y_true=y[split], pred=point,
            quantiles=q if with_quantiles else None,
            meta={"mode": mode, "head_params": hist["params"]["total"],
                  "d_in": int(feats[split].shape[1]),
                  **(extra_meta or {})},
        )
        p.save()
        if split == "eval":
            out["point_metrics"] = p.point_metrics()
            out["quantile_crossing_rate"] = p.meta.get("quantile_crossing_rate")
    return out, net


def main() -> int:
    # torch-only process; LightGBM here would segfault during embed (src/ompguard.py).
    assert_single_omp_runtime("phase3 / tsfm arm")
    seed_everything()
    report: dict = {"phase": 3, "arm": ARM}

    # -------------------------------------------------------------- backbone
    hr("1. BACKBONE LOAD AND FREEZE VERIFICATION")
    pipe, model, binfo = embed.load_backbone()
    print(f"  checkpoint      {binfo['checkpoint']} @ {binfo['revision'][:12]}")
    print(f"  pipeline class  {binfo['pipeline_class']}   device {binfo['device']}")
    print(f"  params          {binfo['n_params']:,}  d_model={binfo['d_model']}  "
          f"layers={binfo['num_layers']}")
    print(f"  ASSERTION 1     every backbone parameter has requires_grad=False "
          f"-> PASSED (trainable={binfo['n_trainable_params']})")
    print(f"  param hash @load  {binfo['param_hash_at_load']}")
    report["backbone"] = binfo

    # Negative control on the freeze assertion.
    import torch
    probe = next(iter(model.parameters()))
    probe.requires_grad_(True)
    n_train_now = sum(1 for p in model.parameters() if p.requires_grad)
    probe.requires_grad_(False)
    print(f"  negative control  un-freezing one tensor makes {n_train_now} params "
          f"trainable, and the check would fail; re-frozen "
          f"({sum(1 for p in model.parameters() if p.requires_grad)} trainable)")

    # -------------------------------------------------------------- extraction
    hr("2. EMBEDDING EXTRACTION (frozen, no gradients)")
    b = dataset.build(C.LOOKBACK, verbose=True)
    print(f"  reductions: {embed.REDUCTIONS}  "
          f"(primary={C.REDUCTION_PRIMARY}, ablation={C.REDUCTION_ABLATION})")
    feats_by_red: dict[str, dict[str, np.ndarray]] = {r: {} for r in embed.REDUCTIONS}
    extract_meta = {}
    t_all = time.perf_counter()
    for split in ("train", "calib", "eval", OFFICIAL_SPLIT):
        f, meta, cached = embed.get_or_extract(pipe, b, split, tag="main")
        extract_meta[split] = meta
        for r in embed.REDUCTIONS:
            feats_by_red[r][split] = f[r]
    # Sum the MEASURED per-split extraction time, not the loop timer: on a cache hit
    # the loop takes ~0.4s and reporting that as "extraction wall-clock" is simply
    # false. The measured values come from the .meta.json sidecars (D-019).
    measured = sum(m.get("wall_seconds", 0.0) for m in extract_meta.values())
    loop_elapsed = time.perf_counter() - t_all
    all_cached = all(m.get("from_cache") for m in extract_meta.values())
    n_win = sum(len(b.idx[s]) for s in ("train", "calib", "eval", OFFICIAL_SPLIT))
    print(f"  extraction wall-clock: {measured:.1f}s measured over {n_win:,} windows"
          + (f"  (this run reused cached embeddings in {loop_elapsed:.1f}s)"
             if all_cached else ""))
    report["extraction"] = {
        "per_split": extract_meta,
        "measured_wall_seconds": measured,
        "loop_elapsed_seconds": loop_elapsed,
        "all_from_cache_this_run": all_cached,
        "n_windows": n_win,
    }

    d_in = feats_by_red[C.REDUCTION_PRIMARY]["train"].shape[1]
    m0 = extract_meta.get("train", {})
    print(f"  embedding dimensionality: per-item embed shape "
          f"{m0.get('per_item_embed_shape')} = (D={m0.get('n_variates')}, "
          f"P+2={m0.get('n_patches')}+2={m0.get('n_tokens')}, h={m0.get('d_model')})")
    print(f"  token axis layout: {m0.get('token_axis_layout')}")
    print(f"  head input dim after reduction: D*h = {b.n_features}*"
          f"{C.BACKBONE_D_MODEL} = {d_in:,}")
    report["embedding_dim"] = {"per_item": m0.get("per_item_embed_shape"),
                               "head_input_dim": int(d_in),
                               "n_variates": b.n_features,
                               "d_model": C.BACKBONE_D_MODEL}

    # -------------------------------------------------------------- heads
    hr("3. HEAD TRAINING (only trainable component)")
    probe_net = head.make_head(d_in, "point", C.SEEDS["head_init"])
    pc = head.head_param_count(probe_net)
    print(f"  architecture: Linear({d_in:,} -> {C.HEAD_HIDDEN}) -> ReLU -> "
          f"Dropout({C.HEAD_DROPOUT}) -> Linear({C.HEAD_HIDDEN} -> n_out) -> ReLU")
    print(f"  point head parameters: {pc['total']:,}  {pc['per_layer']}")
    pcq = head.head_param_count(head.make_head(d_in, "quantile", C.SEEDS["head_init"]))
    print(f"  quantile head parameters: {pcq['total']:,} "
          f"({len(C.QUANTILE_LEVELS)} outputs)")
    print(f"  optimiser: Adam lr={C.HEAD_LR}, {C.HEAD_EPOCHS} epochs, "
          f"batch {C.HEAD_BATCH}; loss = MSE (point) / pinball (quantile)")
    print(f"  backbone params {binfo['n_params']:,} vs head {pc['total']:,} "
          f"= {binfo['n_params']/pc['total']:.0f}x")
    report["head"] = {"point": pc, "quantile": pcq, "hidden": C.HEAD_HIDDEN,
                      "dropout": C.HEAD_DROPOUT, "epochs": C.HEAD_EPOCHS,
                      "lr": C.HEAD_LR, "batch": C.HEAD_BATCH,
                      "backbone_to_head_ratio": binfo["n_params"] / pc["total"]}

    runs: dict = {}
    for variant in ("A", "B"):
        for mode, with_q in (("point", False), ("quantile", True)):
            arm = ARM if mode == "point" else f"{ARM}_q"
            print(f"\n  --- {arm} variant {variant} "
                  f"(reduction={C.REDUCTION_PRIMARY}) ---")
            res, _ = train_and_emit(
                arm, feats_by_red[C.REDUCTION_PRIMARY], b, variant, mode,
                C.SEEDS["head_init"], C.LOOKBACK, with_q,
                extra_meta={"reduction": C.REDUCTION_PRIMARY}, log=print,
            )
            pm = res["point_metrics"]
            print(f"      loss {res['history']['loss_first']:.2f} -> "
                  f"{res['history']['loss_last']:.2f} "
                  f"({res['history']['train_seconds']:.0f}s)")
            print(f"      RMSE={pm['rmse']:.2f} MAE={pm['mae']:.2f} "
                  f"score/n={pm['score_mean']:.2f} "
                  f"bias={pm['bias_mean_signed_error']:+.2f}")
            runs[f"{arm}__{variant}"] = res

    # -------------------------------------------------------------- ablation
    hr("4. POOLING ABLATION (%s vs %s)" % (C.REDUCTION_PRIMARY, C.REDUCTION_ABLATION))
    print(f"  The paper does not specify pooling or layer "
          f"(source_paper.md AMBIGUITY 2/3). Primary = {C.REDUCTION_PRIMARY} "
          f"(mean over the token axis); ablation = {C.REDUCTION_ABLATION} "
          f"([REG] token only). Both keep the variate axis in fixed sensor order,")
    print(f"  which is forced: the backbone is permutation-equivariant across "
          f"variates, so any variate-symmetric pooling would make the Phase 4")
    print(f"  channel-scramble control vacuous (D-011).")
    for variant in ("A", "B"):
        arm = f"{ARM}_abl_{C.REDUCTION_ABLATION}"
        print(f"\n  --- {arm} variant {variant} ---")
        res, _ = train_and_emit(
            arm, feats_by_red[C.REDUCTION_ABLATION], b, variant, "point",
            C.SEEDS["head_init"], C.LOOKBACK, False,
            extra_meta={"reduction": C.REDUCTION_ABLATION}, log=None,
        )
        pm = res["point_metrics"]
        print(f"      RMSE={pm['rmse']:.2f} MAE={pm['mae']:.2f} "
              f"score/n={pm['score_mean']:.2f}")
        runs[f"{arm}__{variant}"] = res
    report["runs"] = runs

    # -------------------------------------------------------------- L=80 secondary
    hr("5. SECONDARY LOOK-BACK L=%d (point accuracy only, D-009)" % C.LOOKBACK_SECONDARY)
    b80 = dataset.build(C.LOOKBACK_SECONDARY, verbose=True)
    f80: dict[str, dict[str, np.ndarray]] = {C.REDUCTION_PRIMARY: {}}
    for split in ("train", "calib", "eval", OFFICIAL_SPLIT):
        f, meta, _ = embed.get_or_extract(pipe, b80, split, tag="main")
        f80[C.REDUCTION_PRIMARY][split] = f[C.REDUCTION_PRIMARY]
    for variant in ("A", "B"):
        res, _ = train_and_emit(
            ARM, f80[C.REDUCTION_PRIMARY], b80, variant, "point",
            C.SEEDS["head_init"], C.LOOKBACK_SECONDARY, False,
            extra_meta={"reduction": C.REDUCTION_PRIMARY}, log=None,
        )
        pm = res["point_metrics"]
        print(f"  L=80 {ARM} variant {variant}: RMSE={pm['rmse']:.2f} "
              f"MAE={pm['mae']:.2f} score/n={pm['score_mean']:.2f}")
        runs[f"{ARM}__L80__{variant}"] = res

    # -------------------------------------------------------------- freeze recheck
    hr("6. FREEZE VERIFICATION AFTER ALL HEAD TRAINING")
    after = embed.param_hash(model)
    same = after == binfo["param_hash_at_load"]
    print(f"  param hash @load  {binfo['param_hash_at_load']}")
    print(f"  param hash @end   {after}")
    print(f"  ASSERTION 2     backbone parameter hash unchanged -> "
          f"{'PASSED' if same else 'FAILED'}")
    trainable_after = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"  ASSERTION 3     requires_grad still False everywhere -> "
          f"{'PASSED' if not trainable_after else 'FAILED ' + str(trainable_after[:3])}")
    if not same or trainable_after:
        return 2
    report["backbone"]["param_hash_after_training"] = after
    report["backbone"]["frozen_verified"] = True

    (C.RESULTS / "phase3_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    manifest.merge_into("phase3", report)
    print(f"\nwrote results/phase3_report.json and predictions "
          f"({len(preds.available())} files total under results/preds/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
