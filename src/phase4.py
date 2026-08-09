"""Phase 4: contamination / attribution controls.

C-MAPSS is one of the most widely mirrored public datasets in existence and TSFM
pretraining corpora are broad, so we cannot rule out that Chronos-2 has seen these
trajectories. None of the three controls below is a contamination *test* -- direct
testing is impossible for a frozen third-party checkpoint whose corpus we cannot
inspect. They bound what the pretrained representation can be credited with.

1. shuffled label     head retrained on permuted training labels. Should collapse to
                      the trivial marginal; anything better means the pipeline leaks.
2. channel scramble    sensor channels permuted within each window before extraction,
                      labels intact.
3. random projection   backbone replaced by a fixed random linear map of the flattened
                      window to the identical output dimensionality, identical head.

IMPORTANT REINTERPRETATION OF CONTROL 2 (see notes/decisions.md D-011, D-015).
Chronos-2 is exactly permutation-equivariant across variates, which we measured. So
permuting sensor channels leaves the *set* of per-variate embeddings bit-identical and
only changes their order in the concatenated head input. Control 2 therefore CANNOT
test whether the backbone uses sensor identity -- it provably does not, by
construction. What it tests is whether the *pipeline*, head included, relies on a
stable sensor-to-slot mapping. That is still worth measuring, but it is a different
claim from the one the brief anticipated, and the paper must say so.

Run: uv run python -m src.phase4
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from . import config as C, dataset, embed, head, manifest, metrics, preds
from .dataset import OFFICIAL_SPLIT
from .features import random_projection
from .ompguard import assert_single_omp_runtime
from .phase1 import hr
from .phase3 import train_and_emit
from .seeding import seed_everything


def scramble_channels(win: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    """Independently permute the channel axis of every window.

    Per-window permutations (not one global permutation) follow the brief's "within
    each window", and are the stronger version: they destroy any stable
    sensor-to-slot mapping rather than merely relabelling it consistently.
    """
    n, L, D = win.shape
    rng = np.random.default_rng(seed)
    out = np.empty_like(win)
    perms = np.empty((n, D), dtype=np.int16)
    n_identity = 0
    for i in range(n):
        p = rng.permutation(D)
        perms[i] = p
        out[i] = win[i][:, p]
        if np.array_equal(p, np.arange(D)):
            n_identity += 1
    return out, {
        "n_windows": int(n),
        "n_channels": int(D),
        "scheme": "independent uniform permutation of the channel axis per window",
        "seed": int(seed),
        "n_identity_permutations": int(n_identity),
    }


def main() -> int:
    assert_single_omp_runtime("phase4 / controls")
    seed_everything()
    report: dict = {"phase": 4, "controls": {}}

    b = dataset.build(C.LOOKBACK, verbose=True)
    d_out = embed.reduced_dim(b.n_features)

    # Reference numbers, loaded from disk so nothing is recomputed inconsistently.
    ref = {}
    for arm in ("trivial", "lgbm_summary", "tsfm"):
        for v in ("A", "B"):
            p = preds.load(f"{arm}__{v}__L{C.LOOKBACK}__eval")
            ref[f"{arm}__{v}"] = metrics.point_metrics(p.y_true, p.pred)
    print(f"  references (L={C.LOOKBACK}, eval RMSE): "
          + "  ".join(f"{k}={v['rmse']:.2f}" for k, v in ref.items()))
    report["references"] = ref

    # -------------------------------------------------------- cached embeddings
    hr("0. EMBEDDINGS (reused from Phase 3 cache where possible)")
    feats_main = {}
    for split in ("train", "calib", "eval", OFFICIAL_SPLIT):
        path = embed.cache_path(C.LOOKBACK, split, "main")
        if not path.exists():
            raise SystemExit(f"missing {path}; run `python -m src.phase3` first")
        feats_main[split] = embed.load_cache(path)[C.REDUCTION_PRIMARY]
        print(f"    {split}: {feats_main[split].shape} (tag=main)")

    # ============================================== CONTROL 1: shuffled labels
    hr("CONTROL 1: SHUFFLED LABELS")
    print("  The head is retrained on a random permutation of the TRAINING labels.")
    print("  Evaluation labels are untouched. Expected: collapse to the trivial")
    print("  marginal. Anything better than trivial means the pipeline leaks.\n")
    c1 = {}
    for variant in ("A", "B"):
        y_tr = b.target("train", variant)
        rng = np.random.default_rng(C.SEEDS["head_shuffle_control"])
        perm = rng.permutation(len(y_tr))
        y_shuf = y_tr[perm]
        frac_moved = float(np.mean(perm != np.arange(len(y_tr))))
        # Sanity: the shuffled labels must be a genuine permutation of the originals.
        assert np.array_equal(np.sort(y_shuf), np.sort(y_tr)), "not a permutation"
        assert frac_moved > 0.99, f"permutation barely moved anything: {frac_moved}"

        class _Shuf:
            """Bundle view whose training target is permuted; eval/calib untouched."""
            def __init__(self, inner, y):
                self._i, self._y = inner, y
                self.lookback, self.idx = inner.lookback, inner.idx
            def target(self, split, v):
                return self._y if split == "train" else self._i.target(split, v)
            def windows(self, split):
                return self._i.windows(split)
            def units(self, split):
                return self._i.units(split)

        res, _ = train_and_emit(
            "control_shufflabel", feats_main, _Shuf(b, y_shuf), variant, "point",
            C.SEEDS["head_init"], C.LOOKBACK, False,
            extra_meta={"control": "shuffled_labels",
                        "shuffle_seed": C.SEEDS["head_shuffle_control"],
                        "frac_labels_moved": frac_moved}, log=None,
        )
        pm = res["point_metrics"]
        c1[variant] = {**res, "frac_labels_moved": frac_moved}
        tr, ts = ref[f"trivial__{variant}"]["rmse"], ref[f"tsfm__{variant}"]["rmse"]
        print(f"  variant {variant}: RMSE={pm['rmse']:.2f}  "
              f"(trivial={tr:.2f}, intact tsfm={ts:.2f})   "
              f"train loss {res['history']['loss_first']:.0f} -> "
              f"{res['history']['loss_last']:.0f}")
        verdict = ("COLLAPSED to trivial as required" if pm["rmse"] >= tr * 0.95
                   else "DID NOT COLLAPSE -- investigate leakage")
        print(f"    -> {verdict}")
        c1[variant]["verdict"] = verdict
    report["controls"]["shuffled_labels"] = c1

    # ============================================ CONTROL 2: channel scramble
    hr("CONTROL 2: CHANNEL SCRAMBLE")
    print("  Sensor channels are independently permuted within every window before")
    print("  extraction; labels are intact. Because Chronos-2 is exactly")
    print("  permutation-equivariant across variates (measured, D-011), the SET of")
    print("  per-variate embeddings is unchanged -- only their order in the head")
    print("  input changes. This control therefore tests whether the PIPELINE relies")
    print("  on a stable sensor-to-slot mapping, not whether the backbone does.\n")
    pipe, model, binfo = embed.load_backbone()
    scram_meta, feats_scram = {}, {}
    for split in ("train", "calib", "eval", OFFICIAL_SPLIT):
        path = embed.cache_path(C.LOOKBACK, split, "scramble")
        if path.exists():
            feats_scram[split] = embed.load_cache(path)[C.REDUCTION_PRIMARY]
            print(f"    {split}: loaded cache {path.name}")
            continue
        w, meta = scramble_channels(
            b.windows(split), C.SEEDS["channel_scramble_control"] + hash(split) % 1000
        )
        scram_meta[split] = meta
        print(f"    {split}: scrambled {meta['n_windows']} windows "
              f"({meta['n_identity_permutations']} identity perms), extracting ...")
        f, em = embed.extract(pipe, w, log=None)
        embed.save_cache(path, f)
        feats_scram[split] = f[C.REDUCTION_PRIMARY]
        print(f"      {em['wall_seconds']:.0f}s ({em['ms_per_window']:.1f} ms/window)")

    # Direct check of the equivariance claim on real data: the sorted embedding
    # values must be preserved even though the head input order is destroyed.
    a = np.sort(feats_main["eval"][0].reshape(b.n_features, -1).sum(axis=1))
    z = np.sort(feats_scram["eval"][0].reshape(b.n_features, -1).sum(axis=1))
    print(f"\n    equivariance check on eval window 0: max abs diff between the "
          f"SORTED per-variate embedding sums = {np.abs(a - z).max():.2e}")
    print(f"    (near zero confirms the backbone output is a permutation, not new "
          f"information)")
    report["controls"]["channel_scramble_equivariance_check"] = float(np.abs(a - z).max())

    c2 = {}
    for variant in ("A", "B"):
        res, _ = train_and_emit(
            "control_chanscramble", feats_scram, b, variant, "point",
            C.SEEDS["head_init"], C.LOOKBACK, False,
            extra_meta={"control": "channel_scramble"}, log=None,
        )
        pm = res["point_metrics"]
        c2[variant] = res
        ts = ref[f"tsfm__{variant}"]["rmse"]
        tr = ref[f"trivial__{variant}"]["rmse"]
        degr = 100 * (pm["rmse"] - ts) / ts
        print(f"  variant {variant}: RMSE={pm['rmse']:.2f}  vs intact tsfm={ts:.2f} "
              f"({degr:+.1f}%)  trivial={tr:.2f}")
    report["controls"]["channel_scramble"] = c2
    report["controls"]["channel_scramble_windows"] = scram_meta

    # ========================================== CONTROL 3: random projection
    hr("CONTROL 3: RANDOM PROJECTION (the sharpest control)")
    print(f"  The backbone is replaced by a FIXED Gaussian random matrix mapping the")
    print(f"  flattened window ({C.LOOKBACK}x{b.n_features}="
          f"{C.LOOKBACK*b.n_features}) to the identical output dimensionality "
          f"({d_out:,}),")
    print(f"  then the IDENTICAL head is trained. If this matches the TSFM arm, the")
    print(f"  gain came from the head and the window, not from pretraining.")
    print(f"  Caveat stated in the paper: the projection is linear, so this arm's")
    print(f"  effective function class is a linear map of the window plus the head's")
    print(f"  nonlinearity -- strictly weaker than the TSFM arm's nonlinear features.")
    print(f"  A tie is therefore a strong result; a loss is weaker evidence.\n")
    c3 = {}
    t0 = time.perf_counter()
    fr = {s: random_projection(b.windows(s), d_out, C.SEEDS["random_projection_control"])
          for s in ("train", "calib", "eval", OFFICIAL_SPLIT)}
    print(f"  projected in {time.perf_counter()-t0:.1f}s; "
          f"shapes { {k: v.shape for k, v in fr.items()} }")
    for variant in ("A", "B"):
        for mode, wq in (("point", False), ("quantile", True)):
            arm = "control_randproj" if mode == "point" else "control_randproj_q"
            res, _ = train_and_emit(
                arm, fr, b, variant, mode, C.SEEDS["head_init"], C.LOOKBACK, wq,
                extra_meta={"control": "random_projection",
                            "projection_seed": C.SEEDS["random_projection_control"],
                            "d_in_flat": C.LOOKBACK * b.n_features}, log=None,
            )
            pm = res["point_metrics"]
            c3[f"{arm}__{variant}"] = res
            ts = ref[f"tsfm__{variant}"]["rmse"]
            print(f"  {arm:<22} variant {variant}: RMSE={pm['rmse']:.2f} "
                  f"MAE={pm['mae']:.2f}  vs intact tsfm={ts:.2f} "
                  f"({100*(pm['rmse']-ts)/ts:+.1f}%)")
    # L=80 secondary for the random projection (D-009).
    b80 = dataset.build(C.LOOKBACK_SECONDARY)
    fr80 = {s: random_projection(b80.windows(s), d_out,
                                 C.SEEDS["random_projection_control"])
            for s in ("train", "calib", "eval", OFFICIAL_SPLIT)}
    for variant in ("A", "B"):
        res, _ = train_and_emit(
            "control_randproj", fr80, b80, variant, "point", C.SEEDS["head_init"],
            C.LOOKBACK_SECONDARY, False,
            extra_meta={"control": "random_projection"}, log=None,
        )
        pm = res["point_metrics"]
        c3[f"control_randproj__L80__{variant}"] = res
        print(f"  control_randproj  L=80 variant {variant}: RMSE={pm['rmse']:.2f} "
              f"MAE={pm['mae']:.2f}")
    report["controls"]["random_projection"] = c3

    # -------------------------------------------------------------- freeze recheck
    hr("FREEZE VERIFICATION AFTER CONTROL HEADS")
    after = embed.param_hash(model)
    ok = after == binfo["param_hash_at_load"]
    print(f"  backbone param hash unchanged -> {'PASSED' if ok else 'FAILED'}")
    if not ok:
        return 2

    (C.RESULTS / "phase4_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    manifest.merge_into("phase4", report)
    print(f"\nwrote results/phase4_report.json "
          f"({len(preds.available())} prediction files total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
