"""Phase 1: evaluation design. Produces the CHECKPOINT 1 evidence.

Run: uv run python -m src.phase1

Emits:
  results/unit_splits.json      explicit unit-id lists per split
  results/window_index_*.parquet the (unit, truncation, label) index per split
  results/phase1_evidence.json  every number quoted at the checkpoint
  results/preprocessor.json     fitted clip/scale stats and dropped columns
  figures/fig_truncation_rul_hist.png
  figures/fig_trajectory_lengths.png
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from . import cmapss, config as C, manifest, preprocess, splits as sp, windows as W
from .seeding import seed_everything


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def fmt_dist(s: pd.Series) -> str:
    return (
        f"n={len(s)} min={s.min():.0f} p05={s.quantile(0.05):.1f} "
        f"q1={s.quantile(0.25):.1f} med={s.median():.1f} mean={s.mean():.1f} "
        f"q3={s.quantile(0.75):.1f} p95={s.quantile(0.95):.1f} max={s.max():.0f} "
        f"sd={s.std(ddof=1):.1f}"
    )


def main() -> int:
    seed_info = seed_everything()
    ev: dict = {"seeds_applied": seed_info}

    # ---------------------------------------------------------------- ingest
    hr("1. DATA INGEST AND FILE VERIFICATION")
    digests = cmapss.verify_files()
    for k, v in digests.items():
        print(f"  sha256 OK  {k}  {v}")
    ev["file_sha256"] = digests

    train_df = cmapss.load_train()
    cmapss.integrity_checks(train_df, "train_FD001")
    print("  integrity checks passed: no NaNs; cycles 1..T contiguous, "
          "strictly increasing, per unit")

    test_df, test_rul = cmapss.load_official_test()
    cmapss.integrity_checks(test_df, "test_FD001")

    ev["train_file"] = cmapss.describe(train_df, "train_FD001")
    ev["official_test_file"] = cmapss.describe(test_df, "test_FD001")
    ev["official_test_file"]["rul_after_last_cycle"] = {
        "n": int(len(test_rul)),
        "min": int(test_rul.min()),
        "median": float(test_rul.median()),
        "mean": float(test_rul.mean()),
        "max": int(test_rul.max()),
    }

    hr("2. MEASURED DATASET COUNTS (from the files, not from any paper)")
    for key in ("train_file", "official_test_file"):
        d = ev[key]
        print(f"  {d['label']}: rows={d['n_rows']:,}  units={d['n_units']}  "
              f"unit ids {d['unit_id_min']}..{d['unit_id_max']}")
        t = d["trajectory_length"]
        print(f"    trajectory length: min={t['min']} q1={t['q1']:.1f} "
              f"med={t['median']:.1f} mean={t['mean']:.2f} q3={t['q3']:.1f} "
              f"max={t['max']} sd={t['std']:.2f} sum={t['sum']:,}")
    rr = ev["official_test_file"]["rul_after_last_cycle"]
    print(f"  RUL_FD001.txt: n={rr['n']} min={rr['min']} med={rr['median']:.1f} "
          f"mean={rr['mean']:.2f} max={rr['max']}")

    train_lengths = cmapss.trajectory_lengths(train_df)

    # ---------------------------------------------------------------- lookback
    hr("3. LOOK-BACK WINDOW LENGTH")
    L = C.LOOKBACK
    n_patches = -(-L // C.BACKBONE_PATCH_SIZE)
    print(f"  L = {L} cycles")
    print(f"  Chronos-2 input_patch_size = {C.BACKBONE_PATCH_SIZE} "
          f"-> ceil({L}/{C.BACKBONE_PATCH_SIZE}) = {n_patches} input patches "
          f"-> embed token axis = {n_patches} + 2 = {n_patches + 2}")
    print(f"  shortest training trajectory = {int(train_lengths.min())} cycles, "
          f"so a {L}-cycle window fits in every unit "
          f"({int((train_lengths < L).sum())} units cannot admit one)")
    print("  Paper's setting is L=5 *hours*; C-MAPSS is indexed in flight cycles, so")
    print("  the number 5 does not transfer as a duration. See notes/decisions.md D-002.")
    ev["lookback"] = {
        "value": L,
        "unit": "flight cycles",
        "backbone_input_patch_size": C.BACKBONE_PATCH_SIZE,
        "n_input_patches": n_patches,
        "embed_token_axis": n_patches + 2,
        "paper_value": 5,
        "paper_unit": "hours (resampling step dt=1h)",
        "units_too_short": int((train_lengths < L).sum()),
        "min_trajectory_length": int(train_lengths.min()),
    }

    # ---------------------------------------------------------------- splits
    hr("4. UNIT-LEVEL SPLITS AND LEAKAGE ASSERTION")
    all_units = np.sort(train_df["unit"].unique())
    splits = sp.make_splits(all_units)
    sp.assert_disjoint(splits, all_units)
    print("  sp.assert_disjoint(splits, all_units) -> PASSED "
          "(pairwise disjoint, no duplicates, partition exhaustive)")
    for name in ("train", "calib", "eval"):
        ids = splits.of(name)
        rows = int(train_df["unit"].isin(ids).sum())
        cyc = int(train_lengths.loc[list(ids)].sum())
        print(f"    {name:<6} units={len(ids):>3} "
              f"({len(ids)/len(all_units):.0%} of {len(all_units)})  rows={rows:,}  "
              f"cycles={cyc:,}")
    print(f"    total   units={len(all_units)} "
          f"rows={sum(int(train_df['unit'].isin(splits.of(n)).sum()) for n in ('train','calib','eval')):,}")

    # Negative control: the assertion must actually fire.
    try:
        bad = sp.UnitSplits(
            train=splits.train + (splits.eval[0],), calib=splits.calib, eval=splits.eval
        )
        sp.assert_disjoint(bad, all_units)
        print("  !! NEGATIVE CONTROL FAILED: leakage was not detected")
        return 2
    except sp.SplitLeakage as e:
        print(f"  negative control: injecting unit {splits.eval[0]} into train ->"
              f" SplitLeakage raised as required")
        print(f"    message: {e}")

    sp.save(splits)
    ev["splits"] = {
        "seed": splits.seed,
        "fractions": C.SPLIT_FRACTIONS,
        "sizes": splits.sizes,
        "units": splits.as_dict(),
        "rows": {
            n: int(train_df["unit"].isin(splits.of(n)).sum())
            for n in ("train", "calib", "eval")
        },
        "cycles": {
            n: int(train_lengths.loc[list(splits.of(n))].sum())
            for n in ("train", "calib", "eval")
        },
        "assertion": "passed; negative control raised SplitLeakage",
    }

    # ---------------------------------------------------------------- preprocessing
    hr("5. PREPROCESSING (fitted on training-split units only)")
    cand = preprocess.candidate_feature_columns()
    train_rows = train_df[train_df["unit"].isin(splits.train)]
    sp.assert_frame_units(train_rows, splits.train, "preprocessor fit rows")
    pre = preprocess.Preprocessor(candidate_cols=cand).fit(train_rows)
    desc = pre.describe()
    print(f"  fitted on {desc['fit_n_units']} training units, {desc['fit_n_rows']:,} rows")
    print(f"  candidates: {len(cand)} columns (3 op settings + 21 sensors)")
    print(f"  DROPPED as constant on the training split ({desc['n_dropped']}): "
          f"{desc['dropped_constant']}")
    print(f"  KEPT ({desc['n_kept']}): {desc['kept_cols']}")
    if desc["zero_std_after_clip"]:
        print(f"  !! zero variance after clipping: {desc['zero_std_after_clip']}")
    # Cross-check: are these columns also constant over the whole file?
    const_all = cmapss.constant_columns(train_df, cand)
    print(f"  cross-check, constant over the *entire* training file: {const_all}")
    if set(const_all) != set(desc["dropped_constant"]):
        print("  note: train-split-constant set differs from whole-file-constant set; "
              "we use the train-split set, per protocol")
    (C.RESULTS / "preprocessor.json").write_text(json.dumps(desc, indent=2) + "\n")
    ev["preprocessing"] = {
        k: desc[k]
        for k in (
            "candidate_cols", "dropped_constant", "n_dropped", "kept_cols",
            "n_kept", "zero_std_after_clip", "fit_n_units", "fit_n_rows",
            "clip_percentiles",
        )
    }
    ev["preprocessing"]["constant_over_whole_train_file"] = const_all
    ev["preprocessing"]["order"] = [
        "drop constant-on-train columns",
        "clip to train 1st/99th percentile",
        "z-score with train mean/std",
    ]
    ev["preprocessing"]["paper_steps_not_applicable"] = [
        "linear interpolation to uniform grid (C-MAPSS already regular)",
        "gap filtering at dt_max (no gaps)",
        "NaN filtering (no NaNs)",
    ]

    # ---------------------------------------------------------------- windows
    hr("6. TRUNCATION-POINT CONSTRUCTION")
    idx: dict[str, W.WindowIndex] = {}
    stats: dict[str, dict] = {}

    for name in ("eval", "calib"):
        lengths = train_lengths.loc[list(splits.of(name))]
        idx[name], stats[name] = W.stratified_truncations(
            lengths, L, C.TRUNCATIONS_PER_UNIT, C.SEEDS["truncation_sampling"]
        )
    lengths_tr = train_lengths.loc[list(splits.train)]
    idx["train"], stats["train"] = W.dense_truncations(lengths_tr, L)

    print(f"  scheme (eval, calib): {idx['eval'].scheme}")
    print(f"  scheme (train):       {idx['train'].scheme}")
    print()
    for name in ("train", "calib", "eval"):
        s, ix = stats[name], idx[name]
        print(f"  {name:<6} windows={len(ix):>6}  units={len(set(ix.unit.tolist())):>3}"
              + (f"  requested={s['requested']} unfilled_strata={s['strata_unfilled']}"
                 if "requested" in s else ""))
        print(f"         cycles in split={s['cycles_total']:,}  "
              f"inadmissible (window does not fit)={s['inadmissible_cycles_total']:,}"
              + (f"  admissible={s['admissible_cycles_total']:,}"
                 if "admissible_cycles_total" in s else ""))

    print()
    print("  DISCARD ACCOUNTING")
    print(f"    A truncation point at cycle t is inadmissible iff t < L = {L}.")
    print(f"    Per unit that is exactly L-1 = {L-1} cycles.")
    tot_disc = sum(stats[n]["inadmissible_cycles_total"] for n in ("train", "calib", "eval"))
    tot_cyc = int(train_lengths.sum())
    print(f"    Across all {len(all_units)} training-file units: "
          f"{tot_disc:,} of {tot_cyc:,} cycles discarded "
          f"({tot_disc/tot_cyc:.2%}).")
    print(f"    Units left with zero admissible truncation point: "
          f"{int((train_lengths < L).sum())}")
    print(f"    Requested eval truncation points: {stats['eval']['requested']}, "
          f"drawn: {stats['eval']['drawn']}, "
          f"unfilled strata: {stats['eval']['strata_unfilled']}")

    # ---------------------------------------------------------------- leakage checks
    hr("7. WINDOW LEAK CHECKS")
    extractors = {}
    for name in ("train", "calib", "eval"):
        rows = train_df[train_df["unit"].isin(splits.of(name))]
        sp.assert_frame_units(rows, splits.of(name), f"{name} rows")
        ex = W.WindowExtractor(pre.transform_frame(rows), pre.kept_cols, L)
        n = ex.assert_no_lookahead(idx[name])
        extractors[name] = ex
        print(f"  {name:<6} assert_no_lookahead over all {n:,} windows -> PASSED "
              f"(window ends exactly at t; t+rul == T_unit)")
    # Spot-check a materialised window's shape and provenance.
    ex, ix = extractors["eval"], idx["eval"]
    w = ex.one(int(ix.unit[0]), int(ix.t[0]))
    print(f"  materialised window shape (unit {ix.unit[0]}, t={ix.t[0]}): {w.shape} "
          f"= (L={L}, n_features={len(pre.kept_cols)})")
    raw_last = train_df[(train_df.unit == int(ix.unit[0])) & (train_df.cycle == int(ix.t[0]))]
    print(f"  raw rows at cycle t for that unit: {len(raw_last)} (expect 1)")

    # Negative control: a window that would overrun must be refused.
    try:
        ex.one(int(ix.unit[0]), L - 1)
        print("  !! NEGATIVE CONTROL FAILED: under-length window was allowed")
        return 2
    except ValueError as e:
        print(f"  negative control: requesting t={L-1} < L -> ValueError as required "
              f"({e})")

    ev["windows"] = {
        name: {
            "n_windows": len(idx[name]),
            "n_units": len(set(idx[name].unit.tolist())),
            "scheme": idx[name].scheme,
            **stats[name],
        }
        for name in ("train", "calib", "eval")
    }
    ev["windows"]["discard_accounting"] = {
        "rule": f"truncation point t inadmissible iff t < L={L}",
        "per_unit_discarded_cycles": L - 1,
        "total_discarded_cycles": tot_disc,
        "total_cycles": tot_cyc,
        "fraction_discarded": tot_disc / tot_cyc,
        "units_with_zero_admissible": int((train_lengths < L).sum()),
    }
    ev["windows"]["leak_checks"] = "assert_no_lookahead passed on all splits; negative controls fired"

    # ---------------------------------------------------------------- targets
    hr("8. TARGET VARIANTS")
    for name in ("train", "calib", "eval"):
        ix = idx[name]
        a, b = ix.target("A"), ix.target("B")
        at_cap = int((ix.rul >= C.RUL_CAP).sum())
        print(f"  {name}:")
        print(f"    Variant A (uncapped): {fmt_dist(pd.Series(a))}")
        print(f"    Variant B (cap {C.RUL_CAP}):   {fmt_dist(pd.Series(b))}")
        print(f"    point mass at cap: {at_cap:,}/{len(ix):,} = {at_cap/len(ix):.1%} "
              f"of windows have true RUL >= {C.RUL_CAP}")
    ev["targets"] = {
        name: {
            "variant_A": {
                "min": float(idx[name].target("A").min()),
                "median": float(np.median(idx[name].target("A"))),
                "mean": float(idx[name].target("A").mean()),
                "max": float(idx[name].target("A").max()),
                "std": float(idx[name].target("A").std(ddof=1)),
            },
            "variant_B": {
                "min": float(idx[name].target("B").min()),
                "median": float(np.median(idx[name].target("B"))),
                "mean": float(idx[name].target("B").mean()),
                "max": float(idx[name].target("B").max()),
                "std": float(idx[name].target("B").std(ddof=1)),
            },
            "n_at_or_above_cap": int((idx[name].rul >= C.RUL_CAP).sum()),
            "frac_at_or_above_cap": float((idx[name].rul >= C.RUL_CAP).mean()),
        }
        for name in ("train", "calib", "eval")
    }
    ev["targets"]["cap"] = C.RUL_CAP

    # RUL-bin occupancy of the eval set: the regime-conditioning figure depends on it.
    hr("9. EVAL RUL-BIN OCCUPANCY (for regime conditioning in Phase 5)")
    ixe = idx["eval"]
    for bname, lo, hi in C.RUL_BINS:
        m = (ixe.rul >= lo) & (ixe.rul < hi) if np.isfinite(hi) else (ixe.rul >= lo)
        n_units = len(set(ixe.unit[m].tolist()))
        print(f"  {bname:<8} [{lo:g}, {hi:g})  windows={int(m.sum()):>4}  units={n_units:>3}")
    ev["eval_rul_bins"] = {
        bname: {
            "n_windows": int((((ixe.rul >= lo) & (ixe.rul < hi)) if np.isfinite(hi)
                              else (ixe.rul >= lo)).sum()),
            "n_units": len(set(ixe.unit[(((ixe.rul >= lo) & (ixe.rul < hi))
                                         if np.isfinite(hi) else (ixe.rul >= lo))].tolist())),
        }
        for bname, lo, hi in C.RUL_BINS
    }

    # ---------------------------------------------------------------- persist
    for name in ("train", "calib", "eval"):
        idx[name].to_frame().to_parquet(C.RESULTS / f"window_index_{name}.parquet")
    (C.RESULTS / "phase1_evidence.json").write_text(json.dumps(ev, indent=2, default=str) + "\n")

    # fresh=True: phase 1 starts a new manifest, so no stale phase survives.
    man = manifest.merge_into("phase1", ev, fresh=True)
    print("\nwrote results/unit_splits.json, results/window_index_{train,calib,eval}.parquet,")
    print("      results/phase1_evidence.json, results/preprocessor.json, run_manifest.json")

    _figures(train_lengths, splits, idx)
    return 0


def _figures(train_lengths, splits, idx) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Trajectory lengths by split.
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.histogram_bin_edges(train_lengths.values, bins=25)
    for name, colour in (("train", "#3b6ea5"), ("calib", "#c88a3d"), ("eval", "#4f8f5c")):
        vals = train_lengths.loc[list(splits.of(name))].values
        ax.hist(vals, bins=bins, alpha=0.65, label=f"{name} (n={len(vals)})", color=colour)
    ax.axvline(C.LOOKBACK, color="crimson", ls="--", lw=1.2,
               label=f"look-back L={C.LOOKBACK}")
    ax.set_xlabel("trajectory length (cycles to failure)")
    ax.set_ylabel("units")
    ax.set_title(f"C-MAPSS {C.SUBSET} training-file trajectory lengths, by unit split")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_trajectory_lengths.png", dpi=180)
    plt.close(fig)

    # Truncation-point RUL histogram.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ixe = idx["eval"]
    axes[0].hist(ixe.rul, bins=40, color="#4f8f5c", edgecolor="white", lw=0.4)
    axes[0].axvline(C.RUL_CAP, color="crimson", ls="--", lw=1.2, label=f"cap={C.RUL_CAP}")
    axes[0].set_title(f"Variant A (uncapped) — eval truncations, n={len(ixe)}")
    axes[0].set_xlabel("true RUL at truncation (cycles)")
    axes[0].set_ylabel("windows")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].hist(ixe.rul_capped, bins=40, color="#3b6ea5", edgecolor="white", lw=0.4)
    axes[1].set_title(f"Variant B (capped at {C.RUL_CAP}) — same truncations")
    axes[1].set_xlabel("capped RUL (cycles)")
    frac = float((ixe.rul >= C.RUL_CAP).mean())
    axes[1].annotate(f"point mass at cap: {frac:.1%}", xy=(0.97, 0.92),
                     xycoords="axes fraction", ha="right", fontsize=9, color="crimson")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Evaluation truncation-point RUL distribution "
                 f"({C.TRUNCATIONS_PER_UNIT} stratified points per eval unit)", fontsize=10)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_truncation_rul_hist.png", dpi=180)
    plt.close(fig)
    print("      figures/fig_trajectory_lengths.png, figures/fig_truncation_rul_hist.png")


if __name__ == "__main__":
    sys.exit(main())
