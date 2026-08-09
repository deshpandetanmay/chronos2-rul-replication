"""Assembles the split / preprocessor / window bundle for a given look-back.

Every arm consumes a `Bundle` and nothing else, so "identical preprocessing and
identical windows across all arms" is enforced structurally. Deterministic: the
same look-back always yields byte-identical windows and labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from . import cmapss, config as C, preprocess, splits as sp, windows as W

SPLIT_NAMES = ("train", "calib", "eval")

# The official held-out test set is a SECONDARY, confirmatory evaluation, reported
# separately and never mixed with the primary eval split. It supplies exactly one
# labelled point per unit, so coverage estimated on it is far too noisy to support C2
# (brief §5.1). Its unit ids are offset so they can never be silently pooled with
# training-file unit ids, which run over the same 1..100 range but are different engines.
OFFICIAL_SPLIT = "test_official"
OFFICIAL_UNIT_OFFSET = 1000


@dataclass
class Bundle:
    lookback: int
    splits: sp.UnitSplits
    pre: preprocess.Preprocessor
    idx: dict[str, W.WindowIndex]
    ext: dict[str, W.WindowExtractor]
    stats: dict[str, dict]
    train_lengths: pd.Series

    @property
    def feature_cols(self) -> list[str]:
        return self.pre.kept_cols

    @property
    def n_features(self) -> int:
        return len(self.pre.kept_cols)

    def windows(self, split: str) -> np.ndarray:
        """(n, lookback, n_features) preprocessed windows for a split."""
        return self.ext[split].batch(self.idx[split])

    def target(self, split: str, variant: str) -> np.ndarray:
        return self.idx[split].target(variant)

    def units(self, split: str) -> np.ndarray:
        return self.idx[split].unit

    def describe(self) -> dict:
        return {
            "lookback": self.lookback,
            "n_features": self.n_features,
            "feature_cols": list(self.feature_cols),
            "dropped_constant": list(self.pre.dropped_constant),
            "splits": self.splits.sizes,
            "windows": {s: len(self.idx[s]) for s in SPLIT_NAMES},
            "window_stats": self.stats,
        }


def build(lookback: int, verbose: bool = False) -> Bundle:
    """Construct the bundle for `lookback`, re-running every leakage assertion."""
    train_df = cmapss.load_train()
    cmapss.integrity_checks(train_df, "train_FD001")
    all_units = np.sort(train_df["unit"].unique())

    splits = sp.make_splits(all_units)
    sp.assert_disjoint(splits, all_units)

    lengths = cmapss.trajectory_lengths(train_df)

    # Preprocessor is fitted on training-split rows only. Calibration rows are
    # unseen here too, or the conformal step would be invalid (D-007).
    train_rows = train_df[train_df["unit"].isin(splits.train)]
    sp.assert_frame_units(train_rows, splits.train, "preprocessor fit rows")
    pre = preprocess.Preprocessor(preprocess.candidate_feature_columns()).fit(train_rows)

    idx, stats, ext = {}, {}, {}
    for name in SPLIT_NAMES:
        sub = lengths.loc[list(splits.of(name))]
        if name == "train":
            idx[name], stats[name] = W.dense_truncations(sub, lookback)
        else:
            idx[name], stats[name] = W.stratified_truncations(
                sub, lookback, C.TRUNCATIONS_PER_UNIT, C.SEEDS["truncation_sampling"]
            )
        rows = train_df[train_df["unit"].isin(splits.of(name))]
        sp.assert_frame_units(rows, splits.of(name), f"{name} rows")
        ext[name] = W.WindowExtractor(pre.transform_frame(rows), pre.kept_cols, lookback)
        ext[name].assert_no_lookahead(idx[name])

    # ---------------------------------------------------- official test (secondary)
    test_df, test_rul = cmapss.load_official_test()
    cmapss.integrity_checks(test_df, "test_FD001")
    test_lengths = cmapss.trajectory_lengths(test_df)
    keep = test_lengths.index[test_lengths >= lookback]
    dropped = [int(u) for u in test_lengths.index if u not in set(keep)]

    off = W.WindowIndex(
        unit=np.array([int(u) + OFFICIAL_UNIT_OFFSET for u in keep], dtype=int),
        t=np.array([int(test_lengths.loc[u]) for u in keep], dtype=int),
        rul=np.array([int(test_rul.loc[u]) for u in keep], dtype=int),
        scheme=(
            "official: one window per test unit, ending at its last observed cycle; "
            "label taken from RUL_FD001.txt (cycles remaining after that cycle)"
        ),
    )
    idx[OFFICIAL_SPLIT] = off
    stats[OFFICIAL_SPLIT] = {
        "n_units_in_file": int(len(test_lengths)),
        "n_units_used": int(len(keep)),
        "n_units_dropped_too_short": len(dropped),
        "dropped_unit_ids": dropped,
        "min_length": int(test_lengths.min()),
        "unit_id_offset": OFFICIAL_UNIT_OFFSET,
    }
    # Build an extractor whose unit ids match the offset index.
    test_shift = test_df.copy()
    test_shift["unit"] = test_shift["unit"] + OFFICIAL_UNIT_OFFSET
    ext[OFFICIAL_SPLIT] = W.WindowExtractor(
        pre.transform_frame(test_shift), pre.kept_cols, lookback
    )
    ext[OFFICIAL_SPLIT].assert_no_lookahead_official(off)

    if verbose:
        print(f"  bundle L={lookback}: features={len(pre.kept_cols)} "
              + " ".join(f"{s}={len(idx[s])}" for s in SPLIT_NAMES)
              + f" {OFFICIAL_SPLIT}={len(off)}"
              + (f" (dropped {len(dropped)} units shorter than L)" if dropped else ""))

    return Bundle(
        lookback=lookback, splits=splits, pre=pre, idx=idx, ext=ext,
        stats=stats, train_lengths=lengths,
    )
