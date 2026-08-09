"""Unit-level splitting and the leakage assertion.

The single most important invariant in this project: an engine unit belongs to
exactly one split. `assert_disjoint` is cheap and is called after every operation
that touches split membership.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C


class SplitLeakage(AssertionError):
    """Raised when a unit id appears in more than one split."""


@dataclass(frozen=True)
class UnitSplits:
    train: tuple[int, ...]
    calib: tuple[int, ...]
    eval: tuple[int, ...]
    seed: int = field(default=C.SEEDS["unit_split"])

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "train": list(self.train),
            "calib": list(self.calib),
            "eval": list(self.eval),
        }

    def of(self, name: str) -> tuple[int, ...]:
        return getattr(self, name)

    @property
    def sizes(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.as_dict().items()}


def assert_disjoint(splits: UnitSplits, all_units: np.ndarray | None = None) -> None:
    """Fail loudly on any unit appearing in more than one split.

    Also checks the partition is exhaustive when `all_units` is supplied, so that
    a unit cannot be silently dropped.
    """
    d = splits.as_dict()
    for name, ids in d.items():
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise SplitLeakage(f"split '{name}' contains duplicate unit ids: {dupes}")
    names = list(d)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = sorted(set(d[a]) & set(d[b]))
            if overlap:
                raise SplitLeakage(
                    f"unit ids appear in both '{a}' and '{b}': {overlap}"
                )
    if all_units is not None:
        union = set().union(*(set(v) for v in d.values()))
        expected = set(int(u) for u in all_units)
        if union != expected:
            missing, extra = sorted(expected - union), sorted(union - expected)
            raise SplitLeakage(
                f"split partition is not exhaustive: missing={missing} extra={extra}"
            )


def assert_frame_units(df: pd.DataFrame, allowed: tuple[int, ...], label: str) -> None:
    """Assert a dataframe contains only units from `allowed`."""
    present = set(int(u) for u in df["unit"].unique())
    stray = sorted(present - set(allowed))
    if stray:
        raise SplitLeakage(f"{label}: contains units outside its split: {stray}")


def make_splits(all_units: np.ndarray, seed: int = C.SEEDS["unit_split"]) -> UnitSplits:
    """Partition units into train/calib/eval by the configured fractions.

    Uses a seeded permutation and largest-remainder allocation so the split sizes
    are deterministic and sum exactly to the number of units.
    """
    units = np.sort(np.asarray(all_units, dtype=int))
    n = len(units)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(units)

    names = ["train", "calib", "eval"]
    exact = {k: C.SPLIT_FRACTIONS[k] * n for k in names}
    counts = {k: int(np.floor(v)) for k, v in exact.items()}
    # Largest-remainder: hand out the leftover units to the largest fractional parts.
    leftover = n - sum(counts.values())
    order = sorted(names, key=lambda k: (-(exact[k] - counts[k]), k))
    for k in order[:leftover]:
        counts[k] += 1
    assert sum(counts.values()) == n, (counts, n)

    out, start = {}, 0
    for k in names:
        out[k] = tuple(int(u) for u in np.sort(perm[start : start + counts[k]]))
        start += counts[k]

    splits = UnitSplits(train=out["train"], calib=out["calib"], eval=out["eval"], seed=seed)
    assert_disjoint(splits, units)
    return splits


def save(splits: UnitSplits, path=None) -> None:
    path = path or (C.RESULTS / "unit_splits.json")
    payload = {
        "subset": C.SUBSET,
        "seed": splits.seed,
        "fractions": C.SPLIT_FRACTIONS,
        "sizes": splits.sizes,
        "units": splits.as_dict(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load(path=None) -> UnitSplits:
    path = path or (C.RESULTS / "unit_splits.json")
    p = json.loads(path.read_text())
    s = UnitSplits(
        train=tuple(p["units"]["train"]),
        calib=tuple(p["units"]["calib"]),
        eval=tuple(p["units"]["eval"]),
        seed=p["seed"],
    )
    assert_disjoint(s)
    return s
