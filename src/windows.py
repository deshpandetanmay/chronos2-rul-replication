"""Truncation-point sampling and look-back window extraction.

Conventions used throughout, fixed here so nothing downstream has to re-derive them:

* A *truncation point* `t` means "the machine has been observed through cycle `t`,
  inclusive, and nothing after". `t` is a cycle index, 1-based, as in the raw file.
* The look-back window for `t` is cycles `[t - L + 1, t]`, so it has exactly `L`
  rows and its last row is `t`. It never reads a cycle after `t`.
* The label is `RUL = T_unit - t`, where `T_unit` is the unit's failure cycle
  (its last observed cycle in the training file). `RUL = 0` at failure.
* A truncation point is *admissible* iff `t >= L` (full window fits) and
  `t <= T_unit`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


@dataclass(frozen=True)
class WindowIndex:
    """A set of (unit, truncation point) pairs with their labels.

    `rul` is the uncapped label (Variant A). `rul_capped` is Variant B.
    """

    unit: np.ndarray  # int, (n,)
    t: np.ndarray  # int, (n,) truncation cycle
    rul: np.ndarray  # int, (n,) uncapped remaining cycles
    scheme: str = ""

    def __len__(self) -> int:
        return len(self.unit)

    @property
    def rul_capped(self) -> np.ndarray:
        return np.minimum(self.rul, C.RUL_CAP)

    def target(self, variant: str) -> np.ndarray:
        """Variant A = uncapped, Variant B = piecewise-linear capped."""
        if variant == "A":
            return self.rul.astype(np.float64)
        if variant == "B":
            return self.rul_capped.astype(np.float64)
        raise ValueError(f"unknown target variant {variant!r} (expected 'A' or 'B')")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "unit": self.unit,
                "t": self.t,
                "rul": self.rul,
                "rul_capped": self.rul_capped,
            }
        )


def admissible_truncations(length: int, lookback: int) -> np.ndarray:
    """All admissible truncation cycles for a trajectory of `length` cycles."""
    if length < lookback:
        return np.empty(0, dtype=int)
    return np.arange(lookback, length + 1, dtype=int)


def inadmissible_count(length: int, lookback: int) -> int:
    """Cycles discarded because a full look-back window does not fit before them."""
    return int(min(lookback - 1, length))


def _contiguous_strata(n_values: int, n_strata: int) -> list[tuple[int, int]]:
    """Partition `range(n_values)` into `n_strata` contiguous, near-equal blocks.

    Returns [start, stop) index pairs. Empty blocks are returned when
    `n_values < n_strata`, so the caller can count the shortfall.
    """
    base, extra = divmod(n_values, n_strata)
    out, start = [], 0
    for i in range(n_strata):
        size = base + (1 if i < extra else 0)
        out.append((start, start + size))
        start += size
    return out


def stratified_truncations(
    lengths: pd.Series,
    lookback: int,
    n_per_unit: int,
    seed: int,
) -> tuple[WindowIndex, dict]:
    """Sample `n_per_unit` truncation points per unit, stratified over the RUL range.

    Scheme: for a unit of length T, the admissible labels are the integers
    RUL = 0 .. T - L. That integer range is cut into `n_per_unit` contiguous,
    near-equal strata ordered from RUL = 0 upward, and exactly one RUL is drawn
    uniformly at random from each stratum. This deliberately covers the whole RUL
    range and guarantees that the lowest stratum -- the near-failure regime, which
    is where a maintenance decision actually gets made -- is always represented,
    which independent uniform sampling over `t` would not guarantee.

    Sampling is per-unit with a seed derived from (seed, unit) so that adding or
    removing a unit does not perturb any other unit's draws.
    """
    units, ts, ruls = [], [], []
    deficit = 0
    for unit, length in lengths.items():
        length = int(length)
        adm = admissible_truncations(length, lookback)
        if len(adm) == 0:
            deficit += n_per_unit
            continue
        # Admissible RULs, ascending from 0 (failure) to length - lookback.
        max_rul = length - lookback
        rng = np.random.default_rng([seed, int(unit)])
        for lo, hi in _contiguous_strata(max_rul + 1, n_per_unit):
            if hi <= lo:
                deficit += 1
                continue
            r = int(rng.integers(lo, hi))
            units.append(int(unit))
            ruls.append(r)
            ts.append(length - r)

    idx = WindowIndex(
        unit=np.asarray(units, dtype=int),
        t=np.asarray(ts, dtype=int),
        rul=np.asarray(ruls, dtype=int),
        scheme=(
            f"stratified: {n_per_unit} contiguous equal-width strata over admissible "
            f"RUL range [0, T_unit - {lookback}], one uniform draw per stratum, "
            f"seed={seed} mixed with unit id"
        ),
    )
    stats = {
        "requested": int(n_per_unit * len(lengths)),
        "drawn": len(idx),
        "strata_unfilled": int(deficit),
        "units_with_no_admissible_point": int((lengths < lookback).sum()),
        "inadmissible_cycles_total": int(
            sum(inadmissible_count(int(v), lookback) for v in lengths.values)
        ),
        "admissible_cycles_total": int(
            sum(len(admissible_truncations(int(v), lookback)) for v in lengths.values)
        ),
        "cycles_total": int(lengths.sum()),
    }
    return idx, stats


def dense_truncations(lengths: pd.Series, lookback: int) -> tuple[WindowIndex, dict]:
    """Every admissible truncation point for every unit (used for head training)."""
    units, ts, ruls = [], [], []
    for unit, length in lengths.items():
        length = int(length)
        for t in admissible_truncations(length, lookback):
            units.append(int(unit))
            ts.append(int(t))
            ruls.append(length - int(t))
    idx = WindowIndex(
        unit=np.asarray(units, dtype=int),
        t=np.asarray(ts, dtype=int),
        rul=np.asarray(ruls, dtype=int),
        scheme=f"dense: every admissible truncation point, lookback={lookback}",
    )
    stats = {
        "drawn": len(idx),
        "inadmissible_cycles_total": int(
            sum(inadmissible_count(int(v), lookback) for v in lengths.values)
        ),
        "cycles_total": int(lengths.sum()),
    }
    return idx, stats


class WindowExtractor:
    """Materialises look-back windows as a dense array, with hard leakage checks."""

    def __init__(self, df: pd.DataFrame, feature_cols: list[str], lookback: int):
        self.feature_cols = list(feature_cols)
        self.lookback = int(lookback)
        # Per-unit contiguous blocks, ordered by cycle, for O(1) slicing.
        self._blocks: dict[int, np.ndarray] = {}
        self._lengths: dict[int, int] = {}
        for unit, g in df.sort_values(["unit", "cycle"]).groupby("unit", sort=True):
            cycles = g["cycle"].to_numpy()
            # Contiguity from 1..T is asserted in cmapss.integrity_checks; re-check
            # cheaply here because window slicing relies on it.
            if cycles[0] != 1 or not np.array_equal(cycles, np.arange(1, len(cycles) + 1)):
                raise ValueError(f"unit {unit}: cycles are not 1..T contiguous")
            self._blocks[int(unit)] = g[self.feature_cols].to_numpy(dtype=np.float32)
            self._lengths[int(unit)] = len(cycles)

    def length_of(self, unit: int) -> int:
        return self._lengths[int(unit)]

    def one(self, unit: int, t: int) -> np.ndarray:
        """Window for (unit, t) as (lookback, n_features), ending at cycle t."""
        unit, t = int(unit), int(t)
        L, T = self.lookback, self._lengths[unit]
        if t < L:
            raise ValueError(f"unit {unit}, t={t}: window of {L} does not fit")
        if t > T:
            raise ValueError(f"unit {unit}, t={t}: beyond trajectory length {T}")
        # cycle c lives at row c-1; window is cycles [t-L+1, t] -> rows [t-L, t).
        return self._blocks[unit][t - L : t]

    def batch(self, idx: WindowIndex) -> np.ndarray:
        """All windows for an index as (n, lookback, n_features)."""
        out = np.empty((len(idx), self.lookback, len(self.feature_cols)), dtype=np.float32)
        for i, (u, t) in enumerate(zip(idx.unit, idx.t)):
            out[i] = self.one(u, t)
        return out

    def assert_no_lookahead_official(self, idx: WindowIndex) -> int:
        """Leak check for the official test split.

        The training-file invariant `t + RUL == T_unit` does NOT hold here: an official
        test trajectory is truncated *before* failure, and `RUL_FD001.txt` gives the
        cycles remaining *after* the last observed cycle. The invariants that do hold
        are that the window ends at the unit's final observed cycle and that the label
        is non-negative. Asserting the wrong invariant here would either fail spuriously
        or, if weakened to pass, stop checking anything.
        """
        for i in range(len(idx)):
            u, t, r = int(idx.unit[i]), int(idx.t[i]), int(idx.rul[i])
            T = self._lengths[u]
            if t != T:
                raise AssertionError(
                    f"official unit {u}: window ends at {t}, not the last cycle {T}"
                )
            if t - self.lookback + 1 < 1:
                raise AssertionError(f"official unit {u}: window underruns cycle 1")
            if r < 0:
                raise AssertionError(f"official unit {u}: negative RUL label {r}")
        return len(idx)

    def assert_no_lookahead(self, idx: WindowIndex, n_check: int | None = None) -> int:
        """Verify no window reads a cycle after its truncation point.

        Checks by reconstructing the cycle span of each window from the raw index
        rather than trusting `one()`. Returns the number of pairs checked.
        """
        n = len(idx) if n_check is None else min(n_check, len(idx))
        for i in range(n):
            u, t, r = int(idx.unit[i]), int(idx.t[i]), int(idx.rul[i])
            T = self._lengths[u]
            first, last = t - self.lookback + 1, t
            if first < 1:
                raise AssertionError(f"unit {u}, t={t}: window starts at cycle {first} < 1")
            if last != t:
                raise AssertionError(f"unit {u}, t={t}: window ends at {last}, not {t}")
            if last > t:
                raise AssertionError(f"unit {u}, t={t}: window reads past truncation")
            if t + r != T:
                raise AssertionError(
                    f"unit {u}, t={t}: label {r} inconsistent with length {T} (t+rul != T)"
                )
        return n
