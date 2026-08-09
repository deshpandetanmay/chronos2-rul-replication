"""Preprocessing, fitted on training-split units only and shared by every arm.

The brief requires preprocessing to be identical across all arms, so this is the
only place any arm is allowed to transform sensor values. Arms consume the output
of `Preprocessor.transform` and nothing else.

Order follows the source paper (§II-A), minus the steps that only apply to
irregular industrial telemetry:
  1. drop columns that are constant on the training split
  2. clip to the training-split 1st/99th percentiles ("global outlier clipping")
  3. z-score using training-split mean/std
The paper's resampling, gap filtering and NaN filtering are inapplicable: C-MAPSS
is already on a regular per-cycle grid and has no missing values (asserted in
`cmapss.integrity_checks`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class Preprocessor:
    """Fit on training-split rows; apply unchanged everywhere else."""

    candidate_cols: list[str]
    dropped_constant: list[str] = field(default_factory=list)
    kept_cols: list[str] = field(default_factory=list)
    clip_lo: np.ndarray | None = None
    clip_hi: np.ndarray | None = None
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    fitted_on_units: tuple[int, ...] = ()
    n_fit_rows: int = 0

    def fit(self, train_df: pd.DataFrame) -> "Preprocessor":
        self.fitted_on_units = tuple(sorted(int(u) for u in train_df["unit"].unique()))
        self.n_fit_rows = int(len(train_df))

        # Constancy is tested with peak-to-peak, not std. A column holding a single
        # repeated value can have std ~1e-13 rather than exactly 0 (the mean does not
        # round-trip in float64), which would survive a `std == 0` test and then be
        # divided by 1e-13 below, amplifying pure rounding noise into a unit-variance
        # feature. max-min is exact for identical values. See notes/decisions.md D-005.
        self.dropped_constant = [
            c for c in self.candidate_cols
            if float(np.ptp(train_df[c].to_numpy(dtype=np.float64))) == 0.0
        ]
        self.kept_cols = [c for c in self.candidate_cols if c not in self.dropped_constant]

        kept = train_df[self.kept_cols].to_numpy(dtype=np.float64)
        self.clip_lo = np.percentile(kept, 1.0, axis=0)
        self.clip_hi = np.percentile(kept, 99.0, axis=0)

        clipped = np.clip(kept, self.clip_lo, self.clip_hi)
        self.mean = clipped.mean(axis=0)
        std = clipped.std(axis=0, ddof=0)
        # A column can survive the constant check yet collapse after clipping if
        # >98% of its mass sits on one value. Same float caveat as above, so the
        # degeneracy test is peak-to-peak and std is only used for scaling.
        ptp_clipped = np.ptp(clipped, axis=0)
        self.zero_std_after_clip = [
            c for c, s, p in zip(self.kept_cols, std, ptp_clipped)
            if (not np.isfinite(s)) or p == 0.0
        ]
        std = np.where(ptp_clipped > 0, std, 1.0)
        self.std = np.where(std > 0, std, 1.0)
        return self

    def transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of `df` with kept feature columns clipped and standardized."""
        self._require_fitted()
        out = df.copy()
        vals = out[self.kept_cols].to_numpy(dtype=np.float64)
        vals = np.clip(vals, self.clip_lo, self.clip_hi)
        vals = (vals - self.mean) / self.std
        out[self.kept_cols] = vals
        return out

    def _require_fitted(self) -> None:
        if self.mean is None:
            raise RuntimeError("Preprocessor.fit must be called before transform")

    def describe(self) -> dict:
        self._require_fitted()
        return {
            "candidate_cols": list(self.candidate_cols),
            "dropped_constant": list(self.dropped_constant),
            "n_dropped": len(self.dropped_constant),
            "kept_cols": list(self.kept_cols),
            "n_kept": len(self.kept_cols),
            "zero_std_after_clip": list(getattr(self, "zero_std_after_clip", [])),
            "fit_n_units": len(self.fitted_on_units),
            "fit_n_rows": self.n_fit_rows,
            "clip_percentiles": [1.0, 99.0],
            "per_column": {
                c: {
                    "clip_lo": float(self.clip_lo[i]),
                    "clip_hi": float(self.clip_hi[i]),
                    "mean": float(self.mean[i]),
                    "std": float(self.std[i]),
                }
                for i, c in enumerate(self.kept_cols)
            },
        }


def candidate_feature_columns() -> list[str]:
    """Operational settings + sensors, before constant-column dropping.

    The 3 operational settings are included as candidates rather than assumed
    uninformative; FD001 is single-condition so we expect the constant check to
    remove most of them, but that is measured, not asserted.
    """
    return list(C.OP_COLS) + list(C.SENSOR_COLS)
