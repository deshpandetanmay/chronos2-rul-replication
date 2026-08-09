"""The one prediction artifact format every arm writes.

Phases 2, 3 and 4 all emit `Prediction` objects; Phase 5 consumes them without
knowing or caring which arm produced them. That keeps the calibration analysis
uniform across arms by construction rather than by discipline.

Columns: unit, t, rul_true_uncapped, y_true (target under the variant),
pred (point estimate), q<level> for each level in config.QUANTILE_LEVELS.
`q*` columns are absent for point-only arms.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C, metrics

PRED_DIR = C.RESULTS / "preds"
PRED_DIR.mkdir(parents=True, exist_ok=True)


def qcol(level: float) -> str:
    return f"q{level:g}"


@dataclass
class Prediction:
    """Predictions for one (arm, target variant, look-back, eval split) cell."""

    arm: str
    variant: str
    lookback: int
    unit: np.ndarray
    t: np.ndarray
    rul_true_uncapped: np.ndarray
    y_true: np.ndarray
    pred: np.ndarray
    quantiles: np.ndarray | None = None  # (n, len(levels))
    levels: tuple[float, ...] = C.QUANTILE_LEVELS
    split: str = "eval"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.y_true)
        for name in ("unit", "t", "rul_true_uncapped", "pred"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{self.arm}/{self.variant}: {name} length != {n}")
        if self.quantiles is not None:
            if self.quantiles.shape != (n, len(self.levels)):
                raise ValueError(
                    f"{self.arm}/{self.variant}: quantiles {self.quantiles.shape} "
                    f"!= ({n}, {len(self.levels)})"
                )
            self.quantiles, crossed = metrics.enforce_monotone_quantiles(self.quantiles)
            self.meta["quantile_crossings_repaired"] = crossed
            self.meta["quantile_crossing_rate"] = crossed / n if n else 0.0

    @property
    def has_intervals(self) -> bool:
        return self.quantiles is not None

    @property
    def key(self) -> str:
        return f"{self.arm}__{self.variant}__L{self.lookback}__{self.split}"

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "unit": self.unit,
                "t": self.t,
                "rul_true_uncapped": self.rul_true_uncapped,
                "y_true": self.y_true,
                "pred": self.pred,
            }
        )
        if self.quantiles is not None:
            for i, lv in enumerate(self.levels):
                df[qcol(lv)] = self.quantiles[:, i]
        return df

    def save(self) -> None:
        self.to_frame().to_parquet(PRED_DIR / f"{self.key}.parquet")
        (PRED_DIR / f"{self.key}.json").write_text(
            json.dumps(
                {
                    "arm": self.arm,
                    "variant": self.variant,
                    "lookback": self.lookback,
                    "split": self.split,
                    "levels": list(self.levels),
                    "has_intervals": self.has_intervals,
                    "n": int(len(self.y_true)),
                    "meta": self.meta,
                },
                indent=2,
                default=str,
            )
            + "\n"
        )

    def point_metrics(self) -> dict:
        return metrics.point_metrics(self.y_true, self.pred)


def load(key: str) -> Prediction:
    df = pd.read_parquet(PRED_DIR / f"{key}.parquet")
    side = json.loads((PRED_DIR / f"{key}.json").read_text())
    levels = tuple(side["levels"])
    qcols = [qcol(lv) for lv in levels]
    q = df[qcols].to_numpy() if side["has_intervals"] else None
    return Prediction(
        arm=side["arm"], variant=side["variant"], lookback=side["lookback"],
        unit=df["unit"].to_numpy(), t=df["t"].to_numpy(),
        rul_true_uncapped=df["rul_true_uncapped"].to_numpy(),
        y_true=df["y_true"].to_numpy(), pred=df["pred"].to_numpy(),
        quantiles=q, levels=levels, split=side["split"], meta=side.get("meta", {}),
    )


def available() -> list[str]:
    return sorted(p.stem for p in PRED_DIR.glob("*.parquet"))
