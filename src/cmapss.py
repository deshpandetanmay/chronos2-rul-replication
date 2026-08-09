"""C-MAPSS FD001 ingest and verification.

Every count reported by this module is measured from the files on disk. No count
is taken from the project brief or from any paper.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from . import config as C


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_files() -> dict[str, str]:
    """Abort if any FD001 file differs from the checksum we ingested against."""
    digests = {}
    for name, expected in C.EXPECTED_SHA256.items():
        path = C.DATA_RAW / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing. Run `bash run.sh fetch-data` (or scripts/fetch_data.sh)."
            )
        got = _sha256(path)
        if got != expected:
            raise ValueError(
                f"SHA-256 mismatch for {name}:\n  expected {expected}\n  got      {got}"
            )
        digests[name] = got
    return digests


def load_split_file(name: str) -> pd.DataFrame:
    """Load one whitespace-delimited C-MAPSS file with the documented layout."""
    path = C.DATA_RAW / name
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] != len(C.ALL_COLS):
        raise ValueError(
            f"{name}: expected {len(C.ALL_COLS)} columns, found {df.shape[1]}"
        )
    df.columns = C.ALL_COLS
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def load_train() -> pd.DataFrame:
    return load_split_file(f"train_{C.SUBSET}.txt")


def load_official_test() -> tuple[pd.DataFrame, pd.Series]:
    """Official held-out test trajectories plus their single end-of-record RUL.

    `RUL_FD001.txt` gives, for each test unit in file order, the number of cycles
    remaining *after* the last observed cycle.
    """
    df = load_split_file(f"test_{C.SUBSET}.txt")
    rul = pd.read_csv(C.DATA_RAW / f"RUL_{C.SUBSET}.txt", header=None).iloc[:, 0]
    units = np.sort(df["unit"].unique())
    if len(rul) != len(units):
        raise ValueError(
            f"RUL file has {len(rul)} rows but test set has {len(units)} units"
        )
    rul.index = units
    rul.index.name = "unit"
    return df, rul.rename("rul_after_last_cycle")


def integrity_checks(df: pd.DataFrame, label: str) -> None:
    """Structural invariants that must hold for every C-MAPSS trajectory file."""
    if df.isna().any().any():
        raise ValueError(f"{label}: unexpected NaNs")
    g = df.groupby("unit")["cycle"]
    # Cycles must start at 1, be contiguous, and be strictly increasing.
    bad_start = g.min()[g.min() != 1]
    if len(bad_start):
        raise ValueError(f"{label}: units not starting at cycle 1: {list(bad_start.index)}")
    lens, maxes = g.size(), g.max()
    bad_contig = lens.index[lens.values != maxes.values]
    if len(bad_contig):
        raise ValueError(f"{label}: non-contiguous cycles for units {list(bad_contig)}")
    if not df.groupby("unit")["cycle"].apply(lambda s: s.is_monotonic_increasing).all():
        raise ValueError(f"{label}: cycles not monotonically increasing within a unit")


def trajectory_lengths(df: pd.DataFrame) -> pd.Series:
    return df.groupby("unit")["cycle"].max().rename("length")


def describe(df: pd.DataFrame, label: str) -> dict:
    """Measured summary of a trajectory file, for the manifest."""
    lens = trajectory_lengths(df)
    return {
        "label": label,
        "n_rows": int(len(df)),
        "n_units": int(df["unit"].nunique()),
        "unit_id_min": int(df["unit"].min()),
        "unit_id_max": int(df["unit"].max()),
        "trajectory_length": {
            "min": int(lens.min()),
            "p05": float(lens.quantile(0.05)),
            "q1": float(lens.quantile(0.25)),
            "median": float(lens.median()),
            "mean": float(lens.mean()),
            "q3": float(lens.quantile(0.75)),
            "p95": float(lens.quantile(0.95)),
            "max": int(lens.max()),
            "std": float(lens.std(ddof=1)),
            "sum": int(lens.sum()),
        },
    }


def constant_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Columns holding a single repeated value over the given rows.

    Uses peak-to-peak rather than std: a single-valued float64 column can report
    std ~1e-13 instead of exactly 0. See notes/decisions.md D-005.
    """
    return [c for c in cols if float(np.ptp(df[c].to_numpy(dtype=np.float64))) == 0.0]
