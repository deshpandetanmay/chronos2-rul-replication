"""Pure-numpy feature builders shared by every arm.

Extracted from `baselines.py` (which imports LightGBM) and `phase4.py` (which imports
torch) so that a consumer needing only the features -- the ridge probe -- can import them
without dragging in either OpenMP-linked library. See src/ompguard.py.

One definition per feature set, imported everywhere, so the LightGBM baseline, the
random-projection control and the ridge probe are provably looking at identical features.
"""

from __future__ import annotations

import numpy as np

SUMMARY_STATS = ("mean", "std", "min", "max", "first", "last", "slope")


def summary_features(win: np.ndarray) -> np.ndarray:
    """Per-channel window summaries -> (n, n_channels * 7).

    Exactly the set the brief specifies: mean, std, min, max, first, last, linear slope.
    The slope is the OLS coefficient of the channel against cycle index, in closed form
    against a fixed mean-centred time base (identical for every window, since every
    window has length L).
    """
    n, L, D = win.shape
    tc = np.arange(L, dtype=np.float64)
    tc -= tc.mean()
    denom = float((tc**2).sum())
    feats = [
        win.mean(axis=1),
        win.std(axis=1, ddof=0),
        win.min(axis=1),
        win.max(axis=1),
        win[:, 0, :],
        win[:, -1, :],
        np.einsum("nld,l->nd", win.astype(np.float64), tc) / denom,
    ]
    return np.concatenate([f.astype(np.float64) for f in feats], axis=1)


def summary_feature_names(channels: list[str]) -> list[str]:
    return [f"{c}__{s}" for s in SUMMARY_STATS for c in channels]


def raw_features(win: np.ndarray) -> np.ndarray:
    """Flattened raw window -> (n, L * n_channels), cycle-major."""
    return win.reshape(win.shape[0], -1).astype(np.float64)


def raw_feature_names(channels: list[str], lookback: int) -> list[str]:
    return [f"{c}__lag{lookback - 1 - i}" for i in range(lookback) for c in channels]


def random_projection(win: np.ndarray, d_out: int, seed: int) -> np.ndarray:
    """Fixed Gaussian projection of the flattened window to `d_out` dims."""
    n, L, D = win.shape
    d_in = L * D
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((d_in, d_out), dtype=np.float32) / np.sqrt(d_in)
    return win.reshape(n, d_in).astype(np.float32) @ R
