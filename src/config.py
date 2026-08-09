"""Central configuration. Every constant that affects a result lives here.

Nothing in this project may introduce randomness that is not seeded from a value
declared in this file.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "cmapss_raw"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
NOTES = ROOT / "notes"
PAPER = ROOT / "paper"
CACHE = ROOT / "data" / "cache"

for _d in (RESULTS, FIGURES, NOTES, PAPER, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- dataset

SUBSET = "FD001"  # scope-locked: FD002/3/4 are explicitly out of scope

OP_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
ID_COLS = ["unit", "cycle"]
ALL_COLS = ID_COLS + OP_COLS + SENSOR_COLS  # 26 whitespace-delimited fields

# SHA-256 of the three FD001 files as extracted from the official NASA PHM
# S3 mirror. Verified at ingest; a mismatch aborts the run.
EXPECTED_SHA256 = {
    "train_FD001.txt": "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8",
    "test_FD001.txt": "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851",
    "RUL_FD001.txt": "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca",
}
DATA_SOURCE_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
DATA_ZIP_SHA256 = "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"

# ---------------------------------------------------------------- backbone

BACKBONE_ID = "amazon/chronos-2"
# Pinned so a silent upstream re-upload cannot change our results.
BACKBONE_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
BACKBONE_PATCH_SIZE = 16  # verified from config.json: input_patch_size == stride == 16
BACKBONE_D_MODEL = 768

# ---------------------------------------------------------------- protocol

# Look-back window, in flight cycles. See notes/decisions.md D-002 for the
# justification and for why the paper's L=5 does not transfer.
LOOKBACK = 30

# Secondary look-back for the point-accuracy-only C1 robustness table (D-009).
# Chosen as the paper's own saturation point (§III-E). Arms are compared by their
# *within-L ranking*: absolute error is not comparable across L, because the
# window-fits constraint couples L to which truncation points exist and therefore
# to the eval label distribution.
LOOKBACK_SECONDARY = 80
SECONDARY_ARMS = ("trivial", "lgbm_summary", "lgbm_raw", "tsfm", "control_randproj")

# Split fractions over *engine units*, never rows.
SPLIT_FRACTIONS = {"train": 0.50, "calib": 0.20, "eval": 0.30}

# Truncation points sampled per evaluation unit.
TRUNCATIONS_PER_UNIT = 15

# Target variants. Variant B is the community-standard piecewise-linear cap.
RUL_CAP = 125

# Calibration analysis.
NOMINAL_LEVELS = (0.50, 0.80, 0.90, 0.95, 0.99)

# Quantile levels every interval-producing arm must emit: the central-interval
# endpoints for each nominal level, plus the median. Derived, not hand-typed, so
# NOMINAL_LEVELS and QUANTILE_LEVELS cannot drift apart.
QUANTILE_LEVELS = tuple(
    sorted({0.5} | {round(q, 6) for c in NOMINAL_LEVELS
                    for q in ((1 - c) / 2, (1 + c) / 2)})
)

# LightGBM hyperparameters. FIXED, no search anywhere in this project (brief §2).
# No early stopping: the only untouched data is the calibration split, and letting
# a base model see it would invalidate the conformal step (D-007).
LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "max_depth": -1,
    "n_jobs": -1,
    "verbose": -1,
    # Reproducibility, not tuning: LightGBM documents `deterministic` as requiring
    # one of force_row_wise/force_col_wise to give bit-identical fits.
    "deterministic": True,
    "force_row_wise": True,
}
RUL_BINS = [
    ("gt100", 100.0, float("inf")),
    ("50to100", 50.0, 100.0),
    ("20to50", 20.0, 50.0),
    ("lt20", 0.0, 20.0),
]
BOOTSTRAP_RESAMPLES = 2000  # by-unit clustered bootstrap

# ---------------------------------------------------------------- seeds

SEEDS = {
    "unit_split": 20260808,
    "truncation_sampling": 11990,
    "head_init": 4242,
    "head_shuffle_control": 5150,
    "channel_scramble_control": 6171,
    "random_projection_control": 7192,
    "lightgbm": 1234,
    "bootstrap": 98765,
    "torch_global": 20260808,
    "numpy_global": 20260808,
}

# ---------------------------------------------------------------- head

# Regression head. Architecture is the paper's (§II-B): two linear layers, hidden
# width m, ReLU, dropout after the first hidden layer, final ReLU to enforce y>=0.
# The paper never gives m or p numerically and its ~300K/~250K parameter counts are
# not reconstructible (notes/source_paper.md AMBIGUITY 5), so both are our choice.
# m=32 is the nearest power of two putting the head in the same order (4.2e5) as the
# paper's stated ~3e5 given our input dim of 17*768=13,056. p=0.1 matches the
# backbone's own configured dropout_rate. See notes/decisions.md D-012.
HEAD_HIDDEN = 32
HEAD_DROPOUT = 0.1
HEAD_EPOCHS = 50          # paper: "a maximum of 50 epochs"
HEAD_LR = 1e-3            # paper: Adam, lr 1e-3
HEAD_BATCH = 64           # paper: batch size 64

# Primary token reduction and the one ablated alternative (brief §7.2).
REDUCTION_PRIMARY = "tokmean"
REDUCTION_ABLATION = "reg"
