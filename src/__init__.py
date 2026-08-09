"""Frozen TSFM embeddings for RUL prediction: replication + calibration study.

NOTE: torch and LightGBM must not be imported into the same process here. See
`src/ompguard.py` for the measured reason and `notes/decisions.md` D-010. Phase 2
is LightGBM-only; Phases 3-4 are torch-only. Nothing is imported at package level
so that each phase controls its own import set.
"""
