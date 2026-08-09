"""Guard against loading two OpenMP runtimes into one process.

This venv ships several copies of libomp (`torch/lib/libomp.dylib`,
`sklearn/.dylibs/libomp.dylib`, plus whatever LightGBM links). On macOS, torch and
LightGBM cannot coexist in a single process: whichever is imported *second*
segfaults as soon as it runs a multithreaded kernel. Measured, both directions:

    import torch    ; import lightgbm ; lgbm.fit(n_jobs=-1)   -> SIGSEGV
    import lightgbm ; import torch    ; pipe.embed(...)       -> SIGSEGV
    import torch    ; import lightgbm ; lgbm.fit(n_jobs=1)    -> OK (single-threaded)
    either one alone                                          -> OK

Import order is therefore not a fix, it only chooses which phase crashes. The fix
is process separation, which the phase layout already gives us: Phase 2 is
LightGBM-only, Phases 3-4 are torch-only, Phases 1 and 5 need neither. This module
makes that requirement explicit and fails loudly instead of segfaulting, so the
constraint cannot silently regress when someone adds an import.

We deliberately do NOT set KMP_DUPLICATE_LIB_OK: it suppresses the duplicate-runtime
warning without making duplicate loading safe, which is how a silent segfault
becomes a silent wrong answer.

See notes/decisions.md D-010.
"""

from __future__ import annotations

import sys


class OpenMPConflict(RuntimeError):
    pass


def assert_single_omp_runtime(context: str = "") -> dict:
    """Fail if both torch and lightgbm are loaded in this process."""
    have_torch = "torch" in sys.modules
    have_lgbm = "lightgbm" in sys.modules
    if have_torch and have_lgbm:
        raise OpenMPConflict(
            f"torch and lightgbm are both imported in this process{' (' + context + ')' if context else ''}. "
            "On macOS this combination segfaults in whichever library was imported "
            "second as soon as it uses threads. Run LightGBM work (Phase 2) and torch "
            "work (Phases 3-4) as separate processes. See src/ompguard.py."
        )
    return {"torch_loaded": have_torch, "lightgbm_loaded": have_lgbm}
