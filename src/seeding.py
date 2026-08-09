"""Global seeding. Called once at the top of every entry point."""

from __future__ import annotations

import os
import random

from . import config as C


def seed_everything() -> dict:
    """Seed every global RNG we can reach, and report what was set."""
    import numpy as np
    import torch

    py = C.SEEDS["numpy_global"]
    random.seed(py)
    np.random.seed(C.SEEDS["numpy_global"])
    torch.manual_seed(C.SEEDS["torch_global"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ.setdefault("PYTHONHASHSEED", str(py))
    return {
        "python_random": py,
        "numpy_global": C.SEEDS["numpy_global"],
        "torch_global": C.SEEDS["torch_global"],
        "deterministic_algorithms": True,
    }
