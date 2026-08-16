"""Run manifest: everything needed to attribute a number to an environment."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

from . import config as C

TRACKED_PACKAGES = [
    "chronos-forecasting",
    "torch",
    "transformers",
    "numpy",
    "pandas",
    "scikit-learn",
    "lightgbm",
    "scipy",
    "matplotlib",
    "safetensors",
    "tokenizers",
]


def _pkg_versions() -> dict[str, str | None]:
    out = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def _git() -> dict[str, str | None]:
    def run(*args):
        # Must check returncode: `git rev-parse HEAD` in a repo with no commits
        # prints the literal string "HEAD" on stdout and exits non-zero. Taking
        # stdout unconditionally records "HEAD" as if it were a commit SHA.
        try:
            r = subprocess.run(
                ["git", *args], cwd=C.ROOT, capture_output=True, text=True, timeout=15
            )
            return (r.stdout.strip() or None) if r.returncode == 0 else None
        except Exception:
            return None

    sha = run("rev-parse", "HEAD")
    out = {
        "sha": sha,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")) if sha else None,
    }
    if sha is None:
        out["note"] = (
            "no commit SHA: the working tree is not a git repository, or has no "
            "commits yet. Commit before depositing so results are attributable to a "
            "revision."
        )
    return out


def _device() -> dict:
    try:
        import torch
    except ImportError:
        return {"available": False}
    # Report the device actually SELECTED by embed.load_backbone, not merely what is
    # available. Reporting "mps" because MPS exists would misattribute every number in
    # this run: MPS is deliberately never used, because it is incompatible with
    # torch.use_deterministic_algorithms(True) for this checkpoint (D-013).
    mps = bool(torch.backends.mps.is_available())
    if torch.cuda.is_available():
        kind, name = "cuda", torch.cuda.get_device_name(0)
    else:
        kind, name = "cpu", platform.processor() or "cpu"
    out = {
        "kind_used": kind,
        "name": name,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": mps,
        "mps_used": False,
        "num_threads": torch.get_num_threads(),
        "selection_rule": "cuda if available, else cpu; never mps",
    }
    if mps:
        out["mps_skipped_reason"] = (
            "MPS is available but unused: loading this checkpoint onto MPS with "
            "deterministic algorithms enabled aborts the process. Determinism is "
            "required; MPS is not, and CPU is faster at the primary look-back."
        )
    return out


def _chip() -> str | None:
    if platform.system() != "Darwin":
        return None
    try:
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip() or None
    except Exception:
        return None


def _redact(value):
    """Rewrite paths under the repository root to `<repo>/...` form, recursively."""
    root = str(C.ROOT)
    if isinstance(value, str):
        return value.replace(root, "<repo>")
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def build(extra: dict | None = None) -> dict:
    man = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project": "tsfmrul",
        "subset": C.SUBSET,
        "protocol": {
            "lookback_cycles": C.LOOKBACK,
            "split_fractions": C.SPLIT_FRACTIONS,
            "truncations_per_eval_unit": C.TRUNCATIONS_PER_UNIT,
            "rul_cap_variant_b": C.RUL_CAP,
            "nominal_levels": list(C.NOMINAL_LEVELS),
            "rul_bins": [[n, lo, hi] for n, lo, hi in C.RUL_BINS],
            "bootstrap_resamples": C.BOOTSTRAP_RESAMPLES,
        },
        "seeds": dict(C.SEEDS),
        "backbone": {
            "checkpoint": C.BACKBONE_ID,
            "revision": C.BACKBONE_REVISION,
            "d_model": C.BACKBONE_D_MODEL,
            "input_patch_size": C.BACKBONE_PATCH_SIZE,
        },
        "data_source": {
            "url": C.DATA_SOURCE_URL,
            "zip_sha256": C.DATA_ZIP_SHA256,
            "file_sha256_expected": dict(C.EXPECTED_SHA256),
        },
        "packages": _pkg_versions(),
        "python": {
            "version": sys.version,
            # Interpreter recorded relative to the repo (or by name) rather than
            # absolutely: this file is published, and an absolute path discloses the
            # operator's home-directory layout without aiding reproduction.
            "executable": _redact(sys.executable),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "chip": _chip(),
        },
        "device": _device(),
        "git": _git(),
    }
    if extra:
        man.update(extra)
    return man


def write(man: dict, path=None) -> None:
    """Write the manifest with repository-root paths redacted.

    Applied at write time rather than at construction so that a section merged in by
    any phase is covered, whatever it happens to contain.
    """
    path = path or (C.ROOT / "run_manifest.json")
    path.write_text(json.dumps(_redact(man), indent=2, default=str) + "\n")


def merge_into(section: str, payload: dict, path=None, fresh: bool = False) -> dict:
    """Add/replace one section of the manifest on disk.

    `fresh=True` rebuilds from scratch, which phase 1 uses so that phase sections from
    a previous, possibly differently-configured run cannot survive into this one.
    """
    path = path or (C.ROOT / "run_manifest.json")
    man = build() if fresh or not path.exists() else json.loads(path.read_text())
    man.setdefault("phases", {})[section] = payload
    write(man, path)
    return man
