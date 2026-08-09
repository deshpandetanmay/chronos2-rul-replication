"""Assemble the Zenodo deposit package.

Contents (brief §10.1 item 6): code, the derived evaluation splits as explicit
unit-id lists, metrics.csv, figures, the run manifest, and CITATION.cff.

Raw C-MAPSS files are deliberately NOT redistributed -- they are NASA's, and the
deposit records the source URL plus SHA-256 of every file instead, so the input can
be reconstructed and verified byte-for-byte without us mirroring someone else's data.

Run: uv run python -m src.package_zenodo
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile

from . import config as C

DIST = C.ROOT / "dist"
NAME = "tsfmrul-cmapss-fd001-replication"

CODE = ["src", "run.sh", "requirements.lock", "pyproject.toml", "CITATION.cff",
        "DEVIATIONS.md", "README.md"]
NOTES = ["source_paper.md", "decisions.md", "future_work.md"]
RESULTS = ["metrics.csv", "unit_splits.json", "phase1_evidence.json",
           "phase2_report.json", "phase3_report.json", "phase4_report.json",
           "c1_comparison.json", "diag_head_budget.json", "preprocessor.json",
           "seed_study.json", "ridge_probe.json", "conformal_cluster.json",
           "equivariance.json", "standard_protocol.json",
           "window_index_train.parquet", "window_index_calib.parquet",
           "window_index_eval.parquet"]
STDOUT = ["phase1_stdout.txt", "phase2_stdout.txt", "phase3_stdout.txt",
          "phase4_stdout.txt", "phase5_stdout.txt", "c1_stdout.txt",
          "diag_head_budget_stdout.txt", "seed_study_stdout.txt",
          "ridge_probe_stdout.txt", "conformal_cluster_stdout.txt",
          "equivariance_stdout.txt", "standard_protocol_stdout.txt"]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    root = DIST / NAME
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    copied: list[str] = []

    def take(src, dst_rel):
        src = C.ROOT / src if not str(src).startswith("/") else src
        if not src.exists():
            print(f"  skip (absent): {dst_rel}")
            return
        dst = root / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc"))
            for f in sorted(dst.rglob("*")):
                if f.is_file():
                    copied.append(str(f.relative_to(root)))
        else:
            shutil.copy2(src, dst)
            copied.append(dst_rel)

    for item in CODE:
        take(item, item)
    for n in NOTES:
        take(C.NOTES / n, f"notes/{n}")
    for r in RESULTS:
        take(C.RESULTS / r, f"results/{r}")
    for s in STDOUT:
        take(C.RESULTS / s, f"results/logs/{s}")
    take(C.RESULTS / "preds", "results/preds")
    for f in sorted(C.FIGURES.glob("*.png")):
        take(f, f"figures/{f.name}")
    for f in sorted(C.PAPER.glob("*.tex")):
        take(f, f"paper/{f.name}")
    # The style file and the compiled PDF, so the deposit is readable without a build.
    take(C.PAPER / "neurips_2024.sty", "paper/neurips_2024.sty")
    take(C.PAPER / "build" / "main.pdf", "paper/main.pdf")
    take(C.ROOT / "run_manifest.json", "run_manifest.json")

    # --------------------------------------------------------------- splits
    # The splits are the single most important derived artifact: without the exact
    # unit-id lists nobody can reproduce a single number, and they are cheap to state.
    splits = json.loads((C.RESULTS / "unit_splits.json").read_text())
    lines = [
        "# Derived evaluation splits (C-MAPSS FD001 train file, by engine unit)",
        "#",
        f"# seed={splits['seed']}  fractions={splits['fractions']}",
        "# Units are disjoint across splits by construction; see src/splits.py.",
        "# The official test file (test_FD001.txt) is a separate secondary evaluation",
        "# and uses all 100 of its own units, offset by +1000 in prediction files.",
        "",
    ]
    for name in ("train", "calib", "eval"):
        ids = splits["units"][name]
        lines.append(f"[{name}] n={len(ids)}")
        lines.append(" ".join(str(i) for i in ids))
        lines.append("")
    (root / "results" / "unit_splits.txt").write_text("\n".join(lines))
    copied.append("results/unit_splits.txt")

    # --------------------------------------------------------------- data pointer
    (root / "DATA_SOURCE.md").write_text(
        "# Input data (not redistributed)\n\n"
        "The raw NASA C-MAPSS files are not included in this deposit. They are\n"
        "obtainable from the official NASA PHM mirror and verified by checksum:\n\n"
        f"    {C.DATA_SOURCE_URL}\n\n"
        f"archive SHA-256: `{C.DATA_ZIP_SHA256}`\n\n"
        "Per-file SHA-256 of the three FD001 files this study used, re-verified at the\n"
        "start of every run (`src/cmapss.verify_files`, which aborts on mismatch):\n\n"
        + "".join(f"    {k}  {v}\n" for k, v in C.EXPECTED_SHA256.items())
        + "\n`bash run.sh fetch-data` downloads and verifies them.\n"
    )
    copied.append("DATA_SOURCE.md")

    # --------------------------------------------------------------- manifest
    inventory = {
        "package": NAME,
        "n_files": len(copied),
        "sha256": {rel: sha256(root / rel) for rel in sorted(set(copied))
                   if (root / rel).is_file()},
    }
    (root / "MANIFEST.sha256.json").write_text(
        json.dumps(inventory, indent=2) + "\n")

    archive = DIST / f"{NAME}.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(root.rglob("*")):
            if f.is_file():
                z.write(f, f"{NAME}/{f.relative_to(root)}")

    size_mb = archive.stat().st_size / 1e6
    print(f"  package tree: {root.relative_to(C.ROOT)}  ({len(copied)} files)")
    print(f"  archive:      {archive.relative_to(C.ROOT)}  ({size_mb:.1f} MB)")
    print(f"  inventory:    {len(inventory['sha256'])} files checksummed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
