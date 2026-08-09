#!/usr/bin/env bash
# Reproduce every result, figure and table from a clean checkout on one machine.
#
#   bash run.sh            # everything
#   bash run.sh fetch-data # data only
#
# PHASES MUST BE SEPARATE PROCESSES. torch and LightGBM cannot coexist in one
# process on macOS: whichever is imported second segfaults on its first threaded
# kernel (see src/ompguard.py and notes/decisions.md D-010). Phase 2 is
# LightGBM-only; phases 3-4 are torch-only. `src/ompguard.py` enforces this and
# fails loudly rather than crashing, so do not merge these invocations.

set -euo pipefail
cd "$(dirname "$0")"

DATA_URL="https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
ZIP_SHA256="c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"

sha_of() { shasum -a 256 "$1" | cut -d' ' -f1; }

fetch_data() {
  mkdir -p data
  if [ ! -f data/cmapss_raw/train_FD001.txt ]; then
    echo "==> fetching C-MAPSS from the NASA PHM mirror"
    curl -sSL --max-time 600 -o data/cmapss.zip "$DATA_URL"
    got="$(sha_of data/cmapss.zip)"
    if [ "$got" != "$ZIP_SHA256" ]; then
      echo "!! zip SHA-256 mismatch: expected $ZIP_SHA256, got $got" >&2
      exit 1
    fi
    (cd data && unzip -oq cmapss.zip \
      && unzip -oq "6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip" \
           -d cmapss_raw)
  fi
  echo "==> data present; per-file checksums are verified again inside phase 1"
}

if [ "${1:-all}" = "fetch-data" ]; then fetch_data; exit 0; fi

echo "==> installing pinned dependencies"
uv sync --frozen 2>/dev/null || uv pip install -r requirements.lock

fetch_data

echo "==> phase 1: evaluation design (splits, windows, targets, preprocessing)"
uv run python -m src.phase1        | tee results/phase1_stdout.txt

echo "==> phase 2: baselines FIRST, before the foundation model (LightGBM process)"
uv run python -m src.phase2        | tee results/phase2_stdout.txt

echo "==> phase 3: frozen Chronos-2 embeddings arm (torch process)"
uv run python -m src.phase3        | tee results/phase3_stdout.txt

echo "==> phase 4: attribution controls (torch process)"
uv run python -m src.phase4        | tee results/phase4_stdout.txt

echo "==> phase 5: calibration analysis + figures"
uv run python -m src.phase5        | tee results/phase5_stdout.txt

echo "==> C1 adjudication (paired clustered bootstrap)"
uv run python -m src.compare_c1    | tee results/c1_stdout.txt

echo "==> diagnostic: head budget is not the cause of the negative C1"
uv run python -m src.diag_head_budget | tee results/diag_head_budget_stdout.txt

echo "==> paper numbers and tables"
uv run python -m src.paper_numbers | tee results/paper_numbers_stdout.txt

echo "==> Zenodo deposit package"
uv run python -m src.package_zenodo

echo
echo "done. key outputs:"
echo "  run_manifest.json          environment, seeds, checkpoint revision"
echo "  results/metrics.csv        long-format metrics (the deposited numbers)"
echo "  results/unit_splits.json   explicit unit-id lists per split"
echo "  figures/*.png              paper figures"
echo "  paper/numbers.tex, tab_*.tex  generated from metrics.csv"
echo "  dist/                      Zenodo deposit package"
