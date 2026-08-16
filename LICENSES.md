# Licensing

This deposit contains two kinds of material under two licences. Zenodo's metadata form
accepts a single licence, so the record as a whole is registered as **CC-BY-4.0**, which
matches the paper; the code remains MIT-licensed as stated here and in `LICENSE`.

| Material | Licence | SPDX |
|---|---|---|
| Code (`src/`, `run.sh`) | MIT — see `LICENSE` | `MIT` |
| Paper, figures, notes, derived metrics (`paper/`, `figures/`, `notes/`, `results/`) | Creative Commons Attribution 4.0 International | `CC-BY-4.0` |

The CC-BY-4.0 legal text is not reproduced here. Canonical version:
<https://creativecommons.org/licenses/by/4.0/legalcode>. Human-readable summary:
<https://creativecommons.org/licenses/by/4.0/>.

## Third-party material, not covered by the above

Nothing in this deposit re-licenses material belonging to others, and no third-party
artefact is redistributed here.

- **NASA C-MAPSS turbofan degradation data.** Not included. Obtained from the NASA
  Prognostics Center of Excellence repository and verified by SHA-256 at every run; the
  source URL and per-file digests are recorded in `DATA_SOURCE.md`. As a work of the US
  federal government it is generally not subject to domestic copyright, but the
  authoritative terms are those of the distributing repository.
- **Chronos-2 (`amazon/chronos-2`).** Weights are not included. Apache-2.0, obtained from
  HuggingFace at the pinned revision recorded in `run_manifest.json`.
- **`paper/neurips_2024.sty`.** The NeurIPS 2024 LaTeX style file, included so the paper
  builds from a clean checkout. Distributed by the NeurIPS organisers under their own
  terms; remove it if redistribution is unwanted.
- **Cited works.** Referenced bibliographically only. No text, table or figure from the
  paper being replicated is reproduced in this deposit.
