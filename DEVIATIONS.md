# Deviations

Every departure from (a) the project brief or (b) the source paper's protocol, with
reasoning. Kept current as phases complete.

Entries cross-reference the decision log as `D-nnn`. That log (`notes/decisions.md`) is
generated alongside a run and is published in the Zenodo deposit rather than tracked in this
repository; `bash run.sh` reproduces it locally.

Status legend: **forced** = the paper's choice is not defined on this dataset;
**required** = the brief mandates it; **chosen** = a genuine judgement call.

---

## A. Deviations from the source paper (arXiv:2606.11990v2)

### A-1 · Dataset — private Nokia telemetry → public NASA C-MAPSS FD001 · *required*
The whole point of the replication (brief §1.4, Gap 1). Consequences that must be carried
into the limitations section: different domain (turbofan simulation vs telecom hardware),
different scale (20,631 rows / 100 units vs 297,345 samples), different channel count
(17 kept vs D=87), different time base (flight cycles vs 1-hour resampling), and RUL
measured in cycles rather than days.

### A-2 · Split protocol — chronological 85:15 → by-unit 50/20/30 · *required / forced*
Paper §III-A cuts one long per-device telemetry stream at a timestamp. C-MAPSS is a set of
100 independent run-to-failure trajectories with one failure each; there is no shared clock
to cut on, and a row-wise cut would leak trajectory information. By-unit splitting is
mandated by the brief and is the standard control here. Note this makes our setting
*harder* than the paper's: their chronological cut keeps train and test on the same
physical devices, whereas our eval units are never seen in any form during training.
The 20% calibration split has no analogue in the paper — it exists for C3.
→ D-001

### A-3 · Look-back window — L=5 (hours) → L=30 (cycles) · *forced / chosen*
The integer 5 does not transfer across time bases. Full reasoning, including the finding
that L=5 is a single padded patch under Chronos-2's `input_patch_size=16`, in D-002 and
`notes/source_paper.md` DISCREPANCY 2. **This is the deviation most likely to affect the
C1 outcome and is flagged as such.**
→ D-002

### A-3b · Context-length sweep — full sweep (L≈0–100) → two points, L=30 and L=80 · *chosen*
Paper §III-E sweeps L and reports saturation at 80. We report two look-backs: L=30 with the
complete protocol, and L=80 with point accuracy only (all 3 baselines + TSFM +
random-projection control, both target variants). A full sweep is out of scope per the
brief's hyperparameter-search lock, and a full dual-L protocol would crowd the calibration
analysis out of 4 pages.

**Cross-L errors are not comparable and must never be tabulated as if they were.** The
window-fits constraint couples L to which truncation points exist, so the L=80 eval set has a
materially different label distribution (mean RUL 63.3 vs 88.7; 10.2% vs 26.4% at the cap).
Only the *within-L ranking of arms* is comparable, which is sufficient for C1. Stated
explicitly in the paper.
→ D-009

### A-4 · Preprocessing — resampling / gap filtering / NaN filtering omitted · *forced*
Paper §II-A steps 1–3 exist to regularise irregular industrial telemetry. C-MAPSS is
already on a regular per-cycle grid with no gaps and no missing values, all three asserted
in `cmapss.integrity_checks`. Steps 4–6 (1st/99th percentile clipping, z-scoring,
train-split-only statistics) are applied exactly as specified.
→ D-007

### A-5 · Target cap — 1000 days → 125 cycles, and an uncapped variant added · *required*
The paper caps (§II-A, `y_max = 1000` days) and reports only the capped setting. 125 cycles
is the C-MAPSS-community analogue. The brief additionally requires an uncapped Variant A,
which the paper has no counterpart for. Measured consequence: 26.4% of eval windows sit at
or above the cap, so Variant B carries a substantial point mass at the ceiling — the exact
artifact the brief warns can masquerade as a calibration failure.

### A-6 · Baselines — 7 tuned baselines → 3 fixed-hyperparameter baselines · *required*
The brief locks scope to a trivial marginal, window-summary LightGBM and raw-window
LightGBM, with fixed hyperparameters and no search. The paper's GrBoost baseline
(per-channel mean/std/min/max/quantiles, last value, first differences, trend slope) is the
direct antecedent of our window-summary arm. We do not implement LSTM/GRU/TCN/Transformer.
Recorded asymmetry: the paper *tunes* its baselines on a ~10% validation split (§III-C)
while we tune nothing, so our baselines are, if anything, weaker than theirs — this favours
the TSFM arm and must not be read as evidence for it.

### A-7 · Metrics — MAE/MSE → MAE, RMSE, prognostics score, plus the full calibration suite · *required*
The paper reports no uncertainty metric of any kind. Coverage, sharpness, pinball loss,
regime-conditioned coverage and conformal repair are additions (brief §1.4, Gap 2). The
paper names this as its own future work (§IV item i), which we cite.

### A-8 · Backbone checkpoint — no substitution required · *n/a, recorded as a non-deviation*
`amazon/chronos-2` is public, ungated, Apache-2.0, and pinned at revision
`29ec3766d36d6f73f0696f85560a422f50e8498c`. The replication claim is **not** weakened on
this axis. Verified: 119,477,664 params, d_model 768, 12 layers.

### A-9 · Embedding extraction — Eq. 6's `H ∈ R^{L×h}` is not obtainable · *forced*
`Chronos2Pipeline.embed` returns `(n_variates, ceil(L/16)+2, 768)`: no per-timestep axis of
length L, and a variate axis the paper's equation omits. Reducing that to a head input is a
choice the paper does not specify. Resolved in Phase 3; ablated per the brief.
→ `notes/source_paper.md` DISCREPANCY 3, AMBIGUITY 2–4

### A-10 · Head hyperparameters m and p are never stated in the paper · *forced*
Architecture is specified (2 linear layers, ReLU, dropout after the first hidden layer,
final ReLU) but the hidden width `m` and dropout rate `p` appear only as symbols. The
~300K/~250K parameter figures do not pin them down and are themselves inconsistent with
Eq. 6 (they differ across devices, which a `L×h` input cannot explain). Our choice is
recorded in Phase 3.
→ `notes/source_paper.md` AMBIGUITY 5

### A-9b · Token reduction chosen as `tokmean`, ablated against `reg` · *forced*
Eq. 6's `H ∈ R^{L×h}` is unobtainable. Primary reduction is the mean over the token axis;
the ablation is the `[REG]` token alone. Both concatenate per-variate embeddings in fixed
sensor order — which is **forced, not chosen**: the backbone is exactly
permutation-equivariant across variates, so any variate-symmetric pooling would make the
Phase 4 channel-scramble control mathematically vacuous.
→ D-011, D-012

### A-10b · Head is `m=32`, `p=0.1`, 417,857 parameters · *forced*
The paper's `m` and `p` are symbols only. Our head is ~1.4x the paper's stated ~300K
budget, on a 13,056-dim input, with a 286:1 backbone:head ratio.
→ D-012

### A-11 · A quantile head is added alongside the paper's MSE point head · *required*
The paper's head is a point predictor trained on MSE and emits no uncertainty at all. The
brief requires intervals, so we run both: the paper-faithful point head (intervals come
only from the Phase 5 conformal step) and an 11-output quantile head trained on pinball
loss. Both are reported.

---

## B. Deviations from the project brief

### B-1 · Calibration windows use the eval scheme, not dense sampling · *chosen*
The brief specifies the 15-per-unit stratified scheme for the eval set and is silent on how
calibration windows are built. Using dense sampling there would break exchangeability
between calibration and eval scores and corrupt the C3 result. Calib is therefore
constructed identically to eval (15 stratified points/unit); only the 50 training units use
dense sampling.
→ D-004

### B-2 · Head training budget diagnostic run outside the reported arms · *chosen*
The brief locks out hyperparameter search. We ran a 4-cell grid (`m ∈ {32,256}` x
`epochs ∈ {50,200}`) purely to exclude "the head was underfit" as an explanation for the
negative C1, because the training loss was still falling at the paper's 50-epoch limit.
No reported arm is selected from this grid; the reported arm stays at the paper's budget.
The grid is published so the exclusion is on the record.
→ D-014

---

## C. Corrections made during the run

### C-1 · Constant-column detection: `std == 0` → `ptp == 0` · *bug, fixed in Phase 1*
The initial implementation would have kept `s1`, `s5` and `s16` — each single-valued — and
z-scored them by std ≈ 1e-13…1e-18, injecting amplified floating-point noise as three input
channels into every arm. Caught by a cross-check that flagged the train-split and
whole-file constant sets disagreeing, which is arithmetically impossible. Fixed before any
model was trained; no result was ever computed under the buggy version.
→ D-005

### C-2 · Import-order "fix" for the OpenMP segfault was wrong and was replaced · *bug*
A package-level `import lightgbm` was added to `src/__init__.py` to stop a Phase 2
segfault. It worked, and moved the identical crash into Phase 3. The conflict is
symmetric: whichever of torch/LightGBM loads second segfaults on its first threaded
kernel. Replaced with process separation plus an explicit guard that fails loudly.
No result was computed under the broken configuration — both failures were crashes.
→ D-010

### C-3 · `n_patches` was read off the variate axis in extraction metadata · *bug, metadata only*
`per_item_embed_shape` is `(n_variates, num_patches+2, d_model)`; the code took
`shape[0]-2`, reporting `n_patches = 15` instead of `2` at L=30. Metadata only — no
feature, label or metric depended on it — but it was wrong in `run_manifest.json`, so it
was fixed and Phase 3 was re-run from scratch with the embedding cache cleared.

---

## D. Phase 5 methodological choices

### D-1 · Conformal method differs by arm (CQR vs absolute residual) · *chosen*
CQR for arms that emit quantiles, absolute-residual conformal for point-only arms.
Applying one method uniformly would either deny the quantile arms their adaptivity or
invent quantiles the point arms do not have.
→ D-015

### D-2 · Interval lower bounds clamped at 0; upper bounds not clamped at the cap · *chosen*
RUL cannot be negative, so the lower clamp is free of coverage cost and applied to every
arm. The Variant B upper bound is left unclamped despite truth being <= 125, so that
width numbers stay interpretable against the point-mass discussion. Variant B widths are
therefore wider than a cap-aware method would give.
→ D-015

### D-3 · C3 claim weakened from "restores nominal coverage" to "approximately nominal" · *required by the evidence*
Mean coverage tracks nominal within 1.4 points at every level, but 19 of 100 cells sit
significantly below nominal, consistent with our points being clustered by unit so that
exchangeability holds at the unit level and the effective calibration sample is 20 units
rather than 300 windows. The stronger claim would exceed the CIs.
→ D-016

### C-4 · `n/a` sentinel in metrics.csv was read back as NaN · *bug*
The `conformal` column used `"n/a"` for non-interval rows; `pandas.read_csv` maps that to
NaN, dropping every point-accuracy row on read-back. Renamed to `"point"`.
→ D-018

### C-5 · Extraction metadata was lost on embedding-cache hits · *bug*
Measured token/patch counts and throughput were discarded when a cached run loaded
embeddings, so the paper generator could not find them. Now persisted in a `.meta.json`
sidecar; caches regenerated so every reported number is measured rather than derived
from config.
→ D-019


---

## E. Post-review revision (2026-08-09)

### E-1 · Head arms reported over 5 seeds, with a combined unit+seed interval · *required by review*
The clustered bootstrap measured evaluation-set uncertainty only. Head arms are now reported
as a mean over 5 training seeds, and the C1 interval resamples units *and* draws a seed per
arm per replicate. C1 survives; the previously reported single seed was at the favourable end
of the TSFM arm's seed distribution.
→ D-021 (M1)

### E-2 · The random-projection pinball result was a favourable seed, and the loss gap has a mechanism · *correction*
Seed-averaged, the random-projection quantile arm is not better than LightGBM
(-1.05 [-4.23, +2.24]), which removes a contradiction between our table and our conclusion.
The MSE-vs-pinball gap is real and caused by the original's final ReLU killing output units
(15.0% vs 6.1% of predictions pinned at zero).
→ D-021 (M2)

### E-3 · Ridge linear probe added; the specified head was costing ~5.5 RMSE · *required by review*
A convex head-free probe shows the original's MLP head is materially suboptimal for these
embeddings, without changing the C1 ranking or the random-projection tie.
→ D-021 (M4)

### E-4 · Paper reframed around variate permutation equivariance · *required by review*
The structural property leads; the replication is its evidence. Promoted to a first-class
reproducible measurement.
→ D-021 (M5)

### E-5 · External FD001 numbers are not quoted; our baseline is run under the standard protocol instead · *chosen*
We could not verify published FD001 figures from primary sources, so rather than quote them
we measured our baseline under the literature's protocol (12.58 RMSE). No claim depends on
that run, which trains on engines held out everywhere else.
→ D-021 (M6)

### C-6 · `extractSeconds` reported the cache-hit loop timer (0 s) instead of measured time · *bug*
Fixed to sum the measured per-split extraction time from the metadata sidecars.
→ D-022


### E-6 · Paper prose rewritten into a conventional academic register · *editorial*
The draft read as machine-written. Audited and fixed six specific patterns (em-dash
density, argumentative paragraph headings, bolded rhetorical sentences, evaluative
vocabulary, second-person imperatives, punchy title). No number, claim or caveat changed;
material relocated to the appendix to hold four pages.
→ D-023
