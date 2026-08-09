"""Frozen Chronos-2 backbone: loading, freeze verification, extraction, reduction.

Verified facts about the installed checkpoint that this module depends on (all
measured, see notes/source_paper.md):

* `Chronos2Pipeline.embed(x)` takes `(batch, n_variates, history_length)` and
  returns one tensor per item of shape `(n_variates, num_patches + 2, d_model)`
  with `num_patches = ceil(L / 16)`.
* The token axis is laid out `[context patches ...] + [REG] + [output patch]`
  (read from `Chronos2Model.encode`), so its members have *heterogeneous roles*:
  left-padded context patches, the architecture's register token, and the token
  the forecast would be decoded from.
* The model is **exactly permutation-equivariant across variates**: permuting the
  variate axis of the input permutes the output identically (max abs diff 1.9e-6,
  i.e. float32 noise). Consequences in `reduce()`.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np

from . import config as C

# Token-axis reductions. Both keep the variate axis, in fixed sensor order.
#
# Keeping the variate axis is NOT a preference, it is forced. Because the backbone
# is exactly permutation-equivariant across variates, ANY reduction that is
# symmetric in the variate axis (mean, sum, max over variates) is exactly invariant
# to permuting the sensor channels. Under such a reduction the Phase 4
# channel-scramble control would be mathematically guaranteed to show zero
# degradation -- a vacuous control that says nothing about the model. Preserving
# sensor identity in the head input is what makes that control informative.
# See notes/decisions.md D-011.
REDUCTIONS = ("tokmean", "reg")


def reduce(emb: np.ndarray, how: str) -> np.ndarray:
    """(n, D, P+2, h) -> (n, D*h), variates concatenated in fixed sensor order.

    tokmean : mean over the whole token axis. The naive reading of the paper's
              Eq. 6-7 ("feed the window of representations"), acknowledging that
              the axis mixes three token roles.
    reg     : the [REG] token only, at index P (P = num_patches). The
              architecture's own register/summary token.
    """
    if emb.ndim != 4:
        raise ValueError(f"expected (n, D, P+2, h), got {emb.shape}")
    n, D, P2, h = emb.shape
    if how == "tokmean":
        red = emb.mean(axis=2)
    elif how == "reg":
        red = emb[:, :, P2 - 2, :]  # [..., REG, output_patch] -> REG is second-to-last
    else:
        raise ValueError(f"unknown reduction {how!r}; expected one of {REDUCTIONS}")
    return red.reshape(n, D * h).astype(np.float32)


def reduced_dim(n_variates: int) -> int:
    return n_variates * C.BACKBONE_D_MODEL


# ------------------------------------------------------------------ backbone


def param_hash(model) -> str:
    """SHA-256 over all parameter bytes, in sorted name order."""
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_backbone(device: str | None = None):
    """Load Chronos-2 at the pinned revision and freeze it.

    Returns (pipeline, info). `info` records the freeze assertions so the
    checkpoint evidence can be printed and persisted.
    """
    import torch
    from chronos import BaseChronosPipeline

    if device is None:
        # CUDA if present, otherwise CPU -- deliberately never MPS. With
        # torch.use_deterministic_algorithms(True) active, loading this checkpoint
        # onto the MPS backend aborts the process (torch 2.13.0, macOS). Determinism
        # is a hard requirement here and MPS is not: measured throughput at the
        # primary look-back is 12.6 ms/window on CPU vs 14.0 on MPS, i.e. CPU is
        # faster anyway. See notes/decisions.md D-013.
        device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()
    pipe = BaseChronosPipeline.from_pretrained(
        C.BACKBONE_ID, revision=C.BACKBONE_REVISION,
        device_map=device, dtype=torch.float32,
    )
    load_s = time.perf_counter() - t0

    model = pipe.inner_model if hasattr(pipe, "inner_model") else pipe.model
    model.eval()
    model.requires_grad_(False)

    # Assertion 1: no backbone parameter is trainable.
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if trainable:
        raise AssertionError(
            f"backbone is not frozen: {len(trainable)} params require grad, "
            f"e.g. {trainable[:5]}"
        )

    n_params = sum(p.numel() for p in model.parameters())
    info = {
        "checkpoint": C.BACKBONE_ID,
        "revision": C.BACKBONE_REVISION,
        "pipeline_class": type(pipe).__name__,
        "device": str(device),
        "load_seconds": load_s,
        "n_params": int(n_params),
        "n_trainable_params": 0,
        "requires_grad_all_false": True,
        "d_model": int(model.config.d_model),
        "num_layers": int(model.config.num_layers),
        "model_context_length": int(pipe.model_context_length),
        "param_hash_at_load": param_hash(model),
    }
    return pipe, model, info


def extract(
    pipe,
    win: np.ndarray,
    reductions=REDUCTIONS,
    batch_size: int = 256,
    log=None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Extract and reduce embeddings for windows `(n, L, D)`.

    Reduces inside the loop so the full `(n, D, P+2, h)` tensor is never
    materialised (it would be ~1.9 GB for the training split).
    """
    import torch

    n, L, D = win.shape
    # embed() wants (batch, n_variates, history_length).
    x = np.ascontiguousarray(win.transpose(0, 2, 1))

    out: dict[str, list[np.ndarray]] = {r: [] for r in reductions}
    # Items per model batch: embed()'s batch_size counts *series*, not items.
    per_call = max(1, batch_size // D)
    shapes_seen, t0 = set(), time.perf_counter()

    for s in range(0, n, per_call):
        chunk = x[s : s + per_call]
        emb, _ = pipe.embed(torch.from_numpy(chunk), batch_size=batch_size)
        arr = np.stack([e.numpy() for e in emb], axis=0)  # (b, D, P+2, h)
        shapes_seen.add(arr.shape[1:])
        for r in reductions:
            out[r].append(reduce(arr, r))
        if log and (s // per_call) % 50 == 0:
            done = min(s + per_call, n)
            log(f"      {done}/{n} windows ({time.perf_counter()-t0:.0f}s)")

    if len(shapes_seen) != 1:
        raise AssertionError(f"inconsistent embedding shapes across batches: {shapes_seen}")
    per_item_shape = shapes_seen.pop()
    wall = time.perf_counter() - t0

    feats = {r: np.concatenate(out[r], axis=0) for r in reductions}
    for r, f in feats.items():
        if f.shape != (n, reduced_dim(D)):
            raise AssertionError(f"{r}: got {f.shape}, expected {(n, reduced_dim(D))}")

    # per_item_shape is (n_variates, num_patches + 2, d_model): the token axis is
    # index 1, not index 0. Reading n_patches off axis 0 would report n_variates - 2.
    meta = {
        "n_windows": int(n),
        "lookback": int(L),
        "n_variates": int(per_item_shape[0]),
        "per_item_embed_shape": [int(v) for v in per_item_shape],
        "n_tokens": int(per_item_shape[1]),
        "n_patches": int(per_item_shape[1] - 2),
        "d_model": int(per_item_shape[2]),
        "token_axis_layout": "[context patches ...] + [REG] + [output patch]",
        "reduced_dim": int(reduced_dim(D)),
        "reductions": list(reductions),
        "wall_seconds": wall,
        "ms_per_window": 1000 * wall / max(n, 1),
        "embed_batch_size": batch_size,
        "items_per_embed_call": int(per_call),
    }
    return feats, meta


# ------------------------------------------------------------------ cache


def cache_path(lookback: int, split: str, tag: str = "main"):
    return C.CACHE / f"emb_L{lookback}_{split}_{tag}.npz"


def meta_path(path):
    return path.with_suffix(".meta.json")


def save_cache(path, feats: dict[str, np.ndarray], meta: dict | None = None) -> None:
    np.savez(path, **feats)
    if meta is not None:
        # Persisted alongside the arrays so a cached run reports the same measured
        # extraction facts (token counts, throughput) as the run that produced them,
        # instead of losing them and having to reconstruct from config.
        import json

        meta_path(path).write_text(json.dumps(meta, indent=2, default=str) + "\n")


def load_cache(path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def load_cache_meta(path) -> dict | None:
    import json

    mp = meta_path(path)
    return json.loads(mp.read_text()) if mp.exists() else None


def get_or_extract(
    pipe, bundle, split: str, tag: str = "main", win: np.ndarray | None = None, log=print
):
    """Cached extraction for one split. Returns (feats, meta, from_cache)."""
    path = cache_path(bundle.lookback, split, tag)
    if path.exists():
        feats = load_cache(path)
        meta = load_cache_meta(path) or {}
        meta = {**meta, "from_cache": True, "path": str(path)}
        log(f"    {split}/{tag}: loaded cache {path.name} "
            f"{ {k: v.shape for k, v in feats.items()} }")
        return feats, meta, True
    if win is None:
        win = bundle.windows(split)
    log(f"    {split}/{tag}: extracting {win.shape} ...")
    feats, meta = extract(pipe, win, log=log)
    save_cache(path, feats, meta)
    log(f"    {split}/{tag}: {meta['n_windows']} windows in {meta['wall_seconds']:.1f}s "
        f"({meta['ms_per_window']:.1f} ms/window), embed shape "
        f"{meta['per_item_embed_shape']} -> reduced {meta['reduced_dim']}")
    return feats, meta, False
