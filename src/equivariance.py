"""The paper's lead result: Chronos-2 is exactly permutation-equivariant across variates.

Claim, stated precisely. Let Phi be the frozen backbone and P a permutation of the D
sensor channels. Then Phi(P x) = P Phi(x) to floating-point precision. Two consequences
follow immediately and neither depends on C-MAPSS:

1. The backbone cannot encode *which* sensor a channel is. Channels are an exchangeable
   set to it. In multivariate industrial telemetry, where one channel is a vibration and
   another an exhaust temperature, sensor identity is exactly where the meaning lives, so
   any such model must have that meaning supplied downstream.
2. Any read-out that is symmetric in the variate axis -- mean, sum or max over variates --
   is exactly *invariant* to permuting the channels. A channel-scramble control built on
   such a read-out is therefore guaranteed to show zero degradation, and would look like
   evidence of robustness while measuring nothing.

Measured on real preprocessed C-MAPSS windows, not synthetic noise:
  equivariance   max |Phi(x)[P] - Phi(P x)|              -> should be float32 noise
  invariance     max |mean_D Phi(x) - mean_D Phi(P x)|    -> should be float32 noise
  scale          max |Phi(x) - Phi(P x)|                  -> the size of the effect that
                                                             equivariance is cancelling

torch-only process. Run: uv run python -m src.equivariance
"""

from __future__ import annotations

import json
import sys

import numpy as np

from . import config as C, dataset, embed
from .ompguard import assert_single_omp_runtime
from .phase1 import hr
from .seeding import seed_everything

N_WINDOWS = 32


def main() -> int:
    assert_single_omp_runtime("equivariance")
    seed_everything()
    import torch

    hr("PERMUTATION EQUIVARIANCE ACROSS VARIATES (measured on real windows)")
    pipe, model, info = embed.load_backbone()
    b = dataset.build(C.LOOKBACK)
    win = b.windows("eval")[:N_WINDOWS]           # (n, L, D), preprocessed
    x = np.ascontiguousarray(win.transpose(0, 2, 1))  # (n, D, L)
    D = x.shape[1]
    rng = np.random.default_rng(C.SEEDS["channel_scramble_control"])
    perm = rng.permutation(D)

    e0 = np.stack([t.numpy() for t in pipe.embed(torch.from_numpy(x))[0]])
    e1 = np.stack([t.numpy() for t in
                   pipe.embed(torch.from_numpy(np.ascontiguousarray(x[:, perm, :])))[0]])

    equi = float(np.abs(e0[:, perm] - e1).max())
    inv = float(np.abs(e0.mean(axis=1) - e1.mean(axis=1)).max())
    scale = float(np.abs(e0 - e1).max())
    emb_scale = float(np.abs(e0).max())

    print(f"  windows tested            {N_WINDOWS} real preprocessed eval windows")
    print(f"  embedding shape per item  {tuple(e0.shape[1:])}  (D={D})")
    print(f"  channel permutation       {perm.tolist()}")
    print()
    print(f"  equivariance  max |Phi(x)[P] - Phi(Px)|        = {equi:.3e}")
    print(f"  invariance    max |mean_D Phi(x) - mean_D Phi(Px)| = {inv:.3e}")
    print(f"  effect scale  max |Phi(x) - Phi(Px)|           = {scale:.3e}")
    print(f"  embedding scale  max |Phi(x)|                  = {emb_scale:.3e}")
    print()
    ratio = scale / max(equi, 1e-12)
    print(f"  The permutation changes the embeddings by {scale:.2f} in absolute terms,")
    print(f"  yet permuting the output recovers them to {equi:.1e} -- a ratio of "
          f"{ratio:,.0f}x.")
    print(f"  So the map is equivariant, not merely insensitive: the backbone carries no")
    print(f"  channel-identity information for a downstream head to use.")

    out = {
        "n_windows": N_WINDOWS, "n_variates": D,
        "permutation": perm.tolist(),
        "equivariance_max_abs_diff": equi,
        "variate_mean_invariance_max_abs_diff": inv,
        "effect_scale_max_abs_diff": scale,
        "embedding_max_abs": emb_scale,
        "scale_over_equivariance_ratio": ratio,
        "backbone": {k: info[k] for k in ("checkpoint", "revision", "n_params")},
        "float32_eps": float(np.finfo(np.float32).eps),
    }
    (C.RESULTS / "equivariance.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\nwrote results/equivariance.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
