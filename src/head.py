"""The trainable regression head, and the only thing in the TSFM arm with gradients.

Architecture follows the paper (§II-B): two linear layers with hidden width m, ReLU,
dropout after the first hidden layer, and a final ReLU enforcing y >= 0. Optimiser
settings also follow the paper: Adam, lr 1e-3, 50 epochs, batch 64, MSE loss.

Two output modes:
  point    -- 1 output, MSE loss. Paper-faithful. Intervals come from Phase 5's
              conformal step, since an MSE point head emits no uncertainty at all.
  quantile -- len(QUANTILE_LEVELS) outputs, pinball loss. Not in the paper; the
              brief asks for it where the head can emit quantiles, so both are run
              and compared.

The same head class is used by the random-projection control, so "identical head"
is enforced by construction.
"""

from __future__ import annotations

import time

import numpy as np

from . import config as C


def make_head(d_in: int, mode: str, seed: int):
    """Build the head. `mode` in {'point', 'quantile'}."""
    import torch
    from torch import nn

    n_out = 1 if mode == "point" else len(C.QUANTILE_LEVELS)
    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(d_in, C.HEAD_HIDDEN),
        nn.ReLU(),
        nn.Dropout(C.HEAD_DROPOUT),
        nn.Linear(C.HEAD_HIDDEN, n_out),
        nn.ReLU(),  # paper: final ReLU enforces yhat >= 0
    )
    return net


def head_param_count(net) -> dict:
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    per_layer = {
        f"{i}:{type(m).__name__}": sum(p.numel() for p in m.parameters())
        for i, m in enumerate(net)
        if any(True for _ in m.parameters())
    }
    return {"total": int(total), "trainable": int(trainable), "per_layer": per_layer}


class FeatureScaler:
    """Per-dimension z-score fitted on training-split features only.

    Applied to whatever feature matrix feeds the head -- Chronos-2 embeddings and
    the random-projection control alike -- so the two remain comparable. This is a
    head-input transform, distinct from the sensor preprocessing in
    `preprocess.Preprocessor`, which is shared by every arm including LightGBM.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean = None
        self.scale = None

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        X = np.asarray(X, np.float64)
        self.mean = X.mean(axis=0)
        sd = X.std(axis=0, ddof=0)
        # Degenerate dims (constant embedding coordinates) pass through unscaled
        # rather than being amplified -- the same trap as D-005.
        self.scale = np.where(sd > self.eps, sd, 1.0)
        self.n_degenerate = int((sd <= self.eps).sum())
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((np.asarray(X, np.float64) - self.mean) / self.scale).astype(np.float32)


def pinball_torch(pred, target, levels):
    """Mean pinball loss over levels. `pred` (n, k), `target` (n,)."""
    import torch

    tau = torch.as_tensor(levels, dtype=pred.dtype, device=pred.device)[None, :]
    diff = target[:, None] - pred
    return torch.maximum(tau * diff, (tau - 1.0) * diff).mean()


def train_head(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    mode: str = "point",
    seed: int = C.SEEDS["head_init"],
    device: str | None = None,
    epochs: int = C.HEAD_EPOCHS,
    log=None,
) -> tuple:
    """Fit the head. Returns (net, scaler, history).

    No early stopping and no validation split: the paper specifies "a maximum of 50
    epochs" without a selection rule (AMBIGUITY 6), and the only untouched data is
    the calibration split, which a base model must not see (D-007). We therefore
    train a fixed number of epochs and report the loss trace.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    device = device or "cpu"
    scaler = FeatureScaler().fit(X_tr)
    Xs = scaler.transform(X_tr)

    net = make_head(Xs.shape[1], mode, seed).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=C.HEAD_LR)
    mse = nn.MSELoss()

    ds = TensorDataset(
        torch.from_numpy(Xs), torch.from_numpy(np.asarray(y_tr, np.float32))
    )
    gen = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=C.HEAD_BATCH, shuffle=True, generator=gen,
                    drop_last=False)

    hist, t0 = [], time.perf_counter()
    net.train()
    for ep in range(epochs):
        tot, n = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            out = net(xb)
            loss = mse(out.squeeze(-1), yb) if mode == "point" else pinball_torch(
                out, yb, C.QUANTILE_LEVELS
            )
            loss.backward()
            opt.step()
            tot += loss.detach().item() * len(yb)
            n += len(yb)
        hist.append(tot / n)
        if log and (ep + 1) % 10 == 0:
            log(f"        epoch {ep+1:>3}/{epochs} loss={hist[-1]:.4f}")

    return net, scaler, {
        "loss_trace": hist,
        "loss_first": hist[0],
        "loss_last": hist[-1],
        "epochs": epochs,
        "mode": mode,
        "seed": seed,
        "train_seconds": time.perf_counter() - t0,
        "n_train": int(len(y_tr)),
        "params": head_param_count(net),
        "scaler_degenerate_dims": scaler.n_degenerate,
    }


def predict_head(net, scaler, X: np.ndarray, mode: str, device: str = "cpu"):
    """Returns (point, quantiles_or_None)."""
    import torch

    net.eval()
    Xs = torch.from_numpy(scaler.transform(X)).to(device)
    with torch.no_grad():
        out = net(Xs).cpu().numpy()
    if mode == "point":
        return out.squeeze(-1), None
    # Median column is the point estimate for the quantile head.
    i_med = list(C.QUANTILE_LEVELS).index(0.5)
    return out[:, i_med], out
