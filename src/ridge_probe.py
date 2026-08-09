"""M4: hyperparameter-free-ish linear probes on the frozen representations.

The standard instrument for measuring frozen-representation quality is a ridge
regression on the frozen features. Routing every arm through a
13,056 -> 32 bottleneck MLP whose width and dropout we had to guess (the original never
states them) leaves that bottleneck -- a 408x compression -- as a live alternative
explanation for the negative C1. A ridge probe removes the head as a confound: it is
convex, has one hyperparameter, and is fit identically for every feature set.

Feature sets probed, all on identical windows and identical preprocessing:
  chronos     frozen Chronos-2 embeddings, the primary reduction (13,056 dims)
  randproj    fixed random linear projection of the flattened window (13,056 dims)
  summary     the 7 per-channel window statistics the LightGBM baseline uses (119)
  raw         the flattened raw window (510)

Protocol: alpha is selected on a held-out subset of *training* units, then the model is
refit on all training units and evaluated once on the evaluation units. The calibration
units are never touched, so the conformal step downstream stays valid.

Implementation note: with 13,056 features and 8,993 samples the primal normal equations
are a 13,056^2 matrix, so those feature sets are solved in the **dual** (kernel) form,
where the system is n x n. One eigendecomposition per fit set then makes every alpha
free. Small feature sets use the primal form. Both give the same estimator.

numpy/scipy only -- neither torch nor LightGBM is imported, so this runs anywhere
(src/ompguard.py).

Run: uv run python -m src.ridge_probe
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from . import config as C, dataset, embed, features as bf, metrics, preds
from .phase1 import hr

ALPHAS = np.logspace(-2, 6, 17)
VAL_UNITS = 15  # of the 50 training units, held out only to choose alpha


class RidgeEig:
    """Ridge for many alphas from one eigendecomposition; picks primal or dual."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.n, self.d = X.shape
        self.xm = X.mean(axis=0)
        self.ym = float(y.mean())
        Xc = (X - self.xm).astype(np.float64)
        self.yc = y.astype(np.float64) - self.ym
        self.dual = self.d > self.n
        t0 = time.perf_counter()
        if self.dual:
            K = Xc @ Xc.T                       # (n, n)
            w, V = np.linalg.eigh(K)
            self.w, self.V, self.Xc = np.maximum(w, 0.0), V, Xc
        else:
            G = Xc.T @ Xc                       # (d, d)
            w, V = np.linalg.eigh(G)
            self.w, self.V = np.maximum(w, 0.0), V
            self.Xty = Xc.T @ self.yc
        self.decomp_seconds = time.perf_counter() - t0

    def predict(self, Xq: np.ndarray, alpha: float) -> np.ndarray:
        Xqc = (Xq - self.xm).astype(np.float64)
        if self.dual:
            # a = (K + aI)^-1 y ; f(x) = x . Xc^T a
            Vt_y = self.V.T @ self.yc
            a = self.V @ (Vt_y / (self.w + alpha))
            return (Xqc @ (self.Xc.T @ a)) + self.ym
        coef = self.V @ ((self.V.T @ self.Xty) / (self.w + alpha))
        return (Xqc @ coef) + self.ym


def main() -> int:
    L = C.LOOKBACK
    hr("M4: RIDGE LINEAR PROBES ON THE FROZEN REPRESENTATIONS")
    b = dataset.build(L)

    # ---- feature blocks, identical windows and preprocessing for every set -----
    win = {s: b.windows(s) for s in ("train", "eval")}
    feats: dict[str, dict[str, np.ndarray]] = {}
    path = embed.cache_path(L, "train", "main")
    if not path.exists():
        raise SystemExit(f"missing {path}; run phase3 first")
    feats["chronos"] = {
        s: embed.load_cache(embed.cache_path(L, s, "main"))[C.REDUCTION_PRIMARY]
        for s in ("train", "eval")
    }
    d_out = embed.reduced_dim(b.n_features)
    feats["randproj"] = {
        s: bf.random_projection(win[s], d_out, C.SEEDS["random_projection_control"])
        for s in ("train", "eval")
    }
    feats["summary"] = {s: bf.summary_features(win[s]) for s in ("train", "eval")}
    feats["raw"] = {s: bf.raw_features(win[s]) for s in ("train", "eval")}
    for k, v in feats.items():
        print(f"  {k:<10} train {v['train'].shape}  eval {v['eval'].shape}")

    # ---- alpha-selection split, over TRAINING units only ----------------------
    tr_units = np.array(b.splits.train)
    rng = np.random.default_rng(C.SEEDS["unit_split"] + 7)
    val_units = set(rng.choice(tr_units, size=VAL_UNITS, replace=False).tolist())
    u_tr = b.units("train")
    fit_mask = ~np.isin(u_tr, list(val_units))
    val_mask = ~fit_mask
    print(f"\n  alpha selected on {VAL_UNITS} held-out TRAINING units "
          f"({int(val_mask.sum())} windows); refit on all {len(tr_units)} "
          f"({len(u_tr)} windows). Calibration units untouched.")
    print(f"  alpha grid: {len(ALPHAS)} values, {ALPHAS[0]:g} .. {ALPHAS[-1]:g}\n")

    report: dict = {"lookback": L, "alphas": ALPHAS.tolist(),
                    "n_val_units": VAL_UNITS, "probes": {}}
    print(f"  {'features':<10}{'v':>2}{'dim':>7}{'alpha*':>10}{'eval RMSE':>11}"
          f"{'[95% CI]':>20}{'MAE':>8}{'decomp s':>10}")
    for variant in ("A", "B"):
        y_tr_all = b.target("train", variant)
        y_ev = b.target("eval", variant)
        for name, blk in feats.items():
            Xa, Xe = blk["train"], blk["eval"]
            # 1. choose alpha on the held-out training units
            sel = RidgeEig(Xa[fit_mask], y_tr_all[fit_mask])
            errs = [float(np.sqrt(np.mean(
                (sel.predict(Xa[val_mask], a) - y_tr_all[val_mask]) ** 2)))
                for a in ALPHAS]
            a_star = float(ALPHAS[int(np.argmin(errs))])
            # 2. refit on all training units, predict eval once
            full = RidgeEig(Xa, y_tr_all)
            pred = full.predict(Xe, a_star)
            pm = metrics.point_metrics(y_ev, pred)
            ci = metrics.metric_ci(b.units("eval"), y_ev, pred, "rmse")
            report["probes"][f"{name}__{variant}"] = {
                "dim": int(Xa.shape[1]), "alpha": a_star,
                "val_rmse_curve": errs, **pm,
                "rmse_ci_lo": ci["ci_lo"], "rmse_ci_hi": ci["ci_hi"],
                "decomp_seconds": full.decomp_seconds, "dual": full.dual,
            }
            cs = "[%.2f, %.2f]" % (ci["ci_lo"], ci["ci_hi"])
            print(f"  {name:<10}{variant:>2}{Xa.shape[1]:>7}{a_star:>10.3g}"
                  f"{pm['rmse']:>11.2f}{cs:>20}{pm['mae']:>8.2f}"
                  f"{full.decomp_seconds:>10.1f}")

    # ---- the comparison the reviewer asks for --------------------------------
    hr("RIDGE PROBE VERDICT: frozen embeddings vs random features vs summaries")
    print("  Paired by-unit clustered bootstrap on the difference of probe RMSEs.\n")
    y = {v: b.target("eval", v) for v in ("A", "B")}
    units = b.units("eval")
    P = {}
    for variant in ("A", "B"):
        y_tr_all = b.target("train", variant)
        for name, blk in feats.items():
            a_star = report["probes"][f"{name}__{variant}"]["alpha"]
            P[f"{name}__{variant}"] = RidgeEig(blk["train"], y_tr_all).predict(
                blk["eval"], a_star)
    # The LightGBM baseline is nonlinear, so ridge-on-summaries is not its equal; the
    # C1 question under a head-free probe is ridge-on-Chronos vs the actual baseline.
    for variant in ("A", "B"):
        P[f"lgbm_summary__{variant}"] = preds.load(
            f"lgbm_summary__{variant}__L{C.LOOKBACK}__eval").pred

    pairs = {}
    print(f"  {'comparison':<34}{'v':>2}{'dRMSE [95% CI]':>28}{'verdict':>24}")
    for variant in ("A", "B"):
        for a_name, b_name in (("chronos", "summary"), ("chronos", "randproj"),
                               ("randproj", "summary"),
                               ("chronos", "lgbm_summary"),
                               ("randproj", "lgbm_summary")):
            pa, pb = P[f"{a_name}__{variant}"], P[f"{b_name}__{variant}"]
            yy = y[variant]

            def stat(rows, pa=pa, pb=pb, yy=yy):
                return (float(np.sqrt(np.mean((pa[rows] - yy[rows]) ** 2)))
                        - float(np.sqrt(np.mean((pb[rows] - yy[rows]) ** 2))))

            d = metrics.clustered_bootstrap(units, stat)
            ez = d["ci_lo"] > 0 or d["ci_hi"] < 0
            pairs[f"{a_name}_minus_{b_name}__{variant}"] = {**d, "excludes_zero": ez}
            ds = "%+.2f [%+.2f, %+.2f]" % (d["point"], d["ci_lo"], d["ci_hi"])
            v = ("indistinguishable" if not ez
                 else (f"{a_name} WORSE" if d["point"] > 0 else f"{a_name} BETTER"))
            print(f"  {a_name + ' - ' + b_name:<34}{variant:>2}{ds:>28}{v:>24}")
    report["paired"] = pairs

    (C.RESULTS / "ridge_probe.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n")
    print("\nwrote results/ridge_probe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
