"""Paper figures, built from results/metrics.csv only.

Palette is the validated categorical order (slots 1-6), checked with the dataviz
validator: adjacent-pair CVD dE 9.1 / normal-vision 19.6 for the 6-series line charts,
all-pairs CVD 9.2 / normal 24.0 for the 3-colour bar chart. Three slots fall below 3:1
contrast on a light surface, so the relief rule applies: every series carries a distinct
marker shape as secondary encoding, a legend is always present, and
`results/metrics.csv` is the table view. Direct labels are used only where a panel has
at most four series -- past that they collide wherever arms converge, which is exactly
where these arms do converge.

Rendered for print on a light page, a deliberate single-look commit.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd

from . import config as C

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GROUP = {"baseline": "#2a78d6", "tsfm": "#eb6834", "control": "#1baf7a"}
GROUP_LABEL = {"baseline": "baseline", "tsfm": "Chronos-2 arm", "control": "control"}
MARKERS = ["o", "s", "^", "D", "v", "P"]
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
SURFACE = "#fcfcfb"

CURVE_ARMS = ["trivial", "lgbm_summary", "lgbm_raw", "tsfm", "tsfm_q",
              "control_randproj_q"]
LABEL = {
    "trivial": "trivial marginal",
    "lgbm_summary": "LGBM summary",
    "lgbm_raw": "LGBM raw",
    "tsfm": "Chronos-2 (point)",
    "tsfm_q": "Chronos-2 (quantile)",
    "tsfm_abl_reg": "Chronos-2 [REG]",
    "control_randproj": "random projection",
    "control_randproj_q": "random proj. (quantile)",
    "control_chanscramble": "channel scramble",
    "control_shufflabel": "shuffled labels",
}
BIN_ORDER = ["gt100", "50to100", "20to50", "lt20"]
BIN_LABEL = {"gt100": ">100", "50to100": "50–100", "20to50": "20–50", "lt20": "<20"}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK3, "axes.linewidth": 0.8,
        "axes.labelcolor": INK, "axes.titlecolor": INK,
        "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.labelsize": 8, "axes.titlesize": 8.5,
        "legend.fontsize": 7, "legend.frameon": False,
        "grid.color": "#e6e5e0", "grid.linewidth": 0.7,
        "lines.linewidth": 1.8, "lines.markersize": 4.5,
        "font.size": 8,
    })
    return mpl


def _clean(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)


def _caption(fig, text, width=140):
    fig.text(0.012, 0.008, "\n".join(textwrap.wrap(text, width)),
             fontsize=6.5, color=INK2, va="bottom")


def _arm_legend(fig, arms, ncol=6, y=0.925):
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=SERIES[i % len(SERIES)], marker=MARKERS[i % len(MARKERS)],
               mec=SURFACE, mew=0.7, lw=1.8, label=LABEL[a])
        for i, a in enumerate(arms)
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, y),
               ncol=ncol, columnspacing=1.3, handlelength=1.8, borderaxespad=0)


def _get(df, **kw):
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= (df[k] == v)
    return df[m]


def _series(df, arm, variant, tag, metric):
    sub = _get(df, arm=arm, variant=variant, conformal=tag, metric=metric,
               rul_bin="all").sort_values("nominal")
    return (sub["nominal"].to_numpy(), sub["value"].to_numpy(),
            sub["ci_lo"].to_numpy(), sub["ci_hi"].to_numpy())


def _vlabel(variant):
    return f"Variant {variant} " + ("(uncapped)" if variant == "A"
                                    else f"(capped at {C.RUL_CAP})")


# ------------------------------------------------------------------ figure 1

def fig_calibration_curves(df, path):
    _style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2), sharex=True, sharey=True)
    for j, variant in enumerate(("A", "B")):
        for i, tag in enumerate(("marginal", "conformal")):
            ax = axes[i][j]
            ax.plot([0.45, 1.0], [0.45, 1.0], color=INK3, ls=(0, (4, 3)), lw=1.0,
                    zorder=1)
            drawn = 0
            for k, arm in enumerate(CURVE_ARMS):
                x, y, lo, hi = _series(df, arm, variant, tag, "coverage")
                if len(x) == 0:
                    continue
                c, mk = SERIES[k % len(SERIES)], MARKERS[k % len(MARKERS)]
                ax.fill_between(x, lo, hi, color=c, alpha=0.12, lw=0, zorder=2)
                ax.plot(x, y, color=c, marker=mk, mec=SURFACE, mew=0.7, zorder=3)
                drawn += 1
            if drawn == 0:
                ax.text(0.5, 0.45, "no arm emits an interval\nbefore conformal",
                        transform=ax.transAxes, ha="center", va="center",
                        color=INK2, fontsize=7.5, style="italic")
            _clean(ax, "both")
            ax.set_xlim(0.46, 1.01)
            ax.set_ylim(0.1, 1.03)
            ax.set_xticks(list(C.NOMINAL_LEVELS))
            # 95% and 99% sit close together on a linear axis; rotate so they do
            # not overprint each other.
            ax.set_xticklabels([f"{lv:.0%}" for lv in C.NOMINAL_LEVELS],
                               rotation=45, ha="right")
            if i == 1:
                ax.set_xlabel("nominal coverage")
            if j == 0:
                ax.set_ylabel(f"empirical coverage\n({tag} intervals)")
            ax.set_title(f"{_vlabel(variant)} — {tag}", loc="left")
    fig.suptitle("Calibration: nominal vs empirical coverage, C-MAPSS FD001 "
                 f"(L={C.LOOKBACK}, 30 held-out units, 450 windows)",
                 fontsize=9.5, x=0.012, ha="left", y=0.985)
    _arm_legend(fig, CURVE_ARMS, ncol=3, y=0.945)
    _caption(fig, "Bands are 95% by-unit clustered bootstrap CIs. Dashed diagonal = "
                  "perfect calibration; below it = intervals too narrow, i.e. "
                  "overconfident. Marker shape encodes arm identity in addition to "
                  "colour.")
    fig.tight_layout(rect=(0, 0.045, 1, 0.885))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ figure 2

def fig_coverage_by_regime(df, path, nominal=0.90):
    _style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0), sharey=True)
    xs = np.arange(len(BIN_ORDER))
    for j, variant in enumerate(("A", "B")):
        for i, tag in enumerate(("marginal", "conformal")):
            ax = axes[i][j]
            ax.axhline(nominal, color=INK3, ls=(0, (4, 3)), lw=1.0, zorder=1)
            for k, arm in enumerate(CURVE_ARMS):
                vals, los, his = [], [], []
                for b in BIN_ORDER:
                    s = _get(df, arm=arm, variant=variant, conformal=tag,
                             metric="coverage", rul_bin=b, nominal=nominal)
                    vals.append(s["value"].iloc[0] if len(s) else np.nan)
                    los.append(s["ci_lo"].iloc[0] if len(s) else np.nan)
                    his.append(s["ci_hi"].iloc[0] if len(s) else np.nan)
                if all(np.isnan(v) for v in vals):
                    continue
                c, mk = SERIES[k % len(SERIES)], MARKERS[k % len(MARKERS)]
                ax.fill_between(xs, los, his, color=c, alpha=0.10, lw=0, zorder=2)
                ax.plot(xs, vals, color=c, marker=mk, mec=SURFACE, mew=0.7, zorder=3)
            ax.annotate(f"nominal {nominal:.0%}", xy=(len(xs) - 1, nominal),
                        xytext=(0, -11), textcoords="offset points", fontsize=6.5,
                        color=INK2, ha="right")
            _clean(ax, "y")
            ax.set_xticks(xs)
            ax.set_xticklabels([BIN_LABEL[b] for b in BIN_ORDER])
            ax.set_xlim(-0.25, len(BIN_ORDER) - 0.75)
            ax.set_ylim(0, 1.05)
            if j == 0:
                ax.set_ylabel(f"empirical coverage\n({tag})")
            ax.set_title(f"{_vlabel(variant)} — {tag}", loc="left")
    fig.supxlabel("true RUL at truncation (cycles):  healthier  ←——→  nearer failure",
                  fontsize=8, y=0.055)
    fig.suptitle(f"Coverage conditioned on health regime, nominal {nominal:.0%}",
                 fontsize=9.5, x=0.012, ha="left", y=0.985)
    _arm_legend(fig, CURVE_ARMS, ncol=3, y=0.945)
    _caption(fig, "Bins are on true uncapped RUL, so they are the same physical regime "
                  "in both variants. Aggregate coverage hides regime-specific failure: "
                  "the near-failure bin is the only one in which a maintenance decision "
                  "is actually taken.")
    fig.tight_layout(rect=(0, 0.075, 1, 0.885))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_coverage_by_regime_compact(df, path, nominal=0.90, variant="A"):
    """Body-sized version of the regime figure: one variant, two panels.

    The 2x2 version is ~0.45 of a NeurIPS page at full column width. Variant A carries
    the headline; Variant B goes to the appendix.
    """
    _style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.55), sharey=True)
    xs = np.arange(len(BIN_ORDER))
    for i, tag in enumerate(("marginal", "conformal")):
        ax = axes[i]
        ax.axhline(nominal, color=INK3, ls=(0, (4, 3)), lw=1.0, zorder=1)
        for k, arm in enumerate(CURVE_ARMS):
            vals, los, his = [], [], []
            for b in BIN_ORDER:
                s_ = _get(df, arm=arm, variant=variant, conformal=tag,
                          metric="coverage", rul_bin=b, nominal=nominal)
                vals.append(s_["value"].iloc[0] if len(s_) else np.nan)
                los.append(s_["ci_lo"].iloc[0] if len(s_) else np.nan)
                his.append(s_["ci_hi"].iloc[0] if len(s_) else np.nan)
            if all(np.isnan(v) for v in vals):
                continue
            c, mk = SERIES[k % len(SERIES)], MARKERS[k % len(MARKERS)]
            ax.fill_between(xs, los, his, color=c, alpha=0.10, lw=0, zorder=2)
            ax.plot(xs, vals, color=c, marker=mk, mec=SURFACE, mew=0.7, zorder=3)
        ax.annotate(f"nominal {nominal:.0%}", xy=(len(xs) - 1, nominal), xytext=(0, -11),
                    textcoords="offset points", fontsize=6.5, color=INK2, ha="right")
        _clean(ax, "y")
        ax.set_xticks(xs)
        ax.set_xticklabels([BIN_LABEL[b] for b in BIN_ORDER])
        ax.set_xlim(-0.25, len(BIN_ORDER) - 0.75)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("true RUL (cycles): healthier $\\leftarrow\\!\\rightarrow$ nearer failure")
        if i == 0:
            ax.set_ylabel("empirical coverage")
        ax.set_title(f"{tag} intervals", loc="left")
    _arm_legend(fig, CURVE_ARMS, ncol=3, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ figure 3

def fig_sharpness_frontier(df, path):
    _style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    for j, variant in enumerate(("A", "B")):
        ax = axes[j]
        for k, arm in enumerate(CURVE_ARMS):
            c, mk = SERIES[k % len(SERIES)], MARKERS[k % len(MARKERS)]
            for tag, ls, alpha in (("marginal", "-", 1.0),
                                   ("conformal", (0, (3, 2)), 0.9)):
                _, cov, _, _ = _series(df, arm, variant, tag, "coverage")
                _, wid, _, _ = _series(df, arm, variant, tag, "width_mean")
                if len(cov) == 0:
                    continue
                ax.plot(wid, cov, color=c, ls=ls, marker=mk, alpha=alpha,
                        mec=SURFACE, mew=0.7, ms=4, zorder=3)
        ax.axhline(0.90, color=INK3, ls=(0, (1, 3)), lw=0.9, zorder=1)
        _clean(ax, "both")
        ax.set_xlabel("mean interval width (cycles)")
        if j == 0:
            ax.set_ylabel("empirical coverage")
        ax.set_ylim(0.1, 1.03)
        ax.set_title(_vlabel(variant), loc="left")
    fig.suptitle("Sharpness–coverage frontier  (solid = arm's own intervals, "
                 "dashed = after split conformal)",
                 fontsize=9.5, x=0.012, ha="left", y=0.985)
    _arm_legend(fig, CURVE_ARMS, ncol=3, y=0.935)
    _caption(fig, "Up and to the left is better: high coverage at narrow width. Each "
                  "line traces one arm across the five nominal levels. Coverage without "
                  "width is gameable by widening, so every coverage claim is reported "
                  "beside its width. Dotted line marks 90% coverage.")
    fig.tight_layout(rect=(0, 0.09, 1, 0.80))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ figure 4

def fig_conformal_before_after(df, path, nominal=0.90):
    _style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    order = ["trivial", "lgbm_summary", "lgbm_raw", "tsfm", "tsfm_q", "tsfm_abl_reg",
             "control_randproj", "control_randproj_q", "control_chanscramble",
             "control_shufflabel"]
    present = [a for a in order
               if len(_get(df, arm=a, variant="A", conformal="conformal",
                           metric="coverage", rul_bin="all", nominal=nominal))]
    ys = np.arange(len(present))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.2), sharey=True)
    for j, variant in enumerate(("A", "B")):
        ax = axes[j]
        ax.axvline(nominal, color=INK3, ls=(0, (4, 3)), lw=1.0, zorder=1)
        for y, arm in zip(ys, present):
            def val(tag, metric):
                s = _get(df, arm=arm, variant=variant, conformal=tag, metric=metric,
                         rul_bin="all", nominal=nominal)
                return s["value"].iloc[0] if len(s) else np.nan
            mc, cc = val("marginal", "coverage"), val("conformal", "coverage")
            mw, cw = val("marginal", "width_mean"), val("conformal", "width_mean")
            grp = ("control" if arm.startswith("control")
                   else "tsfm" if arm.startswith("tsfm") else "baseline")
            colour = GROUP[grp]
            if not np.isnan(mc) and abs(cc - mc) > 0.004:
                ax.annotate("", xy=(cc, y), xytext=(mc, y),
                            arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.5,
                                            shrinkA=0, shrinkB=2, mutation_scale=8),
                            zorder=3)
            if not np.isnan(mc):
                ax.plot([mc], [y], "o", color=SURFACE, mec=colour, mew=1.5, ms=5.5,
                        zorder=4)
            ax.plot([cc], [y], "o", color=colour, mec=SURFACE, mew=0.7, ms=6.5,
                    zorder=5)
            ax.annotate(f"{cw:.0f}" if np.isnan(mw) else f"{mw:.0f}→{cw:.0f}",
                        xy=(1.02, y), xycoords=("axes fraction", "data"),
                        fontsize=6.2, color=INK2, va="center")
        ax.set_yticks(ys)
        ax.set_yticklabels([LABEL[a] for a in present], fontsize=7)
        _clean(ax, "x")
        ax.set_xlim(0.0, 1.04)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 0.9, 1.0])
        ax.set_xticklabels(["0", "25", "50", "75", "90", "100"])
        ax.set_xlabel(f"empirical coverage (%) at nominal {nominal:.0%}")
        ax.set_title(_vlabel(variant), loc="left")
    handles = [Line2D([], [], color=SURFACE, marker="o", mec=INK2, mew=1.5, ls="",
                      ms=5.5, label="arm's own interval"),
               Line2D([], [], color=INK2, marker="o", mec=SURFACE, ls="", ms=6.5,
                      label="after split conformal")]
    handles += [Line2D([], [], color=v, lw=3, label=GROUP_LABEL[k])
                for k, v in GROUP.items()]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.935),
               ncol=5, columnspacing=1.2, handlelength=1.6, borderaxespad=0)
    fig.suptitle("Split-conformal repair at nominal 90%", fontsize=9.5, x=0.012,
                 ha="left", y=0.985)
    _caption(fig, "Grey numbers at the right of each panel are mean interval widths in "
                  "cycles (own→conformal). Arms with no hollow marker emit no interval "
                  "of their own: an MSE-trained point head produces no uncertainty at "
                  "all, so conformal is the only way it gets one.")
    fig.tight_layout(rect=(0, 0.055, 0.955, 0.90))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ figure 5

def fig_controls(df, path):
    _style()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    order = ["trivial", "control_shufflabel", "lgbm_summary", "lgbm_raw", "tsfm",
             "tsfm_abl_reg", "control_chanscramble", "control_randproj"]
    # Independent y-axes: Variant A and B are different targets on different scales,
    # so a shared axis would compress B for no comparative gain.
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9), sharey=False)
    for j, variant in enumerate(("A", "B")):
        ax = axes[j]
        present, vals, los, his, cols = [], [], [], [], []
        for arm in order:
            s = _get(df, arm=arm, variant=variant, metric="rmse", rul_bin="all",
                     conformal="point", split="eval", lookback=C.LOOKBACK)
            if not len(s):
                continue
            present.append(arm)
            vals.append(s["value"].iloc[0])
            los.append(s["ci_lo"].iloc[0])
            his.append(s["ci_hi"].iloc[0])
            cols.append(GROUP["control" if arm.startswith("control")
                              else "tsfm" if arm.startswith("tsfm") else "baseline"])
        xs = np.arange(len(present))
        vals = np.array(vals)
        err = np.vstack([vals - np.array(los), np.array(his) - vals])
        ax.bar(xs, vals, width=0.66, color=cols, zorder=3, edgecolor=SURFACE,
               linewidth=1.2)
        ax.errorbar(xs, vals, yerr=err, fmt="none", ecolor=INK2, elinewidth=0.9,
                    capsize=2.2, zorder=4)
        top = float(np.max(his))
        for x, v, h in zip(xs, vals, his):
            # Anchor the value label above the CI cap, not the bar, so they never collide.
            ax.annotate(f"{v:.1f}", xy=(x, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=6.5,
                        color=INK)
        ax.set_ylim(0, top * 1.16)
        ax.set_xticks(xs)
        ax.set_xticklabels([LABEL[a] for a in present], rotation=34, ha="right",
                           fontsize=6.8)
        _clean(ax, "y")
        ax.set_ylabel("eval RMSE (cycles), lower better")
        ax.set_title(_vlabel(variant), loc="left")
    fig.legend(handles=[Patch(facecolor=v, label=GROUP_LABEL[k])
                        for k, v in GROUP.items()],
               loc="upper left", bbox_to_anchor=(0.012, 0.935), ncol=3,
               handlelength=1.4, borderaxespad=0)
    fig.suptitle("Attribution controls: what is the frozen pretrained representation "
                 f"worth? (L={C.LOOKBACK})", fontsize=9.5, x=0.012, ha="left", y=0.985)
    _caption(fig, "Error bars are 95% by-unit clustered bootstrap CIs. A random "
                  "projection matching the Chronos-2 arm means pretraining contributed "
                  "nothing at this look-back; shuffled labels above the trivial "
                  "marginal confirms the pipeline does not leak.")
    fig.tight_layout(rect=(0, 0.055, 1, 0.90))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ driver

def make_all(df: pd.DataFrame) -> list[str]:
    ev = df[(df.split == "eval") & (df.lookback == C.LOOKBACK)]
    out = []
    for name, fn in (
        ("fig_calibration_curves.png", fig_calibration_curves),
        ("fig_coverage_by_regime.png", fig_coverage_by_regime),
        ("fig_coverage_by_regime_compact.png", fig_coverage_by_regime_compact),
        ("fig_sharpness_frontier.png", fig_sharpness_frontier),
        ("fig_conformal_before_after.png", fig_conformal_before_after),
        ("fig_controls.png", fig_controls),
    ):
        p = C.FIGURES / name
        fn(ev, p)
        out.append(str(p))
        print(f"  wrote {p.relative_to(C.ROOT)}")
    return out
