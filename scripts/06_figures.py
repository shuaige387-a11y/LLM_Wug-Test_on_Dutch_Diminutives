"""Step 6: the three figures for the results summary, from outputs/analysis/*.csv (step 5).

Usage
    python scripts/06_figures.py        # writes figures/fig1..3 as PNG (200 dpi) and PDF
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wug.analysis import wilson  # noqa: E402
from wug.stimuli import SUFFIXES  # noqa: E402

AN, FIG = ROOT / "outputs" / "analysis", ROOT / "figures"
INK, MUTED, GRID, SURFACE = "#1f1f1f", "#6b6b6b", "#e6e6e3", "#fcfcfb"
HUMAN, MODEL = INK, "#2a78d6"
SUFFIX_COLOR = dict(zip(SUFFIXES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def _clean(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def human_acc() -> pd.Series:
    h = pd.read_csv(ROOT / "data" / "human" / "human_accuracy.csv")
    h = h[h.group == "native"].set_index("target")
    return (h.n_correct / h.n_dim).reindex(SUFFIXES)


def fig1():
    """Accuracy by suffix: native speakers vs each model (gray) and the model mean (blue)."""
    a = pd.read_csv(AN / "accuracy_by_suffix.csv")
    a = a[(a.templates == "all") & (a.decoding == "all") & (a.word_type == "nonce") & (a.n_dim >= 5)]
    b = pd.read_csv(AN / "method_b_summary.csv")
    b = b[b.word_type == "nonce"].rename(columns={"p_raw_correct": "accuracy"})
    h = human_acc()
    x = list(range(len(SUFFIXES)))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    panels = [(axes[0], a, "A. Generation (7 models)\naccuracy among diminutive answers"),
              (axes[1], b, "B. Scoring (6 base models)\nmean P(correct suffix)")]
    for ax, d, title in panels:
        _clean(ax)
        for _, g in d.groupby("model"):
            ax.plot(x, g.set_index("condition").accuracy.reindex(SUFFIXES), color="#b5b5b0", lw=1,
                    marker="o", ms=3, zorder=1)
        mean = d.groupby("condition").accuracy.mean().reindex(SUFFIXES)
        ax.plot(x, mean, color=MODEL, lw=2, marker="o", ms=6, zorder=3, label="model mean")
        ax.plot(x, h, color=HUMAN, lw=2, marker="s", ms=6, zorder=4, label="native speakers")
        ax.set_xticks(x, SUFFIXES)
        ax.set_xlim(-0.3, 4.3)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=9, loc="left", color=INK)
    axes[0].set_ylabel("proportion correct (nonce words)")
    axes[0].legend(frameon=False, loc="lower left", fontsize=8)
    fig.text(0.01, -0.03, "Gray lines: individual models. Model cells with fewer than 5 diminutive answers are left out.",
             fontsize=7.5, color=MUTED)
    _save(fig, "fig1_accuracy_by_suffix")


def fig2():
    """What is produced for -kje and -etje nonce targets: native speakers vs all models pooled."""
    hc = pd.read_csv(ROOT / "data" / "human" / "human_confusion.csv")
    hc = hc[hc.group == "native"].assign(who="Native speakers")
    mc = pd.read_csv(AN / "confusion.csv")
    mc = mc[mc.word_type == "nonce"].groupby(["condition", "produced"]).n.sum().reset_index()
    mc = mc.rename(columns={"condition": "target"}).assign(who="Models (pooled)")
    d = pd.concat([hc[["who", "target", "produced", "n"]], mc[["who", "target", "produced", "n"]]])
    d = d[d.produced.isin(SUFFIXES)]
    rows = [(t, w) for t in ("-kje", "-etje") for w in ("Native speakers", "Models (pooled)")]
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    for i, (t, w) in enumerate(rows):
        g = d[(d.target == t) & (d.who == w)].set_index("produced").n.reindex(SUFFIXES).fillna(0)
        share, left, n = g / g.sum(), 0.0, int(g.sum())
        for s in SUFFIXES:
            v = share[s]
            if v > 0:
                ax.barh(i, v, left=left, color=SUFFIX_COLOR[s], height=0.62, edgecolor=SURFACE, lw=2)
                if v >= 0.08:
                    ax.text(left + v / 2, i, f"{s}\n{v:.0%}", ha="center", va="center", fontsize=7, color=INK)
            left += v
        ax.text(1.01, i, f"n = {n}", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(rows)), [f"{w}\ntarget {t}" for t, w in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [matplotlib.patches.Patch(color=SUFFIX_COLOR[s], label=s) for s in SUFFIXES]
    ax.legend(handles=handles, ncol=5, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8,
              title="suffix produced (nonce words; answers with one of the five suffixes)", title_fontsize=8, alignment="left")
    _save(fig, "fig2_kje_etje_answers")


def fig3():
    """Share of nonce -etje errors that follow the stem-final consonant, per model, with Wilson CI."""
    c = pd.read_csv(AN / "consonant_following.csv")
    c = c[(c.templates == "all") & (c.decoding == "all")].sort_values("share")
    hc = pd.read_csv(ROOT / "data" / "human" / "human_consonant_following.csv").set_index("group").loc["native"]
    hlo, hhi = wilson(int(hc.n_follow), int(hc.n_errors))
    labels = [f"{m}  ({int(k)}/{int(n)})" for m, k, n in zip(c.model, c.n_follow, c.n_errors)]
    labels.append(f"native speakers  ({int(hc.n_follow)}/{int(hc.n_errors)})")
    share = list(c.share) + [hc.n_follow / hc.n_errors]
    lo, hi = list(c.wilson_lo) + [hlo], list(c.wilson_hi) + [hhi]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for i, (s, l, h_) in enumerate(zip(share, lo, hi)):
        col = HUMAN if i == len(share) - 1 else MODEL
        ax.plot([l, h_], [i, i], color=col, lw=2, solid_capstyle="round")
        ax.plot(s, i, "s" if col == HUMAN else "o", color=col, ms=7, mec=SURFACE, mew=2)
    ax.axvline(0.25, color=MUTED, ls=(0, (3, 3)), lw=1)
    ax.text(0.26, -0.95, "chance (1/4)", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.3, len(labels) - 0.5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_xlabel("nonce -etje errors whose suffix follows the final consonant (95% Wilson CI)")
    _clean(ax, grid="x")
    _save(fig, "fig3_consonant_following")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print(f"wrote 3 figures to {FIG.relative_to(ROOT)}/")
