"""Step 5: analysis of coded model outputs (method A), scores (method B) and human aggregates.

All functions take plain data frames and return data frames, so each table can be
rebuilt and checked on its own. Confidence intervals:

* Wilson intervals treat every output as independent. Outputs of the same item are not
  independent (11 runs per prompt, 5 templates per item), so these are too narrow.
* `cluster_bootstrap` resamples items (stems) within each suffix condition instead. With
  three nonce items per condition this interval is wide; this is the relevant uncertainty.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from .coding import consonant_suffix
from .stimuli import SUFFIXES

HUMAN_COMPARABLE = ("T1_original", "T2_frame", "T3_two_objects")


# ------------------------------------------------------------------ intervals

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cluster_bootstrap(d: pd.DataFrame, value: str, n_boot: int = 2000, seed: int = 1) -> tuple[float, float]:
    """95% interval of the mean of `value`, resampling stems within each condition."""
    if d.empty:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    per_stem = d.groupby(["condition", "stem"])[value].agg(["sum", "count"]).reset_index()
    groups = [g[["sum", "count"]].to_numpy() for _, g in per_stem.groupby("condition")]
    means = np.empty(n_boot)
    for b in range(n_boot):
        s = c = 0.0
        for g in groups:
            pick = g[rng.integers(0, len(g), len(g))]
            s += pick[:, 0].sum()
            c += pick[:, 1].sum()
        means[b] = s / c
    return tuple(np.quantile(means, [0.025, 0.975]))


# ------------------------------------------------------------------ method A

def select(d: pd.DataFrame, templates: str = "all", decoding: str = "all") -> pd.DataFrame:
    if templates == "human":
        d = d[d.template.isin(HUMAN_COMPARABLE)]
    elif templates == "elicit":
        d = d[~d.template.isin(HUMAN_COMPARABLE)]
    if decoding == "greedy":
        d = d[d.decoding == "greedy"]
    return d


def strategy_table(d: pd.DataFrame) -> pd.DataFrame:
    """Share of each answer strategy, by model and template."""
    t = d.groupby(["model", "template", "strategy"]).size().rename("n").reset_index()
    t["share"] = t.n / t.groupby(["model", "template"]).n.transform("sum")
    return t


def accuracy_table(d: pd.DataFrame, by=("model", "word_type", "condition")) -> pd.DataFrame:
    """Accuracy among diminutive answers, with Wilson and (per model x word type) bootstrap CIs."""
    dim = d[d.strategy == "dim"].assign(correct=lambda x: x.correct.astype(int))
    rows = []
    for key, g in d.groupby(list(by)):
        gd = dim.loc[dim.index.intersection(g.index)]
        k, n = int(gd.correct.sum()), len(gd)
        lo, hi = wilson(k, n)
        rows.append(dict(zip(by, key if isinstance(key, tuple) else (key,)), n_outputs=len(g), n_dim=n,
                         dim_rate=n / len(g) if len(g) else float("nan"),
                         n_correct=k, accuracy=k / n if n else float("nan"), wilson_lo=lo, wilson_hi=hi))
    return pd.DataFrame(rows)


def overall_accuracy(d: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per model x word type, averaged over outputs, with an item-level bootstrap CI."""
    dim = d[d.strategy == "dim"].assign(correct=lambda x: x.correct.astype(int))
    rows = []
    for (m, wt), g in dim.groupby(["model", "word_type"]):
        lo, hi = cluster_bootstrap(g, "correct")
        rows.append(dict(model=m, word_type=wt, n_dim=len(g), accuracy=g.correct.mean(), boot_lo=lo, boot_hi=hi))
    return pd.DataFrame(rows)


def real_nonce_gap(d: pd.DataFrame) -> pd.DataFrame:
    a = overall_accuracy(d).pivot(index="model", columns="word_type", values="accuracy")
    a["gap"] = a.get("real") - a.get("nonce")
    return a.reset_index()


def confusion(d: pd.DataFrame) -> pd.DataFrame:
    dim = d[d.strategy == "dim"]
    return dim.groupby(["model", "word_type", "condition", "produced"]).size().rename("n").reset_index()


def consonant_following(d: pd.DataFrame, by=("model",)) -> pd.DataFrame:
    """Among nonce -etje errors (diminutive answers only), share that follow the final consonant.

    Chance is 1/4 if the error suffix were picked at random among the four wrong suffixes.
    The binomial test against 1/4 treats outputs as independent; the per-stem counts show
    whether one item drives the result.
    """
    e = d[(d.word_type == "nonce") & (d.condition == "-etje") & (d.strategy == "dim") & (d.produced != "-etje")]
    e = e.assign(follow=[p == consonant_suffix(s) for s, p in zip(e.stem, e.produced)])
    rows = []
    for key, g in e.groupby(list(by)):
        k, n = int(g.follow.sum()), len(g)
        lo, hi = wilson(k, n)
        per = {f"follow_{s}": f"{int(x.follow.sum())}/{len(x)}" for s, x in g.groupby("stem")}
        rows.append(dict(zip(by, key if isinstance(key, tuple) else (key,)), n_errors=n, n_follow=k,
                         share=k / n, wilson_lo=lo, wilson_hi=hi,
                         p_vs_chance=stats.binomtest(k, n, 0.25, alternative="greater").pvalue, **per))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ method B

def method_b(scores: pd.DataFrame, column: str = "logprob_total") -> pd.DataFrame:
    """P(suffix | stem) by softmax over the five candidates, raw (frame) and calibrated.

    Calibrated: frame log-prob minus neutral-context log-prob for the same form (a
    pointwise-mutual-information correction for how likely the word is anywhere).
    """
    s = scores.copy()
    wide = s.pivot_table(index=["model", "stem", "condition", "word_type", "candidate_suffix", "is_correct"],
                         columns="context", values=column).reset_index()
    wide["calibrated"] = wide["frame"] - wide["neutral"]

    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    g = wide.groupby(["model", "stem"])
    wide["p_raw"] = g["frame"].transform(softmax)
    wide["p_cal"] = g["calibrated"].transform(softmax)
    return wide


def method_b_summary(wide: pd.DataFrame) -> pd.DataFrame:
    """Per model x word type x condition: mean P(correct) and share of items where it ranks first."""
    out = []
    for (m, wt, c), g in wide.groupby(["model", "word_type", "condition"]):
        row = dict(model=m, word_type=wt, condition=c, n_items=g.stem.nunique())
        for v in ("p_raw", "p_cal"):
            top = g.loc[g.groupby("stem")[v].idxmax()]
            row[f"{v}_correct"] = g.loc[g.is_correct, v].mean()
            row[f"{v}_top1"] = top.is_correct.mean()
        out.append(row)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ ranking vs humans

def ranking_vs_humans(acc: pd.DataFrame, human: pd.DataFrame, value: str = "accuracy",
                      group: str = "native") -> pd.DataFrame:
    """Spearman correlation between a model's per-suffix accuracy and the human one (5 points)."""
    h = human[human.group == group].set_index("target")
    h = (h.n_correct / h.n_dim).reindex(SUFFIXES)
    rows = []
    for m, g in acc.groupby("model"):
        a = g.set_index("condition")[value].reindex(SUFFIXES)
        ok = a.notna()
        rho = stats.spearmanr(a[ok], h[ok]).statistic if ok.sum() >= 3 else float("nan")
        order = " > ".join(a.sort_values(ascending=False).index)
        rows.append(dict(model=m, spearman_rho=rho, n_suffixes=int(ok.sum()), model_order=order))
    rows.append(dict(model=f"human_{group}", spearman_rho=1.0, n_suffixes=5,
                     model_order=" > ".join(h.sort_values(ascending=False).index)))
    return pd.DataFrame(rows)
