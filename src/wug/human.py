"""Load the original Qualtrics export, code every trial, and export only aggregates.

The raw export and the exclusion file stay in data/private/ (git-ignored).
Only the aggregate tables written by `export_summaries` go into the public repo.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .coding import code_answer, consonant_suffix
from .stimuli import SUFFIXES

# Qualtrics column name -> (stem, target suffix). Column names are as exported.
ITEM_COLUMNS = {
    "Tuifje": ("tuif", "-je"), "Duipje": ("duip", "-je"), "Baukje": ("bauk", "-je"),
    "Joertje": ("joer", "-tje"), "Muintje": ("muin", "-tje"), "Duiltje": ("duil", "-tje"),
    "Kuimpje": ("kuim", "-pje"), "Muimpje": ("muim", "-pje"), "Loempje": ("loem", "-pje"),
    "Toddingkje": ("todding", "-kje"), "Benningkje": ("biening", "-kje"), "Mozingkje": ("mozing", "-kje"),
    "Jinetje": ("jin", "-etje"), "Kemetje": ("kem", "-etje"), "Rengetje": ("reng", "-etje"),
}


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the export; Qualtrics puts question text in the first data row."""
    df = pd.read_excel(path, header=0)
    return df.iloc[1:].reset_index(drop=True)


def code_trials(raw: pd.DataFrame, exclusions: dict) -> pd.DataFrame:
    """Return one row per answered trial of every complete, non-preview response.

    exclusions = {"excluded": [ResponseId, ...],          # off-task / joke responses
                  "reclassify_as_L2": [ResponseId, ...],  # self-reported fluent, analysed as L2
                  "other_variety": [ResponseId, ...]}     # e.g. Surinamese Dutch: own group,
                                                          # left out of native and L2 comparisons
    """
    excluded = set(exclusions.get("excluded", []))
    as_l2 = set(exclusions.get("reclassify_as_L2", []))
    variety = set(exclusions.get("other_variety", []))
    keep = raw[(raw["Status"] == "IP Address") & (raw["Finished"].astype(str) == "True")]
    rows = []
    for pid, (_, r) in enumerate(keep.iterrows(), start=1):
        rid = r["ResponseId"]
        fluent = str(r["Q2"]).startswith("Ja")
        if rid in variety:
            group = "other_variety"
        elif fluent and rid not in as_l2:
            group = "native"
        else:
            group = "L2"
        for col, (stem, target) in ITEM_COLUMNS.items():
            ans = r.get(col)
            if pd.isna(ans):
                continue
            c = code_answer(ans, stem)
            rows.append(dict(
                participant=f"P{pid:02d}", response_id=rid, group=group,
                excluded=rid in excluded, stem=stem, target=target,
                answer=str(ans).strip(), strategy=c.strategy, produced=c.produced,
                ambiguous=c.ambiguous,
            ))
    t = pd.DataFrame(rows)
    t["correct"] = (t.strategy == "dim") & (t.produced == t.target)
    return t


def load_exclusions(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------- aggregates

def accuracy_table(trials: pd.DataFrame) -> pd.DataFrame:
    d = trials[(~trials.excluded) & (trials.strategy == "dim")]
    out = d.groupby(["group", "target"]).correct.agg(n_correct="sum", n_dim="count").reset_index()
    out["target"] = pd.Categorical(out.target, SUFFIXES, ordered=True)
    return out.sort_values(["group", "target"]).reset_index(drop=True)


def confusion_table(trials: pd.DataFrame) -> pd.DataFrame:
    d = trials[(~trials.excluded) & (trials.strategy.isin(["dim", "dialect"]))]
    out = d.groupby(["group", "target", "produced"]).size().rename("n").reset_index()
    return out


def strategy_table(trials: pd.DataFrame) -> pd.DataFrame:
    d = trials[~trials.excluded]
    return d.groupby(["group", "strategy"]).size().rename("n").reset_index()


def consonant_following(trials: pd.DataFrame, group: str = "native") -> dict:
    """Among -etje errors (diminutive answers only), share that follow the final consonant."""
    d = trials[(~trials.excluded) & (trials.group == group) & (trials.strategy == "dim")
               & (trials.target == "-etje") & (~trials.correct)]
    follows = [p == consonant_suffix(s) for s, p in zip(d.stem, d.produced)]
    return dict(group=group, n_errors=len(d), n_follow=int(sum(follows)))


def export_summaries(trials: pd.DataFrame, outdir: str | Path) -> list[Path]:
    """Write the aggregate tables that may be published."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # A group of one person would publish that person's answers; keep it private.
    trials = trials[trials.group.isin(["native", "L2"])]
    k = trials[~trials.excluded]
    n_part = k.groupby("group").participant.nunique().rename("n_participants").reset_index()
    files = {
        "human_participants.csv": n_part,
        "human_accuracy.csv": accuracy_table(trials),
        "human_confusion.csv": confusion_table(trials),
        "human_strategy.csv": strategy_table(trials),
        "human_consonant_following.csv": pd.DataFrame(
            [consonant_following(trials, g) for g in ("native", "L2")]),
    }
    paths = []
    for name, df in files.items():
        p = outdir / name
        df.to_csv(p, index=False)
        paths.append(p)
    return paths
