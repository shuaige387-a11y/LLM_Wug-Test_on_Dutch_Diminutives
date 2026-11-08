"""Step 5: build every analysis table from the coded outputs, the scores and the human aggregates.

Usage
    python scripts/05_analyse.py

Reads
    outputs/coded/generation_coded.csv   (step 4)
    outputs/raw/*/scoring.jsonl          (step 3, method B)
    data/human/*.csv                     (step 1, anonymised aggregates)
Writes outputs/analysis/*.csv and prints the headline numbers.

Subsets: "all" = all five templates; "human" = T1-T3 (the human task framings);
"elicit" = T4-T5 (metalinguistic and few-shot). Sensitivity: greedy outputs only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wug import analysis as A  # noqa: E402

OUT = ROOT / "outputs" / "analysis"


def tag(df: pd.DataFrame, **kw) -> pd.DataFrame:
    return df.assign(**kw)[list(kw) + list(df.columns)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(ROOT / "outputs" / "coded" / "generation_coded.csv", keep_default_na=False)
    d["correct"] = d.correct.astype(str).eq("True")
    human_acc = pd.read_csv(ROOT / "data" / "human" / "human_accuracy.csv")
    human_cf = pd.read_csv(ROOT / "data" / "human" / "human_consonant_following.csv")

    acc, overall, cf, gap, rank = [], [], [], [], []
    for templates in ("all", "human", "elicit"):
        for decoding in ("all", "greedy"):
            s = A.select(d, templates, decoding)
            t = dict(templates=templates, decoding=decoding)
            a = A.accuracy_table(s)
            acc.append(tag(a, **t))
            overall.append(tag(A.overall_accuracy(s), **t))
            cf.append(tag(A.consonant_following(s), **t))
            gap.append(tag(A.real_nonce_gap(s), **t))
            rank.append(tag(A.ranking_vs_humans(a[a.word_type == "nonce"], human_acc), **t))
    tables = {
        "accuracy_by_suffix.csv": pd.concat(acc), "accuracy_overall.csv": pd.concat(overall),
        "consonant_following.csv": pd.concat(cf), "real_nonce_gap.csv": pd.concat(gap),
        "ranking_vs_humans.csv": pd.concat(rank), "strategy.csv": A.strategy_table(d),
        "confusion.csv": A.confusion(d),
    }
    sc = [pd.read_json(p, lines=True) for p in sorted((ROOT / "outputs" / "raw").glob("*/scoring.jsonl"))]
    if sc:
        wide = A.method_b(pd.concat(sc, ignore_index=True))
        mb = A.method_b_summary(wide)
        tables["method_b_items.csv"] = wide
        tables["method_b_summary.csv"] = mb
        mbn = mb[mb.word_type == "nonce"].rename(columns={"p_raw_correct": "accuracy"})
        tables["method_b_ranking_vs_humans.csv"] = A.ranking_vs_humans(mbn, human_acc)
    for name, t in tables.items():
        t.to_csv(OUT / name, index=False)

    pd.set_option("display.width", 200)
    main_ = lambda t: t[(t.templates == "all") & (t.decoding == "all")]  # noqa: E731
    print("== human (native): accuracy by suffix")
    print(human_acc[human_acc.group == "native"].to_string(index=False))
    print(human_cf.to_string(index=False))
    print("\n== method A, all templates, all outputs: accuracy among diminutive answers (nonce)")
    a = main_(tables["accuracy_by_suffix.csv"])
    print(a[a.word_type == "nonce"].pivot(index="model", columns="condition", values="accuracy")[A.SUFFIXES].round(2))
    print("\n== overall accuracy with item bootstrap CI")
    print(main_(tables["accuracy_overall.csv"]).round(2).to_string(index=False))
    print("\n== -etje errors that follow the final consonant (chance = .25)")
    print(main_(tables["consonant_following.csv"]).round(3).to_string(index=False))
    print("\n== ranking vs native speakers")
    print(main_(tables["ranking_vs_humans.csv"]).round(2).to_string(index=False))
    if sc:
        print("\n== method B (nonce): mean P(correct), raw / calibrated")
        print(mb[mb.word_type == "nonce"].pivot(index="model", columns="condition",
                                                 values="p_raw_correct")[A.SUFFIXES].round(2))
        print(mb[mb.word_type == "nonce"].pivot(index="model", columns="condition",
                                                 values="p_cal_correct")[A.SUFFIXES].round(2))
        print(tables["method_b_ranking_vs_humans.csv"].round(2).to_string(index=False))
    print(f"\nwrote {len(tables)} tables to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
