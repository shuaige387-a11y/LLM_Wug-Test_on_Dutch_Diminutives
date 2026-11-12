"""Step 4: code every model output and draw a sample for manual checking.

Usage
    python scripts/04_code_outputs.py                  # code all outputs/raw/*/generation.jsonl
    python scripts/04_code_outputs.py --sample 200     # also write a manual-check sample
    python scripts/04_code_outputs.py --agreement outputs/coded/manual_check.csv

Outputs
    outputs/coded/generation_coded.csv   one row per output, with strategy / produced / correct
    outputs/coded/manual_check.csv       stratified random sample; fill in manual_strategy and
                                         manual_produced, then rerun with --agreement
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wug.coding import code_model_output  # noqa: E402


def code_all(raw_dir: Path) -> pd.DataFrame:
    frames = [pd.read_json(p, lines=True) for p in sorted(raw_dir.glob("*/generation.jsonl"))]
    if not frames:
        sys.exit(f"no generation.jsonl files under {raw_dir}")
    d = pd.concat(frames, ignore_index=True)
    codes = [code_model_output(o, s) for o, s in zip(d.output, d.stem)]
    d["strategy"] = [c.strategy for c in codes]
    d["produced"] = [c.produced for c in codes]
    d["coded_word"] = [c.word for c in codes]
    d["ambiguous"] = [c.ambiguous for c in codes]
    d["correct"] = (d.strategy == "dim") & (d.produced == d.condition)
    return d


def draw_sample(d: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratify by model x strategy so rare codes (misread, dialect, klein+N) are checked too."""
    n_groups = d.groupby(["model", "strategy"]).ngroups
    per = max(1, n // max(1, n_groups))
    s = d.sample(frac=1, random_state=seed).groupby(["model", "strategy"]).head(per)[
        ["model", "prompt_id", "decoding", "sample_index", "stem", "condition", "output",
         "strategy", "produced"]].copy()
    s["manual_strategy"] = ""
    s["manual_produced"] = ""
    s["note"] = ""
    return s


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    cats = sorted(set(a) | set(b))
    po = (a.values == b.values).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agreement(path: Path) -> None:
    """Compare manual codes with the automatic ones.

    Accepts the CSV sample itself or the blind Excel workbook (merged back on check_id).
    Suffixes may be written with or without the leading hyphen ('pje' or '-pje').
    """
    if path.suffix == ".xlsx":
        manual = pd.read_excel(path, sheet_name="check", dtype=str).fillna("")
        auto = pd.read_csv(ROOT / "outputs" / "coded" / "manual_check.csv", keep_default_na=False)
        m = auto.drop(columns=["manual_strategy", "manual_produced", "note"], errors="ignore").merge(
            manual[["check_id", "manual_strategy", "manual_produced", "note"]], on="check_id")
    else:
        m = pd.read_csv(path, keep_default_na=False)
    m["manual_strategy"] = m.manual_strategy.str.strip()
    m["manual_produced"] = [("-" + p.strip().lstrip("-")) if p.strip() else "" for p in m.manual_produced]
    m["produced"] = m["produced"].fillna("").astype(str)
    m = m[m.manual_strategy != ""]
    if m.empty:
        sys.exit("no rows filled in yet")
    st = (m.manual_strategy == m.strategy).mean()
    dim = m[m.manual_strategy == "dim"]
    sf = (dim.manual_produced == dim.produced).mean() if len(dim) else float("nan")
    print(f"checked rows: {len(m)}")
    print(f"strategy agreement: {st:.1%}")
    print(f"Cohen's kappa (strategy): {cohen_kappa(m.manual_strategy, m.strategy):.2f}")
    print(f"suffix agreement on manual 'dim' rows: {sf:.1%} (n={len(dim)})")
    bad = m[(m.manual_strategy != m.strategy) | ((m.manual_strategy == "dim") & (m.manual_produced != m.produced))]
    if len(bad):
        print("\ndisagreements:")
        print(bad[["model", "stem", "output", "strategy", "produced", "manual_strategy", "manual_produced"]]
              .to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "outputs" / "raw"))
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--agreement")
    a = ap.parse_args()
    if a.agreement:
        agreement(Path(a.agreement))
        return
    out = ROOT / "outputs" / "coded"
    out.mkdir(parents=True, exist_ok=True)
    d = code_all(Path(a.raw))
    d.to_csv(out / "generation_coded.csv", index=False)
    print(f"coded {len(d)} outputs from {d.model.nunique()} models -> outputs/coded/generation_coded.csv")
    print(pd.crosstab(d.model, d.strategy, normalize="index").round(2).to_string())
    if a.sample:
        s = draw_sample(d, a.sample, a.seed)
        s.to_csv(out / "manual_check.csv", index=False)
        print(f"\nwrote {len(s)} rows to outputs/coded/manual_check.csv for manual checking")


if __name__ == "__main__":
    main()
