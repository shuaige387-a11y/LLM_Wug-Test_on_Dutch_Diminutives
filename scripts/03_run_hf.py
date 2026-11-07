"""Step 3: run one Hugging Face model for method A (generation) or method B (scoring).

Usage (Colab, after `pip install -r requirements-gpu.txt`)
    python scripts/03_run_hf.py --model eurollm-1.7b --method B --limit 3     # dry run
    python scripts/03_run_hf.py --model eurollm-1.7b --method B               # core + real words
    python scripts/03_run_hf.py --model eurollm-1.7b --method B --words all   # + 51 reserve words

Outputs (appended, so an interrupted run resumes where it stopped)
    outputs/raw/<model>/generation.jsonl  or  scoring.jsonl
    outputs/raw/<model>/<method>_meta.json   model revision, versions, decoding settings, GPU
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wug import hf_runner  # noqa: E402


def select(df: pd.DataFrame, words: str) -> pd.DataFrame:
    if words == "core":
        return df[df.is_core]          # 15 core nonce words + 20 real words
    if words == "nonce-core":
        return df[df.is_core & df.word_type.eq("nonce")]
    return df                          # all 86 items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="key in config/models.yaml")
    ap.add_argument("--method", choices=["A", "B"], required=True)
    ap.add_argument("--words", choices=["core", "nonce-core", "all"], default="core")
    ap.add_argument("--limit", type=int, help="only the first N stems (dry run)")
    ap.add_argument("--models-config", default=str(ROOT / "config" / "models.yaml"))
    ap.add_argument("--clear-cache", action="store_true",
                    help="delete this model's Hugging Face download cache afterwards (saves Colab disk)")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.models_config))
    m = cfg["models"][a.model]
    if a.method not in m["methods"]:
        sys.exit(f"{a.model} is not configured for method {a.method}")
    settings = {**cfg["defaults"], **m.get("settings", {})}

    src = ROOT / "prompts" / ("generation_prompts.jsonl" if a.method == "A" else "scoring_items.jsonl")
    items = select(pd.read_json(src, lines=True), a.words)
    if a.limit:
        items = items[items.stem.isin(items.stem.drop_duplicates().head(a.limit))]

    outdir = ROOT / "outputs" / "raw" / a.model
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / ("generation.jsonl" if a.method == "A" else "scoring.jsonl")
    key = "prompt_id" if a.method == "A" else "score_id"
    done = set(pd.read_json(out, lines=True)[key]) if out.exists() and out.stat().st_size else set()
    todo = items[~items[key].isin(done)]
    print(f"{a.model} method {a.method}: {len(items)} items, {len(done)} already done, {len(todo)} to run")
    if todo.empty:
        clear_cache(m["hf_id"], a.clear_cache)
        return

    model, tok = hf_runner.load_model(m["hf_id"], settings["quantization"])
    meta = hf_runner.run_metadata(model, tok, a.model, m["hf_id"], dict(
        method=a.method, kind=m["kind"], words=a.words, **settings,
        chat_template_kwargs=m.get("chat_template_kwargs")))
    (outdir / f"{a.method}_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    if a.method == "A":
        records = hf_runner.generate(model, tok, todo, m["kind"], settings, m.get("chat_template_kwargs"))
    else:
        records = hf_runner.score(model, tok, todo)
    n = 0
    with open(out, "a") as f:
        for r in records:
            r["model"] = a.model
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 50 == 0:
                print(f"  {n} records")
    print(f"wrote {n} records to {out.relative_to(ROOT)}")
    del model
    clear_cache(m["hf_id"], a.clear_cache)


def clear_cache(hf_id: str, enabled: bool) -> None:
    if not enabled:
        return
    hub = Path(os.environ.get("HF_HUB_CACHE") or
               Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    d = hub / f"models--{hf_id.replace('/', '--')}"
    if d.exists():
        shutil.rmtree(d)
        print(f"cleared cache {d}")


if __name__ == "__main__":
    main()
