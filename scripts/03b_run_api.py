"""Step 3b: run one API model for method A (generation).

Usage (needs ANTHROPIC_API_KEY in the environment; on Colab, store it in Secrets)
    python scripts/03b_run_api.py --model claude-sonnet-5-5 --limit 1 --calls 1   # dry run
    python scripts/03b_run_api.py --model claude-sonnet-5-5                       # core + real words

Outputs (appended, so an interrupted run resumes where it stopped)
    outputs/raw/<model>/generation.jsonl
    outputs/raw/<model>/A_meta.json   model id, SDK version, settings, run date
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wug import api_runner  # noqa: E402

SELECT = {"core": lambda d: d[d.is_core],
          "nonce-core": lambda d: d[d.is_core & d.word_type.eq("nonce")],
          "all": lambda d: d}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="key in config/api_models.yaml")
    ap.add_argument("--words", choices=list(SELECT), default="core")
    ap.add_argument("--limit", type=int, help="only the first N stems (dry run)")
    ap.add_argument("--calls", type=int, help="override n_calls (dry run)")
    ap.add_argument("--config", default=str(ROOT / "config" / "api_models.yaml"))
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    m = cfg["models"][a.model]
    settings = {**cfg["defaults"], **m.get("settings", {})}
    if a.calls:
        settings["n_calls"] = a.calls

    items = SELECT[a.words](pd.read_json(ROOT / "prompts" / "generation_prompts.jsonl", lines=True))
    if a.limit:
        items = items[items.stem.isin(items.stem.drop_duplicates().head(a.limit))]

    outdir = ROOT / "outputs" / "raw" / a.model
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "generation.jsonl"
    done = set()
    if out.exists() and out.stat().st_size:
        prev = pd.read_json(out, lines=True)
        done = set(zip(prev.prompt_id, prev.sample_index))
    n_total = len(items) * settings["n_calls"]
    print(f"{a.model}: {n_total} calls planned, {len(done)} already done")

    client = api_runner.make_client(m["provider"], settings["max_retries"])
    meta = api_runner.run_metadata(a.model, m, dict(words=a.words, **settings))
    (outdir / "A_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    n = tok_in = tok_out = 0
    with open(out, "a") as f:
        for r in api_runner.generate(client, items, m, settings, done):
            r["model"] = a.model
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            tok_in += r["input_tokens"]
            tok_out += r["output_tokens"]
            if n % 100 == 0:
                print(f"  {n} calls")
    print(f"wrote {n} records to {out.relative_to(ROOT)} ({tok_in} input / {tok_out} output tokens)")


if __name__ == "__main__":
    main()
