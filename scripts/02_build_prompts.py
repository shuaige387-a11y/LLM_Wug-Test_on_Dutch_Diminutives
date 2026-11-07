"""Step 2: build all method A prompts and method B scoring items.

Usage
    python scripts/02_build_prompts.py            # prints one example per template

Outputs
    prompts/generation_prompts.jsonl   one row per item x template (chat and base text)
    prompts/scoring_items.jsonl        one row per item x candidate suffix x context
    prompts/MANIFEST.json              counts and the template file's SHA-256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wug import prompts, stimuli  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", default=str(prompts.DEFAULT_TEMPLATES))
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--show", default="tuif", help="stem to print as an example")
    a = ap.parse_args()

    cfg = prompts.load_templates(a.templates)
    items = prompts.item_table(stimuli.load_stimuli(), prompts.load_real_words())
    gen = prompts.build_generation_prompts(items, cfg, a.seed)
    sco = prompts.build_scoring_items(items, cfg)

    out = ROOT / "prompts"
    out.mkdir(exist_ok=True)
    gen.to_json(out / "generation_prompts.jsonl", orient="records", lines=True, force_ascii=False)
    sco.to_json(out / "scoring_items.jsonl", orient="records", lines=True, force_ascii=False)
    manifest = dict(
        seed=a.seed,
        templates_sha256=hashlib.sha256(Path(a.templates).read_bytes()).hexdigest(),
        n_items={(wt if wt == "real" else f"{wt}_{'core' if core else 'reserve'}"): int(n)
                 for (wt, core), n in items.groupby(["word_type", "is_core"]).size().items()},
        n_generation_prompts=len(gen), n_scoring_items=len(sco),
    )
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest, indent=2), "\n")
    for _, r in gen[gen.stem == a.show].iterrows():
        print(f"===== {r.template}  (human-comparable: {r.human_comparable})")
        print("--- chat\n" + r.chat_prompt)
        print("--- base\n" + r.base_prompt + "▌\n")
    ex = sco[(sco.stem == a.show) & (sco.context == "frame")]
    print("===== scoring (frame)")
    for _, r in ex.iterrows():
        print(f"{r.prefix}[{r.continuation}]{'  <- correct' if r.is_correct else ''}")


if __name__ == "__main__":
    main()
