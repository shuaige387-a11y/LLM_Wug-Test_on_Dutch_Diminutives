"""Build method A prompts and method B scoring items from the stimuli and templates."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import pandas as pd
import yaml

from .stimuli import REPO_ROOT, SUFFIXES, diminutive

DEFAULT_TEMPLATES = REPO_ROOT / "config" / "templates.yaml"
DEFAULT_REAL_WORDS = REPO_ROOT / "data" / "stimuli" / "real_words.csv"


def load_templates(path: str | Path = DEFAULT_TEMPLATES) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_real_words(path: str | Path = DEFAULT_REAL_WORDS) -> pd.DataFrame:
    rw = pd.read_csv(path, keep_default_na=False)
    bad = [(w, e) for w, c, e in zip(rw.word, rw.condition, rw.expected_diminutive) if diminutive(w, c) != e]
    if bad:
        raise ValueError(f"real_words.csv disagrees with spelling rules: {bad}")
    return rw


def item_table(stimuli: pd.DataFrame, real_words: pd.DataFrame | None = None) -> pd.DataFrame:
    """Stack nonce and real words into one table of items."""
    items = pd.DataFrame(dict(
        stem=stimuli.nonce_base, condition=stimuli.condition,
        word_type="nonce", is_core=stimuli.is_core,
    ))
    if real_words is not None:
        items = pd.concat([items, pd.DataFrame(dict(
            stem=real_words.word, condition=real_words.condition,
            word_type="real", is_core=True,
        ))], ignore_index=True)
    return items


def _item_seed(stem: str, base_seed: int) -> int:
    """Stable per-item seed, independent of item order and Python hash randomisation."""
    return int(hashlib.sha256(f"{base_seed}:{stem}".encode()).hexdigest()[:8], 16)


def _examples(cfg: dict, stem: str, seed: int) -> str:
    ex = list(cfg["fewshot_examples"])
    random.Random(_item_seed(stem, seed)).shuffle(ex)
    return "\n".join(f"{w} → {d}" for w, d in ex)


def render(cfg: dict, template_id: str, stem: str, form: str, seed: int = 20261107) -> str:
    """Render one template for one stem; form is 'chat' or 'base'."""
    t = cfg["templates"][template_id]
    big, small = cfg["survey_objects"].get(stem, [1, 4])
    text = t[form].format(stem=stem, big=big, small=small,
                          examples=_examples(cfg, stem, seed) if "{examples}" in t[form] else "")
    reg = t.get("register", "none")
    if form == "chat" and reg in cfg["answer_instruction"]:
        text = f"{text}\n{cfg['answer_instruction'][reg]}"
    return text


def build_generation_prompts(items: pd.DataFrame, cfg: dict, seed: int = 20261107) -> pd.DataFrame:
    rows = []
    for _, it in items.iterrows():
        for tid, t in cfg["templates"].items():
            rows.append(dict(
                prompt_id=f"{it.word_type}:{it.stem}:{tid}",
                stem=it.stem, condition=it.condition, word_type=it.word_type,
                is_core=it.is_core, template=tid, human_comparable=t["human_comparable"],
                chat_prompt=render(cfg, tid, it.stem, "chat", seed),
                base_prompt=render(cfg, tid, it.stem, "base", seed),
            ))
    return pd.DataFrame(rows)


def build_scoring_items(items: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for _, it in items.iterrows():
        for suf in SUFFIXES:
            cand = diminutive(it.stem, suf)
            for ctx, spec in cfg["scoring"].items():
                rows.append(dict(
                    score_id=f"{it.word_type}:{it.stem}:{suf}:{ctx}",
                    stem=it.stem, condition=it.condition, word_type=it.word_type,
                    is_core=it.is_core, candidate_suffix=suf, candidate_form=cand,
                    is_correct=suf == it.condition, context=ctx,
                    prefix=spec["prefix"].format(stem=it.stem),
                    continuation=spec["continuation"].format(candidate=cand),
                ))
    return pd.DataFrame(rows)
