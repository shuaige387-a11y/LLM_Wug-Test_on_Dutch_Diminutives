"""Load the nonce-word stimuli and build the five candidate diminutive forms.

The five Dutch diminutive allomorphs are -je, -tje, -pje, -kje and -etje.
For every stem we build all five forms with the spelling rules a Dutch writer
would apply, so that scoring (method B) and coding (method A) compare like with
like:

* -kje after a stem in -ng turns ng into nk (koning -> koninkje).
* -etje doubles a single final consonant after a single short vowel letter
  (ring -> ringetje, but kam -> kammetje, vin -> vinnetje).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

SUFFIXES = ["-je", "-tje", "-pje", "-kje", "-etje"]
VOWELS = "aeiouy"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STIMULI = REPO_ROOT / "data" / "stimuli" / "nonce_words.csv"


def _short_vowel_single_coda(stem: str) -> bool:
    """True for stems like 'kem', 'jin', 'bor': consonant, one vowel letter, one consonant."""
    return bool(re.search(rf"(^|[^{VOWELS}])[{VOWELS}][^{VOWELS}]$", stem)) and not stem.endswith("ng")


def diminutive(stem: str, suffix: str) -> str:
    """Spell `stem` + `suffix` with Dutch orthographic adjustments."""
    stem = stem.lower()
    if suffix == "-je":
        return stem + "je"
    if suffix == "-tje":
        return stem + "tje"
    if suffix == "-pje":
        return stem + "pje"
    if suffix == "-kje":
        return (stem[:-1] + "kje") if stem.endswith("ng") else stem + "kje"
    if suffix == "-etje":
        return stem + (stem[-1] if _short_vowel_single_coda(stem) else "") + "etje"
    raise ValueError(f"unknown suffix {suffix!r}")


def load_stimuli(path: str | Path = DEFAULT_STIMULI, core_only: bool = False) -> pd.DataFrame:
    """Read nonce_words.csv and check it is internally consistent."""
    df = pd.read_csv(path)
    required = {"condition", "status", "rank", "nonce_base", "expected_diminutive", "wuggy_template"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"stimulus file lacks columns: {sorted(missing)}")
    bad_cond = set(df.condition) - set(SUFFIXES)
    if bad_cond:
        raise ValueError(f"unknown conditions: {bad_cond}")
    if df.nonce_base.duplicated().any():
        raise ValueError(f"duplicate stems: {df.nonce_base[df.nonce_base.duplicated()].tolist()}")
    rebuilt = [diminutive(s, c) for s, c in zip(df.nonce_base, df.condition)]
    mismatch = df.loc[[r != e for r, e in zip(rebuilt, df.expected_diminutive)]]
    if len(mismatch):
        raise ValueError(f"expected_diminutive disagrees with spelling rules:\n{mismatch}")
    df["is_core"] = df.status.eq("recommended")
    if core_only:
        df = df[df.is_core]
    return df.reset_index(drop=True)


def candidate_table(stimuli: pd.DataFrame) -> pd.DataFrame:
    """One row per stem x candidate suffix, flagging the correct one."""
    rows = []
    for _, r in stimuli.iterrows():
        for suf in SUFFIXES:
            rows.append(
                dict(
                    nonce_base=r.nonce_base,
                    condition=r.condition,
                    is_core=r.is_core,
                    candidate_suffix=suf,
                    candidate_form=diminutive(r.nonce_base, suf),
                    is_correct=suf == r.condition,
                )
            )
    return pd.DataFrame(rows)
