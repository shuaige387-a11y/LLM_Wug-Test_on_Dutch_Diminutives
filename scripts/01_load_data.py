"""Step 1: load and validate the stimuli; code the human data and export aggregates.

Usage
    python scripts/01_load_data.py                       # stimuli only
    python scripts/01_load_data.py --human-raw data/private/raw_qualtrics_export.xlsx \
                                   --exclusions data/private/human_exclusions.yaml

Outputs
    data/stimuli/candidates.csv      all 5 candidate forms for every stem
    data/human/*.csv                 aggregate human tables (public)
    data/private/human_trials.csv    trial-level coding (private, git-ignored)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wug import human, stimuli  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", default=str(stimuli.DEFAULT_STIMULI))
    ap.add_argument("--human-raw")
    ap.add_argument("--exclusions")
    a = ap.parse_args()

    st = stimuli.load_stimuli(a.stimuli)
    print(f"Stimuli: {len(st)} stems ({st.is_core.sum()} core, {(~st.is_core).sum()} reserve)")
    print(st.groupby(["condition", "status"]).size().unstack(fill_value=0).to_string(), "\n")
    cand = stimuli.candidate_table(st)
    cand.to_csv(ROOT / "data" / "stimuli" / "candidates.csv", index=False)
    print("Candidate forms for the core words:")
    core = cand[cand.is_core].pivot(index="nonce_base", columns="candidate_suffix", values="candidate_form")
    print(core[stimuli.SUFFIXES].to_string(), "\n")

    if not a.human_raw:
        return
    excl = human.load_exclusions(a.exclusions) if a.exclusions else {}
    trials = human.code_trials(human.load_raw(a.human_raw), excl)
    trials.to_csv(ROOT / "data" / "private" / "human_trials.csv", index=False)
    kept = trials[~trials.excluded]
    print(f"Human: {trials.participant.nunique()} complete responses, "
          f"{kept.participant.nunique()} after exclusion, {len(kept)} trials")
    print(kept.groupby("group").participant.nunique().to_string(), "\n")
    print("Accuracy among diminutive answers:")
    print(human.accuracy_table(trials).to_string(index=False), "\n")
    for g in ("native", "L2"):
        print(human.consonant_following(trials, g))
    amb = kept[kept.ambiguous]
    if len(amb):
        print(f"\n{len(amb)} answers offered alternatives; the first was coded:")
        print(amb[["participant", "stem", "answer", "produced"]].to_string(index=False))
    for p in human.export_summaries(trials, ROOT / "data" / "human"):
        print("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
