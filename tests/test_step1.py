"""Checks for step 1. The human test runs only when the private raw file is present."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wug import human, stimuli  # noqa: E402
from wug.coding import code_answer  # noqa: E402

RAW = ROOT / "data" / "private" / "raw_qualtrics_export.xlsx"
EXCL = ROOT / "data" / "private" / "human_exclusions.yaml"


def test_stimuli_consistent():
    st = stimuli.load_stimuli()
    assert st.is_core.sum() == 15
    assert st[st.is_core].groupby("condition").size().eq(3).all()


@pytest.mark.parametrize("stem,suffix,form", [
    ("jin", "-etje", "jinnetje"), ("kem", "-etje", "kemmetje"), ("reng", "-etje", "rengetje"),
    ("todding", "-kje", "toddinkje"), ("todding", "-etje", "toddingetje"),
    ("tuif", "-etje", "tuifetje"), ("muim", "-pje", "muimpje"), ("bor", "-etje", "borretje"),
])
def test_spelling(stem, suffix, form):
    assert stimuli.diminutive(stem, suffix) == form


@pytest.mark.parametrize("answer,stem,strategy,produced", [
    ("Tuifje", "tuif", "dim", "-je"), ("Baukje", "bauk", "dim", "-je"),
    ("Toddingkje", "todding", "dim", "-kje"), ("Toddinkje", "todding", "dim", "-kje"),
    ("Kemmetje", "kem", "dim", "-etje"), ("Loemetje", "loem", "dim", "-etje"),
    ("Rengeltje", "reng", "dim", "-eltje"), ("Baukie", "bauk", "dialect", "-ie"),
    ("Kleine duip", "duip", "klein+N", None), ("een duil", "duil", "bare", None),
    ("kerntje", "kem", "misread", None), ("Vlek", "duip", "other", None),
    ("Renkje of Rengetje", "reng", "dim", "-kje"), ("kleine rengje", "reng", "dim", "-je"),
    ("klein kem", "kem", "klein+N", None),
])
def test_coder(answer, stem, strategy, produced):
    c = code_answer(answer, stem)
    assert (c.strategy, c.produced) == (strategy, produced)


@pytest.mark.skipif(not (RAW.exists() and EXCL.exists()), reason="private human data not present")
def test_reproduces_slides():
    """Slides 10-13 of the original presentation."""
    t = human.code_trials(human.load_raw(RAW), human.load_exclusions(EXCL))
    kept = t[~t.excluded]
    assert kept.groupby("group").participant.nunique().to_dict() == {"L2": 8, "native": 26, "other_variety": 1}
    acc = human.accuracy_table(t)
    nat = acc[acc.group == "native"].set_index("target")
    assert nat.loc[["-je", "-tje", "-pje", "-kje", "-etje"], ["n_correct", "n_dim"]].values.tolist() == \
        [[19, 19], [16, 16], [15, 20], [14, 20], [7, 19]]
    l2 = acc[acc.group == "L2"].set_index("target")
    assert l2.loc[["-je", "-tje", "-pje", "-kje", "-etje"], ["n_correct", "n_dim"]].values.tolist() == \
        [[4, 4], [3, 3], [3, 4], [3, 4], [0, 4]]
    assert human.consonant_following(t, "native") == dict(group="native", n_errors=12, n_follow=9)
