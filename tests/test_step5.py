"""Step 5 checks on small hand-made tables."""
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wug import analysis as A  # noqa: E402


def test_wilson_known_value():
    lo, hi = A.wilson(9, 12)
    assert math.isclose(lo, 0.4677, abs_tol=1e-3) and math.isclose(hi, 0.9111, abs_tol=1e-3)
    assert all(math.isnan(x) for x in A.wilson(0, 0))


def _coded():
    rows = []
    for stem, cond, produced in [("kem", "-etje", "-pje"), ("kem", "-etje", "-etje"), ("reng", "-etje", "-je"),
                                 ("jin", "-etje", "-tje"), ("tuif", "-je", "-je"), ("tuif", "-je", None)]:
        rows.append(dict(model="m", stem=stem, condition=cond, word_type="nonce", template="T4_metalinguistic",
                         decoding="greedy", strategy="dim" if produced else "bare", produced=produced or "",
                         correct=produced == cond))
    return pd.DataFrame(rows)


def test_consonant_following_counts():
    cf = A.consonant_following(_coded()).iloc[0]
    assert (cf.n_errors, cf.n_follow) == (3, 2)        # kem->pje and jin->tje follow; reng->je does not
    assert cf.follow_reng == "0/1"


def test_accuracy_only_counts_diminutive_answers():
    a = A.accuracy_table(_coded()).set_index("condition")
    assert a.loc["-je", "n_outputs"] == 2 and a.loc["-je", "n_dim"] == 1 and a.loc["-je", "accuracy"] == 1
    assert a.loc["-etje", "accuracy"] == 0.25


def test_method_b_probabilities_sum_to_one():
    rows = []
    for ctx, base in (("frame", 0.0), ("neutral", -1.0)):
        for i, s in enumerate(["-je", "-tje", "-pje", "-kje", "-etje"]):
            rows.append(dict(model="m", stem="tuif", condition="-je", word_type="nonce", candidate_suffix=s,
                             is_correct=s == "-je", context=ctx, logprob_total=base - i))
    w = A.method_b(pd.DataFrame(rows))
    assert math.isclose(w.p_raw.sum(), 1) and math.isclose(w.p_cal.sum(), 1)
    s = A.method_b_summary(w).iloc[0]
    assert s.p_raw_top1 == 1.0
