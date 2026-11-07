"""Answer extraction from model outputs; cases taken from the Colab pilot (EuroLLM-1.7B-Instruct)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wug.coding import code_model_output  # noqa: E402


@pytest.mark.parametrize("output,stem,strategy,produced", [
    # pilot outputs
    ("Object 4 is een tuif.", "tuif", "bare", None),
    ("Dit is een tuif.", "tuif", "bare", None),
    ("Het kleine voorwerp is een tuif.", "tuif", "bare", None),
    ("Het verkleinwoord van 'tuif' is 'tuifje'.", "tuif", "dim", "-je"),
    ("Object 6 is een verkleinde versie van object 2.", "bauk", "other", None),
    ("Dit is een huis.", "bauk", "other", None),
    ("Het kleine voorwerp is een bakje.", "bauk", "misread", None),
    ("Bauwke", "bauk", "misread", None),  # bauk read as bauw + dialectal -ke
    ('Het antwoord is: "duip".', "duip", "bare", None),
    ("duipduip", "duip", "other", None),
    # other shapes
    ("tuifje", "tuif", "dim", "-je"),
    (" kemmetje. Dit is een", "kem", "dim", "-etje"),
    ("Een kleine kem.", "kem", "klein+N", None),
    ("Een klein kemmetje", "kem", "dim", "-etje"),
    ("Het verkleinwoord van 'todding' is 'toddinkje'.", "todding", "dim", "-kje"),
    ("Ik zou het een jintje noemen.", "jin", "dim", "-tje"),
])
def test_extraction(output, stem, strategy, produced):
    c = code_model_output(output, stem)
    assert (c.strategy, c.produced) == (strategy, produced)


def test_alternatives_take_first_and_flag():
    c = code_model_output("rengetje of renkje", "reng")
    assert (c.produced, c.ambiguous) == ("-etje", True)
