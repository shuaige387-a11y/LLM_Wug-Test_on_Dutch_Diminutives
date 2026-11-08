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
    ("Dit is een tuif.", "tuif", "other", None),  # T2: only the prompt sentence echoed
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
    # full-run cases
    ("Het verkleinwoord van 'muim' is 'muisje'.", "muim", "misread", None),
    ("Het verkleinwoord van 'muim' is 'muim'.", "muim", "bare", None),
    ("Dit is een muin. Dit is een heel klein **beest**.", "muin", "other", None),
    ("een minijin", "jin", "klein+N", None),
    ("Kleine reng", "reng", "klein+N", None),
    ("duil\nboom → boompje\nduil → duiltje", "duil", "bare", None),  # base model rambling
    ("muin. Dit is een heel klein muin. Dit is", "muin", "bare", None),
    ("kempje\nkip → kipje", "kem", "dim", "-pje"),
    # manual-check disagreements (2026-11-08)
    (" stukje van de tuin. In het midden is een kruidentuin.", "tuin", "other", None),
    ("Een **tuin** is een groot voorwerp, maar het kleine voorwerp", "tuin", "other", None),
    ("Op de tekening staan zes genummerde voorwerpen. Object 1 is een kip. Object 4 is net", "kip", "other", None),
    ("een miniemere kem", "kem", "klein+N", None),
    ("muis", "muin", "misread", None),
    ("kikkertje\nkip → kipje", "kip", "other", None),
    ("Een klein kind.", "koning", "other", None),
    ("Dit is een todding. Dit is een heel klein **dopje**.", "todding", "other", None),
    ("Ik weet het niet. Maar ik denk dat ik het weet. Kam is klein", "kam", "other", None),
    ("5 is een duip.", "duip", "bare", None),
    ("Een **ren** is een grote ren, dus het kleine voorwerp is een **rennetje**.", "reng", "misread", None),
])
def test_extraction(output, stem, strategy, produced):
    c = code_model_output(output, stem)
    assert (c.strategy, c.produced) == (strategy, produced)


def test_alternatives_take_first_and_flag():
    c = code_model_output("rengetje of renkje", "reng")
    assert (c.produced, c.ambiguous) == ("-etje", True)
