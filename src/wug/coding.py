"""Rule-based coding of a single elicited answer, shared by human and model data.

The scheme follows the original human study:

strategy   meaning
---------  -------------------------------------------------------------
dim        a -je family diminutive of the stem (suffix in `produced`)
dialect    a dialectal -ie / -ke diminutive of the stem (baukie, loempie)
klein+N    periphrasis: klein(e) / kleinere / small / mini / baby + noun
bare       the stem repeated without a suffix
misread    a diminutive or bare form of a different but similar stem
other      anything else (unrelated word, description, joke, empty)

Accuracy is computed only on strategy == "dim". Spelling of the stem-suffix
juncture is ignored, as in the original (mozingkje counts as -kje).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

PERIPHRASIS = re.compile(r"^(heel\s+|hele\s+|zeer\s+)?(klein|kleine|kleinere|small|smaller|mini|baby)\b", re.I)
ARTICLES = re.compile(r"^((een|ook een|het|de|a|an)\s+)+", re.I)
DIALECT_ENDINGS = ("ie", "pie", "tie", "kie", "mie", "ke")


@dataclass
class Code:
    strategy: str
    produced: str | None = None  # suffix label when strategy == "dim" or "dialect"
    word: str = ""                # the token that was coded
    ambiguous: bool = False       # answer offered alternatives ("X of Y")


def first_answer(text: str) -> tuple[str, bool]:
    """Normalise an answer and keep the first alternative."""
    t = str(text).strip().split("\n")[0].strip().lower()
    t = re.sub(r"[\"'.!?,;:()]", " ", t).strip()
    parts = re.split(r"\s+(?:of|or|/)\s+", t)
    return parts[0].strip(), len(parts) > 1


def _stem_variants(stem: str) -> list[str]:
    v = [stem]
    if stem.endswith("ng"):
        v.append(stem[:-1] + "k")  # toddink-je
    return v


def code_answer(text: str, stem: str) -> Code:
    stem = stem.lower()
    if text is None or str(text).strip() == "" or str(text).lower() == "nan":
        return Code("other")
    t, amb = first_answer(text)
    m = PERIPHRASIS.match(t)
    if m:
        # "kleine rengje": a diminutive was still produced, so code the noun;
        # otherwise it is a periphrasis with a bare or unrelated noun.
        rest = t[m.end():].strip()
        inner = code_answer(rest, stem) if rest else Code("other")
        if inner.strategy in ("dim", "dialect"):
            inner.ambiguous = inner.ambiguous or amb
            return inner
        return Code("klein+N", word=t, ambiguous=amb)
    t = ARTICLES.sub("", t).strip()
    if not t:
        return Code("other", ambiguous=amb)
    w = t.split()[0]
    st = next((s for s in _stem_variants(stem) if w.startswith(s)), None)
    if st is None:
        similar = w[:2] == stem[:2] or w[1:3] == stem[1:3]
        return Code("misread" if similar else "other", word=w, ambiguous=amb)
    rem = w[len(st):]
    if st != stem:  # nk-variant of an -ng stem
        return Code("dim", "-kje", w, amb) if rem == "je" else Code("other", None, w, amb)
    if rem == "":
        return Code("bare", word=w, ambiguous=amb)
    if rem in DIALECT_ENDINGS:
        return Code("dialect", "-ie", w, amb)
    if rem == "etje" or (len(rem) == 5 and rem[0] == stem[-1] and rem[1:] == "etje"):
        return Code("dim", "-etje", w, amb)
    table = {"je": "-je", "tje": "-tje", "pje": "-pje", "kje": "-kje", "eltje": "-eltje"}
    if rem in table:
        return Code("dim", table[rem], w, amb)
    return Code("other", word=w, ambiguous=amb)


# Which allomorph the final consonant would select after a long vowel; used for
# the RQ2 "consonant-following" index on -etje errors.
CONSONANT_SUFFIX = {"n": "-tje", "l": "-tje", "r": "-tje", "m": "-pje", "ng": "-kje"}


def consonant_suffix(stem: str) -> str | None:
    stem = stem.lower()
    if stem.endswith("ng"):
        return CONSONANT_SUFFIX["ng"]
    return CONSONANT_SUFFIX.get(stem[-1])


# ------------------------------------------------------------ model outputs

WORD = re.compile(r"[a-zà-ÿ]+")
KLEIN_WORDS = {"klein", "kleine", "kleinere", "small", "smaller", "mini", "baby"}
DIM_ENDINGS = ("je", "ie", "ke")


def _answer_text(text: str, stem: str) -> str:
    """Keep the answer part of an output: the first non-empty line, minus echoed prompt text."""
    lines = [ln.strip() for ln in str(text).splitlines()]
    lines = [ln for ln in lines if ln and ln.lower().rstrip(":") not in ("antwoord", "answer")]
    t = lines[0].lower() if lines else ""
    t = re.sub(r"^antwoord\s*:\s*", "", t)
    st = re.escape(stem)
    # echoes of the prompt, not answers: "Dit is een tuif." (T2) and "van 'tuif'" (T4/T5)
    t = re.sub(rf"\bdit is een {st}\s*[.!]", " ", t)
    t = re.sub(rf"\bvan\s+['\"‘’“”]?{st}['\"‘’“”]?", " ", t)
    return t


def code_model_output(text: str, stem: str) -> Code:
    """Code a model's free-text output, which is often a whole sentence.

    Models answer with sentences such as "Het verkleinwoord van 'tuif' is 'tuifje'."
    or "Object 4 is een tuif.", echo parts of the prompt, and base models keep writing
    after the answer. The rule:

    0. Keep only the first non-empty line and remove echoed prompt text
       ("Dit is een tuif." and "van 'tuif'").
    1. Among the words built on the stem, take the first one that is not the bare
       stem and code it with the human rules. Several different forms: the first is
       coded and the answer is flagged ambiguous, as for the human data.
    2. A compound of mini/klein/baby with the stem ("minijin") is 'klein+N'.
    3. A diminutive-looking word sharing the stem's first two letters is 'misread'
       (e.g. 'bakje' for bauk, 'muisje' for muim).
    4. Otherwise, if the bare stem occurs, the answer is 'klein+N' when a word like
       klein(e) stands right before it, and 'bare' if not. Anything else is 'other'.
    """
    stem = stem.lower()
    if text is None or not str(text).strip():
        return Code("other")
    words = WORD.findall(_answer_text(text, stem))
    variants = _stem_variants(stem)
    built = [i for i, w in enumerate(words) if any(w.startswith(v) for v in variants)]
    derived = [i for i in built if words[i] != stem]
    if derived:
        first = words[derived[0]]
        c = code_answer(first, stem)
        c.ambiguous = len({words[i] for i in derived}) > 1
        return c
    for w in words:
        if w.endswith(stem) and w[: -len(stem)] in ("mini", "klein", "kleine", "baby"):
            return Code("klein+N", word=w)
    for w in words:
        if w[:2] == stem[:2] and w.endswith(DIM_ENDINGS) and len(w) > len(stem):
            return Code("misread", word=w)
    if built:
        i = built[0]
        if i > 0 and words[i - 1] in KLEIN_WORDS:
            return Code("klein+N", word=f"{words[i - 1]} {stem}")
        return Code("bare", word=stem)
    return Code("other")
