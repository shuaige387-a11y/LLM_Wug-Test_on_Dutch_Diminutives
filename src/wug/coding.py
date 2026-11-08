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
KLEIN_WORDS = {"klein", "kleine", "kleinere", "kleiner", "small", "smaller", "mini", "baby",
               "miniem", "mini", "miniemere", "piepklein", "piepkleine", "minuscule", "minuscuul"}
KLEIN_PREFIXES = ("mini", "klein", "kleine", "baby", "piepklein")
DIM_ENDINGS = ("tje", "pje", "kje", "je", "ie", "ke")
ARTICLE = r"(?:een|de|het)"


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _strip_dim(w: str) -> str:
    for e in ("etje", "tje", "pje", "kje", "je", "ie", "ke"):
        if w.endswith(e) and len(w) > len(e) + 1:
            return w[: -len(e)]
    return w


def _answer_text(text: str, stem: str) -> str:
    """Keep the answer part of an output: the first non-empty line, minus echoed prompt text."""
    lines = [ln.strip() for ln in str(text).splitlines()]
    lines = [ln for ln in lines if ln and ln.lower().rstrip(":") not in ("antwoord", "answer")]
    t = lines[0].lower() if lines else ""
    t = t.replace("*", "")
    t = re.sub(r"^antwoord\s*:\s*", "", t)
    st = re.escape(stem)
    # echoes of the prompt, not answers
    t = re.sub(rf"\bdit is een {st}\s*[.!]", " ", t)                       # T2
    t = re.sub(rf"\bvan\s+['\"‘’“”]?{st}['\"‘’“”]?", " ", t)              # T4/T5
    if re.search(r"op (een|de) tekening staan zes genummerde voorwerpen", t):  # T1 restated
        t = re.sub(r"op (een|de) tekening staan zes genummerde voorwerpen\.?", " ", t)
        t = re.sub(rf"object \d+ is een {st}\s*[.!]?", " ", t, count=1)
    t = re.sub(r"het grote voorwerp is een \w+\s*[.!]?", " ", t)           # T3 restated
    return t


def _is_bare_answer(t: str, stem: str) -> bool:
    """The bare stem is given as the answer, not just mentioned somewhere in a sentence."""
    st = re.escape(stem)
    q = r"['\"‘’“”]?"
    if re.match(rf"^\W*(?:{ARTICLE}\s+)?{q}{st}{q}\W*(?:[.!?,;:]|$)", t):  # the answer is the stem
        return True
    return bool(re.search(rf"\b(?:is|heet|heten|noem\w*|wordt)\s*:?\s+(?:{ARTICLE}\s+)?{q}{st}{q}(?:\W|$)", t))


def code_model_output(text: str, stem: str) -> Code:
    """Code a model's free-text output, which is often a whole sentence.

    0. Keep the first non-empty line; drop echoed prompt text ("Dit is een tuif.",
       "van 'tuif'", a restated T1/T3 question).
    1. The first word built on the stem that is not the bare stem is coded with the
       human rules (several different forms: first one, flagged ambiguous).
    2. The bare stem given as the answer ("tuif", "een tuif", "Object 4 is een tuif",
       "Het antwoord is: 'tuif'") is 'bare'; preceded by klein(e)/mini/miniemere it is
       'klein+N'; a mini-/klein- compound ("minijin") is also 'klein+N'.
    3. A form of a different but similar stem (edit distance <= 2 after removing a
       diminutive ending, e.g. 'bakje' for bauk, 'muisje' or 'muis' for muim) is 'misread'.
    4. Anything else, including a stem that is only mentioned inside a longer sentence
       ("stukje van de tuin") or klein + an unrelated noun ("klein kind"), is 'other'.
    """
    stem = stem.lower()
    if text is None or not str(text).strip():
        return Code("other")
    t = _answer_text(text, stem)
    words = WORD.findall(t)
    variants = _stem_variants(stem)
    built = [i for i, w in enumerate(words) if any(w.startswith(v) for v in variants)]
    derived = [i for i in built if words[i] != stem]
    if derived:
        first = words[derived[0]]
        c = code_answer(first, stem)
        if c.strategy in ("dim", "dialect"):
            c.ambiguous = len({words[i] for i in derived}) > 1
            return c
    for w in words:
        if w.endswith(stem) and w[: -len(stem)] in KLEIN_PREFIXES:
            return Code("klein+N", word=w)
    if built:
        if re.match(rf"^\W*{re.escape(stem)}\W*(?:[.!?,;:]|$)", t):  # base model filled the blank
            return Code("bare", word=stem)
        i = built[0]
        if i > 0 and words[i - 1] in KLEIN_WORDS:
            return Code("klein+N", word=f"{words[i - 1]} {stem}")
        if _is_bare_answer(t, stem):
            return Code("bare", word=stem)
    content = [w for w in words if w not in ("een", "de", "het") and w not in KLEIN_WORDS]
    for w in content:
        if w == stem or len(w) < 3:
            continue
        core = _strip_dim(w)
        looks_dim = core != w
        if w[0] == stem[0] or core[:2] == stem[:2]:
            if (looks_dim and _edit_distance(core, stem) <= 2) or \
               (not looks_dim and len(content) <= 2 and _edit_distance(w, stem) == 1):
                return Code("misread", word=w)
    return Code("other")
