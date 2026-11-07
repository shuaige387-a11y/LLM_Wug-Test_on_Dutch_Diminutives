# Dutch diminutive wug-test: humans vs. LLMs

Extends a human wug-test on the five Dutch diminutive allomorphs (-je, -tje, -pje, -kje, -etje)
to large language models. Work in progress; modules are added one step at a time.

## Layout

```
data/stimuli/nonce_words.csv   15 core + 51 reserve nonce words (Wuggy, orthographic_dutch)
data/stimuli/candidates.csv    all five candidate forms per stem (generated, step 1)
data/human/                    aggregate human results only (no trial-level data)
data/private/                  raw survey export + exclusion list (git-ignored, never published)
src/wug/stimuli.py             stimulus loading, Dutch diminutive spelling rules
src/wug/coding.py              rule-based answer coder shared by human and model data
src/wug/human.py               human data coding and aggregate export
src/wug/prompts.py             prompt rendering (method A) and scoring items (method B)
config/templates.yaml          the five prompt templates, few-shot examples, scoring frames
data/stimuli/real_words.csv    20 real-word baseline items (Wuggy templates, 4 per condition)
prompts/                       generated prompts + MANIFEST.json (step 2)
scripts/01_load_data.py        step 1
scripts/02_build_prompts.py    step 2
tests/                         pytest checks
```

## Step 1: stimuli and human data

```bash
pip install -r requirements.txt
python scripts/01_load_data.py                     # stimuli only
python scripts/01_load_data.py --human-raw data/private/raw_qualtrics_export.xlsx \
                               --exclusions data/private/human_exclusions.yaml
pytest -q
```

With the private files present, the coder reproduces the native-speaker results of the original
study exactly (35 participants and 175 trials after exclusion: 26 native speakers, 8 L2 learners,
and 1 Surinamese Dutch speaker who is analysed separately and not published; native accuracy among
diminutive answers -je 19/19, -tje 16/16, -pje 15/20, -kje 14/20, -etje 7/19; 9 of 12 -etje errors
follow the stem-final consonant). An answer like "kleine rengje" counts as a diminutive, and when an
answer offers alternatives ("Renkje of Rengetje") the first one is coded. `tests/test_step1.py::test_reproduces_slides` checks this.

## Step 2: prompts

```bash
python scripts/02_build_prompts.py               # writes prompts/, prints every template for 'tuif'
python scripts/02_build_prompts.py --show reng
```

Five templates (`config/templates.yaml`), each with a chat form (instruct and API models, one user
turn, no system prompt) and a base form (plain text continued by a base model):

| id | what it is | comparable with humans |
|---|---|---|
| T1_original | survey question; the picture is replaced by "object 4 is the same kind of object as object 1, but much smaller"; original object numbers | yes |
| T2_frame | "Dit is een tuif. Dit is een heel klein ..." (uninflected *klein* cues a neuter noun, i.e. a diminutive) | yes |
| T3_two_objects | a big and a very small object of the same kind described in words | yes |
| T4_metalinguistic | "Wat is het verkleinwoord van 'tuif'?" | no (asks for a diminutive) |
| T5_fewshot | five real-word examples, one per allomorph, in item-specific seeded order | no (asks for a diminutive) |

Chat prompts of T1, T3 and T4 end with "Geef alleen uw/je antwoord." so answers stay short; this asks
for brevity, not for a diminutive.

Method B scores each of the five candidate forms as the continuation of
"Dit is een {stem}. Dit is een heel klein" (the continuation includes the final period, so the word
boundary is scored), and again after the neutral prefix "Het woord is" for calibration.

Sizes: 86 items (15 core + 51 reserve nonce words, 20 real words) x 5 templates = 430 generation
prompts; 86 x 5 candidates x 2 contexts = 860 scoring items. Planned API budget per model:
15 core words x 5 templates x (1 greedy + 10 samples) = 825 calls, plus 20 real words x 5 templates
greedy only = 100 calls.
