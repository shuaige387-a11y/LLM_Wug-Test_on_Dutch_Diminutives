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
scripts/01_load_data.py        step 1
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
