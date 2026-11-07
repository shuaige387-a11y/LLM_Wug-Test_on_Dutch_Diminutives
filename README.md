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
config/models.yaml             open models, decoding settings, seed
src/wug/hf_runner.py           4-bit loading, generation (method A), log-prob scoring (method B)
scripts/03_run_hf.py           step 3, one model and method per call; resumable
notebooks/03_run_models_colab.ipynb   step 3 on a free Colab T4
outputs/raw/<model>/           raw model outputs + run metadata
scripts/04_code_outputs.py     step 4: code model outputs, manual-check sample, agreement
outputs/coded/                 coded model outputs (step 4)
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

## Step 3: run the open models

Open `notebooks/03_run_models_colab.ipynb` in Colab (T4 GPU) and run it top to bottom, or call the
script directly:

```bash
pip install -r requirements-gpu.txt
python scripts/03_run_hf.py --model eurollm-1.7b --method B --limit 3   # dry run on 3 words
python scripts/03_run_hf.py --model eurollm-1.7b --method B             # 15 core + 20 real words
python scripts/03_run_hf.py --model eurollm-1.7b --method B --words all # + 51 reserve words
```

- **Models** (`config/models.yaml`): EuroLLM 1.7B and 9B, Gemma 4 E2B, E4B and 12B, fietje-2.
  Method B (scoring) runs on the base models. Method A (generation) runs on the instruct models,
  plus EuroLLM-1.7B base and fietje-2 as base-model checks. All are loaded in 4-bit NF4, so size
  is not confounded with quantization.
- **Decoding** for method A: 1 greedy output plus 10 samples at temperature 0.7, 24 new tokens at
  most. The sampling seed is derived from the global seed and the prompt id, so a rerun on the
  same hardware and library versions gives the same samples. GPU kernels are not guaranteed to be
  bit-identical across hardware.
- **Scoring** for method B: the full string (prefix + candidate + ".") is tokenized once, and the
  candidate's tokens are found by character offsets. The output keeps the total log-probability,
  the per-token mean, the tokens themselves and a flag for a token that crosses the boundary.
  `tests/test_step3.py` checks the result against log P(prefix + continuation) − log P(prefix) on
  a tiny local model.
- **Metadata**: every run writes `<method>_meta.json` with the model's commit hash, the UTC time,
  library versions, the GPU and the decoding settings.

## Step 4: code the model outputs

```bash
python scripts/04_code_outputs.py --sample 200      # code everything + draw a manual-check sample
# fill in manual_strategy / manual_produced in outputs/coded/manual_check.csv, then:
python scripts/04_code_outputs.py --agreement outputs/coded/manual_check.csv
```

Models often answer with a sentence ("Het verkleinwoord van 'tuif' is 'tuifje'.", "Object 4 is
een tuif.") and echo the stem from the question. `code_model_output` in `src/wug/coding.py` takes
the first word built on the stem that is not the bare stem, and codes it with the same rules as the
human answers. If there is none, the answer is bare (or klein + noun when *klein(e)* stands right
before the stem). A diminutive-looking word with the stem's first two letters is a misread (e.g.
*bakje* for *bauk*), and anything else is "other". The manual-check sample is stratified by model
and strategy, so rare codes get checked too. The test cases in `tests/test_step4.py` come from the
Colab pilot run.
