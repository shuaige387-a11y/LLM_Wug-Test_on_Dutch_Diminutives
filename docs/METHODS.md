# Methods and pipeline

Step-by-step description of the pipeline. Each step is one script and can be run on its own.

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
prompts; 86 x 5 candidates x 2 contexts = 860 scoring items. API model: 35 core items (15 nonce + 20 real words)
x 5 templates x 11 calls = 1,925 calls (see step 3b).

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

## Step 3b: run the API model

One API model, Claude Sonnet 5.5 (`config/api_models.yaml`), runs method A only, since closed
models do not expose the log-probabilities that method B needs. Current Claude models do not accept
`temperature` or `top_p`, so a greedy run and T = 0.7 sampling are not possible. Each prompt is
sent 11 times at the model's default sampling instead, which gives the same number of outputs per
prompt as the open models, and is recorded as `decoding = "default"`. Extended thinking is
switched off (the model's lowest thinking setting), so that the model answers directly like the
open models and the participants. The model id, SDK version and run date are written to
`outputs/raw/claude-sonnet-5-5/A_meta.json`.

```bash
export ANTHROPIC_API_KEY=...        # or Colab Secrets, see notebooks/03b_run_api_colab.ipynb
python scripts/03b_run_api.py --model claude-sonnet-5-5 --limit 1 --calls 1   # dry run, 5 calls
python scripts/03b_run_api.py --model claude-sonnet-5-5                       # 1,925 calls
```

## Step 4: code the model outputs

```bash
python scripts/04_code_outputs.py --sample 200      # code everything + draw a manual-check sample
# fill in manual_strategy / manual_produced in outputs/coded/manual_check.csv, then:
python scripts/04_code_outputs.py --agreement outputs/coded/manual_check_adjudicated.csv
```

Models often answer with a sentence ("Het verkleinwoord van 'tuif' is 'tuifje'.", "Object 4 is
een tuif.") and echo the stem from the question. `code_model_output` in `src/wug/coding.py` takes
the first word built on the stem that is not the bare stem, and codes it with the same rules as the
human answers. If there is none, the answer is bare (or klein + noun when *klein(e)* stands right
before the stem). *Klein* + an unrelated noun (*een klein kind*) is "other". A word counts as a
misread only if, without its diminutive ending, it is at most two edits from the stem (*kemp* for
*kem*, *hariksje* for *haring*); anything further away (*haantje*, *ruitje*) is "other". The
manual-check sample is stratified by model and strategy, so rare codes get checked too.

Manual check (210 rows, coded blind by the first author). Before the coder was revised, strategy
agreement was 86.7% (Cohen's kappa .84). After the two rules above were added it is 91.0% (kappa
.89). The 23 remaining disagreements were then settled row by row
(`outputs/coded/manual_check_adjudicated.csv`, which holds the manual codes and an `adj_note` column). Most were manual slips, such as
*riemje* coded as -pje. Against the settled codes the coder agrees on 209 of 210 rows (99.5%).
That figure is optimistic, because the rules were revised on this same sample.
The test cases in `tests/test_step4.py` come from the Colab pilot run.

## Step 5: analysis

```bash
python scripts/05_analyse.py      # writes outputs/analysis/*.csv and prints the headline numbers
```

`src/wug/analysis.py` builds each table from the coded outputs, the method B scores and the human
aggregates. Accuracy is the share of correct suffixes among diminutive answers. Each table is
given for all templates, for T1-T3 (the human task framings) and for T4-T5, and for all outputs
and for greedy outputs only. Wilson intervals treat outputs as independent and are therefore too
narrow. The item bootstrap (resampling stems within each suffix) is the main interval, and it is
wide with three nonce items per suffix. The consonant-following index counts nonce *-etje* errors
whose suffix matches the stem's final consonant (n, l, r to -tje; m to -pje; ng to -kje; chance
= 1/4), with per-item counts. Method B gives P(suffix | stem) by softmax over the five candidates,
raw and calibrated against the neutral context.

## Step 6: figures

```bash
python scripts/06_figures.py      # figures/fig1-3 as PNG and PDF, from outputs/analysis/
```

1. Accuracy by suffix for native speakers, each model and the model mean, for methods A and B.
2. Which suffix is produced for *-kje* and *-etje* nonce targets, native speakers vs models pooled.
3. Share of nonce *-etje* errors that follow the final consonant, per model, with Wilson intervals.

