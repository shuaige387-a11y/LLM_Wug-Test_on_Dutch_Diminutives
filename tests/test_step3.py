"""Step 3 checks with a tiny randomly initialised GPT-2 and a locally trained tokenizer.

No download is needed, so these run anywhere torch and transformers are installed.
They check the bookkeeping (which tokens are scored, how outputs are recorded), not
any linguistic result.
"""
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wug import hf_runner, prompts, stimuli  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    corpus = ["Dit is een tuif. Dit is een heel klein tuifje.", "Het woord is boompje.",
              "Wat is het verkleinwoord van huis? huisje", "kemmetje jinnetje toddinkje rengetje"] * 20
    t = Tokenizer(models.BPE(unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=300, special_tokens=["<unk>", "<eos>"],
                                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=t, eos_token="<eos>", unk_token="<unk>")
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tok), n_layer=2, n_head=2, n_embd=32, n_positions=128))
    model.eval()
    return model, tok


def _full_logprob(model, tok, text):
    ids = tok(text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        lp = torch.log_softmax(model(ids).logits[0], -1)
    return sum(lp[i - 1, ids[0, i]].item() for i in range(1, ids.shape[1])), ids.shape[1]


def test_continuation_equals_difference_of_full_logprobs(tiny):
    model, tok = tiny
    prefix, cont = "Dit is een kem. Dit is een heel klein", " kemmetje."
    r = hf_runner.continuation_logprob(model, tok, prefix, cont)
    full, _ = _full_logprob(model, tok, prefix + cont)
    pre, _ = _full_logprob(model, tok, prefix)
    assert not r["straddles_boundary"]
    assert math.isclose(r["logprob_total"], full - pre, rel_tol=1e-5, abs_tol=1e-5)
    assert "".join(r["tokens"]).replace("Ġ", " ") == cont
    assert math.isclose(r["logprob_mean"], r["logprob_total"] / r["n_tokens"])


def test_scoring_pipeline_and_normalisation(tiny):
    model, tok = tiny
    cfg = prompts.load_templates()
    st = stimuli.load_stimuli(core_only=True).head(2)
    items = prompts.build_scoring_items(prompts.item_table(st), cfg)
    s = pd.DataFrame(hf_runner.score(model, tok, items))
    assert len(s) == 2 * 5 * 2
    s["p"] = hf_runner.normalise(s)
    assert s.groupby(["stem", "context"]).p.sum().round(6).eq(1).all()


def test_generation_records(tiny):
    model, tok = tiny
    cfg = prompts.load_templates()
    st = stimuli.load_stimuli(core_only=True).head(1)
    gp = prompts.build_generation_prompts(prompts.item_table(st), cfg).head(2)
    settings = dict(max_new_tokens=4, greedy=True, n_samples=3, temperature=0.7, seed=1)
    a = pd.DataFrame(hf_runner.generate(model, tok, gp, "base", settings))
    assert len(a) == 2 * (1 + 3)
    assert set(a.decoding) == {"greedy", "sample"}
    b = pd.DataFrame(hf_runner.generate(model, tok, gp, "base", settings))
    assert a.output.tolist() == b.output.tolist()  # same seed -> same samples (CPU)
