"""Run Hugging Face models: method A (generation) and method B (log-probability scoring).

Both functions take an already loaded (model, tokenizer) so they can be tested with a
tiny local model; `load_model` does the 4-bit loading used for the real runs.
"""
from __future__ import annotations

import hashlib
import math
import platform
from datetime import datetime, timezone

import pandas as pd


def load_model(hf_id: str, quantization: str = "4bit"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id)
    kwargs = dict(device_map="auto")
    if quantization == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    else:
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    model.eval()
    return model, tok


def run_metadata(model, tok, model_key: str, hf_id: str, settings: dict) -> dict:
    import torch
    import transformers
    return dict(
        model_key=model_key, hf_id=hf_id,
        revision=getattr(model.config, "_commit_hash", None),
        run_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        torch=torch.__version__, transformers=transformers.__version__,
        python=platform.python_version(),
        device=str(next(model.parameters()).device),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        **settings,
    )


def _seed_for(text: str, base_seed: int) -> int:
    return int(hashlib.sha256(f"{base_seed}:{text}".encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ method A

def build_input(tok, prompt_row, kind: str, chat_template_kwargs: dict | None = None) -> str:
    """Instruct models get the chat prompt through their chat template; base models the base prompt."""
    if kind == "instruct":
        msgs = [{"role": "user", "content": prompt_row["chat_prompt"]}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       **(chat_template_kwargs or {}))
    return prompt_row["base_prompt"]


def generate(model, tok, prompts: pd.DataFrame, kind: str, settings: dict,
             chat_template_kwargs: dict | None = None):
    """Yield one record per (prompt, decoding run): 1 greedy + n_samples sampled outputs."""
    import torch

    for _, p in prompts.iterrows():
        text = build_input(tok, p, kind, chat_template_kwargs)
        add_special = kind != "instruct"  # chat templates already contain BOS
        enc = tok(text, return_tensors="pt", add_special_tokens=add_special).to(model.device)
        n_in = enc["input_ids"].shape[1]
        common = dict(max_new_tokens=settings["max_new_tokens"], pad_token_id=tok.pad_token_id or tok.eos_token_id)
        runs = []
        if settings.get("greedy", True):
            with torch.no_grad():
                out = model.generate(**enc, do_sample=False, **common)
            runs.append(("greedy", 0, out[0]))
        if settings.get("n_samples", 0):
            seed = _seed_for(p["prompt_id"], settings["seed"])
            torch.manual_seed(seed)
            with torch.no_grad():
                out = model.generate(**enc, do_sample=True, temperature=settings["temperature"],
                                     top_p=settings.get("top_p", 1.0),
                                     num_return_sequences=settings["n_samples"], **common)
            runs += [("sample", i + 1, o) for i, o in enumerate(out)]
        for decoding, idx, seq in runs:
            yield dict(
                prompt_id=p["prompt_id"], stem=p["stem"], condition=p["condition"],
                word_type=p["word_type"], is_core=bool(p["is_core"]), template=p["template"],
                decoding=decoding, sample_index=idx,
                output=tok.decode(seq[n_in:], skip_special_tokens=True),
            )


# ------------------------------------------------------------------ method B

def continuation_logprob(model, tok, prefix: str, continuation: str) -> dict:
    """Log-probability of `continuation` given `prefix`, token by token.

    The full string is tokenized once and the continuation tokens are found by
    character offsets, so the stem and suffix are tokenized as they would be in
    running text. A token that straddles the prefix/continuation boundary is
    counted with the continuation and flagged.
    """
    import torch

    full = prefix + continuation
    enc = tok(full, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    ids = enc["input_ids"][0]
    enc = enc.to(model.device)
    with torch.no_grad():
        logits = model(**enc).logits[0].float()
    logprobs = torch.log_softmax(logits, dim=-1)

    cut = len(prefix)
    cont_positions = [i for i, (s, e) in enumerate(offsets) if e > cut and e > s]
    straddles = any(s < cut < e for s, e in offsets)
    if not cont_positions or cont_positions[0] == 0:
        raise ValueError(f"cannot locate continuation tokens for {full!r}")
    lps = [logprobs[i - 1, ids[i]].item() for i in cont_positions]
    tokens = [tok.convert_ids_to_tokens(int(ids[i])) for i in cont_positions]
    return dict(
        logprob_total=sum(lps), n_tokens=len(lps),
        logprob_mean=sum(lps) / len(lps),
        tokens=tokens, token_logprobs=lps, straddles_boundary=straddles,
    )


def score(model, tok, items: pd.DataFrame):
    """Yield one record per scoring item (stem x candidate x context)."""
    for _, it in items.iterrows():
        r = continuation_logprob(model, tok, it["prefix"], it["continuation"])
        yield dict(
            score_id=it["score_id"], stem=it["stem"], condition=it["condition"],
            word_type=it["word_type"], is_core=bool(it["is_core"]),
            candidate_suffix=it["candidate_suffix"], candidate_form=it["candidate_form"],
            is_correct=bool(it["is_correct"]), context=it["context"], **r,
        )


def normalise(scores: pd.DataFrame, column: str = "logprob_total") -> pd.Series:
    """Softmax over the five candidates of each stem x context: P(suffix | stem)."""
    def _softmax(x):
        m = x.max()
        e = (x - m).map(math.exp)
        return e / e.sum()
    return scores.groupby(["stem", "context"])[column].transform(_softmax)
