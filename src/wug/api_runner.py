"""Run API models for method A (generation).

`generate` takes an already built client, so it can be tested with a fake one.
Only the Anthropic Messages API is supported for now.
"""
from __future__ import annotations

import platform
from datetime import datetime, timezone

import pandas as pd


def make_client(provider: str, max_retries: int = 5):
    if provider != "anthropic":
        raise ValueError(f"unsupported provider: {provider}")
    import anthropic
    return anthropic.Anthropic(max_retries=max_retries)  # key from ANTHROPIC_API_KEY


def run_metadata(model_key: str, model_cfg: dict, settings: dict) -> dict:
    import anthropic
    return dict(
        model_key=model_key, model_id=model_cfg["model_id"], provider=model_cfg["provider"],
        run_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        anthropic_sdk=anthropic.__version__, python=platform.python_version(),
        thinking=model_cfg.get("thinking"), decoding="default (sampling parameters not accepted)",
        **settings,
    )


def _request(model_cfg: dict, settings: dict, prompt: str) -> dict:
    req = dict(model=model_cfg["model_id"], max_tokens=settings["max_tokens"],
               messages=[{"role": "user", "content": prompt}])
    if model_cfg.get("thinking"):
        req["thinking"] = model_cfg["thinking"]
    return req


def generate(client, prompts: pd.DataFrame, model_cfg: dict, settings: dict, done: set | None = None):
    """Yield one record per (prompt, call). `done` holds (prompt_id, sample_index) pairs to skip."""
    done = done or set()
    for _, p in prompts.iterrows():
        for idx in range(1, settings["n_calls"] + 1):
            if (p["prompt_id"], idx) in done:
                continue
            r = client.messages.create(**_request(model_cfg, settings, p["chat_prompt"]))
            text = "".join(b.text for b in r.content if b.type == "text")
            yield dict(
                prompt_id=p["prompt_id"], stem=p["stem"], condition=p["condition"],
                word_type=p["word_type"], is_core=bool(p["is_core"]), template=p["template"],
                decoding="default", sample_index=idx, output=text,
                stop_reason=r.stop_reason, response_model=r.model, response_id=r.id,
                input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens,
            )
