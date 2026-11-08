"""Step 3b checks with a fake Anthropic client (no network, no key)."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wug import api_runner  # noqa: E402
from wug.coding import code_model_output  # noqa: E402


class FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **req):
        self.calls.append(req)
        content = [NS(type="thinking", thinking=""), NS(type="text", text="tuifje")]
        return NS(content=content, stop_reason="end_turn", model=req["model"], id=f"msg_{len(self.calls)}",
                  usage=NS(input_tokens=50, output_tokens=3))


CFG = dict(provider="anthropic", model_id="claude-sonnet-5-5", thinking={"type": "between_tools"})
SET = dict(n_calls=3, max_tokens=64)


def _prompts():
    d = pd.read_json(ROOT / "prompts" / "generation_prompts.jsonl", lines=True)
    return d[d.stem.eq("tuif")].head(2)


def test_records_and_request():
    client = NS(messages=FakeMessages())
    recs = pd.DataFrame(api_runner.generate(client, _prompts(), CFG, SET))
    assert len(recs) == 2 * 3
    assert recs.sample_index.tolist() == [1, 2, 3, 1, 2, 3]
    assert set(recs.decoding) == {"default"}
    assert (recs.output == "tuifje").all()            # thinking blocks are not part of the answer
    req = client.messages.calls[0]
    assert not {"temperature", "top_p", "top_k"} & set(req)
    assert req["thinking"] == {"type": "between_tools"}
    assert code_model_output(recs.output[0], "tuif").strategy == "dim"


def test_resume_skips_done_calls():
    client = NS(messages=FakeMessages())
    p = _prompts()
    done = {(p.prompt_id.iloc[0], 1), (p.prompt_id.iloc[0], 2)}
    recs = list(api_runner.generate(client, p, CFG, SET, done))
    assert len(recs) == 6 - 2
    assert len(client.messages.calls) == 4
