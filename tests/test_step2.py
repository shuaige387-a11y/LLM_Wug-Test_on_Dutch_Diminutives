import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wug import prompts, stimuli  # noqa: E402

CFG = prompts.load_templates()
ITEMS = prompts.item_table(stimuli.load_stimuli(), prompts.load_real_words())


def test_counts():
    gen = prompts.build_generation_prompts(ITEMS, CFG)
    assert len(gen) == (66 + 20) * 5
    assert gen.prompt_id.is_unique
    sco = prompts.build_scoring_items(ITEMS, CFG)
    assert len(sco) == (66 + 20) * 5 * 2
    assert sco.groupby(["stem", "context"]).is_correct.sum().eq(1).all()


def test_no_unfilled_placeholders():
    gen = prompts.build_generation_prompts(ITEMS, CFG)
    for col in ("chat_prompt", "base_prompt"):
        assert not gen[col].str.contains(r"[{}]").any()


def test_survey_object_numbers():
    assert "Object 3 is een reng" in prompts.render(CFG, "T1_original", "reng", "chat")
    assert "object 4 noemen" in prompts.render(CFG, "T1_original", "reng", "chat")


def test_fewshot_has_no_overlap_and_is_reproducible():
    shots = {w for w, _ in CFG["fewshot_examples"]}
    templates = set(stimuli.load_stimuli().wuggy_template)
    assert not shots & templates
    assert not shots & set(prompts.load_real_words().word)
    a = prompts.render(CFG, "T5_fewshot", "tuif", "chat", seed=1)
    assert a == prompts.render(CFG, "T5_fewshot", "tuif", "chat", seed=1)
    for w, d in CFG["fewshot_examples"]:
        assert f"{w} → {d}" in a


def test_answer_instruction_only_in_chat():
    assert prompts.render(CFG, "T1_original", "tuif", "chat").endswith("Geef alleen uw antwoord.")
    assert "Geef alleen" not in prompts.render(CFG, "T1_original", "tuif", "base")
