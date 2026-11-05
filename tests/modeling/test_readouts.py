import pytest
import torch

from metiscraft.config import ReadoutConfig
from metiscraft.data.contract import Record
from metiscraft.modeling.prompts import PromptBuilder, build_codebook
from metiscraft.train.collate import Collator
from tests.modeling.tiny import (
    choice_row,
    multilabel_row,
    noul_row,
    score_row,
    tiny_decision_model,
    tiny_tokenizer,
)

ANSWER = {"kind": "answer_token"}
LIST = {"kind": "candidate_scalar", "layout": "list", "shuffle_train": False}
PAIRS = {"kind": "candidate_scalar", "layout": "pairs"}


def test_codebook_uses_single_tokens_in_order():
    tok = tiny_tokenizer()
    codes = build_codebook(tok, 30)
    assert [c for c, _ in codes[:3]] == ["A", "B", "C"]
    assert codes[26][0] == "AA"
    assert all(tok.decode([i]) == c for c, i in codes)


def test_codebook_reports_shortage():
    tok = tiny_tokenizer()
    with pytest.raises(ValueError, match="single-token option codes"):
        build_codebook(tok, 10_000)


def test_list_positions_follow_targets_under_permutation():
    tok = tiny_tokenizer()
    builder = PromptBuilder(tok, 256)
    record = Record.model_validate(choice_row(0))
    plain = builder.candidate_list(record)
    shuffled = builder.candidate_list(record, order=[2, 0, 1])
    names = ["billing", "tech", "refund"]
    for j, name in enumerate(names):
        assert tok.convert_ids_to_tokens(plain.input_ids[plain.positions[j]]) == name
        assert tok.convert_ids_to_tokens(shuffled.input_ids[shuffled.positions[j]]) == name
    assert plain.positions != shuffled.positions


def test_truncation_keeps_question_and_options():
    tok = tiny_tokenizer()
    record = Record.model_validate(choice_row(0, state=" ".join(["red"] * 500)))
    enc = PromptBuilder(tok, 80).answer_token(record)
    assert len(enc.input_ids) == 80
    tail = tok.convert_ids_to_tokens(enc.input_ids[-6:])
    assert tail[-2:] == ["Answer", ":"]


def test_noul_answer_token_equals_linear_head():
    """① on noul is the linear head (W_yes - W_no) . h_last (design §12, equivalence 1)."""
    model, tok = tiny_decision_model("qwen3", ANSWER)
    batch = Collator(tok, ReadoutConfig(**ANSWER), 128)([noul_row(0), noul_row(1, False)])
    with torch.no_grad():
        scores = model(batch)[:, 0]
        h = model.hidden(batch["input_ids"], batch["attention_mask"], batch["position_ids"])[:, -1]
        w = model.backbone.get_output_embeddings().weight
        yes, no = tok.convert_tokens_to_ids(["yes", "no"])
        manual = h @ (w[yes] - w[no])
    torch.testing.assert_close(scores, manual, rtol=1e-5, atol=1e-5)


def test_list_and_pairs_share_prefix_states():
    model, tok = tiny_decision_model("qwen3", LIST)
    record = Record.model_validate(choice_row(0))
    builder = PromptBuilder(tok, 256)
    lst = builder.candidate_list(record)
    pairs = builder.candidate_pairs(record)
    shared = 0
    while (
        shared < min(len(lst.input_ids), len(pairs.pairs[0]))
        and lst.input_ids[shared] == pairs.pairs[0][shared]
    ):
        shared += 1
    assert 5 < shared < len(lst.input_ids)
    with torch.no_grad():
        a = model.hidden(torch.tensor([lst.input_ids]), None, None)[0, :shared]
        b = model.hidden(torch.tensor([pairs.pairs[0]]), None, None)[0, :shared]
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_pairs_scores_ignore_unrelated_candidates():
    model, tok = tiny_decision_model("qwen3", PAIRS)
    full = choice_row(0)
    fewer = choice_row(0)
    fewer["question"]["options"] = fewer["question"]["options"][:2]
    collator = Collator(tok, ReadoutConfig(**PAIRS), 256)
    with torch.no_grad():
        a = model(collator([full]))[0, :2]
        b = model(collator([fewer]))[0, :2]
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("arch", ["qwen3", "llama"])
@pytest.mark.parametrize("readout", [ANSWER, LIST, PAIRS])
def test_composability_matrix(arch, readout):
    """Same rows, any architecture x readout, only the config changes (design §12)."""
    model, tok = tiny_decision_model(arch, readout)
    rows = [choice_row(0), noul_row(1), score_row(2)]
    if readout["kind"] == "candidate_scalar":
        rows.append(multilabel_row(3))
    batch = Collator(tok, ReadoutConfig(**readout), 256, train=True)(rows)
    scores = model(batch)
    assert scores.shape == batch["mask"].shape
    assert torch.isfinite(scores[batch["mask"]]).all()
    from metiscraft.judgments import batch_loss

    loss, n = batch_loss(scores, batch["mask"], batch["targets"], batch["kinds"])
    loss.backward()
    assert n == len(rows) and torch.isfinite(loss)


def test_answer_token_rejects_multi_label():
    tok = tiny_tokenizer()
    with pytest.raises(ValueError, match="multi_label"):
        Collator(tok, ReadoutConfig(**ANSWER), 256)([multilabel_row(0)])


CHAT = {**ANSWER, "prompt": {"style": "chat_json", "system": "Pick one."}}


class Spy:
    """The tiny tokenizer, recording what reaches its chat template."""

    def __init__(self):
        self.tok, self.calls = tiny_tokenizer(), []

    def __getattr__(self, name):
        return getattr(self.tok, name)

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.tok.apply_chat_template(messages, **kwargs)


def test_chat_json_prompt_is_the_payload_in_the_chat_template():
    spy = Spy()
    row = choice_row(0)
    row["input"]["context"]["state"] = {"amount": 3, "note": "late"}  # key order is kept
    row["question"]["options"][0]["text"] = "Billing team"
    enc = PromptBuilder(spy, 256, ReadoutConfig(**CHAT).prompt).answer_token(Record(**row))
    ((messages, kwargs),) = spy.calls
    user = (
        '{"state": {"amount": 3, "note": "late"}, "question": "Which team should handle this ?", '
        '"options": [{"label": "A", "key": "billing", "description": "Billing team"}, '
        '{"label": "B", "key": "tech", "description": null}, '
        '{"label": "C", "key": "refund", "description": null}]}'
    )
    assert messages == [
        {"role": "system", "content": "Pick one."},
        {"role": "user", "content": user},
    ]
    assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    text = spy.tok.apply_chat_template(messages, **kwargs)
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert enc.input_ids == spy.tok.encode(text, add_special_tokens=False)
    assert enc.codes == [t for _, t in build_codebook(spy.tok, 3)]


def test_chat_json_letters_levels_and_takes_no_noul():
    tok = tiny_tokenizer()
    builder = PromptBuilder(tok, 256, ReadoutConfig(**CHAT).prompt)
    assert builder.answer_token(Record(**score_row(0))).codes == tok.convert_tokens_to_ids(
        ["A", "B", "C"]
    )
    with pytest.raises(ValueError, match="choice and score"):
        builder.answer_token(Record(**noul_row(0)))
    with pytest.raises(ValueError, match="needs kind=answer_token"):
        ReadoutConfig(**{**LIST, "prompt": {"style": "chat_json"}})
