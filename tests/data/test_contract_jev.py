import pytest
from pydantic import ValidationError

from metiscraft.data.contract import Record, content_hash, normalize_text
from metiscraft.data.jev import JevFormatError, question_to_jev, request_to_records


def rec(question, supervision=None, candidates=None):
    row = {
        "id": "x",
        "task_id": "t",
        "input": {"query": "q", "context": {"state": "s"}},
        "question": question,
    }
    if supervision is not None:
        row["supervision"] = supervision
    if candidates is not None:
        row["candidates"] = candidates
    return Record.model_validate(row)


OPTS = [{"id": "a"}, {"id": "b"}]


def test_valid_records():
    rec({"type": "noul"}, {"kind": "noul", "labels": True, "soft": [0.8]})
    rec({"type": "choice", "options": OPTS}, {"kind": "choice", "labels": "b", "soft": [0.3, 0.7]})
    rec({"type": "choice", "options": OPTS, "allow_null": True}, {"kind": "choice", "labels": None})
    rec({"type": "score", "levels": OPTS}, {"kind": "score", "labels": 1})
    rec({"type": "multi_label"}, {"kind": "multi_label", "labels": ["b"]}, candidates=OPTS)


@pytest.mark.parametrize("kind", ["choice", "score"])
@pytest.mark.parametrize("soft", [[0.0, 0.0], [1.0, 1.0], [0.3, 0.6]])
def test_categorical_soft_targets_must_be_normalized(kind, soft):
    question = {"type": kind, "options" if kind == "choice" else "levels": OPTS}
    with pytest.raises(ValidationError, match="must sum to 1"):
        rec(question, {"kind": kind, "soft": soft})


@pytest.mark.parametrize("kind", ["choice", "score"])
def test_categorical_soft_targets_tolerate_floating_point_rounding(kind):
    question = {"type": kind, "options" if kind == "choice" else "levels": OPTS}
    rec(question, {"kind": kind, "soft": [0.3, 0.7000002]})


@pytest.mark.parametrize("soft", [[0.0, 0.0], [1.0, 1.0]])
def test_multilabel_probabilities_are_independent(soft):
    rec({"type": "multi_label"}, {"kind": "multi_label", "soft": soft}, candidates=OPTS)
    rec({"type": "noul"}, {"kind": "noul", "soft": [soft[0]]})


@pytest.mark.parametrize("allow_null", [False, True])
def test_user_options_cannot_use_the_reserved_null_id(allow_null):
    with pytest.raises(ValidationError, match="reserved for the null answer"):
        rec({"type": "choice", "options": [{"id": "__null__"}, {"id": "b"}],
             "allow_null": allow_null})


@pytest.mark.parametrize(
    "question,supervision,candidates",
    [
        ({"type": "choice", "options": OPTS}, {"kind": "choice", "labels": "z"}, None),
        ({"type": "score", "levels": OPTS}, {"kind": "score", "labels": 2}, None),
        ({"type": "score", "levels": OPTS}, {"kind": "score", "labels": True}, None),
        ({"type": "noul"}, {"kind": "noul", "labels": "yes"}, None),
        ({"type": "noul"}, {"kind": "choice", "labels": True}, None),
        (
            {"type": "choice", "options": OPTS},
            {"kind": "choice", "labels": "a", "soft": [1.0]},
            None,
        ),
        ({"type": "multi_label"}, {"kind": "multi_label", "labels": ["c"]}, OPTS),
        ({"type": "multi_label"}, {"kind": "multi_label", "labels": []}, None),
        ({"type": "choice", "options": [{"id": "a"}, {"id": "a"}]}, None, None),
        ({"type": "choice", "options": OPTS, "criteria": {"true": "x"}}, None, None),
    ],
)
def test_invalid_records(question, supervision, candidates):
    with pytest.raises(ValidationError):
        rec(question, supervision, candidates)


def test_content_hash_normalises():
    assert normalize_text("  Hello\n\tWORLD ") == "hello world"
    assert content_hash("ＡＢＣ  x") == content_hash("abc x")
    assert content_hash("a", "b") != content_hash("ab")
    assert content_hash({"k": 1, "j": 2}) == content_hash({"j": 2, "k": 1})


REQUEST = {
    "model": "any",
    "state": {"ticket": "My card was charged twice."},
    "questions": {
        "category": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "Charges and refunds.", "tech": ""},
        },
        "severity": {"type": "score", "instructions": "How severe?", "criteria": ["low", "high"]},
        "needs_human": {"type": "noul", "instructions": "Escalate?"},
    },
}
GOLD = {
    "category": {
        "type": "choice",
        "label": "billing",
        "probabilities": {"billing": 0.9, "tech": 0.1},
    },
    "severity": {"type": "score", "label": "1", "probabilities": {"0": 0.2, "1": 0.8}},
    "needs_human": {"type": "noul", "label": "false", "probability_true": 0.25},
}


def test_request_to_records():
    records = request_to_records(REQUEST, task_id="t", request_id="r1", gold=GOLD)
    by_name = {r.id.split(":")[-1]: r for r in records}
    assert set(by_name) == {"category", "severity", "needs_human"}
    assert {r.group.id for r in records} == {"r1"}
    cat = by_name["category"]
    assert [o.id for o in cat.question.options] == ["billing", "tech"]
    assert cat.question.options[1].text is None
    assert cat.supervision.labels == "billing" and cat.supervision.soft == [0.9, 0.1]
    assert by_name["severity"].supervision.labels == 1
    assert by_name["needs_human"].supervision.labels is False
    assert by_name["needs_human"].supervision.soft == [0.25]


def test_request_format_errors():
    bad = {"state": "s", "questions": {"q": {"type": "choice", "criteria": ["a", "b"]}}}
    with pytest.raises(JevFormatError, match="object mapping"):
        request_to_records(bad, task_id="t", request_id="r")
    with pytest.raises(JevFormatError, match="not supported"):
        request_to_records(
            {"state": "s", "questions": {"q": {"type": "box"}}}, task_id="t", request_id="r"
        )
    with pytest.raises(JevFormatError, match="state"):
        request_to_records({"questions": {}}, task_id="t", request_id="r")


def test_question_round_trip():
    records = request_to_records(REQUEST, task_id="t", request_id="r1")
    by_name = {r.id.split(":")[-1]: r for r in records}
    spec = question_to_jev(by_name["category"].question, by_name["category"].input.query)
    assert spec["criteria"] == {"billing": "Charges and refunds.", "tech": ""}
    assert question_to_jev(by_name["severity"].question, "")["criteria"] == ["low", "high"]
