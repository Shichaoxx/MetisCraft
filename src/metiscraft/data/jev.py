"""Jev typed-decision format <-> internal records (design §4.4).

Official request (OpenAPI 0.2.0)::

    {"model": "...", "state": <string | object | array>,
     "questions": {"<name>": {"type": ..., "instructions": ..., "criteria": ...}}}

- choice: ``criteria`` is an object mapping option names to descriptions (1-255 entries);
- score:  ``criteria`` is an array of level descriptions; the position is the score (1-10);
- noul:   ``criteria`` is optional, ``{"true": ..., "false": ...}``;
- box:    image localisation, not read in v0.1.

One request becomes one record per question; all records of a request share a group, so
they always land in the same split. ``model`` is not required when reading datasets.
"""

from __future__ import annotations

import json
from typing import Any

from metiscraft.data.contract import (
    Group,
    Input,
    Option,
    Provenance,
    Question,
    Record,
    Supervision,
    content_hash,
)

SUPPORTED_TYPES = ("noul", "choice", "score")


class JevFormatError(ValueError):
    """A request or gold answer does not follow the Jev typed-decision format."""


def as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def question_from_jev(name: str, spec: dict[str, Any]) -> Question:
    kind = spec.get("type")
    if kind not in SUPPORTED_TYPES:
        raise JevFormatError(f"question {name!r}: type {kind!r} is not supported in v0.1")
    criteria = spec.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise JevFormatError(
                f"question {name!r}: choice criteria must be an object mapping option names "
                "to descriptions"
            )
        options = [
            Option(id=str(key), text=None if value in (None, "") else as_text(value))
            for key, value in criteria.items()
        ]
        return Question(type="choice", options=options)
    if kind == "score":
        if not isinstance(criteria, list) or not criteria:
            raise JevFormatError(f"question {name!r}: score criteria must be a non-empty array")
        levels = [
            Option(id=str(i), text=None if value in (None, "") else as_text(value))
            for i, value in enumerate(criteria)
        ]
        return Question(type="score", levels=levels)
    noul_criteria = None
    if criteria is not None:
        if not isinstance(criteria, dict):
            raise JevFormatError(f"question {name!r}: noul criteria must be an object")
        noul_criteria = {str(k).lower(): as_text(v) for k, v in criteria.items()}
    return Question(type="noul", criteria=noul_criteria)


def parse_request(request: dict[str, Any]) -> tuple[Any, dict[str, Question], dict[str, str]]:
    if "state" not in request:
        raise JevFormatError("a request needs a state")
    questions = request.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise JevFormatError("a request needs a non-empty questions object")
    parsed: dict[str, Question] = {}
    instructions: dict[str, str] = {}
    for name, spec in questions.items():
        if not isinstance(spec, dict):
            raise JevFormatError(f"question {name!r} must be an object")
        parsed[name] = question_from_jev(name, spec)
        instructions[name] = as_text(spec.get("instructions", ""))
    return request["state"], parsed, instructions


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def supervision_from_gold(question: Question, gold: dict[str, Any]) -> Supervision:
    """Read a gold answer written in the official response shape."""
    probs = gold.get("probabilities")
    label = gold.get("label")
    if question.type == "noul":
        p = gold.get("probability_true", gold.get("probability"))
        if p is None and isinstance(probs, dict):
            p = probs.get("true")
        if label is None:
            if p is None:
                raise JevFormatError("noul gold needs a label or probability_true")
            value = float(p) >= 0.5
        else:
            value = label if isinstance(label, bool) else str(label).strip().lower() == "true"
        return Supervision(kind="noul", labels=value, soft=None if p is None else [float(p)])
    ids = [o.id for o in (question.options if question.type == "choice" else question.levels) or []]
    soft = [float(probs.get(i, 0.0)) for i in ids] if isinstance(probs, dict) else None
    if question.type == "choice":
        if label is None:
            if soft is None:
                raise JevFormatError("choice gold needs a label or probabilities")
            label = ids[_argmax(soft)]
        return Supervision(kind="choice", labels=str(label), soft=soft)
    if label is None:
        if soft is None:
            raise JevFormatError("score gold needs a label or probabilities")
        label = _argmax(soft)
    return Supervision(kind="score", labels=int(label), soft=soft)


def request_to_records(
    request: dict[str, Any],
    *,
    task_id: str,
    request_id: str,
    gold: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> list[Record]:
    state, questions, instructions = parse_request(request)
    input_hash = content_hash(state)
    records = []
    for name in sorted(questions):
        question = questions[name]
        supervision = None
        if gold is not None and name in gold:
            supervision = supervision_from_gold(question, gold[name])
        prov = None
        if provenance is not None:
            extra = {**provenance.get("extra", {}), "question_name": name}
            prov = Provenance(**{**provenance, "input_hash": input_hash, "extra": extra})
        records.append(
            Record(
                id=f"{request_id}:{name}",
                task_id=task_id,
                input=Input(query=instructions[name], context={"state": state}),
                question=question,
                group=Group(id=request_id, split_key=request_id),
                supervision=supervision,
                provenance=prov,
            )
        )
    return records


def question_to_jev(question: Question, instructions: str) -> dict[str, Any]:
    spec: dict[str, Any] = {"type": question.type}
    if instructions:
        spec["instructions"] = instructions
    if question.type == "choice":
        spec["criteria"] = {o.id: o.text or "" for o in question.options or []}
    elif question.type == "score":
        spec["criteria"] = [lv.text or "" for lv in question.levels or []]
    elif question.criteria:
        spec["criteria"] = dict(question.criteria)
    return spec
