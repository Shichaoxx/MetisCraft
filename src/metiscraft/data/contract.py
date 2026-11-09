"""The internal record contract: one JSONL row per question (design §4.3).

Labels by judgment type:
  noul         -> bool
  choice       -> option id, or None when ``allow_null`` and the answer is "none of these"
  score        -> level index, starting at 0
  multi_label  -> list of positive candidate ids
``soft`` holds probabilities aligned with the targets (see ``Record.target_size``).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

JudgmentType = Literal["noul", "choice", "score", "multi_label"]
JUDGMENT_TYPES: tuple[str, ...] = ("noul", "choice", "score", "multi_label")
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
NULL_OPTION_ID = "__null__"
HASH_VERSION = "nfkc-lower-ws-sha256-v1"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Option(_Strict):
    id: str
    text: str | None = None


def _check_unique(items: list[Option], what: str) -> None:
    ids = [item.id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{what} ids must be unique")


class Question(_Strict):
    type: JudgmentType
    options: list[Option] | None = None
    levels: list[Option] | None = None
    criteria: dict[str, str] | None = None
    allow_null: bool = False

    @model_validator(mode="after")
    def _check(self) -> Question:
        if self.type == "choice":
            if not self.options:
                raise ValueError("a choice question needs options")
            if len(self.options) > MAX_CHOICE_OPTIONS:
                raise ValueError(f"a choice question takes at most {MAX_CHOICE_OPTIONS} options")
            _check_unique(self.options, "option")
            if any(option.id == NULL_OPTION_ID for option in self.options):
                raise ValueError(f"option id {NULL_OPTION_ID!r} is reserved for the null answer")
        elif self.options is not None:
            raise ValueError(f"a {self.type} question takes no options")
        if self.type == "score":
            if not self.levels or len(self.levels) > MAX_SCORE_LEVELS:
                raise ValueError(f"a score question needs 1-{MAX_SCORE_LEVELS} levels")
            _check_unique(self.levels, "level")
        elif self.levels is not None:
            raise ValueError(f"a {self.type} question takes no levels")
        if self.criteria is not None and self.type != "noul":
            raise ValueError("criteria are only for noul; choice and score carry options or levels")
        if self.allow_null and self.type != "choice":
            raise ValueError("allow_null is only for choice")
        return self


class Input(_Strict):
    query: str
    context: dict[str, Any] | None = None


class Group(_Strict):
    id: str
    split_key: str | None = None


class Supervision(_Strict):
    kind: JudgmentType
    labels: bool | int | str | list[str] | None = None
    soft: list[float] | None = None
    # None preserves legacy records; explicit flags survive exclude_none serialization.
    has_hard_label: bool | None = Field(default=None, strict=True)


class Provenance(_Strict):
    dataset: str
    source: str | None = None
    config: str | None = None
    split: str | None = None
    index: int | str | None = None
    converter: str
    converter_version: str
    input_hash: str
    label_form: str = "hard"
    teacher: dict[str, Any] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Record(_Strict):
    id: str
    task_id: str
    input: Input
    question: Question
    candidates: list[Option] | None = None
    group: Group | None = None
    supervision: Supervision | None = None
    provenance: Provenance | None = None

    def hard_label_status(self) -> bool | None:
        if self.supervision is None:
            return False
        return hard_label_status(
            self.question.type,
            self.supervision.labels,
            self.supervision.soft,
            allow_null=self.question.allow_null,
            declared=self.supervision.has_hard_label,
        )

    def target_size(self) -> int:
        q = self.question
        if q.type == "noul":
            return 1
        if q.type == "choice":
            return len(q.options or []) + (1 if q.allow_null else 0)
        if q.type == "score":
            return len(q.levels or [])
        return len(self.candidates or [])

    @model_validator(mode="after")
    def _check(self) -> Record:
        q = self.question
        if q.type == "multi_label":
            if not self.candidates:
                raise ValueError("a multi_label question needs candidates")
            _check_unique(self.candidates, "candidate")
        elif self.candidates is not None:
            raise ValueError("candidates are only for multi_label questions")
        sup = self.supervision
        if sup is None:
            return self
        if sup.kind != q.type:
            raise ValueError("supervision.kind must equal question.type")
        if sup.soft is not None:
            if len(sup.soft) != self.target_size():
                raise ValueError(f"soft has {len(sup.soft)} entries, expected {self.target_size()}")
            if any(not 0.0 <= p <= 1.0 for p in sup.soft):
                raise ValueError("soft probabilities must lie in [0, 1]")
            if q.type in {"choice", "score"} and not math.isclose(
                math.fsum(sup.soft), 1.0, rel_tol=0.0, abs_tol=1e-6
            ):
                raise ValueError("choice/score soft probabilities must sum to 1 (tolerance 1e-6)")
        if sup.has_hard_label is False:
            if sup.soft is None or sup.labels is not None:
                raise ValueError("has_hard_label=false requires soft and no hard label")
        else:
            _check_labels(self, sup.labels)
        return self


def _check_labels(record: Record, labels: Any) -> None:
    q = record.question
    if labels is None:
        if q.type == "choice" and q.allow_null:
            return
        if (
            record.supervision is not None
            and record.supervision.soft is not None
            and record.supervision.has_hard_label is not True
        ):
            return
        raise ValueError("supervision needs labels or soft")
    if q.type == "noul":
        if not isinstance(labels, bool):
            raise ValueError("noul labels must be true or false")
    elif q.type == "choice":
        ids = {o.id for o in q.options or []}
        if not isinstance(labels, str) or labels not in ids:
            raise ValueError(f"choice label {labels!r} is not an option id")
    elif q.type == "score":
        if isinstance(labels, bool) or not isinstance(labels, int):
            raise ValueError("score labels must be a level index")
        if not 0 <= labels < len(q.levels or []):
            raise ValueError(f"score label {labels} is out of range")
    else:
        ids = {c.id for c in record.candidates or []}
        if not isinstance(labels, list) or any(not isinstance(x, str) for x in labels):
            raise ValueError("multi_label labels must be a list of candidate ids")
        unknown = set(labels) - ids
        if unknown:
            raise ValueError(f"multi_label labels are not candidates: {sorted(unknown)}")


def hard_label_status(
    kind: str,
    labels: Any,
    soft: list[float] | None,
    *,
    allow_null: bool = False,
    declared: bool | None = None,
) -> bool | None:
    """Whether hard gold is known; None is an irrecoverably ambiguous legacy null.

    Old writers omitted None values. A null-capable choice with soft supervision
    therefore cannot tell an explicit hard null from absent hard supervision.
    """
    if declared is not None:
        return declared
    if labels is not None:
        return True  # False and [] are valid labels, not missing values.
    if kind == "choice" and allow_null:
        return True if soft is None else None
    return False


_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return _WHITESPACE.sub(" ", text).strip()


def content_hash(*parts: Any) -> str:
    """Hash of the judged input, used for deduplication (design §4.6)."""
    pieces = []
    for part in parts:
        if not isinstance(part, str):
            part = json.dumps(part, sort_keys=True, ensure_ascii=False)
        pieces.append(normalize_text(part))
    return hashlib.sha256("\x1f".join(pieces).encode("utf-8")).hexdigest()


def validate_record(row: dict[str, Any]) -> Record:
    return Record.model_validate(row)
