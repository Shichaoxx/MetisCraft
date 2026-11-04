"""Turn records into token sequences for the two readouts (design §6).

Every prompt is assembled from separately tokenised segments, so the position of
each candidate's last token is known exactly and nothing merges across segments.
Only the state segment is truncated when a sequence is too long.

Answer-token readout (①)::

    Input:
    <state>

    Question: <query>

    Options:
    A: <option>
    ...
    Answer:
    <code>          <- logits of the code tokens are read at the last position

Candidate-scalar readout (③), list layout: the shared prefix is followed by one
segment per candidate; each candidate is scored from the hidden state of its last
token. The pairs layout builds one sequence per candidate instead.
"""

from __future__ import annotations

import json
import string
from dataclasses import dataclass, field
from itertools import product
from typing import Any

from metiscraft.data.contract import Option, Record

NULL_OPTION = Option(id="__null__", text="None of the above")
YES, NO = "yes", "no"


def render_state(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def option_label(option: Option) -> str:
    if option.text is None:
        return option.id
    if option.text.lower() == option.id.replace("_", " ").lower():
        return option.text
    return f"{option.id}: {option.text}"


def choices_of(record: Record) -> list[Option]:
    """The ordered targets of a record: options (+ null), levels, or candidates."""
    q = record.question
    if q.type == "choice":
        return list(q.options or []) + ([NULL_OPTION] if q.allow_null else [])
    if q.type == "score":
        return list(q.levels or [])
    if q.type == "multi_label":
        return list(record.candidates or [])
    return []


# --------------------------------------------------------------------------- codebook


def code_candidates() -> list[str]:
    upper = list(string.ascii_uppercase)
    return upper + ["".join(p) for p in product(upper, repeat=2)]


def single_token_id(tokenizer, text: str) -> int | None:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1 or ids[0] == tokenizer.unk_token_id:
        return None
    if tokenizer.decode(ids).strip() != text:
        return None
    return ids[0]


def build_codebook(tokenizer, k: int) -> list[tuple[str, int]]:
    """The first ``k`` option codes that are single tokens for this tokenizer."""
    codes: list[tuple[str, int]] = []
    seen: set[int] = set()
    for code in code_candidates():
        tid = single_token_id(tokenizer, code)
        if tid is not None and tid not in seen:
            codes.append((code, tid))
            seen.add(tid)
            if len(codes) == k:
                return codes
    raise ValueError(f"the tokenizer has only {len(codes)} single-token option codes, need {k}")


def level_codes(tokenizer, k: int) -> list[tuple[str, int]]:
    codes = []
    for i in range(k):
        tid = single_token_id(tokenizer, str(i))
        if tid is None:
            raise ValueError(f"level code {i} is not a single token for this tokenizer")
        codes.append((str(i), tid))
    return codes


def yes_no_codes(tokenizer) -> list[tuple[str, int]]:
    """[(yes, id), (no, id)]; the noul logit is z_yes - z_no."""
    out = []
    for word in (YES, NO):
        tid = single_token_id(tokenizer, word)
        if tid is None:
            raise ValueError(f"{word!r} is not a single token for this tokenizer")
        out.append((word, tid))
    return out


# --------------------------------------------------------------------------- encoding


@dataclass
class Encoded:
    """One record as token ids plus what the readout needs."""

    input_ids: list[int]
    kind: str
    codes: list[int] = field(default_factory=list)  # ①: token ids to read, one per target
    positions: list[int] = field(default_factory=list)  # ③ list: last-token index per target
    pairs: list[list[int]] = field(default_factory=list)  # ③ pairs: one sequence per target


class PromptBuilder:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tok = tokenizer
        self.max_length = max_length

    def _ids(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def _assemble(self, record: Record, tail: list[list[int]]) -> list[int]:
        """Prefix (state truncated to fit) followed by the tail segments."""
        context = record.input.context or {}
        head = self._ids("Input:\n") if "state" in context else []
        state = self._ids(render_state(context["state"])) if "state" in context else []
        question = self._ids(f"\n\nQuestion: {record.input.query}")
        criteria = record.question.criteria or {}
        notes = []
        if criteria.get("true"):
            notes.append(f"Answer yes if: {criteria['true']}")
        if criteria.get("false"):
            notes.append(f"Answer no if: {criteria['false']}")
        question += self._ids("".join("\n" + n for n in notes))
        fixed = len(head) + len(question) + sum(len(t) for t in tail)
        budget = self.max_length - fixed
        if budget < 0:
            raise ValueError(f"{record.id}: question and options need {fixed} tokens > max_length")
        return head + state[:budget] + question + [i for seg in tail for i in seg]

    def answer_token(self, record: Record) -> Encoded:
        q = record.question
        if q.type == "multi_label":
            raise ValueError("the answer-token readout does not support multi_label")
        if q.type == "noul":
            codes = yes_no_codes(self.tok)
            tail = [self._ids("\n\nAnswer yes or no.\nAnswer:\n")]
            return Encoded(self._assemble(record, tail), q.type, codes=[t for _, t in codes])
        choices = choices_of(record)
        if q.type == "score":
            codes = level_codes(self.tok, len(choices))
            header = "\n\nLevels:"
            instruction = "\n\nAnswer with the number of one level.\nAnswer:\n"
        else:
            codes = build_codebook(self.tok, len(choices))
            header = "\n\nOptions:"
            instruction = "\n\nAnswer with the code of one option.\nAnswer:\n"
        lines = "".join(
            f"\n{code}: {option_label(o)}" for (code, _), o in zip(codes, choices, strict=True)
        )
        tail = [self._ids(header + lines), self._ids(instruction)]
        return Encoded(self._assemble(record, tail), q.type, codes=[t for _, t in codes])

    def candidate_list(self, record: Record, order: list[int] | None = None) -> Encoded:
        """List layout; ``order`` permutes candidates in the prompt, positions stay aligned
        with the record's targets."""
        q = record.question
        if q.type == "noul":
            ids = self._assemble(record, [self._ids("\nAnswer:")])
            return Encoded(ids, q.type, positions=[len(ids) - 1])
        choices = choices_of(record)
        order = list(range(len(choices))) if order is None else list(order)
        if sorted(order) != list(range(len(choices))):
            raise ValueError("order must be a permutation of the candidates")
        segments = [self._ids(f"\n- {option_label(choices[j])}") for j in order]
        ids = self._assemble(record, [self._ids("\n\nCandidates:")] + segments)
        ends = []
        end = len(ids)
        for seg in reversed(segments):
            ends.append(end - 1)
            end -= len(seg)
        ends.reverse()
        positions = [0] * len(choices)
        for slot, j in enumerate(order):
            positions[j] = ends[slot]
        return Encoded(ids, q.type, positions=positions)

    def candidate_pairs(self, record: Record) -> Encoded:
        q = record.question
        if q.type == "noul":
            return Encoded([], q.type, pairs=[self._assemble(record, [self._ids("\nAnswer:")])])
        pairs = [
            self._assemble(
                record, [self._ids("\n\nCandidate:"), self._ids(f"\n- {option_label(o)}")]
            )
            for o in choices_of(record)
        ]
        return Encoded([], q.type, pairs=pairs)
