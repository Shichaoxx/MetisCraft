"""Tev1 decision records -> internal records.

Tev1 writes one lettered single-choice task per row: ``state``, ``question``, ``options``
(``label``, ``key``, ``description``), the gold letter ``answer`` and its ``answer_key``.
Options keep Tev1's order, so its prompt can be rendered again byte for byte. Rows of
kind ``score`` (SST-5) become score questions with the levels in their canonical order;
the others, BoolQ included (Tev1 shows it as two lettered options), become choice
questions. Upstream split names are build paths such as ``new-v1/train``, read from
``raw/data/<build>/records/<split>.jsonl``.
"""

from __future__ import annotations

import string
from collections.abc import Iterator
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
from metiscraft.data.converters import Converter, ConverterContext, register
from metiscraft.data.converters.common import provenance
from metiscraft.data.store import read_jsonl


@register
class Tev1Records(Converter):
    name = "tev1"
    version = "2"

    def iter_split(self, ctx: ConverterContext, split: str) -> Iterator[dict[str, Any]]:
        build, _, part = split.rpartition("/")
        path = ctx.raw_dir / "data" / build / "records" / f"{part}.jsonl"
        for row in read_jsonl(path, ctx.max_rows):
            ctx.stats[f"{split} {row['source']}"] += 1
            yield self.record(ctx, split, row).model_dump(mode="json", exclude_none=True)

    def record(self, ctx: ConverterContext, split: str, row: dict[str, Any]) -> Record:
        letters = [o["label"] for o in row["options"]]
        if letters != list(string.ascii_uppercase[: len(letters)]):
            raise ValueError(f"{row['id']}: options are not lettered A, B, ... in order")
        keys = [o["key"] for o in row["options"]]
        if keys[letters.index(row["answer"])] != row["answer_key"]:
            raise ValueError(f"{row['id']}: answer {row['answer']} is not {row['answer_key']}")
        options = [Option(id=o["key"], text=o["description"]) for o in row["options"]]
        if row["kind"] == "score":
            question = Question(type="score", levels=options)
            label: Any = keys.index(row["answer_key"])
        else:
            question = Question(type="choice", options=options)
            label = row["answer_key"]
        extra = {
            "tev1_source": row["source"],
            "tev1_kind": row["kind"],
            "answer_letter": row["answer"],
            "token_count": row.get("token_count"),
        }
        return Record(
            id=f"{ctx.entry.name}:{row['id']}",
            task_id=f"{ctx.entry.name}/{row['source']}",
            input=Input(query=row["question"], context={"state": row["state"]}),
            question=question,
            group=Group(id=row["group_id"]),
            supervision=Supervision(has_hard_label=True, kind=question.type, labels=label),
            provenance=Provenance(
                **provenance(ctx, self, split, row["id"], content_hash(row["state"]), extra=extra)
            ),
        )
