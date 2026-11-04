"""Converters for Jev-style typed-decision datasets (field mapping: research/datasets.md §4)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from pydantic import ValidationError

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
from metiscraft.data.converters.common import (
    argmax,
    is_one_hot,
    iter_parquet,
    parquet_files,
    provenance,
)
from metiscraft.data.jev import request_to_records


@register
class JevRequests(Converter):
    """LocalLLaMA typed-decisions: one row = one official request with several questions."""

    name = "jev_requests"

    def iter_split(self, ctx: ConverterContext, split: str) -> Iterator[dict[str, Any]]:
        for row in iter_parquet(parquet_files(ctx, split), ctx.max_rows):
            request = {"state": json.loads(row["state"]), "questions": json.loads(row["questions"])}
            gold = json.loads(row["gold"]) if row.get("gold") else None
            base = provenance(
                ctx,
                self,
                split,
                row["id"],
                "",
                label_form="soft_teacher_mean3",
                teacher={"id": "undisclosed", "license": "unknown"},
                extra={"workflow": row.get("workflow")},
            )
            records = request_to_records(
                request,
                task_id=f"{ctx.entry.name}/{row.get('workflow', 'all')}",
                request_id=f"{ctx.entry.name}:{row['id']}",
                gold=gold,
                provenance=base,
            )
            for record in records:
                yield record.model_dump(mode="json", exclude_none=True)


def _flat_question(kind: str, options: list[str], target: list[float], parse_described: bool):
    """Build (question, label, soft) from one flat row; returns None for unsupported rows."""
    if kind == "choice":
        if parse_described:
            parsed = [o.split(": ", 1) if ": " in o else [o, ""] for o in options]
            names = [name for name, _ in parsed]
            if len(set(names)) == len(names):
                opts = [Option(id=n, text=d or None) for n, d in parsed]
            else:
                opts = [Option(id=str(i), text=o) for i, o in enumerate(options)]
        else:
            if len(set(options)) != len(options):
                return None
            opts = [Option(id=o) for o in options]
        question = Question(type="choice", options=opts)
        soft = None if is_one_hot(target) else target
        return question, opts[argmax(target)].id, soft
    if kind == "score":
        question = Question(
            type="score", levels=[Option(id=str(i), text=o) for i, o in enumerate(options)]
        )
        soft = None if is_one_hot(target) else target
        return question, argmax(target), soft
    if kind == "noul":
        p_true = target[-1]
        soft = None if p_true in (0.0, 1.0) else [p_true]
        return Question(type="noul"), p_true >= 0.5, soft
    return None


class _FlatRows(Converter):
    """Shared logic for datasets with one question per row."""

    parse_described = False

    def state(self, row: dict[str, Any]) -> Any:
        raise NotImplementedError

    def keep(self, ctx: ConverterContext, split: str, row: dict[str, Any]) -> bool:
        return True

    def extra(self, row: dict[str, Any]) -> dict[str, Any]:
        return {}

    def iter_split(self, ctx: ConverterContext, split: str) -> Iterator[dict[str, Any]]:
        for i, row in enumerate(iter_parquet(parquet_files(ctx, split), ctx.max_rows)):
            if not self.keep(ctx, split, row):
                ctx.stats[f"{split}:filtered"] += 1
                continue
            target = [float(x) for x in row["target"]]
            try:
                built = _flat_question(
                    row["kind"], list(row["options"] or []), target, self.parse_described
                )
            except ValidationError:
                built = None
            if built is None:
                ctx.stats[f"{split}:unsupported"] += 1
                continue
            question, label, soft = built
            state = self.state(row)
            record = Record(
                id=f"{ctx.entry.name}:{row['id']}",
                task_id=f"{ctx.entry.name}/{row['source']}",
                input=Input(query=row["question"], context={"state": state}),
                question=question,
                group=Group(id=row["group_id"], split_key=row["group_id"]),
                supervision=Supervision(kind=question.type, labels=label, soft=soft),
                provenance=Provenance(
                    **provenance(
                        ctx,
                        self,
                        split,
                        row.get("original_line_number", row.get("source_index", i)),
                        content_hash(state),
                        label_form="hard" if soft is None else "soft",
                        extra={"source": row["source"], **self.extra(row)},
                    )
                ),
            )
            yield record.model_dump(mode="json", exclude_none=True)


@register
class OpenJev(_FlatRows):
    """Open-Jev: options are "name: description" strings; noul targets are [p_no, p_yes]."""

    name = "open_jev"
    parse_described = True

    def state(self, row: dict[str, Any]) -> Any:
        return json.loads(row["state_json"])


@register
class TasksourceJev(_FlatRows):
    """tasksource-jev: options are bare names; noul targets are [p_true]."""

    name = "tasksource_jev"

    def state(self, row: dict[str, Any]) -> Any:
        return row["state"]

    def keep(self, ctx: ConverterContext, split: str, row: dict[str, Any]) -> bool:
        params = ctx.entry.params
        key = "train_filter" if split == ctx.entry.splits.get("train") else "eval_filter"
        rules = params.get(key) or {}
        return all(row.get(field) in allowed for field, allowed in rules.items())

    def extra(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "variant": row.get("variant"),
            "license": row.get("license"),
            "license_use": row.get("license_use"),
            "example_id": row.get("example_id"),
        }
