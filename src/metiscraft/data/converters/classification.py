"""Converters for mainstream classification datasets."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from metiscraft.data.contract import (
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
    class_names,
    humanize,
    iter_parquet,
    parquet_files,
    provenance,
    read_card,
)


def _dump(record: Record) -> dict[str, Any]:
    return record.model_dump(mode="json", exclude_none=True)


@register
class BoolQ(Converter):
    """BoolQ: a yes/no question about a passage -> noul."""

    name = "boolq"
    version = "2"

    def iter_split(self, ctx: ConverterContext, split: str) -> Iterator[dict[str, Any]]:
        for i, row in enumerate(iter_parquet(parquet_files(ctx, split), ctx.max_rows)):
            question = row["question"].strip()
            question = question[:1].upper() + question[1:]
            if not question.endswith("?"):
                question += "?"
            passage = row["passage"]
            digest = content_hash(question, passage)
            yield _dump(
                Record(
                    id=f"{ctx.entry.name}:{split}:{i}",
                    task_id=ctx.entry.name,
                    input=Input(query=question, context={"state": passage}),
                    question=Question(type="noul"),
                    supervision=Supervision(
                        has_hard_label=True, kind="noul", labels=bool(row["answer"])
                    ),
                    provenance=Provenance(**provenance(ctx, self, split, i, digest)),
                )
            )


@register
class TextChoice(Converter):
    """Single-label text classification (Banking77, CLINC150, AG News) -> choice.

    params: text_field, label_field, instructions, optional null_label (mapped to
    "none of the options" via allow_null, e.g. CLINC's out-of-scope intent).
    """

    name = "text_choice"
    version = "2"

    def iter_split(self, ctx: ConverterContext, split: str) -> Iterator[dict[str, Any]]:
        p = ctx.entry.params
        names = class_names(read_card(ctx.raw_dir), p["label_field"], ctx.entry.config)
        null = p.get("null_label")
        if null is not None and null not in names:
            raise ValueError(f"{ctx.entry.name}: null_label {null!r} is not a class")
        question = Question(
            type="choice",
            options=[Option(id=n, text=humanize(n)) for n in names if n != null],
            allow_null=null is not None,
        )
        for i, row in enumerate(iter_parquet(parquet_files(ctx, split), ctx.max_rows)):
            name = names[row[p["label_field"]]]
            text = row[p["text_field"]]
            yield _dump(
                Record(
                    id=f"{ctx.entry.name}:{split}:{i}",
                    task_id=ctx.entry.name,
                    input=Input(query=p["instructions"], context={"state": text}),
                    question=question,
                    supervision=Supervision(
                        has_hard_label=True, kind="choice", labels=None if name == null else name
                    ),
                    provenance=Provenance(**provenance(ctx, self, split, i, content_hash(text))),
                )
            )


@register
class PairChoice(Converter):
    """Sentence-pair classification (MNLI) -> choice; params name the two text fields."""

    name = "pair_choice"
    version = "2"

    def iter_split(self, ctx: ConverterContext, split: str) -> Iterator[dict[str, Any]]:
        p = ctx.entry.params
        names = class_names(read_card(ctx.raw_dir), p["label_field"], ctx.entry.config)
        texts = p.get("option_texts", {})
        question = Question(
            type="choice", options=[Option(id=n, text=texts.get(n, humanize(n))) for n in names]
        )
        first, second = p["first_field"], p["second_field"]
        for i, row in enumerate(iter_parquet(parquet_files(ctx, split), ctx.max_rows)):
            label = row[p["label_field"]]
            if label is None or label < 0:
                ctx.stats[f"{split}:unlabelled"] += 1
                continue
            state = f"{first.capitalize()}: {row[first]}\n{second.capitalize()}: {row[second]}"
            yield _dump(
                Record(
                    id=f"{ctx.entry.name}:{split}:{i}",
                    task_id=ctx.entry.name,
                    input=Input(query=p["instructions"], context={"state": state}),
                    question=question,
                    supervision=Supervision(
                        has_hard_label=True, kind="choice", labels=names[label]
                    ),
                    provenance=Provenance(
                        **provenance(ctx, self, split, i, content_hash(row[first], row[second]))
                    ),
                )
            )


@register
class SST5(Converter):
    """SST-5 sentence sentiment (SetFit jsonl files) -> score with five levels."""

    name = "sst5"
    version = "2"

    def iter_split(self, ctx: ConverterContext, split: str) -> Iterator[dict[str, Any]]:
        p = ctx.entry.params
        question = Question(
            type="score", levels=[Option(id=str(i), text=t) for i, t in enumerate(p["levels"])]
        )
        path = ctx.raw_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"{ctx.entry.name}: missing {path.name}")
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if ctx.max_rows is not None and i >= ctx.max_rows:
                    break
                row = json.loads(line)
                text = row["text"]
                yield _dump(
                    Record(
                        id=f"{ctx.entry.name}:{split}:{i}",
                        task_id=ctx.entry.name,
                        input=Input(query=p["instructions"], context={"state": text}),
                        question=question,
                        supervision=Supervision(
                            has_hard_label=True, kind="score", labels=int(row["label"])
                        ),
                        provenance=Provenance(
                            **provenance(ctx, self, split, i, content_hash(text))
                        ),
                    )
                )


@register
class TextMultiLabel(Converter):
    """Multi-label text classification (GoEmotions) -> multi_label over the label set."""

    name = "text_multilabel"
    version = "2"

    def iter_split(self, ctx: ConverterContext, split: str) -> Iterator[dict[str, Any]]:
        p = ctx.entry.params
        names = class_names(read_card(ctx.raw_dir), p["labels_field"], ctx.entry.config)
        candidates = [Option(id=n, text=humanize(n)) for n in names]
        question = Question(type="multi_label")
        for i, row in enumerate(iter_parquet(parquet_files(ctx, split), ctx.max_rows)):
            text = row[p["text_field"]]
            labels = sorted({names[j] for j in row[p["labels_field"]]})
            yield _dump(
                Record(
                    id=f"{ctx.entry.name}:{split}:{row.get(p.get('id_field', ''), i)}",
                    task_id=ctx.entry.name,
                    input=Input(query=p["instructions"], context={"state": text}),
                    question=question,
                    candidates=candidates,
                    supervision=Supervision(has_hard_label=True, kind="multi_label", labels=labels),
                    provenance=Provenance(**provenance(ctx, self, split, i, content_hash(text))),
                )
            )
