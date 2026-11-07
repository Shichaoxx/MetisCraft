"""Vertical scenarios: turn a labelled dataset into business questions via a mapping table.

params:
  text_field, label_field: the source text and its class label
  questions: {name: {type, instructions, criteria, map | true_for}}
    choice: ``map`` sends each answer to the source labels it covers; every source label
            must land on exactly one answer, so a typo in the table is an error, not a gap.
    score:  ``map`` sends each level index ("0", "1", ...) to source labels, same rule.
    noul:   ``true_for`` lists the source labels whose answer is yes.

Each source row becomes one Jev request (all questions about the same text), so its
records share a group and always land in the same split.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from metiscraft.data.converters import Converter, ConverterContext, register
from metiscraft.data.converters.common import (
    class_names,
    iter_parquet,
    parquet_files,
    provenance,
    read_card,
)
from metiscraft.data.jev import request_to_records


def _table(
    qname: str, mapping: dict[str, list[str]], names: list[str], allowed: set[str]
) -> dict[str, str]:
    table: dict[str, str] = {}
    for answer, sources in mapping.items():
        if str(answer) not in allowed:
            raise ValueError(f"{qname}: answer {answer!r} is not one of its criteria")
        for source in sources:
            if source in table:
                raise ValueError(f"{qname}: source label {source!r} is mapped twice")
            table[source] = str(answer)
    missing = [n for n in names if n not in table]
    if missing:
        raise ValueError(f"{qname}: source labels without an answer: {missing}")
    return table


@register
class LabelMapping(Converter):
    name = "label_mapping"
    version = "2"

    def _plan(self, ctx: ConverterContext):
        p = ctx.entry.params
        names = class_names(read_card(ctx.raw_dir), p["label_field"], ctx.entry.config)
        questions: dict[str, dict[str, Any]] = {}
        rules: dict[str, tuple[str, Any]] = {}
        for qname, spec in p["questions"].items():
            kind = spec["type"]
            jev: dict[str, Any] = {"type": kind, "instructions": spec["instructions"]}
            if kind == "choice":
                jev["criteria"] = dict(spec["criteria"])
                rules[qname] = ("table", _table(qname, spec["map"], names, set(jev["criteria"])))
            elif kind == "score":
                jev["criteria"] = list(spec["criteria"])
                levels = {str(i) for i in range(len(jev["criteria"]))}
                rules[qname] = ("table", _table(qname, spec["map"], names, levels))
            elif kind == "noul":
                true_for = set(spec.get("true_for", []))
                unknown = true_for - set(names)
                if unknown:
                    raise ValueError(
                        f"{qname}: true_for names unknown source labels {sorted(unknown)}"
                    )
                rules[qname] = ("true_for", true_for)
            else:
                raise ValueError(f"{qname}: unsupported question type {kind!r}")
            questions[qname] = jev
        return names, questions, rules

    def iter_split(self, ctx: ConverterContext, split: str) -> Iterator[dict[str, Any]]:
        names, questions, rules = self._plan(ctx)
        p = ctx.entry.params
        for i, row in enumerate(iter_parquet(parquet_files(ctx, split), ctx.max_rows)):
            source = names[row[p["label_field"]]]
            gold = {}
            for qname, (how, rule) in rules.items():
                label = rule[source] if how == "table" else source in rule
                gold[qname] = {"type": questions[qname]["type"], "label": label}
            base = provenance(ctx, self, split, i, "", extra={"source_label": source})
            records = request_to_records(
                {"state": row[p["text_field"]], "questions": questions},
                task_id=ctx.entry.name,
                request_id=f"{ctx.entry.name}:{split}:{i}",
                gold=gold,
                provenance=base,
            )
            for record in records:
                yield record.model_dump(mode="json", exclude_none=True)
