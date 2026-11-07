"""Eval-first deduplication (design §4.6).

A training row is dropped when its input hash, or its group, also appears in any
calib, dev, test or eval_only set; every other row of a hit group goes with it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from metiscraft.data.contract import HASH_VERSION


def _group_id(row: Mapping[str, Any]) -> str | None:
    group = row.get("group")
    return group.get("id") if isinstance(group, Mapping) else None


def eval_first_filter(
    train: Sequence[Mapping[str, Any]],
    eval_sets: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    eval_hashes: set[str] = set()
    eval_groups: set[str] = set()
    for rows in eval_sets.values():
        for row in rows:
            eval_hashes.add(row["provenance"]["input_hash"])
            gid = _group_id(row)
            if gid is not None:
                eval_groups.add(gid)

    hit_groups: set[str] = set()
    same_input = same_group = 0
    for row in train:
        gid = _group_id(row)
        if row["provenance"]["input_hash"] in eval_hashes:
            same_input += 1
            if gid is not None:
                hit_groups.add(gid)
        elif gid is not None and gid in eval_groups:
            same_group += 1
            hit_groups.add(gid)

    kept = []
    for row in train:
        gid = _group_id(row)
        if row["provenance"]["input_hash"] in eval_hashes:
            continue
        if gid is not None and gid in hit_groups:
            continue
        kept.append(row)

    report = {
        "hash_version": HASH_VERSION,
        "rules": [
            "drop training rows whose input hash appears in an evaluation set",
            "drop training rows whose group appears in an evaluation set",
            "drop every training row of a group with a hit",
        ],
        "eval_rows": {role: len(rows) for role, rows in eval_sets.items()},
        "train_before": len(train),
        "train_after": len(kept),
        "hits_same_input": same_input,
        "hits_same_group": same_group,
        "dropped": len(train) - len(kept),
    }
    return kept, report
