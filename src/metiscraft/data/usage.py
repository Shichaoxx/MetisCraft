"""Fail-fast separation of training, calibration and held-out evaluation data."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from itertools import chain
from typing import Any

from metiscraft.config import Recipe
from metiscraft.data.contract import content_hash
from metiscraft.data.store import ROLES, DatasetVersion, read_jsonl

EVALUATION_ROLES = frozenset({"dev", "test", "eval_only"})


def record_input_hash(row: Mapping[str, Any]) -> str:
    """Compute an identity from the actual input, independent of supplied provenance."""
    value = {key: val for key, val in row["input"].items() if val is not None}
    return content_hash(value)


def validate_split_isolation(roles: Mapping[str, Iterable[Mapping[str, Any]]]) -> None:
    """Reject shared inputs/groups between fitting roles and any held-out split.

    Evaluation splits may overlap each other: neither is used to fit a model or
    calibrator. Their overlap must not be mistaken for independent evidence.
    """
    seen: dict[str, dict[str, set[str]]] = {}
    for role, rows in roles.items():
        keys: dict[str, set[str]] = {"input": set(), "group": set(), "split_key": set()}
        for row in rows:
            keys["input"].add(record_input_hash(row))
            provenance = row.get("provenance") or {}
            if provenance.get("input_hash"):
                keys["input"].add(provenance["input_hash"])
            group = row.get("group") or {}
            if group.get("id"):
                keys["group"].add(group["id"])
            split_key = group.get("split_key") or group.get("id")
            if split_key:
                keys["split_key"].add(split_key)
        for other, other_keys in seen.items():
            if role in EVALUATION_ROLES and other in EVALUATION_ROLES:
                continue
            for kind, values in keys.items():
                if values & other_keys[kind]:
                    raise ValueError(
                        f"data leakage: {other} and {role} share {kind}; "
                        "keep training, calibration and evaluation inputs/groups separate"
                    )
        seen[role] = keys


def validate_usage(recipe: Recipe, dv: DatasetVersion) -> None:
    """Validate roles, file declarations and input/group isolation before model loading.

    The caller verifies the sealed dataset with ``dv.verify()`` first. Check all
    available roles, including held-out splits not selected for this run.
    """
    if recipe.data.train_split != "train":
        raise ValueError("data.train_split must be 'train'; held-out labels cannot train a model")
    # The prior baseline also reads this split when training data is unavailable.
    if recipe.calibrate.split != "calib" or recipe.data.calib_split != "calib":
        raise ValueError("calibration must use the 'calib' split, never train/dev/test/eval_only")
    if recipe.data.dev_split != "dev" or recipe.data.test_split != "test":
        raise ValueError("data.dev_split and data.test_split must keep their declared roles")
    if not set(recipe.eval.splits) <= EVALUATION_ROLES:
        raise ValueError("eval.splits may contain only dev, test and eval_only")

    required = {"train", *recipe.eval.splits}
    if recipe.calibrate.method == "temperature":
        required.add("calib")
    for role in sorted(required):
        if not dv.has_split(role):
            raise ValueError(f"{dv.name}@{dv.version}: required {role} split is missing")

    meta_splits = dv.meta().get("splits", {})
    rows_by_role = {}
    files: dict[tuple[int, int], str] = {}
    for role in ROLES:
        path = dv.split_path(role)
        if not path.is_file():
            continue
        relative = path.relative_to(dv.root).as_posix()
        if meta_splits.get(role, {}).get("file") != relative:
            raise ValueError(f"{role}: dataset.yaml must declare {relative}")
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in files:
            raise ValueError(f"split files must be distinct: {files[identity]} and {role}")
        files[identity] = role
        rows = read_jsonl(path)
        first = next(rows, None)
        if role in required and first is None:
            raise ValueError(f"{role}: required split is empty")
        rows_by_role[role] = chain([first], rows) if first is not None else ()

    expected = {
        "algorithm_input.json": {
            dv.split_path(role).relative_to(dv.root).as_posix()
            for role in rows_by_role
            if role not in {"test", "eval_only"}
        },
        "eval_only.json": {
            dv.split_path(role).relative_to(dv.root).as_posix()
            for role in rows_by_role
            if role in {"test", "eval_only"}
        },
    }
    for name, paths in expected.items():
        manifest = dv.manifests_dir / name
        if not manifest.is_file():
            raise ValueError(f"missing dataset purpose manifest: {name}")
        declared = json.loads(manifest.read_text(encoding="utf-8")).get("files")
        if not isinstance(declared, list) or set(declared) != paths:
            raise ValueError(f"{name}: file declarations do not match dataset split roles")
    validate_split_isolation(rows_by_role)
