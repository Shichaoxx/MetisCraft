"""Prepare a registered dataset: download, convert, assign roles, deduplicate, write.

The version directory is built in a temporary staging directory, checksummed and
verified there, and only then copied under ``data_root``. An existing version is
never overwritten; preparing it again only verifies it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from metiscraft import __version__
from metiscraft.data.converters import ConverterContext, get_converter
from metiscraft.data.dedup import eval_first_filter
from metiscraft.data.fetch import FetchResult
from metiscraft.data.registry import DatasetEntry, get_entry
from metiscraft.data.store import (
    EVAL_ONLY_ROLES,
    DatasetVersion,
    write_jsonl,
    write_sha256sums,
)

Fetcher = Callable[[DatasetEntry, Path, Path], list[str] | FetchResult]


def bucket(row: dict[str, Any]) -> float:
    """A stable number in [0, 1) from the row's group (or input hash), for carving splits."""
    group = row.get("group") or {}
    key = group.get("split_key") or group.get("id") or row["provenance"]["input_hash"]
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0x100000000


def assign_roles(
    entry: DatasetEntry, upstream: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    splits = entry.splits
    roles: dict[str, list[dict[str, Any]]] = {}
    origin: dict[str, str] = {}
    for role in ("train", "calib", "dev", "test", "eval_only"):
        if role in splits:
            roles[role] = upstream[splits[role]]
            origin[role] = splits[role]
    if "calib" in splits or "dev" in splits:
        return roles, origin
    if "val" in splits:
        val = upstream[splits["val"]]
        roles["calib"] = [r for r in val if bucket(r) < 0.5]
        roles["dev"] = [r for r in val if bucket(r) >= 0.5]
        origin["calib"] = f"{splits['val']} (first half by hash)"
        origin["dev"] = f"{splits['val']} (second half by hash)"
    elif "train" in roles:
        f = entry.carve_fraction
        train = roles["train"]
        roles["calib"] = [r for r in train if bucket(r) < f]
        roles["dev"] = [r for r in train if f <= bucket(r) < 2 * f]
        roles["train"] = [r for r in train if bucket(r) >= 2 * f]
        origin["calib"] = f"{splits['train']} (carved {f:.0%} by hash)"
        origin["dev"] = f"{splits['train']} (carved {f:.0%} by hash)"
    return roles, origin


def _make_read_only(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            mode = path.stat().st_mode
            path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def prepare_dataset(
    name: str,
    data_root: Path,
    *,
    max_rows: int | None = None,
    fetcher: Fetcher | None = None,
    staging_parent: Path | None = None,
) -> DatasetVersion:
    entry = get_entry(name)
    if entry.converter is None:
        raise NotImplementedError(f"{name} is registered but has no converter in v0.1")
    data_root = Path(data_root)
    version = (
        entry.default_version if max_rows is None else f"{entry.default_version}-max{max_rows}"
    )
    target = DatasetVersion(data_root / name / version)
    if target.root.exists():
        target.verify()
        return target

    if fetcher is None:
        from metiscraft.data.fetch import fetch_raw as fetcher
    converter = get_converter(entry.converter)
    with tempfile.TemporaryDirectory(prefix=f"metiscraft-{name}-", dir=staging_parent) as tmp:
        stage = DatasetVersion(Path(tmp) / name / version)
        stage.raw_dir.mkdir(parents=True)
        fetched = fetcher(entry, data_root, stage.raw_dir)
        if not isinstance(fetched, FetchResult):
            fetched = FetchResult(files=list(fetched))
        ctx = ConverterContext(entry=entry, raw_dir=stage.raw_dir, max_rows=max_rows)
        upstream: dict[str, list[dict[str, Any]]] = {}
        for split in dict.fromkeys(entry.splits.values()):
            upstream[split] = list(converter.iter_split(ctx, split))
        roles, origin = assign_roles(entry, upstream)
        eval_sets = {role: rows for role, rows in roles.items() if role != "train"}
        dedup_report: dict[str, Any] | None = None
        if "train" in roles:
            roles["train"], dedup_report = eval_first_filter(roles["train"], eval_sets)

        split_meta = {}
        for role, rows in roles.items():
            path = stage.split_path(role)
            count = write_jsonl(path, rows)
            split_meta[role] = {
                "file": path.relative_to(stage.root).as_posix(),
                "rows": count,
                "from": origin.get(role),
            }
        stage.manifests_dir.mkdir(parents=True)
        algorithm_input = [m["file"] for r, m in split_meta.items() if r not in EVAL_ONLY_ROLES]
        eval_only = [m["file"] for r, m in split_meta.items() if r in EVAL_ONLY_ROLES]
        (stage.manifests_dir / "algorithm_input.json").write_text(
            json.dumps({"files": algorithm_input}, indent=2) + "\n", encoding="utf-8"
        )
        (stage.manifests_dir / "eval_only.json").write_text(
            json.dumps({"files": eval_only}, indent=2) + "\n", encoding="utf-8"
        )
        for manifest, content in fetched.manifests.items():
            text = content if isinstance(content, str) else json.dumps(content, indent=2) + "\n"
            (stage.manifests_dir / manifest).write_text(text, encoding="utf-8")
        if dedup_report is not None:
            (stage.manifests_dir / "dedup_report.json").write_text(
                json.dumps(dedup_report, indent=2) + "\n", encoding="utf-8"
            )
        meta = {
            "name": name,
            "version": version,
            "status": "ready",
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metiscraft_version": __version__,
            "tier": entry.tier,
            "source": {
                "hf": entry.repo if entry.hf else None,
                **({"git": entry.git} if entry.git else {}),
                "revision": entry.revision,
                "config": entry.config,
                "files": fetched.files,
                **fetched.source,
            },
            "license": entry.license,
            "terms": entry.terms,
            "judgment": entry.judgments,
            "converter": {"name": converter.name, "version": converter.version},
            "max_rows": max_rows,
            "splits": split_meta,
            "conversion_stats": dict(ctx.stats),
            "eval_frozen": True,
            "lineage": entry.lineage,
        }
        stage.meta_path.write_text(
            yaml.safe_dump(meta, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        _make_read_only(stage.raw_dir)
        write_sha256sums(stage.root)
        stage.verify()
        target.root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(stage.root, target.root)
        # copytree keeps the read-only bits, so the TemporaryDirectory cleanup needs them back.
        for path in stage.root.rglob("*"):
            if path.is_file():
                os.chmod(path, path.stat().st_mode | stat.S_IWUSR)
    target.verify()
    return target
