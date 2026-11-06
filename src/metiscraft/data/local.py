"""Import user-owned Record JSONL files into an immutable DatasetVersion."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from metiscraft import __version__
from metiscraft.config import canonical_sha256
from metiscraft.data.contract import HASH_VERSION, Record
from metiscraft.data.store import ROLES, DatasetVersion, write_jsonl, write_sha256sums
from metiscraft.data.usage import record_input_hash, validate_split_isolation


class LocalManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    source: str = Field(min_length=1)
    license: str = Field(min_length=1)
    terms: list[str]
    splits: dict[str, str]

    @field_validator("name", "version")
    @classmethod
    def safe_component(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
            raise ValueError("name/version must be a single alphanumeric-led path component")
        return value

    @field_validator("splits")
    @classmethod
    def split_roles(cls, value: dict[str, str]) -> dict[str, str]:
        if not value or set(value) - set(ROLES):
            raise ValueError(f"splits must use one or more roles from {ROLES}")
        return value


def import_dataset(manifest_path: Path, data_root: Path) -> DatasetVersion:
    """Import relative JSONL files; identical re-imports verify, conflicting ones fail.

    Splits are explicit and preserved. Cross-purpose duplicates are rejected, not
    silently discarded. Source files are copied into the sealed version so their
    identity remains inspectable after the original files move or change.
    """
    manifest_path = Path(manifest_path).resolve()
    spec = LocalManifest.model_validate(yaml.safe_load(manifest_path.read_text("utf-8")))
    raw: dict[str, bytes] = {}
    source_hashes = {}
    rows_by_role: dict[str, list[dict[str, Any]]] = {}
    ids: set[str] = set()
    source_files: set[tuple[int, int]] = set()
    for role, relative in spec.splits.items():
        rel = Path(relative)
        path = (manifest_path.parent / rel).resolve()
        if rel.is_absolute() or not path.is_relative_to(manifest_path.parent):
            raise ValueError(f"{role}: split path must stay relative to the manifest directory")
        stat = path.stat()
        inode = (stat.st_dev, stat.st_ino)
        if inode in source_files:
            raise ValueError(f"{role}: the same source file cannot supply multiple split roles")
        source_files.add(inode)
        raw[role] = path.read_bytes()
        source_hashes[role] = hashlib.sha256(raw[role]).hexdigest()
        rows = []
        for index, line in enumerate(raw[role].decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                rec = Record.model_validate(value)
                if not rec.id.strip() or not rec.task_id.strip():
                    raise ValueError("record id and task_id must be nonempty")
                if rec.id in ids:
                    raise ValueError(f"duplicate record id {rec.id!r}")
                if rec.supervision is None:
                    raise ValueError("dataset records must include supervision")
                ids.add(rec.id)
                row = rec.model_dump(mode="json", exclude_none=True)
                original = row.get("provenance")
                row["provenance"] = {
                    "dataset": spec.name,
                    "source": spec.source,
                    "split": role,
                    "index": index,
                    "converter": "local_record_jsonl",
                    "converter_version": "1",
                    "input_hash": record_input_hash(row),
                    "label_form": "soft" if rec.supervision.soft is not None else "hard",
                    "extra": {"imported_provenance": original} if original else {},
                }
                rows.append(Record.model_validate(row).model_dump(mode="json", exclude_none=True))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{relative}:{index}: {exc}") from exc
        if not rows:
            raise ValueError(f"{role}: split must contain at least one supervised record")
        rows_by_role[role] = rows
    validate_split_isolation(rows_by_role)

    fingerprint = canonical_sha256({"manifest": spec.model_dump(), "files": source_hashes})
    target = DatasetVersion(Path(data_root) / spec.name / spec.version)
    if target.root.exists():
        target.verify()
        if target.meta().get("local_import_sha256") != fingerprint:
            raise FileExistsError(
                f"{target.root}: version exists with different local inputs; choose a new version"
            )
        return target

    target.root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".local-import-", dir=target.root.parent) as tmp:
        stage = DatasetVersion(Path(tmp) / spec.version)
        stage.raw_dir.mkdir(parents=True)
        stage.manifests_dir.mkdir(parents=True)
        (stage.raw_dir / "manifest.yaml").write_text(
            yaml.safe_dump(spec.model_dump(), sort_keys=False, allow_unicode=True),
            "utf-8",
        )
        split_meta = {}
        for role, rows in rows_by_role.items():
            (stage.raw_dir / f"{role}.jsonl").write_bytes(raw[role])
            path = stage.split_path(role)
            split_meta[role] = {
                "file": path.relative_to(stage.root).as_posix(),
                "rows": write_jsonl(path, rows),
                "from": spec.splits[role],
            }
        for name, roles in (
            ("algorithm_input.json", {"train", "calib", "dev"}),
            ("eval_only.json", {"test", "eval_only"}),
        ):
            (stage.manifests_dir / name).write_text(
                json.dumps(
                    {"files": [m["file"] for r, m in split_meta.items() if r in roles]},
                    indent=2,
                )
                + "\n",
                "utf-8",
            )
        (stage.manifests_dir / "local_import.json").write_text(
            json.dumps(
                {
                    "sha256": fingerprint,
                    "hash_version": HASH_VERSION,
                    "source_files": source_hashes,
                },
                indent=2,
            )
            + "\n",
            "utf-8",
        )
        meta = {
            "name": spec.name,
            "version": spec.version,
            "status": "ready",
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metiscraft_version": __version__,
            "tier": "local",
            "source": {"local": spec.source, "files": source_hashes},
            "license": spec.license,
            "terms": spec.terms,
            "judgment": sorted(
                {r["question"]["type"] for rows in rows_by_role.values() for r in rows}
            ),
            "converter": {"name": "local_record_jsonl", "version": "1"},
            "splits": split_meta,
            "eval_frozen": True,
            "local_import_sha256": fingerprint,
        }
        stage.meta_path.write_text(
            yaml.safe_dump(meta, sort_keys=False, allow_unicode=True), "utf-8"
        )
        write_sha256sums(stage.root)
        stage.verify()
        # Rename only a completed version; never expose a partially copied dataset.
        if target.root.exists():
            raise FileExistsError(f"{target.root}: another import created this version")
        stage.root.rename(target.root)
    return target
