"""Versioned dataset directories (design §4.1).

    <data_root>/<dataset>/<version>/
      dataset.yaml  raw/  derived/  derived/eval_only/  manifests/  SHA256SUMS

A version directory is written once and never modified: preparing a dataset that
already exists only verifies it.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SUMS_FILE = "SHA256SUMS"
META_FILE = "dataset.yaml"
ROLES = ("train", "calib", "dev", "test", "eval_only")
EVAL_ONLY_ROLES = ("test", "eval_only")


class IntegrityError(RuntimeError):
    """A dataset version or artifact does not match its checksums."""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_files(root: Path) -> list[Path]:
    """All regular files under ``root`` except the checksum file; symlinks are refused."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            if path.is_symlink():
                raise IntegrityError(f"symlinks are not allowed: {path}")
        for name in filenames:
            path = Path(dirpath) / name
            if path.relative_to(root).as_posix() != SUMS_FILE:
                files.append(path)
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def write_sha256sums(root: Path) -> Path:
    lines = [f"{sha256_file(p)}  {p.relative_to(root).as_posix()}" for p in list_files(root)]
    sums = root / SUMS_FILE
    sums.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return sums


def verify_sha256sums(root: Path) -> int:
    sums = root / SUMS_FILE
    if not sums.is_file():
        raise IntegrityError(f"{root}: missing {SUMS_FILE}")
    listed: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, rel = line.partition("  ")
        if not rel or rel.startswith("/") or ".." in Path(rel).parts:
            raise IntegrityError(f"{root}: bad entry in {SUMS_FILE}: {line!r}")
        listed[rel] = digest
    actual = {p.relative_to(root).as_posix(): p for p in list_files(root)}
    problems = []
    for rel in sorted(set(listed) - set(actual)):
        problems.append(f"missing {rel}")
    for rel in sorted(set(actual) - set(listed)):
        problems.append(f"unlisted {rel}")
    for rel in sorted(set(listed) & set(actual)):
        if sha256_file(actual[rel]) != listed[rel]:
            problems.append(f"changed {rel}")
    if problems:
        raise IntegrityError(f"{root}: " + "; ".join(problems))
    return len(listed)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """One JSON object per line. Keys keep their order, so object states stay as the upstream
    wrote them and a prompt rendered from a record can match the upstream's byte for byte."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError("limit must be a positive integer")
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            if line.strip():
                yield json.loads(line)


@dataclass(frozen=True)
class DatasetVersion:
    root: Path

    @property
    def name(self) -> str:
        return self.root.parent.name

    @property
    def version(self) -> str:
        return self.root.name

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def derived_dir(self) -> Path:
        return self.root / "derived"

    @property
    def manifests_dir(self) -> Path:
        return self.root / "manifests"

    @property
    def meta_path(self) -> Path:
        return self.root / META_FILE

    def split_path(self, role: str) -> Path:
        if role not in ROLES:
            raise ValueError(f"unknown split role {role!r}; roles: {', '.join(ROLES)}")
        if role in EVAL_ONLY_ROLES:
            return self.derived_dir / "eval_only" / f"{role}.jsonl"
        return self.derived_dir / f"{role}.jsonl"

    def meta(self) -> dict[str, Any]:
        with open(self.meta_path, encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def has_split(self, role: str) -> bool:
        return self.split_path(role).is_file()

    def load(self, role: str, limit: int | None = None) -> list[dict[str, Any]]:
        path = self.split_path(role)
        if not path.is_file():
            raise FileNotFoundError(f"{self.name}@{self.version} has no {role} split")
        return list(read_jsonl(path, limit))

    def verify(self) -> int:
        if not self.meta_path.is_file():
            raise IntegrityError(f"{self.root}: missing {META_FILE}")
        count = verify_sha256sums(self.root)
        status = self.meta().get("status")
        if status != "ready":
            raise IntegrityError(f"{self.root}: status is {status!r}, not 'ready'")
        return count


def list_versions(data_root: Path, name: str) -> list[DatasetVersion]:
    base = Path(data_root) / name
    if not base.is_dir():
        return []
    return [
        DatasetVersion(p)
        for p in sorted(base.iterdir())
        if p.is_dir() and not p.name.startswith((".", "_")) and (p / META_FILE).is_file()
    ]


def resolve_version(data_root: Path, name: str, version: str | None = None) -> DatasetVersion:
    if version:
        dv = DatasetVersion(Path(data_root) / name / version)
        if not dv.meta_path.is_file():
            raise FileNotFoundError(f"{name}@{version} is not prepared under {data_root}")
        return dv
    versions = list_versions(data_root, name)
    if not versions:
        raise FileNotFoundError(
            f"{name} is not prepared under {data_root}; run metiscraft data prepare {name}"
        )
    if len(versions) > 1:
        names = ", ".join(v.version for v in versions)
        raise ValueError(f"{name} has several versions ({names}); set data.version")
    return versions[0]
