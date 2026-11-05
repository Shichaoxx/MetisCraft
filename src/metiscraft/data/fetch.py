"""Download a registered dataset at its pinned revision."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from metiscraft.data.registry import DatasetEntry

DOWNLOAD_DIR = ("_downloads", "metiscraft")


@dataclass
class FetchResult:
    """What a fetcher put into ``raw/``, plus anything worth recording about how.

    ``source`` is merged into the ``source`` block of ``dataset.yaml``; each entry of
    ``manifests`` is written to ``manifests/<name>`` (text as is, anything else as JSON).
    """

    files: list[str]
    source: dict[str, Any] = field(default_factory=dict)
    manifests: dict[str, Any] = field(default_factory=dict)


def download_root(data_root: Path) -> Path:
    return Path(data_root).joinpath(*DOWNLOAD_DIR)


def download_cache(data_root: Path) -> Path:
    return download_root(data_root) / "hf-cache"


def fetch_raw(entry: DatasetEntry, data_root: Path, raw_dir: Path) -> list[str] | FetchResult:
    """Download the files listed in the registry entry and copy them into ``raw_dir``.

    The Hugging Face cache lives under ``<data_root>/_downloads/metiscraft/``; ``raw_dir``
    receives plain copies (no symlinks) with the upstream names and formats unchanged.
    Entries with ``git`` and ``params.build`` are built instead (``metiscraft.data.external``).
    """
    if entry.git and "build" in entry.params:
        from metiscraft.data.external import fetch_build

        return fetch_build(entry, data_root, raw_dir)
    if not entry.hf:
        raise NotImplementedError(f"{entry.name}: a git source without params.build")
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=entry.repo,
            repo_type="dataset",
            revision=entry.revision,
            allow_patterns=entry.files or None,
            cache_dir=download_cache(data_root),
        )
    )
    copied = []
    for path in sorted(snapshot.rglob("*")):
        rel = path.relative_to(snapshot)
        if path.is_dir() or rel.parts[0].startswith("."):
            continue
        dest = raw_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path.resolve(), dest)
        copied.append(rel.as_posix())
    if not copied:
        raise RuntimeError(f"{entry.name}: nothing matched {entry.files} at {entry.revision}")
    return copied
