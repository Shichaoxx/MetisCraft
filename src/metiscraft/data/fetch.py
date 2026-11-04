"""Download a registered dataset at its pinned revision."""

from __future__ import annotations

import shutil
from pathlib import Path

from metiscraft.data.registry import DatasetEntry

DOWNLOAD_DIR = ("_downloads", "metiscraft")


def download_cache(data_root: Path) -> Path:
    return Path(data_root).joinpath(*DOWNLOAD_DIR, "hf-cache")


def fetch_raw(entry: DatasetEntry, data_root: Path, raw_dir: Path) -> list[str]:
    """Download the files listed in the registry entry and copy them into ``raw_dir``.

    The Hugging Face cache lives under ``<data_root>/_downloads/metiscraft/``; ``raw_dir``
    receives plain copies (no symlinks) with the upstream names and formats unchanged.
    """
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
