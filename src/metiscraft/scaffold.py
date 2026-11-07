"""Export installed-package examples without requiring a source checkout."""

from __future__ import annotations

from importlib import resources
from pathlib import Path


def init_project(directory: Path, template: str = "quickstart") -> Path:
    """Copy a bundled template into a new directory; never overwrite existing work."""
    if template != "quickstart":
        raise ValueError(f"unknown template {template!r}; available: quickstart")
    source = resources.files("metiscraft").joinpath("templates").joinpath(template)
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=False)

    def copy_tree(src, dst: Path) -> None:
        for item in src.iterdir():
            target = dst / item.name
            if item.is_dir():
                target.mkdir()
                copy_tree(item, target)
            else:
                target.write_bytes(item.read_bytes())

    copy_tree(source, destination)
    return destination
