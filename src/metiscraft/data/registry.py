"""Load the data registry shipped with the package (``registry.yaml``)."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

SPLIT_KEYS = ("train", "val", "calib", "dev", "test", "eval_only")
ROLES = ("train", "calib", "dev", "test", "eval_only")


class DatasetEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    tier: Literal["must", "optional", "eval_only"]
    hf: str | None = None
    git: str | None = None
    config: str | None = None
    files: list[str] = Field(default_factory=list)
    extra_sources: dict[str, str] = Field(default_factory=dict)
    judgment: str | list[str]
    converter: str | None = None
    license: str
    terms: list[str] = Field(default_factory=list)
    splits: dict[str, str] = Field(default_factory=dict)
    lineage: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    carve_fraction: float = 0.05

    @property
    def repo(self) -> str:
        if not self.hf:
            raise ValueError(f"{self.name} has no Hugging Face source")
        return self.hf.split("@", 1)[0]

    @property
    def revision(self) -> str:
        source = self.hf or self.git
        if not source or "@" not in source:
            raise ValueError(f"{self.name} has no pinned revision")
        return source.split("@", 1)[1]

    @property
    def default_version(self) -> str:
        return f"rev-{self.revision[:8]}"

    @property
    def judgments(self) -> list[str]:
        return [self.judgment] if isinstance(self.judgment, str) else list(self.judgment)


@lru_cache(maxsize=1)
def load_registry() -> dict[str, DatasetEntry]:
    text = resources.files("metiscraft.data").joinpath("registry.yaml").read_text("utf-8")
    raw = yaml.safe_load(text)["datasets"]
    entries = {}
    for name, body in raw.items():
        unknown = set(body.get("splits", {})) - set(SPLIT_KEYS)
        if unknown:
            raise ValueError(f"registry entry {name}: unknown split keys {sorted(unknown)}")
        entries[name] = DatasetEntry(name=name, **body)
    return entries


def get_entry(name: str) -> DatasetEntry:
    registry = load_registry()
    try:
        return registry[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; known: {', '.join(sorted(registry))}") from None
