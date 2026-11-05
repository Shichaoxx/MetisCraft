"""Dataset converters: upstream split -> internal records (design §4.5).

A converter yields record dicts for one upstream split. It never decides roles
(train / calib / dev / test); that is the pipeline's job, driven by the registry.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from metiscraft.data.registry import DatasetEntry
from metiscraft.registry import Registry


@dataclass
class ConverterContext:
    entry: DatasetEntry
    raw_dir: Path
    max_rows: int | None = None
    stats: Counter = field(default_factory=Counter)


class Converter:
    name: str = ""
    version: str = "1"

    def iter_split(self, ctx: ConverterContext, split: str) -> Iterator[dict[str, Any]]:
        raise NotImplementedError


CONVERTERS: Registry[type[Converter]] = Registry("converter")


def register(cls: type[Converter]) -> type[Converter]:
    return CONVERTERS.register(cls.name)(cls)


def get_converter(name: str) -> Converter:
    from metiscraft.data.converters import classification, jev_sets, scenario, tev1  # noqa: F401

    return CONVERTERS.get(name)()
