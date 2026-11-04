"""Helpers shared by converters: file discovery, streaming reads, dataset cards."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from metiscraft.data.converters import Converter, ConverterContext

SHARD = re.compile(r"^(?P<split>.+?)-\d{5}-of-\d{5}\.parquet$")


def parquet_dir(ctx: ConverterContext) -> Path:
    """The raw directory holding the configured parquet shards, taken from ``files``."""
    for pattern in ctx.entry.files:
        if pattern.endswith(".parquet"):
            parent = Path(pattern).parent
            return ctx.raw_dir / parent
    raise ValueError(f"{ctx.entry.name}: no parquet pattern in files")


def parquet_files(ctx: ConverterContext, split: str) -> list[Path]:
    base = parquet_dir(ctx)
    files = sorted(
        p for p in base.glob("*.parquet") if (m := SHARD.match(p.name)) and m["split"] == split
    )
    if not files:
        raise FileNotFoundError(
            f"{ctx.entry.name}: no parquet shards for split {split!r} in {base}"
        )
    return files


def iter_parquet(files: list[Path], max_rows: int | None = None) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    count = 0
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                if max_rows is not None and count >= max_rows:
                    return
                yield row
                count += 1


def read_card(raw_dir: Path) -> dict[str, Any]:
    """YAML front matter of the dataset card (``README.md``), or ``{}``."""
    path = raw_dir / "README.md"
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
    return yaml.safe_load(match.group(1)) or {} if match else {}


def _find_class_label(spec: Any) -> list[str] | None:
    if isinstance(spec, dict):
        if "class_label" in spec:
            names = spec["class_label"]["names"]
            if isinstance(names, dict):
                return [str(names[k]) for k in sorted(names, key=lambda k: int(k))]
            return [str(n) for n in names]
        for value in spec.values():
            found = _find_class_label(value)
            if found is not None:
                return found
    return None


def class_names(card: dict[str, Any], feature: str, config: str | None = None) -> list[str]:
    info = card.get("dataset_info")
    blocks = info if isinstance(info, list) else [info]
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if config is not None and block.get("config_name") not in (config, None):
            continue
        for feat in block.get("features") or []:
            if feat.get("name") == feature:
                names = _find_class_label(feat)
                if names is not None:
                    return names
    raise ValueError(f"no class names for feature {feature!r} (config {config!r}) in the card")


def humanize(label: str) -> str:
    return label.replace("_", " ").strip()


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def is_one_hot(values: list[float]) -> bool:
    return all(v in (0.0, 1.0) for v in values) and sum(values) == 1.0


def provenance(
    ctx: ConverterContext,
    converter: Converter,
    split: str,
    index: int | str,
    input_hash: str,
    *,
    label_form: str = "hard",
    teacher: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "dataset": ctx.entry.name,
        "source": ctx.entry.hf,
        "config": ctx.entry.config,
        "split": split,
        "index": index,
        "converter": converter.name,
        "converter_version": converter.version,
        "input_hash": input_hash,
        "label_form": label_form,
        "teacher": teacher,
        "extra": extra or {},
    }
