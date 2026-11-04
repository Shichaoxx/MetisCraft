from pathlib import Path

import pytest
from pydantic import ValidationError

from metiscraft.config import (
    Paths,
    apply_overrides,
    load_recipe,
    require_path,
    resolve_paths,
)

RECIPE = """
task: banking77
data: {dataset: banking77}
model: {base: qwen3/0.6b}
readout: {kind: candidate_scalar}
train: {lr: 2.0e-4}
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "recipe.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_recipe_fills_defaults(tmp_path):
    recipe = load_recipe(write(tmp_path, RECIPE))
    assert recipe.readout.layout == "list"
    assert recipe.model.finetune.kind == "lora"
    assert recipe.calibrate.method == "temperature"


def test_overrides_are_parsed_as_yaml(tmp_path):
    recipe = load_recipe(
        write(tmp_path, RECIPE),
        ["train.lr=1e-3", "train.bf16=false", "readout.layout=pairs", "eval.splits=[dev, test]"],
    )
    assert recipe.train.lr == pytest.approx(1e-3)
    assert recipe.train.bf16 is False
    assert recipe.readout.layout == "pairs"
    assert recipe.eval.splits == ["dev", "test"]


def test_unknown_keys_are_rejected(tmp_path):
    with pytest.raises(ValidationError):
        load_recipe(write(tmp_path, RECIPE), ["train.learning_rate=1e-3"])


def test_bad_override_form():
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])


def test_recipe_hash_changes_with_content(tmp_path):
    path = write(tmp_path, RECIPE)
    assert load_recipe(path).sha256() == load_recipe(path).sha256()
    assert load_recipe(path).sha256() != load_recipe(path, ["train.seed=7"]).sha256()


def test_resolve_paths_priority(tmp_path, monkeypatch):
    user_file = tmp_path / "paths.yaml"
    user_file.write_text("data_root: /from/file\nruns_root: /from/file/runs\n", encoding="utf-8")
    monkeypatch.setenv("METISCRAFT_DATA_ROOT", "/from/env")
    monkeypatch.delenv("METISCRAFT_RUNS_ROOT", raising=False)
    monkeypatch.delenv("METISCRAFT_MODELS_ROOT", raising=False)
    paths = resolve_paths(Paths(models_root=Path("/explicit")), user_file=user_file)
    assert paths.models_root == Path("/explicit")
    assert paths.data_root == Path("/from/env")
    assert paths.runs_root == Path("/from/file/runs")


def test_require_path_explains_missing(tmp_path):
    with pytest.raises(ValueError, match="METISCRAFT_DATA_ROOT"):
        require_path(Paths(), "data_root")
