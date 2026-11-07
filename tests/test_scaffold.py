import os
import subprocess
import sys

import pytest

from metiscraft.config import load_recipe
from metiscraft.data.local import import_dataset
from metiscraft.data.store import verify_sha256sums
from metiscraft.data.usage import validate_usage
from metiscraft.scaffold import init_project


def test_scaffold_refuses_to_overwrite_any_existing_directory(tmp_path):
    destination = tmp_path / "owned"
    destination.mkdir()
    sentinel = destination / "recipe.yaml"
    sentinel.write_text("user content")
    with pytest.raises(FileExistsError):
        init_project(destination)
    assert sentinel.read_text() == "user content"


def test_unknown_template_does_not_create_destination(tmp_path):
    destination = tmp_path / "project"
    with pytest.raises(ValueError, match="unknown template"):
        init_project(destination, "../elsewhere")
    assert not destination.exists()


def test_exported_quickstart_builds_offline_inputs_without_test_helpers(tmp_path):
    destination = init_project(tmp_path / "project")
    script = (destination / "prepare.py").read_text()
    assert "tests." not in script and "from_pretrained" not in script
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    subprocess.run([sys.executable, "prepare.py"], cwd=destination, env=env, check=True, timeout=60)
    assert verify_sha256sums(destination / "models" / "tiny-qwen3" / "v1") > 2
    dataset = import_dataset(destination / "dataset.yaml", destination / "data")
    validate_usage(load_recipe(destination / "recipe.yaml"), dataset)
    assert len(dataset.load("train")) == 48
    assert len(dataset.load("calib")) == 24
    assert (destination / "request.json").is_file()
    assert (destination / "records.jsonl").is_file()
    result = subprocess.run(
        [sys.executable, "prepare.py"],
        cwd=destination,
        env=env,
        timeout=60,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "already exists" in result.stderr
