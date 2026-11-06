import json
import os

import pytest
import yaml

from metiscraft.config import Recipe
from metiscraft.data.local import import_dataset
from metiscraft.data.store import IntegrityError, write_jsonl, write_sha256sums
from metiscraft.data.usage import validate_usage


def row(role, index=0):
    return {
        "id": f"{role}-{index}",
        "task_id": "routing",
        "input": {
            "query": "Which team?",
            "context": {"state": f"{role} ticket {index}"},
        },
        "question": {
            "type": "choice",
            "options": [{"id": "billing"}, {"id": "support"}],
        },
        "supervision": {"kind": "choice", "labels": "support"},
    }


@pytest.fixture
def local_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    spec = {
        "name": "own-data",
        "version": "v1",
        "source": "owned-tickets",
        "license": "proprietary",
        "terms": ["internal_only"],
        "splits": {role: f"{role}.jsonl" for role in ("train", "calib", "dev", "test")},
    }
    for role, path in spec["splits"].items():
        write_jsonl(source / path, [row(role)])
    manifest = source / "dataset.yaml"
    manifest.write_text(yaml.safe_dump(spec))
    return manifest, tmp_path / "data"


def recipe(**changes):
    value = {
        "task": "routing",
        "data": {"dataset": "own-data", "version": "v1"},
        "model": {"base": "never-loaded"},
        "readout": {"kind": "candidate_scalar"},
        "eval": {"splits": ["dev", "test"]},
    }
    for key, val in changes.items():
        value.setdefault(key, {}).update(val)
    return Recipe.model_validate(value)


def test_import_seals_source_and_is_idempotent(local_source):
    manifest, root = local_source
    dataset = import_dataset(manifest, root)
    assert dataset.verify() > 5
    assert dataset.load("train")[0]["provenance"]["source"] == "owned-tickets"
    assert dataset.load("train")[0]["provenance"]["input_hash"]
    assert dataset.meta()["terms"] == ["internal_only"]
    assert (
        dataset.split_path("test")
        .relative_to(dataset.root)
        .as_posix()
        .startswith("derived/eval_only/")
    )
    before = (dataset.root / "SHA256SUMS").read_bytes()
    assert import_dataset(manifest, root) == dataset
    assert (dataset.root / "SHA256SUMS").read_bytes() == before
    validate_usage(recipe(), dataset)


def test_changed_source_requires_new_version(local_source):
    manifest, root = local_source
    dataset = import_dataset(manifest, root)
    before = (dataset.root / "SHA256SUMS").read_bytes()
    write_jsonl(manifest.parent / "train.jsonl", [row("train", 1)])
    with pytest.raises(FileExistsError, match="different local inputs"):
        import_dataset(manifest, root)
    assert (dataset.root / "SHA256SUMS").read_bytes() == before
    dataset.verify()


def test_invalid_label_reports_file_and_line(local_source):
    manifest, root = local_source
    bad = row("train")
    bad["supervision"]["labels"] = "missing-team"
    write_jsonl(manifest.parent / "train.jsonl", [bad])
    with pytest.raises(
        ValueError,
        match="train.jsonl:1:.*",
    ):
        import_dataset(manifest, root)
    assert not (root / "own-data" / "v1").exists()


@pytest.mark.parametrize("kind", ["input", "group", "split_key"])
def test_import_rejects_cross_purpose_overlap(local_source, kind):
    manifest, root = local_source
    train, calib = row("train"), row("calib")
    if kind == "input":
        calib["input"] = train["input"]
    elif kind == "group":
        train["group"] = calib["group"] = {"id": "conversation-1"}
    else:
        train["group"] = {"id": "a", "split_key": "document-1"}
        calib["group"] = {"id": "b", "split_key": "document-1"}
    write_jsonl(manifest.parent / "train.jsonl", [train])
    write_jsonl(manifest.parent / "calib.jsonl", [calib])
    with pytest.raises(ValueError, match=f"share {kind}"):
        import_dataset(manifest, root)


def test_source_paths_cannot_escape_manifest_directory(local_source):
    manifest, root = local_source
    spec = yaml.safe_load(manifest.read_text())
    spec["splits"]["train"] = "../outside.jsonl"
    manifest.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match="relative to the manifest directory"):
        import_dataset(manifest, root)


def test_hardlinked_sources_cannot_supply_two_roles(local_source):
    manifest, root = local_source
    (manifest.parent / "calib.jsonl").unlink()
    os.link(manifest.parent / "train.jsonl", manifest.parent / "calib.jsonl")
    with pytest.raises(ValueError, match="same source file"):
        import_dataset(manifest, root)


@pytest.mark.parametrize(
    "changes",
    [
        {"data": {"train_split": "test"}},
        {"data": {"train_split": "eval_only"}},
        {"calibrate": {"split": "test"}},
        {"calibrate": {"split": "eval_only"}},
        {"calibrate": {"method": "none", "split": "test"}},
        {"eval": {"splits": ["train"]}},
        {"eval": {"splits": ["calib"]}},
    ],
)
def test_usage_rejects_role_overrides_before_any_model_is_needed(local_source, changes):
    dataset = import_dataset(*local_source)
    with pytest.raises(ValueError):
        validate_usage(recipe(**changes), dataset)


def test_usage_catches_forged_provenance_hash(local_source):
    dataset = import_dataset(*local_source)
    leaked = dataset.load("train")[0]
    leaked["id"] = "new-test-id"
    leaked["provenance"]["input_hash"] = "forged-different-hash"
    write_jsonl(dataset.split_path("test"), [leaked])
    write_sha256sums(dataset.root)
    dataset.verify()
    with pytest.raises(ValueError, match="share input"):
        validate_usage(recipe(eval={"splits": []}), dataset)


def test_usage_rejects_misdeclared_purpose_manifest(local_source):
    dataset = import_dataset(*local_source)
    path = dataset.manifests_dir / "algorithm_input.json"
    value = json.loads(path.read_text())
    value["files"].append("derived/eval_only/test.jsonl")
    path.write_text(json.dumps(value))
    write_sha256sums(dataset.root)
    with pytest.raises(ValueError, match="file declarations"):
        validate_usage(recipe(), dataset)


def test_usage_rejects_missing_calibration_instead_of_silently_skipping(local_source):
    dataset = import_dataset(*local_source)
    dataset.split_path("calib").unlink()
    with pytest.raises(ValueError, match="required calib split is missing"):
        validate_usage(recipe(), dataset)


def test_imported_source_is_covered_by_checksums(local_source):
    dataset = import_dataset(*local_source)
    (dataset.raw_dir / "train.jsonl").write_text("tampered\n")
    with pytest.raises(IntegrityError, match="changed raw/train.jsonl"):
        dataset.verify()
