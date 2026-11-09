"""Base identities bind model bytes, not a directory name or checksum-file claim."""

import copy
import hashlib
import json
import shutil

import pytest

from metiscraft.data.store import IntegrityError
from metiscraft.modeling.identity import capture_identity, verify_identity


def seal(root, *, include_source=True):
    files = sorted(path for path in root.rglob("*") if path.is_file())
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}"
        for path in files
        if path.name != "SHA256SUMS" and (include_source or path.name != "SOURCE.json")
    ]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def base(tmp_path):
    root = tmp_path / "base"
    root.mkdir()
    for name, content in {
        "SOURCE.json": '{"repo":"test/tiny","revision":"v1.0-local"}',
        "config.json": '{"model_type":"tiny"}',
        "model.safetensors": "synthetic weights; no model is loaded by these tests",
        "tokenizer.json": '{"model":{"type":"WordLevel","vocab":{"a":0}}}',
        "tokenizer_config.json": '{"model_max_length":128}',
        "generation_config.json": '{"eos_token_id":1}',
    }.items():
        (root / name).write_text(content, encoding="utf-8")
    seal(root)
    return root


def test_identity_survives_relocation_and_manifest_reordering(base, tmp_path):
    expected = capture_identity(base)
    relocated = tmp_path / "elsewhere" / "same-bytes"
    shutil.copytree(base, relocated)
    lines = (relocated / "SHA256SUMS").read_text().splitlines()
    (relocated / "SHA256SUMS").write_text("\n".join(reversed(lines)) + "\n")
    verify_identity(relocated, expected)
    assert capture_identity(relocated) == expected
    assert str(base) not in json.dumps(expected)
    assert set(expected) == {"schema", "files", "sha256"}


@pytest.mark.parametrize(
    "name", ["model.safetensors", "config.json", "tokenizer.json", "SOURCE.json"]
)
def test_resealing_changed_bytes_does_not_reuse_an_artifacts_identity(base, name):
    expected = capture_identity(base)
    with (base / name).open("a", encoding="utf-8") as handle:
        handle.write(" ")
    seal(base)
    with pytest.raises(IntegrityError, match=f"base identity mismatch: changed {name}"):
        verify_identity(base, expected)


def test_old_manifest_cannot_claim_mutated_weight_bytes(base):
    (base / "model.safetensors").write_bytes(b"different model")
    with pytest.raises(IntegrityError, match="changed model.safetensors"):
        capture_identity(base)


@pytest.mark.parametrize("name", ["model.safetensors", "config.json", "tokenizer.json"])
def test_required_model_files_cannot_be_omitted_from_the_manifest(base, name):
    lines = (base / "SHA256SUMS").read_text().splitlines()
    (base / "SHA256SUMS").write_text("\n".join(x for x in lines if not x.endswith(f"  {name}")))
    with pytest.raises(IntegrityError, match=f"unlisted model file {name}"):
        capture_identity(base)


@pytest.mark.parametrize(
    "name",
    [
        "pytorch_model.bin",
        "model-00002-of-00002.safetensors",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "processor_config.json",
        "chat_templates/custom.jinja",
    ],
)
def test_new_model_files_need_checksums_and_change_the_recorded_identity(base, name):
    expected = capture_identity(base)
    path = base / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    with pytest.raises(IntegrityError, match="unlisted model file"):
        verify_identity(base, expected)
    seal(base)
    with pytest.raises(IntegrityError, match="base identity mismatch: added"):
        verify_identity(base, expected)


def test_source_is_recorded_even_when_written_after_the_manifest(base):
    seal(base, include_source=False)
    expected = capture_identity(base)
    assert (
        expected["files"]["SOURCE.json"]
        == hashlib.sha256((base / "SOURCE.json").read_bytes()).hexdigest()
    )
    (base / "SOURCE.json").write_text('{"repo":"different/source"}')
    with pytest.raises(IntegrityError, match="changed SOURCE.json"):
        verify_identity(base, expected)


@pytest.mark.parametrize("content", ["invalid json", "null", "[]", '"source"'])
def test_source_must_be_a_json_object(base, content):
    (base / "SOURCE.json").write_text(content)
    seal(base)
    with pytest.raises(IntegrityError):
        capture_identity(base)


@pytest.mark.parametrize("name", ["SOURCE.json", "SHA256SUMS", "config.json", "tokenizer.json"])
def test_incomplete_base_is_not_a_verified_identity(base, name):
    (base / name).unlink()
    if name != "SHA256SUMS":
        seal(base)
    with pytest.raises(IntegrityError, match="missing"):
        capture_identity(base)


def test_tokenizer_config_without_vocabulary_is_insufficient(base):
    (base / "tokenizer.json").unlink()
    seal(base)
    with pytest.raises(IntegrityError, match="missing tokenizer"):
        capture_identity(base)


def test_documentation_is_not_part_of_model_identity_but_listed_files_are_checked(base):
    expected = capture_identity(base)
    (base / "README.md").write_text("Instructions")
    seal(base)
    verify_identity(base, expected)
    (base / "README.md").write_text("Modified instructions")
    with pytest.raises(IntegrityError, match="changed README.md"):
        capture_identity(base)


@pytest.mark.parametrize(
    "name", ["../outside.bin", "/tmp/outside.bin", "./config.json", "a/../config.json", "a//b"]
)
def test_manifest_paths_must_be_canonical_and_contained(base, name):
    with (base / "SHA256SUMS").open("a") as handle:
        handle.write(f"{'0' * 64}  {name}\n")
    with pytest.raises(IntegrityError, match="unsafe relative file path"):
        capture_identity(base)


def test_duplicate_manifest_entries_are_rejected(base):
    line = (base / "SHA256SUMS").read_text().splitlines()[0]
    with (base / "SHA256SUMS").open("a") as handle:
        handle.write(line + "\n")
    with pytest.raises(IntegrityError, match="duplicate entry"):
        capture_identity(base)


@pytest.mark.parametrize("line", ["bad-hash  config.json", "0" * 64 + " config.json", ""])
def test_bad_or_empty_manifests_are_rejected(base, line):
    (base / "SHA256SUMS").write_text(line)
    with pytest.raises(IntegrityError, match="bad entry|empty"):
        capture_identity(base)


@pytest.mark.parametrize("name", ["model.safetensors", "SOURCE.json", "SHA256SUMS"])
def test_symlink_files_are_rejected_even_if_the_bytes_match(base, tmp_path, name):
    target = tmp_path / "outside"
    (base / name).rename(target)
    (base / name).symlink_to(target)
    with pytest.raises(IntegrityError, match="symlinks are not allowed"):
        capture_identity(base)


def test_symlink_base_or_ancestor_is_rejected(base, tmp_path):
    linked = tmp_path / "linked"
    linked.symlink_to(base, target_is_directory=True)
    with pytest.raises(IntegrityError, match="symlinks are not allowed"):
        capture_identity(linked)
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(IntegrityError, match="symlinks are not allowed"):
        capture_identity(parent_link / "base")


def test_symlink_subdirectory_cannot_hide_model_files(base, tmp_path):
    target = tmp_path / "extra"
    target.mkdir()
    (base / "other").symlink_to(target, target_is_directory=True)
    with pytest.raises(IntegrityError, match="symlinks are not allowed"):
        capture_identity(base)


def test_sharded_weights_bind_both_index_and_all_shards(base):
    (base / "model.safetensors").unlink()
    shard = "model-00001-of-00001.safetensors"
    index = "model.safetensors.index.json"
    (base / shard).write_bytes(b"weights")
    (base / index).write_text(json.dumps({"weight_map": {"layer.weight": shard}}))
    seal(base)
    expected = capture_identity(base)
    assert {shard, index} <= expected["files"].keys()
    (base / shard).unlink()
    seal(base)
    with pytest.raises(IntegrityError):
        verify_identity(base, expected)


@pytest.mark.parametrize("shard", ["../outside.safetensors", "absent.safetensors", "config.json"])
def test_weight_index_cannot_reference_files_outside_the_verified_weights(base, shard):
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": shard}})
    )
    seal(base)
    with pytest.raises(IntegrityError, match="unsafe relative file path|missing or invalid weight"):
        capture_identity(base)


def test_changed_recorded_files_with_old_summary_are_rejected_before_accessing_base(base):
    expected = capture_identity(base)
    expected["files"]["model.safetensors"] = "0" * 64
    with pytest.raises(IntegrityError, match="summary does not match"):
        verify_identity(base / "does-not-exist", expected)


@pytest.mark.parametrize("field,value", [("schema", "unknown/1"), ("files", {}), ("sha256", "bad")])
def test_malformed_identity_is_not_treated_as_legacy(base, field, value):
    expected = capture_identity(base)
    expected[field] = value
    with pytest.raises(IntegrityError):
        verify_identity(base, expected)


def test_missing_identity_and_unsafe_recorded_paths_are_rejected(base):
    with pytest.raises(IntegrityError, match="missing or malformed"):
        verify_identity(base, None)
    expected = copy.deepcopy(capture_identity(base))
    expected["files"]["../outside.bin"] = "0" * 64
    with pytest.raises(IntegrityError, match="unsafe relative file path"):
        verify_identity(base, expected)
