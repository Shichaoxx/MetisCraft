"""Release gates: no real weights, downloads, training or GPU use.

Suggested destination: tests/test_release_guards.py. These tests intentionally
seal synthetic file bytes and intercept every real model/tokenizer loader.
"""

import json
import math
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import torch
import yaml

from metiscraft.calibrate import calibration_for, fit_calibration, mean_nll
from metiscraft.cli import build_parser
from metiscraft.config import Recipe
from metiscraft.data.local import import_dataset
from metiscraft.data.store import IntegrityError, write_jsonl, write_sha256sums
from metiscraft.modeling.identity import capture_identity
from metiscraft.serve.predictor import Predictor
from metiscraft.trace import artifact_sha, load_calibration, save_calibration


@pytest.fixture
def sealed_base(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    for name, text in {
        "SOURCE.json": '{"source":"synthetic-test-only"}',
        "config.json": '{"model_type":"synthetic-test-only"}',
        "model.safetensors": "placeholder bytes; never passed to a model loader",
        "tokenizer.json": '{"model":{"type":"WordLevel","vocab":{"test":0}}}',
    }.items():
        (base / name).write_text(text, encoding="utf-8")
    write_sha256sums(base)
    return base


def seal_artifact(tmp_path, base, version="2", identity=None):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    metadata = {
        "format": f"metiscraft-artifact/{version}",
        "task": "release-guard",
        "task_ids": ["release-guard"],
        "base": {"path": str(base)},
        "model": {"base": str(base), "dtype": "fp32", "max_length": 32},
        "readout": {"kind": "answer_token"},
        "finetune": {"kind": "frozen"},
    }
    if identity is not None:
        metadata["base"]["identity"] = identity
    (artifact / "artifact.json").write_text(json.dumps(metadata), encoding="utf-8")
    write_sha256sums(artifact)
    return artifact


@pytest.fixture
def forbid_model_loading(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("tokenizer/model loading must not start before the validation error")

    monkeypatch.setattr("metiscraft.serve.predictor.load_tokenizer", forbidden)
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", forbidden)


@pytest.mark.parametrize("allow_legacy", [False, True])
def test_format_two_missing_identity_cannot_use_legacy_escape_hatch(
    tmp_path, sealed_base, forbid_model_loading, allow_legacy
):
    artifact = seal_artifact(tmp_path, sealed_base, version="2")
    with pytest.raises(IntegrityError, match="no verified base identity"):
        Predictor.load(artifact, allow_legacy_artifact=allow_legacy, device=torch.device("cpu"))


def test_format_one_missing_identity_requires_explicit_opt_in(
    tmp_path, sealed_base, forbid_model_loading
):
    artifact = seal_artifact(tmp_path, sealed_base, version="1")
    with pytest.raises(IntegrityError, match="allow_legacy_artifact=True"):
        Predictor.load(artifact, device=torch.device("cpu"))


def test_explicit_legacy_load_remains_visibly_unverified(tmp_path, sealed_base, monkeypatch):
    artifact = seal_artifact(tmp_path, sealed_base, version="1")
    calls = []

    class FakeDecisionModel:
        def __init__(self, backbone, readout, hidden_size):
            self.readout = readout
            self.head = None

        def to(self, device):
            assert device == torch.device("cpu")
            return self

        def eval(self):
            return self

    def tokenizer(path):
        calls.append("tokenizer")
        return SimpleNamespace(eos_token_id=0, pad_token_id=0)

    def backbone(*args, **kwargs):
        calls.append("backbone")
        return SimpleNamespace(config=SimpleNamespace(hidden_size=4))

    monkeypatch.setattr("metiscraft.serve.predictor.load_tokenizer", tokenizer)
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", backbone)
    monkeypatch.setattr("metiscraft.serve.predictor.DecisionModel", FakeDecisionModel)
    monkeypatch.setattr("metiscraft.serve.predictor.Collator", lambda *args, **kwargs: object())
    with pytest.warns(UserWarning, match="Historical base identity is unverified"):
        predictor = Predictor.load(artifact, allow_legacy_artifact=True, device=torch.device("cpu"))
    assert calls == ["tokenizer", "backbone"]
    assert predictor.meta["base_verification"] == "legacy_unverified"


@pytest.mark.parametrize("version", ["1", "2"])
def test_present_but_malformed_identity_never_downgrades_to_legacy(
    tmp_path, sealed_base, forbid_model_loading, version
):
    artifact = seal_artifact(tmp_path, sealed_base, version=version, identity={})
    with pytest.raises(IntegrityError, match="malformed base identity"):
        Predictor.load(artifact, allow_legacy_artifact=True, device=torch.device("cpu"))


@pytest.mark.parametrize("version", ["1", "2"])
def test_changed_base_fails_before_loading_even_with_legacy_flag(
    tmp_path, sealed_base, forbid_model_loading, version
):
    identity = capture_identity(sealed_base)
    artifact = seal_artifact(tmp_path, sealed_base, version=version, identity=identity)
    # Even a freshly recomputed manifest cannot repair the historical identity.
    (sealed_base / "tokenizer.json").write_text('{"changed":true}', encoding="utf-8")
    write_sha256sums(sealed_base)
    with pytest.raises(IntegrityError, match="identity mismatch: changed tokenizer.json"):
        Predictor.load(artifact, allow_legacy_artifact=True, device=torch.device("cpu"))


def test_corrupt_artifact_fails_before_loading(tmp_path, sealed_base, forbid_model_loading):
    artifact = seal_artifact(tmp_path, sealed_base, identity=capture_identity(sealed_base))
    with (artifact / "artifact.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(IntegrityError, match="changed artifact.json"):
        Predictor.load(artifact, allow_legacy_artifact=True, device=torch.device("cpu"))


def test_explicit_relocated_base_reaches_both_verifier_and_tokenizer(
    tmp_path, sealed_base, forbid_model_loading, monkeypatch
):
    from metiscraft.modeling.identity import verify_identity

    artifact = seal_artifact(tmp_path, sealed_base, identity=capture_identity(sealed_base))
    relocated = tmp_path / "relocated"
    shutil.copytree(sealed_base, relocated)
    observed = []

    class ReachedTokenizer(Exception):
        pass

    def verify(path, expected):
        assert Path(path).resolve() == relocated.resolve()
        observed.append("verify")
        verify_identity(path, expected)

    def tokenizer(path):
        assert Path(path).resolve() == relocated.resolve()
        observed.append("tokenizer")
        raise ReachedTokenizer

    monkeypatch.setattr("metiscraft.serve.predictor.verify_identity", verify)
    monkeypatch.setattr("metiscraft.serve.predictor.load_tokenizer", tokenizer)
    with pytest.raises(ReachedTokenizer):
        Predictor.load(artifact, base_path=relocated, device=torch.device("cpu"))
    # The recorded original still exists: the explicit path must take precedence.
    assert sealed_base.exists()
    assert observed == ["verify", "tokenizer"]


@pytest.mark.parametrize("override_state", ["missing", "changed"])
def test_bad_explicit_base_does_not_fall_back_to_valid_recorded_base(
    tmp_path, sealed_base, forbid_model_loading, override_state
):
    artifact = seal_artifact(tmp_path, sealed_base, identity=capture_identity(sealed_base))
    relocated = tmp_path / "relocated"
    if override_state == "changed":
        shutil.copytree(sealed_base, relocated)
        (relocated / "tokenizer.json").write_text('{"different":true}', encoding="utf-8")
        write_sha256sums(relocated)
    with pytest.raises(IntegrityError):
        Predictor.load(
            artifact, base_path=relocated, allow_legacy_artifact=True, device=torch.device("cpu")
        )


@pytest.mark.parametrize(
    "command,arguments",
    [
        ("predict", ["artifact", "records.jsonl"]),
        ("eval", ["artifact", "dataset"]),
        ("serve", ["artifact"]),
    ],
)
def test_base_path_cli_option_reaches_predictor(command, arguments, tmp_path, monkeypatch):
    from metiscraft.cli import main

    relocated = tmp_path / "relocated"

    class ReachedLoad(Exception):
        pass

    def load(*args, **kwargs):
        assert Path(kwargs["base_path"]) == relocated
        raise ReachedLoad

    monkeypatch.setattr(Predictor, "load", load)
    with pytest.raises(ReachedLoad):
        main([command, *arguments, "--base-path", str(relocated)])


@pytest.mark.parametrize(
    "command,arguments",
    [
        ("predict", ["artifact", "records.jsonl"]),
        ("eval", ["artifact", "dataset"]),
        ("serve", ["artifact"]),
    ],
)
def test_legacy_cli_option_is_explicit_and_defaults_off(command, arguments):
    parser = build_parser()
    assert parser.parse_args([command, *arguments]).allow_legacy_artifact is False
    assert (
        parser.parse_args([command, *arguments, "--allow-legacy-artifact"]).allow_legacy_artifact
        is True
    )


@pytest.fixture
def isolated_dataset(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    spec = {
        "name": "release-guard",
        "version": "v1",
        "source": "synthetic",
        "license": "CC0-1.0",
        "terms": [],
        "splits": {role: f"{role}.jsonl" for role in ("train", "calib", "dev", "test")},
    }
    for role, name in spec["splits"].items():
        write_jsonl(
            source / name,
            [
                {
                    "id": role,
                    "task_id": "release-guard",
                    "input": {"query": f"Is the {role} item red?"},
                    "question": {"type": "noul"},
                    "supervision": {"kind": "noul", "labels": True},
                }
            ],
        )
    manifest = source / "dataset.yaml"
    manifest.write_text(yaml.safe_dump(spec), encoding="utf-8")
    data_root = tmp_path / "data"
    return import_dataset(manifest, data_root), data_root


def guarded_recipe(tmp_path, data_root, *, split="calib"):
    return Recipe.model_validate(
        {
            "task": "release-guard",
            "data": {"dataset": "release-guard", "version": "v1"},
            "model": {"base": "must-not-load"},
            "readout": {"kind": "answer_token"},
            "calibrate": {"method": "temperature", "split": split},
            "paths": {"data_root": data_root, "runs_root": tmp_path / "runs"},
        }
    )


@pytest.mark.parametrize("split", ["train", "dev", "test", "eval_only"])
def test_calibration_cannot_read_other_roles_before_devices_or_model_load(
    tmp_path, isolated_dataset, monkeypatch, split
):
    from metiscraft.pipeline import run_recipe

    _, data_root = isolated_dataset
    monkeypatch.setattr("metiscraft.pipeline.init_distributed", lambda: pytest.fail("device init"))
    with pytest.raises(ValueError, match="calibration must use"):
        run_recipe(
            guarded_recipe(tmp_path, data_root, split=split),
            model_factory=lambda *args: pytest.fail("model loading"),
        )
    assert not (tmp_path / "runs").exists()


def test_resealed_cross_split_leak_still_fails_before_device_or_model_load(
    tmp_path, isolated_dataset, monkeypatch
):
    from metiscraft.pipeline import run_recipe

    dataset, data_root = isolated_dataset
    leaked = dataset.load("train")[0]
    leaked["id"] = "different-test-id"
    leaked["provenance"]["input_hash"] = "different-but-false-input-hash"
    write_jsonl(dataset.split_path("test"), [leaked])
    write_sha256sums(dataset.root)
    monkeypatch.setattr("metiscraft.pipeline.init_distributed", lambda: pytest.fail("device init"))
    with pytest.raises(ValueError, match="share input"):
        run_recipe(
            guarded_recipe(tmp_path, data_root),
            model_factory=lambda *args: pytest.fail("model loading"),
        )
    assert not (tmp_path / "runs").exists()


def test_task_temperatures_are_actually_fitted_and_remain_bound_to_the_artifact(
    tmp_path, sealed_base
):
    # Soft labels give a deterministic known optimum, avoiding sampling tolerances.
    def examples(task, temperature, count):
        positive = 1.0 / (1.0 + math.exp(-2.0 / temperature))
        return [
            {
                "task_id": task,
                "kind": "choice",
                "scores": [2.0, 0.0],
                "target": [positive, 1.0 - positive],
            }
            for _ in range(count)
        ]

    cold = examples("cold", 0.5, 24)
    hot = examples("hot", 2.0, 24)
    rare = examples("rare", 4.0, 1)
    fitted = fit_calibration(cold + hot + rare)
    assert fitted["rows"] == {"*|choice": 49, "cold|choice": 24, "hot|choice": 24}
    assert fitted["temperatures"]["cold|choice"] == pytest.approx(0.5, rel=0.05)
    assert fitted["temperatures"]["hot|choice"] == pytest.approx(2.0, rel=0.05)
    for task, values in (("cold", cold), ("hot", hot)):
        chosen = calibration_for(fitted, task, "choice")
        assert chosen["group"] == f"{task}|choice"
        assert chosen["fallback_reason"] is None
        assert mean_nll("choice", values, chosen["temperature"]) < mean_nll("choice", values)
    fallback = calibration_for(fitted, "rare", "choice")
    assert fallback["group"] == "*|choice"
    assert fallback["fallback_reason"] == "task_temperature_unavailable"
    assert all(math.isfinite(t) and t > 0 for t in fitted["temperatures"].values())

    artifact = seal_artifact(tmp_path, sealed_base, identity=capture_identity(sealed_base))
    calibration = save_calibration(fitted, tmp_path / "calibration", artifact, "calib")
    loaded = load_calibration(calibration, artifact)
    assert loaded["temperatures"] == fitted["temperatures"]
    assert loaded["fitted_on"] == "calib"
    assert loaded["parent_artifact_sha256"] == artifact_sha(artifact)
