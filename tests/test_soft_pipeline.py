"""The documented local import -> training -> calibrated evaluation accepts soft-only gold."""

import json

import yaml

from metiscraft.config import Recipe
from metiscraft.data.local import import_dataset
from metiscraft.data.store import write_jsonl, write_sha256sums
from metiscraft.pipeline import run_recipe
from tests.modeling.tiny import tiny_backbone, tiny_tokenizer
from tests.test_runtime_contracts import record


def test_soft_only_import_train_calibrate_and_evaluate(tmp_path):
    tok = tiny_tokenizer()
    base = tmp_path / "models" / "tiny" / "v1"
    tiny_backbone("qwen3", len(tok)).save_pretrained(base)
    tok.save_pretrained(base)
    (base / "SOURCE.json").write_text('{"channel":"synthetic-test-only"}')
    write_sha256sums(base)
    splits = {}
    for role in ("train", "calib", "dev"):
        rows = []
        for kind in ("noul", "choice", "score", "multi_label"):
            for i in range(3):
                row = record(kind, soft=[0.8] if kind == "noul" else [0.2, 0.8])
                row["id"] = f"{role}-{kind}-{i}"
                row["input"]["query"] = f"{role} {kind} item {i}"
                rows.append(row)
        splits[role] = f"{role}.jsonl"
        write_jsonl(tmp_path / splits[role], rows)
    manifest = tmp_path / "dataset.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "name": "soft",
                "version": "v1",
                "source": "synthetic",
                "license": "CC0-1.0",
                "terms": [],
                "splits": splits,
            }
        )
    )
    dataset = import_dataset(manifest, tmp_path / "data")
    assert all(row["supervision"]["has_hard_label"] is False for row in dataset.load("train"))
    recipe = Recipe.model_validate(
        {
            "task": "soft",
            "data": {"dataset": "soft", "version": "v1"},
            "model": {
                "base": str(base),
                "dtype": "fp32",
                "max_length": 128,
                "finetune": {"kind": "lora", "r": 4, "alpha": 8},
            },
            "readout": {"kind": "candidate_scalar"},
            "train": {"max_steps": 2, "batch_size": 4, "grad_accum": 2, "bf16": False},
            "paths": {"data_root": tmp_path / "data", "runs_root": tmp_path / "runs"},
        }
    )
    run = run_recipe(recipe)
    payload = (run / "eval/metrics.json").read_text()

    def reject_constant(value):
        raise AssertionError(f"Non-JSON constant {value}")

    metrics = json.loads(payload, parse_constant=reject_constant)
    assert set(metrics["dev"]) == {"model", "model + temperature", "prior"}
    for method in metrics["dev"].values():
        assert set(method) == {"noul", "choice", "score", "multi_label"}
        for result in method.values():
            assert result["n_hard"] == 0
            assert result["n_soft"] == 3
            assert result["nll"] is None
            assert result["nll_soft"] > 0
    report = (run / "report.md").read_text()
    assert "n_soft" in report and "no_known_hard_labels" in report
