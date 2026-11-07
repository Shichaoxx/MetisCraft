"""Small CPU regressions for preview training, supervision and provenance contracts."""

import json
import math
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from torch.utils.data import SequentialSampler

from metiscraft.cli import main
from metiscraft.config import (
    DataConfig,
    FinetuneConfig,
    HeadConfig,
    ModelConfig,
    ObjectiveConfig,
    Paths,
    TrainConfig,
)
from metiscraft.data.contract import Record
from metiscraft.data.local import import_dataset
from metiscraft.data.store import write_jsonl
from metiscraft.evaluate.metrics import evaluate, evaluate_prior, metrics_for_kind
from metiscraft.judgments import answer_index, target_vector
from metiscraft.train import loop
from metiscraft.train.loop import pos_weight_for


@pytest.mark.parametrize(
    "model,field,value",
    [
        (TrainConfig, "grad_accum", 0),
        (TrainConfig, "batch_size", -1),
        (TrainConfig, "logging_steps", 0),
        (TrainConfig, "max_steps", -1),
        (TrainConfig, "epochs", 0),
        (TrainConfig, "epochs", float("inf")),
        (TrainConfig, "lr", float("nan")),
        (TrainConfig, "head_lr", -1),
        (TrainConfig, "warmup_ratio", 1.1),
        (TrainConfig, "weight_decay", -1),
        (TrainConfig, "max_grad_norm", 0),
        (TrainConfig, "seed", -1),
        (TrainConfig, "seed", 2**32),
        (TrainConfig, "batch_size", True),
        (ModelConfig, "max_length", 0),
        (DataConfig, "max_train", -1),
        (DataConfig, "max_eval", 0),
        (HeadConfig, "hidden", 0),
        (HeadConfig, "dropout", 1.1),
        (FinetuneConfig, "r", 0),
        (FinetuneConfig, "alpha", -1),
        (FinetuneConfig, "dropout", float("nan")),
        (ObjectiveConfig, "pos_weight", -1),
        (ObjectiveConfig, "pos_weight", 0),
        (ObjectiveConfig, "pos_weight", float("inf")),
        (ObjectiveConfig, "brier_weight", -1),
        (ObjectiveConfig, "ordinal_sigma", 0),
    ],
)
def test_invalid_numeric_config_is_rejected(model, field, value):
    defaults = {"base": "unused"} if model is ModelConfig else {}
    if model is DataConfig:
        defaults = {"dataset": "unused"}
    with pytest.raises(ValidationError):
        model.model_validate({**defaults, field: value})


def test_zero_learning_rates_and_frozen_configuration_are_valid():
    cfg = TrainConfig(lr=0, head_lr=0, weight_decay=0, warmup_ratio=1)
    assert cfg.lr == cfg.head_lr == 0
    assert ModelConfig(base="unused", finetune={"kind": "frozen"}).finetune.kind == "frozen"


class Scalar(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        return self.weight * batch["x"]


def collate(rows):
    n = len(rows)
    return {
        "ids": [row["id"] for row in rows],
        "kinds": ["noul"] * n,
        "x": torch.tensor([[float(row["id"] + 1)] for row in rows]),
        "targets": torch.zeros(n, 1),
        "mask": torch.ones(n, 1, dtype=torch.bool),
    }


@pytest.mark.parametrize(
    "n,batch,accum,epochs,max_steps,expected",
    [
        (5, 2, 2, 1, None, [[0, 1, 2, 3], [4]]),
        (2, 2, 4, 1, None, [[0, 1]]),
        (5, 2, 2, 0.5, None, [[0, 1, 2, 3]]),
        (5, 2, 2, 1.5, None, [[0, 1, 2, 3], [4], [0, 1, 2, 3]]),
        (5, 2, 2, 2, None, [[0, 1, 2, 3], [4], [0, 1, 2, 3], [4]]),
        (5, 2, 2, 2.2, None, [[0, 1, 2, 3], [4], [0, 1, 2, 3], [4], [0, 1]]),
        (5, 2, 2, 99, 3, [[0, 1, 2, 3], [4], [0, 1, 2, 3]]),
        (5, 2, 2, 0.1, 4, [[0, 1, 2, 3], [4], [0, 1, 2, 3], [4]]),
    ],
)
def test_epoch_and_step_budgets_flush_tails(
    monkeypatch, n, batch, accum, epochs, max_steps, expected
):
    monkeypatch.setattr(loop, "RandomSampler", lambda rows, generator: SequentialSampler(rows))
    original = loop.accumulate
    seen, grads = [], []

    def observe(net, micro, objective, **kwargs):
        seen.append([i for part in micro for i in part["ids"]])
        result = original(net, micro, objective, **kwargs)
        grads.append(float(net.weight.grad))
        return result

    monkeypatch.setattr(loop, "accumulate", observe)
    model = Scalar()
    cfg = TrainConfig(
        epochs=epochs,
        max_steps=max_steps,
        batch_size=batch,
        grad_accum=accum,
        lr=0,
        bf16=False,
        max_grad_norm=1000,
    )
    result = loop.train_loop(
        model,
        [{"id": i} for i in range(n)],
        collate,
        cfg,
        ObjectiveConfig(),
        device=torch.device("cpu"),
    )
    assert seen == expected
    assert result["steps"] == len(expected)
    assert result["sample_presentations"] == sum(map(len, expected))
    for ids, grad in zip(expected, grads, strict=True):
        reference = sum((1 / (1 + math.exp(-(i + 1)))) * (i + 1) for i in ids) / len(ids)
        assert grad == pytest.approx(reference, rel=1e-6)


def record(kind, labels="missing", soft=None, declared="missing", allow_null=False):
    q = {"type": kind}
    opts = [{"id": "a"}, {"id": "b"}]
    row = {"id": "r", "task_id": "t", "input": {"query": "q"}, "question": q}
    if kind in ("choice", "score"):
        q["options" if kind == "choice" else "levels"] = opts
    if allow_null:
        q["allow_null"] = True
    if kind == "multi_label":
        row["candidates"] = opts
    sup = {"kind": kind}
    if labels != "missing":
        sup["labels"] = labels
    if soft is not None:
        sup["soft"] = soft
    if declared != "missing":
        sup["has_hard_label"] = declared
    row["supervision"] = sup
    return row


@pytest.mark.parametrize("kind", ["noul", "choice", "score", "multi_label"])
def test_soft_only_has_independent_metrics_and_no_invented_hard_gold(kind):
    soft = [0.8] if kind == "noul" else [0.2, 0.8]
    rec = Record.model_validate(record(kind, soft=soft, declared=False))
    assert target_vector(rec) == soft
    pred = prediction(rec)
    metrics = metrics_for_kind(kind, [pred], [soft])
    assert metrics["n"] == metrics["n_soft"] == 1
    assert metrics["n_hard"] == 0
    assert metrics["nll"] is metrics["brier"] is None
    assert metrics["brier_soft"] == pytest.approx(0)
    assert metrics["nll_soft"] > 0
    assert metrics["undefined_reasons"]["nll"] == "no_known_hard_labels"
    json.dumps(metrics, allow_nan=False)
    with pytest.raises(ValueError, match="one hard label"):
        answer_index(rec)


def prediction(rec):
    from metiscraft.modeling.prompts import choices_of

    sup = rec.supervision
    result = {
        "kind": rec.question.type,
        "task_id": rec.task_id,
        "choice_ids": [x.id for x in choices_of(rec)],
        "labels": sup.labels,
        "has_hard_label": rec.hard_label_status(),
        "target": target_vector(rec),
        "scores": [0.0] * rec.target_size(),
    }
    if sup.soft is not None:
        result["soft"] = sup.soft
    return result


@pytest.mark.parametrize(
    "row,expected",
    [
        (record("noul", labels=False), True),
        (record("multi_label", labels=[]), True),
        (record("choice", labels=None, allow_null=True), True),
        (record("choice", soft=[0.1, 0.2, 0.7], allow_null=True), None),
        (record("choice", labels=None, soft=[0.1, 0.2, 0.7], declared=True, allow_null=True), True),
        (record("choice", soft=[0.1, 0.2, 0.7], declared=False, allow_null=True), False),
    ],
)
def test_hard_gold_survives_serialization_including_null_and_empty(row, expected):
    rec = Record.model_validate(row)
    assert rec.hard_label_status() is expected
    restored = Record.model_validate(rec.model_dump(mode="json", exclude_none=True))
    assert restored.hard_label_status() is expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"labels": "a", "soft": [0.2, 0.8], "declared": False},
        {"declared": False},
        {"soft": [0.2, 0.8], "declared": True},
    ],
)
def test_conflicting_explicit_supervision_is_rejected(kwargs):
    with pytest.raises(ValidationError):
        Record.model_validate(record("choice", **kwargs))


def test_mixed_hard_soft_and_legacy_null_counts():
    rows = [
        prediction(Record.model_validate(record("choice", labels="a", soft=[0.2, 0.8]))),
        prediction(Record.model_validate(record("choice", soft=[0.2, 0.8], declared=False))),
        prediction(Record.model_validate(record("choice", soft=[0.2, 0.3, 0.5], allow_null=True))),
    ]
    metrics = metrics_for_kind("choice", rows, [[1, 0], [0.2, 0.8], [0.2, 0.3, 0.5]])
    assert (metrics["n"], metrics["n_hard"], metrics["n_soft"], metrics["n_hard_unknown"]) == (
        3,
        1,
        3,
        1,
    )
    assert metrics["accuracy"] == 1
    assert metrics["brier_soft"] == pytest.approx(1.28 / 3)
    for actual in (evaluate(rows), evaluate_prior(rows, {})):
        assert actual["choice"]["n_hard"] == 1
        assert actual["choice"]["n_soft"] == 3
        json.dumps(actual, allow_nan=False)


def test_one_class_auc_is_null_with_reason():
    rec = Record.model_validate(record("noul", labels=False))
    metrics = metrics_for_kind("noul", [prediction(rec)], [[0.2]])
    assert metrics["auc"] is None
    assert metrics["undefined_reasons"]["auc"] == "hard_labels_have_only_one_class"
    json.dumps(metrics, allow_nan=False)


def test_soft_multilabel_pos_weight_uses_training_targets():
    row = record("multi_label", soft=[0.1, 0.3], declared=False)
    assert float(pos_weight_for([row], ObjectiveConfig(pos_weight="auto"))) == pytest.approx(4)


def test_local_import_uses_original_json_key_presence(tmp_path):
    import yaml

    data = tmp_path / "source"
    data.mkdir()
    raw = [
        record("choice", labels=None, soft=[0.1, 0.2, 0.7], allow_null=True),
        record("choice", soft=[0.1, 0.2, 0.7], allow_null=True),
    ]
    for i, row in enumerate(raw):
        row["id"] = f"r{i}"
        row["input"]["query"] = f"q{i}"
    write_jsonl(data / "train.jsonl", raw)
    manifest = data / "dataset.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "name": "local",
                "version": "v1",
                "source": "test",
                "license": "CC0",
                "terms": [],
                "splits": {"train": "train.jsonl"},
            }
        )
    )
    dataset = import_dataset(manifest, tmp_path / "datasets")
    assert [row["supervision"]["has_hard_label"] for row in dataset.load("train")] == [True, False]
    before = (dataset.root / "SHA256SUMS").read_bytes()
    assert import_dataset(manifest, tmp_path / "datasets") == dataset
    assert (dataset.root / "SHA256SUMS").read_bytes() == before


def test_cli_preserves_mixed_input_order(tmp_path, monkeypatch, capsys):
    from metiscraft.serve.predictor import Predictor

    class Fake:
        def answer_request(self, row):
            return {"id": row["id"]}

        def predict_records(self, rows):
            return [{"id": row["id"]} for row in rows]

    monkeypatch.setattr(Predictor, "load", lambda *args, **kwargs: Fake())
    source = tmp_path / "mixed.jsonl"
    rows = [{"id": 1}, {"id": 2, "questions": {}}, {"id": 3}, {"id": 4}]
    write_jsonl(source, rows)
    assert main(["predict", "unused", str(source)]) == 0
    assert [row["id"] for row in map(json.loads, capsys.readouterr().out.splitlines())] == [
        1,
        2,
        3,
        4,
    ]


def test_resolved_paths_expand_and_become_absolute(tmp_path, monkeypatch):
    from metiscraft.config import resolve_paths

    monkeypatch.chdir(tmp_path)
    got = resolve_paths(
        Paths(data_root="data", models_root="~/models", runs_root="runs"),
        user_file=tmp_path / "absent",
    )
    assert got.data_root == tmp_path / "data"
    assert got.models_root == Path("~/models").expanduser()


def _distributed_tail_worker(rank, rendezvous, output):
    from datetime import timedelta

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=40),
    )
    try:
        model = Scalar()
        net = DistributedDataParallel(model)
        original = loop.accumulate
        seen, grads = [], []

        def observe(net, micro, objective, **kwargs):
            seen.extend(i for part in micro for i in part["ids"])
            result = original(net, micro, objective, **kwargs)
            grads.append(float(model.weight.grad))
            return result

        loop.accumulate = observe
        summary = loop.train_loop(
            model,
            [{"id": i} for i in range(5)],
            collate,
            TrainConfig(batch_size=2, grad_accum=3, lr=0, bf16=False, max_grad_norm=1000),
            ObjectiveConfig(),
            device=torch.device("cpu"),
            net=net,
        )
        Path(output, f"rank-{rank}.json").write_text(
            json.dumps({"summary": summary, "seen": seen, "grads": grads})
        )
    finally:
        dist.destroy_process_group()


def test_cpu_ddp_tail_uses_real_global_question_count(tmp_path):
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("CPU gloo is unavailable")
    mp.spawn(_distributed_tail_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2)
    results = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    seen = [i for result in results for i in result["seen"]]
    assert len(seen) == 6  # DistributedSampler pads five examples to equal rank lengths.
    reference = sum((1 / (1 + math.exp(-(i + 1)))) * (i + 1) for i in seen) / len(seen)
    for result in results:
        assert result["summary"]["steps"] == 1
        assert result["summary"]["sample_presentations"] == 6
        assert result["grads"] == pytest.approx([reference], rel=1e-6)


@pytest.mark.parametrize(
    "command",
    [
        ["data", "show", "dataset", "-n", "0"],
        ["data", "prepare", "dataset", "--max-rows", "-1"],
        ["eval", "artifact", "dataset", "--max-eval", "0"],
    ],
)
def test_cli_row_limits_fail_at_argument_parsing(command):
    from metiscraft.cli import build_parser

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(command)
    assert exc.value.code == 2


@pytest.mark.parametrize("kind", ["noul", "choice", "score"])
def test_probability_only_jev_gold_does_not_invent_hard_labels(kind):
    from metiscraft.data.jev import supervision_from_gold

    rec = Record.model_validate(record(kind, soft=[0.8] if kind == "noul" else [0.2, 0.8]))
    gold = {"probability_true": 0.8} if kind == "noul" else {"probabilities": {"a": 0.2, "b": 0.8}}
    sup = supervision_from_gold(rec.question, gold)
    assert sup.has_hard_label is False
    assert sup.labels is None
    assert sup.soft == rec.supervision.soft


@pytest.mark.parametrize("kind", ["noul", "choice", "score"])
def test_flat_probability_targets_keep_only_observed_supervision(kind):
    from metiscraft.data.converters.jev_sets import _flat_question

    _, label, soft = _flat_question(kind, ["a", "b"], [0.2, 0.8], False)
    assert label is None
    assert soft == ([0.8] if kind == "noul" else [0.2, 0.8])
    _, label, soft = _flat_question(kind, ["a", "b"], [0.0, 1.0], False)
    assert label is None
    assert soft == ([1.0] if kind == "noul" else [0.0, 1.0])


def test_identical_legacy_import_returns_before_new_supervision_normalization(tmp_path):
    import hashlib

    import yaml

    from metiscraft.config import canonical_sha256
    from metiscraft.data.local import LocalManifest
    from metiscraft.data.store import DatasetVersion, write_sha256sums

    source = tmp_path / "train.jsonl"
    # Historically valid: None plus soft on noul meant absent hard supervision.
    write_jsonl(source, [record("noul", labels=None, soft=[0.8])])
    manifest = tmp_path / "manifest.yaml"
    spec = LocalManifest(
        name="legacy",
        version="v1",
        source="synthetic",
        license="CC0",
        terms=[],
        splits={"train": "train.jsonl"},
    )
    manifest.write_text(yaml.safe_dump(spec.model_dump()))
    existing = DatasetVersion(tmp_path / "data" / "legacy" / "v1")
    existing.root.mkdir(parents=True)
    write_jsonl(existing.split_path("train"), [record("noul", labels=None, soft=[0.8])])
    fingerprint = canonical_sha256(
        {
            "manifest": spec.model_dump(),
            "files": {"train": hashlib.sha256(source.read_bytes()).hexdigest()},
        }
    )
    existing.meta_path.write_text(
        yaml.safe_dump(
            {
                "status": "ready",
                "local_import_sha256": fingerprint,
                "converter": {"name": "local_record_jsonl", "version": "1"},
            }
        )
    )
    write_sha256sums(existing.root)
    before = {
        p.relative_to(existing.root): p.read_bytes()
        for p in existing.root.rglob("*")
        if p.is_file()
    }
    assert import_dataset(manifest, tmp_path / "data") == existing
    assert before == {
        p.relative_to(existing.root): p.read_bytes()
        for p in existing.root.rglob("*")
        if p.is_file()
    }
    # A new version normalizes non-choice null plus soft as absent hard gold.
    spec.version = "v2"
    manifest.write_text(yaml.safe_dump(spec.model_dump()))
    new = import_dataset(manifest, tmp_path / "data")
    assert new.load("train")[0]["supervision"]["has_hard_label"] is False
    assert existing.meta()["converter"]["version"] == "1"
    assert new.meta()["converter"]["version"] == "2"


@pytest.mark.parametrize("kind", ["noul", "score", "multi_label"])
def test_new_import_accepts_default_dump_null_with_soft(kind, tmp_path):
    import yaml

    row = record(kind, labels=None, soft=[0.8] if kind == "noul" else [0.2, 0.8])
    source = tmp_path / "train.jsonl"
    write_jsonl(source, [row])
    manifest = tmp_path / "dataset.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "name": "null-soft",
                "version": "v1",
                "source": "synthetic",
                "license": "CC0",
                "terms": [],
                "splits": {"train": "train.jsonl"},
            }
        )
    )
    dataset = import_dataset(manifest, tmp_path / "data")
    assert dataset.load("train")[0]["supervision"]["has_hard_label"] is False
