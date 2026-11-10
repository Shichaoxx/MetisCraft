import json

import pytest

from metiscraft.config import Recipe
from metiscraft.data.prepare import prepare_dataset
from metiscraft.data.store import write_sha256sums
from metiscraft.pipeline import run_recipe
from metiscraft.trace import load_calibration, read_artifact_meta
from tests.data.fixtures import fake_fetcher
from tests.modeling.tiny import tiny_backbone, tiny_tokenizer


@pytest.fixture
def roots(tmp_path):
    roots = {name: tmp_path / name for name in ("data", "models", "runs")}
    tok = tiny_tokenizer()
    base = roots["models"] / "tiny-qwen3" / "v1"
    tiny_backbone("qwen3", len(tok)).save_pretrained(base)
    tok.save_pretrained(base)
    (base / "SOURCE.json").write_text(
        json.dumps({"channel": "test", "repo": "tiny", "revision": "0"})
    )
    write_sha256sums(base)
    for name in ("banking77", "sst5", "go_emotions"):
        prepare_dataset(name, roots["data"], fetcher=fake_fetcher, staging_parent=tmp_path)
    return roots


def recipe(roots, dataset, readout, objective=None, **train):
    return Recipe.model_validate(
        {
            "task": f"{dataset}-test",
            "objective": objective or {},
            "data": {"dataset": dataset},
            "model": {
                "base": "tiny-qwen3/v1",
                "dtype": "fp32",
                "max_length": 160,
                "finetune": {"kind": "lora", "r": 4, "alpha": 8},
            },
            "readout": readout,
            "train": {"max_steps": 2, "batch_size": 4, "bf16": False, **train},
            "eval": {"splits": ["dev", "test"]},
            "paths": {
                "data_root": str(roots["data"]),
                "models_root": str(roots["models"]),
                "runs_root": str(roots["runs"]),
            },
        }
    )


def test_candidate_scalar_run_is_complete_and_traceable(roots):
    run = run_recipe(recipe(roots, "banking77", {"kind": "candidate_scalar"}))
    for name in (
        "config.yaml",
        "config.sha256",
        "git.json",
        "env.json",
        "data.json",
        "model.json",
        "train_log.jsonl",
        "report.md",
    ):
        assert (run / name).is_file(), name
    metrics = json.loads((run / "eval" / "metrics.json").read_text())
    assert set(metrics) == {"dev", "test"}
    assert {"model", "model + temperature", "prior"} <= set(metrics["dev"])
    assert "choice" in metrics["dev"]["model"]
    meta = read_artifact_meta(run / "artifact")
    assert meta["commercial_use"] is True and meta["readout"]["kind"] == "candidate_scalar"
    assert (run / "artifact" / "adapter").is_dir() and (
        run / "artifact" / "head.safetensors"
    ).is_file()
    calibration = load_calibration(run / "calibration", run / "artifact")
    assert calibration["fitted_on"] == "calib" and "*|choice" in calibration["temperatures"]
    assert "## dev" in (run / "report.md").read_text()


def test_answer_token_run_marks_research_only_data(roots):
    run = run_recipe(recipe(roots, "sst5", {"kind": "answer_token"}))
    meta = read_artifact_meta(run / "artifact")
    assert meta["commercial_use"] is False
    assert "NOT permitted" in (run / "artifact" / "NOTICE").read_text()
    assert not (run / "artifact" / "head.safetensors").exists()
    metrics = json.loads((run / "eval" / "metrics.json").read_text())
    assert "mae_expected" in metrics["dev"]["model"]["score"]


def test_predictor_reload_matches_training_process(roots):
    from metiscraft.data.store import read_jsonl, resolve_version
    from metiscraft.serve.predictor import Predictor

    run = run_recipe(recipe(roots, "banking77", {"kind": "candidate_scalar"}))
    saved = {p["id"]: p for p in read_jsonl(run / "eval" / "dev.predictions.jsonl")}
    predictor = Predictor.load(run / "artifact")
    assert predictor.calibration is not None
    rows = resolve_version(roots["data"], "banking77").load("dev")
    for result in predictor.predict_records(rows):
        assert result["raw_scores"] == pytest.approx(saved[result["id"]]["scores"], abs=1e-5)
        assert abs(sum(result["probabilities"]) - 1) < 1e-6


def test_pos_weight_is_undone_at_inference(roots):
    import math

    from metiscraft.data.store import read_jsonl, resolve_version
    from metiscraft.evaluate.infer import predict
    from metiscraft.serve.predictor import Predictor

    readout = {"kind": "candidate_scalar", "shuffle_train": False}
    run = run_recipe(recipe(roots, "go_emotions", readout, {"kind": "bce", "pos_weight": "auto"}))
    train = resolve_version(roots["data"], "go_emotions").load("train")
    pos = sum(len(r["supervision"]["labels"]) for r in train)
    neg = sum(len(r["candidates"]) for r in train) - pos
    w = read_artifact_meta(run / "artifact")["train"]["pos_weight"]
    assert w == pytest.approx(min(max(neg / pos, 1.0), 100.0))
    saved = {p["id"]: p for p in read_jsonl(run / "eval" / "dev.predictions.jsonl")}
    predictor = Predictor.load(run / "artifact")
    rows = resolve_version(roots["data"], "go_emotions").load("dev")
    unshifted = predict(
        predictor.model, rows, predictor.collator, batch_size=4, device=predictor.device
    )
    for raw, result in zip(unshifted, predictor.predict_records(rows), strict=True):
        expected = [s - math.log(w) for s in raw["scores"]]
        assert saved[raw["id"]]["scores"] == pytest.approx(expected, abs=1e-5)
        assert result["raw_scores"] == pytest.approx(expected, abs=1e-5)


def test_http_service_speaks_the_official_format(roots):
    from fastapi.testclient import TestClient

    from metiscraft.serve.predictor import Predictor
    from metiscraft.serve.server import create_app

    run = run_recipe(recipe(roots, "banking77", {"kind": "candidate_scalar"}))
    client = TestClient(create_app(Predictor.load(run / "artifact")))
    request = {
        "model": "anything",
        "state": "card arrival message",
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team ?",
                "criteria": {"billing": "", "tech": "Bugs"},
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent ?",
                "criteria": ["low", "high"],
            },
            "human": {"type": "noul", "instructions": "Needs a human ?"},
        },
    }
    body = client.post("/v1/systemone", json=request).json()
    assert body["model"].startswith("metiscraft/") and set(body["answers"]) == {
        "team",
        "urgency",
        "human",
    }
    team = body["answers"]["team"]
    assert team["type"] == "choice" and team["choice"] in ("billing", "tech")
    assert abs(sum(team["probabilities"].values()) - 1) < 1e-6
    assert body["answers"]["urgency"]["legend"] == {"0": "low", "1": "high"}
    assert 0 <= body["answers"]["human"]["noul"] <= 1
    assert body["usage"]["output_tokens"] == 0 and body["usage"]["input_tokens"] > 0
    bad = dict(request, questions={"q": {"type": "choice", "criteria": ["a", "b"]}})
    assert client.post("/v1/systemone", json=bad).status_code == 400


def test_evaluate_artifact_matches_the_run(roots, monkeypatch):
    from metiscraft.pipeline import evaluate_artifact

    monkeypatch.setenv("METISCRAFT_DATA_ROOT", str(roots["data"]))
    run = run_recipe(recipe(roots, "banking77", {"kind": "candidate_scalar"}))
    again = evaluate_artifact(run / "artifact", "banking77", splits=["dev"])
    saved = json.loads((run / "eval" / "metrics.json").read_text())
    for method in ("model", "model + temperature"):
        for key in ("accuracy", "nll", "ece"):
            assert again["dev"][method]["choice"][key] == pytest.approx(
                saved["dev"][method]["choice"][key], abs=1e-6
            )


def test_chat_json_completion_run_reloads_with_its_prompt(roots):
    from metiscraft.data.store import read_jsonl, resolve_version
    from metiscraft.serve.predictor import Predictor

    readout = {"kind": "answer_token", "prompt": {"style": "chat_json", "system": "Pick one."}}
    objective = {"kind": "completion"}
    run = run_recipe(recipe(roots, "banking77", readout, objective, scheduler="cosine"))
    meta = read_artifact_meta(run / "artifact")
    assert meta["readout"]["prompt"] == {"style": "chat_json", "system": "Pick one."}
    log = [json.loads(x) for x in (run / "train_log.jsonl").read_text().splitlines()]
    assert log[-1]["lr"] == pytest.approx(0.0)  # cosine reaches zero on the last step
    saved = {p["id"]: p for p in read_jsonl(run / "eval" / "dev.predictions.jsonl")}
    predictor = Predictor.load(run / "artifact")
    rows = resolve_version(roots["data"], "banking77").load("dev")
    for result in predictor.predict_records(rows):
        assert result["raw_scores"] == pytest.approx(saved[result["id"]]["scores"], abs=1e-5)


def test_raw_and_resolved_configuration_identities_are_recomputable(roots, monkeypatch):
    from metiscraft.cli import main
    from metiscraft.config import Paths, load_recipe

    for field, value in roots.items():
        monkeypatch.setenv(f"METISCRAFT_{field.upper()}_ROOT", str(value))
    cfg = recipe(roots, "banking77", {"kind": "candidate_scalar"})
    cfg = cfg.model_copy(update={"paths": Paths()})
    run = run_recipe(cfg)
    raw = load_recipe(run / "config.raw.yaml")
    resolved = load_recipe(run / "config.yaml")
    assert raw.sha256() == cfg.sha256()
    assert raw.sha256() == (run / "config.raw.sha256").read_text().strip()
    assert resolved.sha256() == (run / "config.sha256").read_text().strip()
    assert raw.sha256() != resolved.sha256()
    assert run.name.endswith(raw.sha256()[:8])
    meta = read_artifact_meta(run / "artifact")
    assert meta["recipe_sha256"] == raw.sha256()
    assert meta["resolved_config_sha256"] == resolved.sha256()
    assert main(["config", "show", str(run / "config.raw.yaml")]) == 0


def _failing_recipe_worker(rank, config, phase, rendezvous, output):
    """Exercise the actual recipe path and let a failed rank disconnect its peer."""
    import gc
    import weakref
    from datetime import timedelta
    from pathlib import Path

    import torch
    import torch.distributed as dist

    import metiscraft.pipeline as pipeline
    from metiscraft.train.distributed import init_process_group
    from tests.test_distributed_lifecycle import _gloo_thread_names

    torch.set_num_threads(1)

    def exercise():
        group_refs = []

        def initialize():
            init_process_group(
                "gloo",
                init_method=f"file://{rendezvous}",
                rank=rank,
                world_size=2,
                timeout=timedelta(seconds=20),
            )
            group_refs.append(weakref.ref(dist.group.WORLD))
            return rank, 2, torch.device("cpu")

        original_train = pipeline.train_loop

        def train(*args, **kwargs):
            result = original_train(*args, **kwargs)
            if rank == 0 and phase == "train":
                raise RuntimeError("injected train failure after backward")
            return result

        def model_factory(*args):
            if rank == 0 and phase == "model":
                raise RuntimeError("injected model failure")
            return pipeline.build_model(*args)

        def evaluate(*args):
            raise RuntimeError("injected evaluate failure")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(pipeline, "init_distributed", initialize)
            patch.setattr(pipeline, "env_info", lambda: {})
            patch.setattr(pipeline, "git_info", lambda: ({"test": True}, ""))
            patch.setattr(pipeline, "train_loop", train)
            if phase == "evaluate":
                patch.setattr(pipeline, "evaluate_run", evaluate)
            try:
                pipeline.run_recipe(Recipe.model_validate(config), model_factory=model_factory)
            except Exception as error:
                message = str(error)
            else:
                pytest.fail("a failed rank must not return a successful recipe")
        return group_refs[0], message

    group_ref, message = exercise()
    # Tracebacks may form cycles. Check after the caller has handled the error,
    # without retaining the exception or training frames in the assertion scope.
    gc.collect()
    assert not dist.is_initialized()
    assert group_ref() is None, "failed recipe retained its process group"
    assert not _gloo_thread_names(), "failed recipe retained Gloo threads"
    if rank == 0:
        assert f"injected {phase} failure" in message
    Path(output, f"failure-rank-{rank}.json").write_text(
        json.dumps({"rank": rank, "error": message, "group_released": True})
    )


@pytest.mark.parametrize("phase", ["model", "train", "evaluate"])
def test_failed_distributed_recipe_releases_owned_group_and_peer(roots, tmp_path, phase):
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("CPU gloo is unavailable")
    config = recipe(roots, "banking77", {"kind": "candidate_scalar"}, max_steps=1)
    mp.spawn(
        _failing_recipe_worker,
        args=(config.to_dict(), phase, str(tmp_path / "failure-rendezvous"), str(tmp_path)),
        nprocs=2,
    )
    for rank in range(2):
        result = json.loads((tmp_path / f"failure-rank-{rank}.json").read_text())
        assert result["group_released"] and result["error"]
