import json

import pytest

from metiscraft.config import Recipe
from metiscraft.data.prepare import prepare_dataset
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
