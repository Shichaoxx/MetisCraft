"""HTTP and batch requests must choose the same calibration, including fallback."""

from types import SimpleNamespace

import pytest

from metiscraft.calibrate import calibration_for
from metiscraft.config import ReadoutConfig, Recipe
from metiscraft.data.jev import request_to_records
from metiscraft.serve.predictor import Predictor
from metiscraft.serve.server import create_app
from tests.modeling.tiny import tiny_tokenizer


@pytest.fixture
def predictor(monkeypatch):
    def scores(model, rows, *args, **kwargs):
        return [
            {
                "task_id": r.task_id,
                "kind": "choice",
                "scores": [1.0, 0.0],
                "choice_ids": ["yes", "no"],
            }
            for r in rows
        ]

    monkeypatch.setattr("metiscraft.serve.predictor.predict", scores)
    return Predictor(
        SimpleNamespace(readout=ReadoutConfig(kind="answer_token")),
        tiny_tokenizer(),
        {
            "task": "mixed",
            "task_ids": ["a", "b"],
            "model": {"max_length": 128},
            "serve": {"task_map": {"first": "a", "second": "b"}},
        },
        {"temperatures": {"a|choice": 0.5, "b|choice": 2.0, "*|choice": 1.1}},
        "cpu",
    )


def request():
    spec = {"type": "choice", "criteria": {"yes": "Yes", "no": "No"}}
    # Intentionally reverse insertion order; conversion sorts question names.
    return {"state": "Example", "questions": {"second": spec, "first": spec}}


def test_http_and_records_match_for_distinct_task_temperatures(predictor):
    from fastapi.testclient import TestClient

    rows = request_to_records(request(), task_id="ignored", request_id="same")
    for row in rows:
        row.task_id = predictor.task_map[row.id.split(":", 1)[1]]
    batch = predictor.predict_records(rows)
    response = TestClient(create_app(predictor)).post("/v1/systemone", json=request())
    assert response.status_code == 200
    for row, result in zip(rows, batch, strict=True):
        answer = response.json()["answers"][row.id.split(":", 1)[1]]
        assert list(answer["probabilities"].values()) == pytest.approx(result["probabilities"])
        assert result["calibration"]["group"] == f"{row.task_id}|choice"
        assert result["calibration"]["fallback_reason"] is None
    assert batch[0]["probabilities"] != batch[1]["probabilities"]


def test_ambiguous_multi_task_request_is_rejected(predictor):
    from fastapi.testclient import TestClient

    predictor.task_map = {}
    response = TestClient(create_app(predictor)).post("/v1/systemone", json=request())
    assert response.status_code == 400
    assert "task_map" in response.json()["detail"]


def test_single_task_uses_data_task_id_not_recipe_name(predictor):
    predictor.task_ids = {"a"}
    predictor.task_map = {}
    answer = predictor.answer_request(request())
    assert answer["answers"]["first"]["confidence"] > 0.85


def test_missing_task_temperature_reports_kind_fallback():
    result = calibration_for({"temperatures": {"*|choice": 1.7}}, "rare", "choice")
    assert result == {
        "temperature": 1.7,
        "group": "*|choice",
        "fallback_reason": "task_temperature_unavailable",
    }


def test_unimplemented_permutations_are_not_silently_accepted():
    with pytest.raises(ValueError):
        ReadoutConfig(kind="answer_token", infer_permutations=2)


@pytest.mark.parametrize("field,value", [("train_split", "test"), ("train_split", "eval_only")])
def test_invalid_training_role_fails_before_model_loading(tmp_path, field, value, monkeypatch):
    from metiscraft.data.prepare import prepare_dataset
    from metiscraft.pipeline import run_recipe
    from tests.data.fixtures import fake_fetcher

    prepare_dataset("banking77", tmp_path / "data", fetcher=fake_fetcher)
    monkeypatch.setattr("metiscraft.pipeline.init_distributed", lambda: pytest.fail("too late"))
    cfg = Recipe.model_validate(
        {
            "task": "bad",
            "data": {"dataset": "banking77", field: value},
            "model": {"base": "not-loaded"},
            "readout": {"kind": "answer_token"},
            "paths": {"data_root": tmp_path / "data", "runs_root": tmp_path / "runs"},
        }
    )
    with pytest.raises(ValueError, match="train_split"):
        run_recipe(cfg, model_factory=lambda *args: pytest.fail("model was loaded"))
    assert not (tmp_path / "runs").exists()
