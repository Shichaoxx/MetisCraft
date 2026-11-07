"""Focused CPU regressions for legacy routing and path persistence; no model loads."""

from types import SimpleNamespace

import pytest

from metiscraft.modeling.backbone import base_identity
from metiscraft.serve.predictor import Predictor


def construct(monkeypatch, task_map=None):
    monkeypatch.setattr(
        "metiscraft.serve.predictor.Collator",
        lambda *args, **kwargs: SimpleNamespace(encode=lambda row: SimpleNamespace(input_ids=[1])),
    )
    meta = {
        "format": "metiscraft-artifact/1",
        "task": "mixed",
        "model": {"max_length": 64},
        "serve": {"task_map": task_map or {}},
    }
    return Predictor(
        SimpleNamespace(readout=None),
        None,
        meta,
        {"temperatures": {"a|choice": 0.25, "*|choice": 1.5}},
        "cpu",
    )


def test_legacy_task_inventory_cannot_be_inferred_from_temperature_keys(monkeypatch):
    predictor = construct(monkeypatch)
    predictor.predict_records = lambda rows: [
        {
            "kind": "choice",
            "calibration": None,
            "probabilities": [0.8, 0.2],
            "choice_ids": ["yes", "no"],
        }
        for row in rows
    ]
    request = {
        "state": "input",
        "questions": {
            "task_b_question": {"type": "choice", "criteria": {"yes": "Yes", "no": "No"}},
        },
    }
    with pytest.raises(ValueError, match="task_map"):
        predictor.answer_request(request)


def test_explicit_legacy_map_can_declare_task_without_specific_temperature(monkeypatch):
    predictor = construct(monkeypatch, {"task_b_question": "b"})
    assert predictor.task_map["task_b_question"] == "b"


def test_base_identity_persists_absolute_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models" / "tiny").mkdir(parents=True)
    monkeypatch.setattr("metiscraft.modeling.identity.capture_identity", lambda path: {})
    assert base_identity("models/tiny")["path"] == str(tmp_path / "models" / "tiny")
