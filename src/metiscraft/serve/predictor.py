"""Load a sealed artifact and answer records or official Jev requests.

Per-type temperatures from the run's calibration are applied here, which is the reason
v0.1 serves through its own Predictor rather than exporting to inference engines.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from metiscraft.calibrate import temperature_for
from metiscraft.config import Paths, ReadoutConfig, resolve_paths
from metiscraft.data.contract import Record
from metiscraft.data.jev import parse_request, request_to_records
from metiscraft.evaluate.infer import predict
from metiscraft.judgments import decide, probabilities
from metiscraft.modeling.backbone import DTYPES, load_tokenizer, resolve_base
from metiscraft.modeling.model import DecisionModel
from metiscraft.trace import load_calibration, read_artifact_meta
from metiscraft.train.collate import Collator

SEMANTICS = {
    "noul": "bernoulli: probability that the answer is yes / the statement is true",
    "choice": "categorical over options; sums to 1",
    "score": "categorical over ordered levels; sums to 1",
    "multi_label": "independent bernoulli per candidate; does not sum to 1",
}


class Predictor:
    def __init__(
        self, model: DecisionModel, tokenizer, meta: dict[str, Any], calibration, device
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.meta = meta
        self.calibration = calibration
        self.device = device
        self.collator = Collator(tokenizer, model.readout, meta["model"]["max_length"], train=False)

    @property
    def name(self) -> str:
        return f"metiscraft/{self.meta['task']}"

    @classmethod
    def load(
        cls,
        artifact_dir: str | Path,
        *,
        calibration_dir: str | Path | None = None,
        paths: Paths | None = None,
        device: torch.device | None = None,
    ) -> Predictor:
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM

        artifact = Path(artifact_dir)
        meta = read_artifact_meta(artifact)
        base_path = meta.get("base", {}).get("path")
        if not base_path or not Path(base_path).exists():
            base_path = resolve_base(meta["model"]["base"], resolve_paths(paths))
        dtype = DTYPES[meta["model"]["dtype"]]
        tokenizer = load_tokenizer(base_path)
        if (artifact / "model").is_dir():
            backbone = AutoModelForCausalLM.from_pretrained(artifact / "model", dtype=dtype)
        else:
            backbone = AutoModelForCausalLM.from_pretrained(base_path, dtype=dtype)
            if (artifact / "adapter").is_dir():
                from peft import PeftModel

                backbone = PeftModel.from_pretrained(backbone, artifact / "adapter")
        readout = ReadoutConfig(**meta["readout"])
        model = DecisionModel(backbone, readout, backbone.config.hidden_size)
        if model.head is not None:
            model.head.load_state_dict(load_file(str(artifact / "head.safetensors")))
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        if calibration_dir is None and (artifact.parent / "calibration").is_dir():
            calibration_dir = artifact.parent / "calibration"
        calibration = load_calibration(Path(calibration_dir), artifact) if calibration_dir else None
        return cls(model, tokenizer, meta, calibration, device)

    def predict_records(
        self, records: Sequence[Record | dict[str, Any]], batch_size: int = 8
    ) -> list[dict[str, Any]]:
        rows = [r if isinstance(r, Record) else Record.model_validate(r) for r in records]
        raw = predict(self.model, rows, self.collator, batch_size=batch_size, device=self.device)
        out = []
        for row, pred in zip(rows, raw, strict=True):
            kind = pred["kind"]
            t = temperature_for(self.calibration, pred["task_id"], kind)
            probs = probabilities(kind, pred["scores"], t)
            top = max(probs) if kind != "noul" else max(probs[0], 1 - probs[0])
            out.append(
                {
                    "id": row.id,
                    "task_id": row.task_id,
                    "kind": kind,
                    "choice_ids": pred["choice_ids"],
                    "raw_scores": pred["scores"],
                    "probabilities": probs,
                    "probability_semantics": SEMANTICS[kind],
                    "calibration": None
                    if self.calibration is None
                    else {"method": "temperature", "temperature": t},
                    "decision": decide(kind, probs, pred["choice_ids"]),
                    "abstained": False,
                    "uncertainty": 1.0 - top,
                }
            )
        return out

    def answer_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Answer an official ``/v1/systemone`` request in the official response shape."""
        _, questions, _ = parse_request(request)
        records = request_to_records(request, task_id=self.meta["task"], request_id="request")
        results = self.predict_records(records)
        answers: dict[str, Any] = {}
        for record, res in zip(records, results, strict=True):
            name = record.id.split(":", 1)[1]
            probs, ids = res["probabilities"], res["choice_ids"]
            if res["kind"] == "noul":
                answers[name] = {"type": "noul", "noul": probs[0]}
            elif res["kind"] == "choice":
                best = max(range(len(probs)), key=probs.__getitem__)
                answers[name] = {
                    "type": "choice",
                    "choice": ids[best],
                    "confidence": probs[best],
                    "probabilities": dict(zip(ids, probs, strict=True)),
                }
            else:
                levels = questions[name].levels or []
                answers[name] = {
                    "type": "score",
                    "score": sum(i * p for i, p in enumerate(probs)),
                    "confidence": max(probs),
                    "legend": {lv.id: lv.text or "" for lv in levels},
                    "probabilities": dict(zip(ids, probs, strict=True)),
                }
        input_tokens = sum(len(self.collator.encode(r).input_ids) or 0 for r in records)
        return {
            "model": self.name,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }
