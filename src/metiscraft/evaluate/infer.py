"""Run a DecisionModel over records and keep raw scores with aligned targets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from metiscraft.data.contract import Record
from metiscraft.judgments import target_vector
from metiscraft.modeling.prompts import choices_of
from metiscraft.train.collate import Collator, to_device


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    rows: Sequence[dict[str, Any]],
    collator: Collator,
    *,
    batch_size: int,
    device: torch.device,
    bf16: bool = True,
) -> list[dict[str, Any]]:
    was_training = model.training
    model.eval()
    use_bf16 = bf16 and device.type == "cuda"
    out: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        chunk = [Record.model_validate(r) for r in rows[start : start + batch_size]]
        batch = collator(chunk)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            scores = model(to_device(batch, device))
        scores = scores.float().cpu()
        for j, rec in enumerate(chunk):
            entry: dict[str, Any] = {
                "id": rec.id,
                "task_id": rec.task_id,
                "kind": rec.question.type,
                "scores": scores[j, : rec.target_size()].tolist(),
                "choice_ids": [c.id for c in choices_of(rec)],
            }
            if rec.supervision is not None:
                entry["target"] = target_vector(rec)
                entry["labels"] = rec.supervision.labels
                if rec.supervision.soft is not None:
                    entry["soft"] = list(rec.supervision.soft)
            out.append(entry)
    if was_training:
        model.train()
    return out
