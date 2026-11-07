"""Run a DecisionModel over records and keep raw scores with aligned targets.

``offsets`` (from :func:`metiscraft.judgments.prior_shift`) are subtracted per judgment type,
so the stored scores sit on the data's own prior scale even when training reweighted classes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    offsets: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
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
            shift = (offsets or {}).get(rec.question.type, 0.0)
            entry: dict[str, Any] = {
                "id": rec.id,
                "task_id": rec.task_id,
                "kind": rec.question.type,
                "scores": (scores[j, : rec.target_size()] - shift).tolist(),
                "choice_ids": [c.id for c in choices_of(rec)],
            }
            if rec.supervision is not None:
                entry["target"] = target_vector(rec)
                entry["labels"] = rec.supervision.labels
                entry["has_hard_label"] = rec.hard_label_status()
                if rec.supervision.soft is not None:
                    entry["soft"] = list(rec.supervision.soft)
            out.append(entry)
    if was_training:
        model.train()
    return out
