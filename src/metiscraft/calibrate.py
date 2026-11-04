"""Temperature scaling per (task, judgment type), fitted on the calib split only (design §8)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import torch

from metiscraft.judgments import batch_loss

MIN_T, MAX_T = 0.05, 20.0


def _stack(preds: Sequence[dict[str, Any]]):
    k = max(len(p["scores"]) for p in preds)
    scores = torch.zeros((len(preds), k), dtype=torch.float64)
    targets = torch.zeros((len(preds), k), dtype=torch.float64)
    mask = torch.zeros((len(preds), k), dtype=torch.bool)
    for i, p in enumerate(preds):
        n = len(p["scores"])
        scores[i, :n] = torch.tensor(p["scores"], dtype=torch.float64)
        targets[i, :n] = torch.tensor(p["target"], dtype=torch.float64)
        mask[i, :n] = True
    return scores, mask, targets


def mean_nll(kind: str, preds: Sequence[dict[str, Any]], temperature: float = 1.0) -> float:
    scores, mask, targets = _stack(preds)
    loss, n = batch_loss(scores / temperature, mask, targets, [kind] * len(preds))
    return float(loss) / n


def fit_temperature(kind: str, preds: Sequence[dict[str, Any]]) -> float:
    scores, mask, targets = _stack(preds)
    kinds = [kind] * len(preds)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss, n = batch_loss(scores / log_t.exp(), mask, targets, kinds)
        loss = loss / n
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.detach().exp().clamp(MIN_T, MAX_T))


def fit_calibration(preds: Sequence[dict[str, Any]], min_rows: int = 20) -> dict[str, Any]:
    """Temperatures keyed ``"<task_id>|<kind>"``, plus a per-kind fallback ``"*|<kind>"``."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in preds:
        if p.get("target") is None:
            continue
        groups[(p["task_id"], p["kind"])].append(p)
        by_kind[p["kind"]].append(p)
    temperatures: dict[str, float] = {}
    counts: dict[str, int] = {}
    for kind, rows in by_kind.items():
        temperatures[f"*|{kind}"] = fit_temperature(kind, rows)
        counts[f"*|{kind}"] = len(rows)
    for (task, kind), rows in groups.items():
        if len(rows) >= min_rows:
            temperatures[f"{task}|{kind}"] = fit_temperature(kind, rows)
            counts[f"{task}|{kind}"] = len(rows)
    return {"method": "temperature", "temperatures": temperatures, "rows": counts}


def temperature_for(calibration: dict[str, Any] | None, task_id: str, kind: str) -> float:
    if not calibration:
        return 1.0
    temps = calibration["temperatures"]
    return temps.get(f"{task_id}|{kind}", temps.get(f"*|{kind}", 1.0))
