"""Metrics per judgment type (design §5, §8). One place defines every metric."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from metiscraft.calibrate import temperature_for
from metiscraft.judgments import probabilities
from metiscraft.modeling.prompts import NULL_OPTION

ECE_BINS = 15
ECE_NOTE = (
    "ECE uses 15 equal-width bins; confidence is the top-1 probability for choice and score, "
    "and the positive-class probability for noul and for each multi_label candidate."
)
EPS = 1e-12


def ece(confidence: np.ndarray, outcome: np.ndarray, bins: int = ECE_BINS) -> float:
    confidence = np.asarray(confidence, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if confidence.size == 0:
        return float("nan")
    index = np.minimum((confidence * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        sel = index == b
        if sel.any():
            total += abs(outcome[sel].mean() - confidence[sel].mean()) * sel.sum()
    return float(total / confidence.size)


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos, neg = labels.sum(), (~labels).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _f1(tp: float, fp: float, fn: float) -> float:
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def macro_f1(gold: Sequence[int], pred: Sequence[int]) -> float:
    """Macro-F1 over the classes seen in gold or predictions (sklearn's default labels)."""
    gold_a, pred_a = np.asarray(gold), np.asarray(pred)
    classes = np.union1d(gold_a, pred_a)
    scores = []
    for c in classes:
        tp = np.sum((gold_a == c) & (pred_a == c))
        fp = np.sum((gold_a != c) & (pred_a == c))
        fn = np.sum((gold_a == c) & (pred_a != c))
        scores.append(_f1(tp, fp, fn))
    return float(np.mean(scores))


def _gold_index(pred: dict[str, Any]) -> int:
    ids = pred["choice_ids"]
    if pred["kind"] == "score":
        return int(pred["labels"])
    return ids.index(NULL_OPTION.id if pred["labels"] is None else pred["labels"])


def metrics_for_kind(
    kind: str, preds: Sequence[dict[str, Any]], probs: Sequence[Sequence[float]]
) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(preds)}
    if not preds:
        return out
    if kind == "noul":
        p = np.clip(np.array([q[0] for q in probs]), EPS, 1 - EPS)
        y = np.array([1.0 if r["labels"] else 0.0 for r in preds])
        out.update(
            accuracy=float(np.mean((p >= 0.5) == (y == 1))),
            auc=binary_auc(p, y),
            nll=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            brier=float(np.mean((p - y) ** 2)),
            ece=ece(p, y),
        )
        soft = [r["soft"][0] for r in preds if "soft" in r]
        if len(soft) == len(preds):
            out["brier_soft"] = float(np.mean((p - np.array(soft)) ** 2))
        return out

    if kind in ("choice", "score"):
        gold = [_gold_index(r) for r in preds]
        top = [int(np.argmax(q)) for q in probs]
        conf = np.array([max(q) for q in probs])
        correct = np.array([g == t for g, t in zip(gold, top, strict=True)], dtype=float)
        p_gold = np.clip(np.array([q[g] for q, g in zip(probs, gold, strict=True)]), EPS, 1.0)
        brier = [
            sum((pi - (1.0 if i == g else 0.0)) ** 2 for i, pi in enumerate(q))
            for q, g in zip(probs, gold, strict=True)
        ]
        out.update(
            accuracy=float(correct.mean()),
            nll=float(-np.mean(np.log(p_gold))),
            brier=float(np.mean(brier)),
            ece=ece(conf, correct),
        )
        if len({tuple(r["choice_ids"]) for r in preds}) == 1:
            out["macro_f1"] = macro_f1(gold, top)
        if kind == "score":
            expected = [sum(i * pi for i, pi in enumerate(q)) for q in probs]
            out["mae_expected"] = float(np.mean(np.abs(np.array(expected) - np.array(gold))))
        if all("soft" in r for r in preds):
            out["brier_soft"] = float(
                np.mean(
                    [
                        sum((a - b) ** 2 for a, b in zip(q, r["soft"], strict=True))
                        for q, r in zip(probs, preds, strict=True)
                    ]
                )
            )
        return out

    p_all, y_all, exact = [], [], []
    shared = len({tuple(r["choice_ids"]) for r in preds}) == 1
    for q, r in zip(probs, preds, strict=True):
        positives = set(r["labels"] or [])
        y = [1.0 if cid in positives else 0.0 for cid in r["choice_ids"]]
        p_all.append(np.array(q))
        y_all.append(np.array(y))
        exact.append(all((pi >= 0.5) == (yi == 1) for pi, yi in zip(q, y, strict=True)))
    p_flat = np.clip(np.concatenate(p_all), EPS, 1 - EPS)
    y_flat = np.concatenate(y_all)
    hat = p_flat >= 0.5
    tp = float(np.sum(hat & (y_flat == 1)))
    fp = float(np.sum(hat & (y_flat == 0)))
    fn = float(np.sum(~hat & (y_flat == 1)))
    out.update(
        exact_match=float(np.mean(exact)),
        micro_f1=_f1(tp, fp, fn),
        nll=float(-np.mean(y_flat * np.log(p_flat) + (1 - y_flat) * np.log(1 - p_flat))),
        brier=float(np.mean((p_flat - y_flat) ** 2)),
        ece=ece(p_flat, y_flat),
    )
    if shared:
        P, Y = np.stack(p_all) >= 0.5, np.stack(y_all) == 1
        out["macro_f1"] = float(
            np.mean(
                [
                    _f1(
                        np.sum(P[:, j] & Y[:, j]),
                        np.sum(P[:, j] & ~Y[:, j]),
                        np.sum(~P[:, j] & Y[:, j]),
                    )
                    for j in range(P.shape[1])
                ]
            )
        )
    return out


def evaluate(
    preds: Sequence[dict[str, Any]], calibration: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Metrics per judgment type; probabilities use the calibration's temperatures if given."""
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in preds:
        if "labels" in p:
            by_kind[p["kind"]].append(p)
    result = {}
    for kind, rows in sorted(by_kind.items()):
        probs = [
            probabilities(kind, r["scores"], temperature_for(calibration, r["task_id"], kind))
            for r in rows
        ]
        result[kind] = metrics_for_kind(kind, rows, probs)
    return result


def prior_probabilities(train_preds: Sequence[dict[str, Any]]) -> dict[tuple, list[float]]:
    """Mean training target per (task, kind, option set): the no-information baseline."""
    sums: dict[tuple, np.ndarray] = {}
    counts: dict[tuple, int] = defaultdict(int)
    for p in train_preds:
        key = (p["task_id"], p["kind"], tuple(p["choice_ids"]))
        vec = np.array(p["target"], dtype=float)
        sums[key] = sums.get(key, np.zeros_like(vec)) + vec
        counts[key] += 1
    return {k: (sums[k] / counts[k]).tolist() for k in sums}


def evaluate_prior(
    preds: Sequence[dict[str, Any]], priors: dict[tuple, list[float]]
) -> dict[str, Any]:
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in preds:
        if "labels" in p:
            by_kind[p["kind"]].append(p)
    result = {}
    for kind, rows in sorted(by_kind.items()):
        probs = []
        for r in rows:
            k = max(1, len(r["choice_ids"]))
            default = (
                [0.5] if kind == "noul" else ([1.0 / k] * k if kind != "multi_label" else [0.5] * k)
            )
            probs.append(priors.get((r["task_id"], kind, tuple(r["choice_ids"])), default))
        result[kind] = metrics_for_kind(kind, rows, probs)
    return result
