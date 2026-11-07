"""Judgment types: targets, losses, probabilities and decisions (design §5).

Scores come as ``[B, K]`` with a boolean mask: noul uses column 0 (a logit), choice and
score use a softmax over the valid columns, multi_label uses an independent sigmoid
per candidate. Losses return a *sum* over questions together with the question count,
so the training loop can normalise by the global number of questions.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from metiscraft.data.contract import Record
from metiscraft.modeling.prompts import NULL_OPTION, choices_of

SOFTMAX_TYPES = ("choice", "score")


def target_vector(record: Record, ordinal_sigma: float | None = None) -> list[float]:
    sup = record.supervision
    if sup is None:
        raise ValueError(f"{record.id} has no supervision")
    q = record.question
    if sup.soft is not None:
        return list(sup.soft)
    if q.type == "noul":
        return [1.0 if sup.labels else 0.0]
    choices = choices_of(record)
    vec = [0.0] * len(choices)
    if q.type == "multi_label":
        positives = set(sup.labels or [])
        return [1.0 if c.id in positives else 0.0 for c in choices]
    if q.type == "choice":
        target_id = NULL_OPTION.id if sup.labels is None else sup.labels
        vec[[c.id for c in choices].index(target_id)] = 1.0
        return vec
    label = int(sup.labels)
    if ordinal_sigma:
        weights = [math.exp(-((i - label) ** 2) / (2 * ordinal_sigma**2)) for i in range(len(vec))]
        total = sum(weights)
        return [w / total for w in weights]
    vec[label] = 1.0
    return vec


def answer_index(record: Record) -> int:
    """The target an answer-token prompt should be completed with: yes/no for noul (codes
    are [yes, no]), otherwise the labelled option or level."""
    sup = record.supervision
    if sup is None or sup.soft is not None:
        raise ValueError(f"{record.id}: completion training needs one hard label")
    if record.question.type == "noul":
        return 0 if sup.labels else 1
    return target_vector(record).index(1.0)


def completion_loss(logits: torch.Tensor, answer_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Next-token cross-entropy over the whole vocabulary on [answer code, EOS].

    ``logits`` [B, 2, V] predict ``answer_ids`` [B, 2]. Every question has two answer tokens,
    so the mean of its two losses summed over questions, divided by the question count,
    equals the per-token mean that supervised fine-tuning reports.
    """
    ce = F.cross_entropy(logits.flatten(0, 1), answer_ids.flatten(), reduction="none")
    return ce.view(-1, 2).mean(-1).sum(), logits.size(0)


def batch_loss(
    scores: torch.Tensor,
    mask: torch.Tensor,
    targets: torch.Tensor,
    kinds: list[str],
    *,
    brier_weight: float = 0.0,
    pos_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    total = scores.new_zeros(())
    for kind in sorted(set(kinds)):
        idx = torch.tensor([i for i, k in enumerate(kinds) if k == kind], device=scores.device)
        s, m, t = scores[idx], mask[idx], targets[idx]
        if kind in SOFTMAX_TYPES:
            logp = torch.log_softmax(s.masked_fill(~m, float("-inf")), dim=-1).masked_fill(~m, 0.0)
            loss = -(t * logp).sum(-1)
            if brier_weight:
                loss = loss + brier_weight * (((logp.exp() - t) ** 2) * m).sum(-1)
            total = total + loss.sum()
        elif kind == "noul":
            total = total + F.binary_cross_entropy_with_logits(s[:, 0], t[:, 0], reduction="sum")
        else:
            per = F.binary_cross_entropy_with_logits(s, t, reduction="none", pos_weight=pos_weight)
            total = total + ((per * m).sum(-1) / m.sum(-1).clamp(min=1)).sum()
    return total, len(kinds)


def prior_shift(train_summary: dict[str, Any] | None) -> dict[str, float]:
    """Per-type logit offsets that undo a positive-class weight used in training.

    BCE with ``pos_weight=w`` multiplies the odds of every positive by ``w``, so the learned
    multi_label logits sit ``ln w`` above the data's own prior scale. Subtracting it before
    calibration puts the probabilities back on that scale (design §7).
    """
    w = (train_summary or {}).get("pos_weight")
    return {"multi_label": math.log(w)} if w and w != 1.0 else {}


def probabilities(kind: str, scores: list[float], temperature: float = 1.0) -> list[float]:
    s = torch.tensor(scores, dtype=torch.float64) / temperature
    if kind == "noul":
        return [float(torch.sigmoid(s[0]))]
    if kind in SOFTMAX_TYPES:
        return torch.softmax(s, dim=-1).tolist()
    return torch.sigmoid(s).tolist()


def decide(kind: str, probs: list[float], choice_ids: list[str]) -> Any:
    if kind == "noul":
        return probs[0] >= 0.5
    if kind == "multi_label":
        return [cid for cid, p in zip(choice_ids, probs, strict=True) if p >= 0.5]
    best = max(range(len(probs)), key=probs.__getitem__)
    if kind == "score":
        return best
    return None if choice_ids[best] == NULL_OPTION.id else choice_ids[best]
