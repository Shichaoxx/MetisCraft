"""Feedback interface (design §1): a reward with TRL's ``reward_funcs`` shape.

``reward(prompts, completions, **kwargs) -> list[float]``: ``completions`` are Predictor
results and ``kwargs["labels"]`` the observed outcomes. The reward is the negative Brier
score, a proper scoring rule, so it never pays for overstated confidence. v0.1 only
guarantees this shape; reinforcement learning itself is out of scope.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from metiscraft.modeling.prompts import NULL_OPTION


def _brier(result: dict[str, Any], label: Any) -> float:
    kind, probs, ids = result["kind"], result["probabilities"], result["choice_ids"]
    if kind == "noul":
        return (probs[0] - (1.0 if label else 0.0)) ** 2
    if kind == "multi_label":
        positives = set(label or [])
        return sum(
            (p - (1.0 if cid in positives else 0.0)) ** 2 for p, cid in zip(probs, ids, strict=True)
        ) / len(ids)
    gold = int(label) if kind == "score" else ids.index(NULL_OPTION.id if label is None else label)
    return sum((p - (1.0 if i == gold else 0.0)) ** 2 for i, p in enumerate(probs))


def brier_reward() -> Callable[..., list[float]]:
    def reward(
        prompts: Sequence[Any], completions: Sequence[dict[str, Any]], **kwargs: Any
    ) -> list[float]:
        labels = kwargs["labels"]
        if len(completions) != len(labels):
            raise ValueError("completions and labels must have the same length")
        return [-_brier(result, label) for result, label in zip(completions, labels, strict=True)]

    reward.__name__ = "brier_reward"
    return reward
