import inspect

import pytest

from metiscraft.feedback import brier_reward


def test_reward_has_the_trl_shape():
    reward = brier_reward()
    params = list(inspect.signature(reward).parameters)
    assert params[:2] == ["prompts", "completions"]
    completions = [
        {"kind": "noul", "probabilities": [0.9], "choice_ids": []},
        {"kind": "choice", "probabilities": [0.2, 0.8], "choice_ids": ["a", "b"]},
        {"kind": "score", "probabilities": [0.1, 0.1, 0.8], "choice_ids": ["0", "1", "2"]},
        {"kind": "multi_label", "probabilities": [0.9, 0.2], "choice_ids": ["x", "y"]},
    ]
    out = reward(["p"] * 4, completions, labels=[True, "b", 2, ["x"]])
    assert len(out) == 4 and all(isinstance(v, float) for v in out)
    assert out[0] == pytest.approx(-0.01)
    assert out[1] == pytest.approx(-(0.2**2 + 0.2**2))


def test_confident_mistakes_cost_more():
    reward = brier_reward()
    sure = {"kind": "noul", "probabilities": [0.99], "choice_ids": []}
    unsure = {"kind": "noul", "probabilities": [0.6], "choice_ids": []}
    low, high = reward([None, None], [sure, unsure], labels=[False, False])
    assert low < high
