import copy
import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss

from metiscraft.calibrate import fit_temperature, mean_nll
from metiscraft.config import ObjectiveConfig, ReadoutConfig, TrainConfig
from metiscraft.data.contract import Record
from metiscraft.evaluate.metrics import ece, macro_f1, metrics_for_kind
from metiscraft.judgments import (
    answer_index,
    completion_loss,
    decide,
    prior_shift,
    probabilities,
)
from metiscraft.train.collate import Collator
from metiscraft.train.loop import accumulate, train_loop
from tests.modeling.tiny import (
    choice_row,
    multilabel_row,
    noul_row,
    score_row,
    tiny_decision_model,
)

LIST = {"kind": "candidate_scalar", "layout": "list", "shuffle_train": False}
ANSWER = {"kind": "answer_token"}
CHAT = {**ANSWER, "prompt": {"style": "chat_json", "system": "Pick one."}}


def _rows(n):
    labels = ["billing", "tech", "refund"]
    rows = []
    for i in range(n):
        if i % 3 == 0:
            rows.append(noul_row(i, i % 2 == 0))
        elif i % 3 == 1:
            rows.append(multilabel_row(i))
        else:
            rows.append(choice_row(i, labels[i % 3]))
    return rows


def _grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("split", [(4, 4), (3, 5), (1, 7)])
def test_gradient_accumulation_matches_one_big_batch(split):
    """GAS equivalence (design §12): uneven micro-batches still give the global mean."""
    model, tok = tiny_decision_model("qwen3", LIST)
    other = copy.deepcopy(model)
    collator = Collator(tok, ReadoutConfig(**LIST), 256)
    rows = _rows(8)
    a, b = split
    objective = ObjectiveConfig()
    cpu = torch.device("cpu")
    accumulate(model, [collator(rows[:a]), collator(rows[a:])], objective, device=cpu)
    accumulate(other, [collator(rows)], objective, device=cpu)
    ga, gb = _grads(model), _grads(other)
    assert ga.keys() == gb.keys() and ga
    for name in ga:
        torch.testing.assert_close(ga[name], gb[name], rtol=1e-5, atol=1e-6, msg=name)


def test_train_loop_runs_and_logs(tmp_path):
    model, tok = tiny_decision_model("llama", LIST)
    collator = Collator(tok, ReadoutConfig(**LIST), 256, train=True)
    cfg = TrainConfig(
        batch_size=4, grad_accum=2, max_steps=3, lr=1e-3, head_lr=1e-2, bf16=False, logging_steps=1
    )
    log = tmp_path / "log.jsonl"
    summary = train_loop(
        model, _rows(12), collator, cfg, ObjectiveConfig(), device=torch.device("cpu"), log_path=log
    )
    assert summary["steps"] == 3
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(lines) == 3 and all(np.isfinite(x["loss"]) for x in lines)


@pytest.mark.parametrize("readout", [ANSWER, CHAT])
def test_completion_predicts_the_answer_code_then_eos(readout):
    model, tok = tiny_decision_model("qwen3", readout)
    model.eval()
    rows = [choice_row(0, "tech"), score_row(1)]
    if readout is ANSWER:
        rows.append(noul_row(2, False))  # chat_json takes no noul
    cfg = ReadoutConfig(**readout)
    plain = Collator(tok, cfg, 256)(rows)
    batch = Collator(tok, cfg, 256, train=True, completion=True)(rows)
    gold = [int(plain["codes"][i, answer_index(Record(**r))]) for i, r in enumerate(rows)]
    assert batch["input_ids"][:, -1].tolist() == gold  # the prompt, then its answer code
    assert torch.equal(batch["input_ids"][:, :-1], plain["input_ids"])
    assert batch["answer_ids"].tolist() == [[g, tok.eos_token_id] for g in gold]
    with torch.no_grad():
        logits = model(batch)
        loss, n = completion_loss(logits, batch["answer_ids"])
        full = model.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
        ).logits
        h = model.hidden(plain["input_ids"], plain["attention_mask"], plain["position_ids"])
        read = model.backbone.get_output_embeddings()(h[:, -1])
    target = batch["answer_ids"]
    per_token = F.cross_entropy(full[:, -2], target[:, 0], reduction="none") + F.cross_entropy(
        full[:, -1], target[:, 1], reduction="none"
    )
    assert n == len(rows)
    torch.testing.assert_close(loss, (per_token / 2).sum())
    # the readout position sees only the prompt, so the appended code changes nothing there
    torch.testing.assert_close(logits[:, 0], read, rtol=1e-4, atol=1e-5)


def test_completion_gradient_accumulation_matches_one_big_batch():
    model, tok = tiny_decision_model("qwen3", ANSWER)
    other = copy.deepcopy(model)
    collator = Collator(tok, ReadoutConfig(**ANSWER), 256, train=True, completion=True)
    rows = [choice_row(i, ["billing", "tech", "refund"][i % 3]) for i in range(4)]
    rows += [score_row(4), noul_row(5)]
    objective = ObjectiveConfig(kind="completion")
    cpu = torch.device("cpu")
    accumulate(model, [collator(rows[:1]), collator(rows[1:])], objective, device=cpu)
    accumulate(other, [collator(rows)], objective, device=cpu)
    ga, gb = _grads(model), _grads(other)
    assert ga.keys() == gb.keys() and ga
    for name in ga:
        torch.testing.assert_close(ga[name], gb[name], rtol=1e-5, atol=1e-6, msg=name)


@pytest.mark.parametrize("scheduler", ["linear", "cosine"])
def test_learning_rate_follows_the_schedule(tmp_path, scheduler):
    model, tok = tiny_decision_model("qwen3", ANSWER)
    collator = Collator(tok, ReadoutConfig(**ANSWER), 256, train=True, completion=True)
    cfg = TrainConfig(
        batch_size=2, max_steps=4, lr=1e-3, scheduler=scheduler, bf16=False, logging_steps=1
    )
    log = tmp_path / "log.jsonl"
    rows = [choice_row(i) for i in range(8)]
    objective = ObjectiveConfig(kind="completion")
    train_loop(model, rows, collator, cfg, objective, device=torch.device("cpu"), log_path=log)
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    shape = {"linear": lambda p: 1 - p, "cosine": lambda p: 0.5 * (1 + math.cos(math.pi * p))}
    expected = [1e-3 * shape[scheduler](step / 4) for step in range(1, 5)]
    assert [x["lr"] for x in lines] == pytest.approx(expected)
    assert all(np.isfinite(x["loss"]) for x in lines)


def _choice_preds(n=4000, true_t=2.0, k=4, seed=0):
    rng = np.random.default_rng(seed)
    preds = []
    for _ in range(n):
        logits = rng.normal(0, 3, size=k)
        p = np.exp(logits / true_t) / np.exp(logits / true_t).sum()
        y = int(rng.choice(k, p=p))
        target = [1.0 if i == y else 0.0 for i in range(k)]
        preds.append(
            {"task_id": "t", "kind": "choice", "scores": logits.tolist(), "target": target}
        )
    return preds


def test_temperature_recovers_known_scale_and_keeps_argmax():
    preds = _choice_preds()
    t = fit_temperature("choice", preds)
    assert t == pytest.approx(2.0, rel=0.1)
    assert mean_nll("choice", preds, t) < mean_nll("choice", preds, 1.0)
    for p in preds[:50]:
        a = probabilities("choice", p["scores"], 1.0)
        b = probabilities("choice", p["scores"], t)
        assert int(np.argmax(a)) == int(np.argmax(b))


def test_prior_shift_undoes_pos_weight():
    # Fit one logit with torch's weighted BCE on labels with base rate 0.04: the optimum
    # inflates the odds by w, and subtracting ln w brings the probability back to the rate.
    rate, w = 0.04, 23.0
    y = torch.zeros(1000)
    y[: int(rate * 1000)] = 1.0
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([b], lr=1.0, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            b.expand(1000), y, pos_weight=torch.tensor(w)
        )
        loss.backward()
        return loss

    opt.step(closure)
    shift = prior_shift({"pos_weight": w})
    assert set(shift) == {"multi_label"}
    assert torch.sigmoid(b).item() == pytest.approx(w * rate / (w * rate + 1 - rate), rel=1e-3)
    assert torch.sigmoid(b - shift["multi_label"]).item() == pytest.approx(rate, rel=1e-3)
    assert prior_shift({"pos_weight": None}) == {} and prior_shift(None) == {}


def test_choice_metrics_match_sklearn():
    rng = np.random.default_rng(1)
    ids = ["a", "b", "c"]
    probs = rng.dirichlet(np.ones(3), size=200)
    gold = rng.integers(0, 3, size=200)
    preds = [{"kind": "choice", "choice_ids": ids, "labels": ids[g]} for g in gold]
    m = metrics_for_kind("choice", preds, probs.tolist())
    assert m["accuracy"] == pytest.approx(accuracy_score(gold, probs.argmax(1)))
    assert m["macro_f1"] == pytest.approx(f1_score(gold, probs.argmax(1), average="macro"))
    assert m["nll"] == pytest.approx(log_loss(gold, probs, labels=[0, 1, 2]), rel=1e-6)
    onehot = np.eye(3)[gold]
    assert m["brier"] == pytest.approx(np.mean(((probs - onehot) ** 2).sum(1)))


def test_noul_and_multilabel_metrics_match_sklearn():
    rng = np.random.default_rng(2)
    p = rng.uniform(size=300)
    y = rng.uniform(size=300) < p
    preds = [{"kind": "noul", "labels": bool(v)} for v in y]
    m = metrics_for_kind("noul", preds, [[v] for v in p])
    assert m["brier"] == pytest.approx(brier_score_loss(y, p))
    assert m["nll"] == pytest.approx(log_loss(y, p), rel=1e-6)

    ids = ["x", "y", "z", "w"]
    P = rng.uniform(size=(100, 4))
    Y = rng.uniform(size=(100, 4)) < 0.4
    preds = [
        {"kind": "multi_label", "choice_ids": ids, "labels": [ids[j] for j in range(4) if Y[i, j]]}
        for i in range(100)
    ]
    m = metrics_for_kind("multi_label", preds, P.tolist())
    assert m["micro_f1"] == pytest.approx(f1_score(Y, P >= 0.5, average="micro"))
    assert m["macro_f1"] == pytest.approx(f1_score(Y, P >= 0.5, average="macro", zero_division=0))


def test_ece_and_helpers():
    assert ece(np.array([0.9, 0.9]), np.array([1.0, 0.0])) == pytest.approx(0.4)
    assert ece(np.array([0.25, 0.75]), np.array([0.0, 1.0])) == pytest.approx(0.25)
    assert macro_f1([0, 1, 1], [0, 1, 0]) == pytest.approx(
        f1_score([0, 1, 1], [0, 1, 0], average="macro")
    )
    assert decide("choice", [0.1, 0.9], ["a", "__null__"]) is None
    assert decide("multi_label", [0.6, 0.4], ["a", "b"]) == ["a"]
