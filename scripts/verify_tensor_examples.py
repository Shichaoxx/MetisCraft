#!/usr/bin/env python3
"""Verify the Concepts & Tensors examples on CPU without downloads or test helpers.

验证“原理与张量解析”的算子示例：仅使用本地随机小模型，不下载权重、不衡量任务效果。
Run from a source checkout with its runtime and HTTP test dependencies installed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import transformers  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from metiscraft.calibrate import calibration_for, fit_temperature  # noqa: E402
from metiscraft.config import ReadoutConfig  # noqa: E402
from metiscraft.data.contract import Record  # noqa: E402
from metiscraft.data.jev import request_to_records  # noqa: E402
from metiscraft.judgments import (  # noqa: E402
    batch_loss,
    completion_loss,
    probabilities,
)
from metiscraft.modeling.model import DecisionModel  # noqa: E402
from metiscraft.serve.predictor import Predictor  # noqa: E402
from metiscraft.serve.server import create_app  # noqa: E402
from metiscraft.train.collate import Collator  # noqa: E402

ATOL = 2e-6


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    """A standalone teaching vocabulary; not a tokenizer from an evaluated model.

    独立教学词表；不使用真实实验模型的 tokenizer。
    """
    words = (
        "[PAD] [UNK] [EOS] A B C D yes no 0 1 2 3 "
        "Question Options Levels Answer Candidates Candidate "
        "with the code of one option number level or "
        "customer needs help route request is urgent low medium high "
        "alpha beta gamma delta red green blue gold calm short longer "
        "billing refund technical which select state true false "
        ": - . ?"
    ).split()
    vocab = {word: i for i, word in enumerate(dict.fromkeys(words))}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="[EOS]",
        padding_side="left",
    )


def tiny_model(tok, kind: str, layout: str = "list") -> DecisionModel:
    torch.manual_seed(20261109)
    config = Qwen3Config(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    config._attn_implementation = "eager"
    backbone = Qwen3ForCausalLM(config).to("cpu")
    readout = ReadoutConfig(
        kind=kind, layout=layout, head={"hidden": 16, "dropout": 0.0}, shuffle_train=False
    )
    return DecisionModel(backbone, readout, config.hidden_size).to("cpu").eval()


def record(kind: str, n: int = 3, *, name: str = "example") -> Record:
    items = [{"id": word, "text": word} for word in ("alpha", "beta", "gamma", "delta")[:n]]
    question: dict = {"type": kind}
    extra: dict = {}
    if kind == "choice":
        question["options"] = items
        label = "beta"
    elif kind == "score":
        question["levels"] = items
        label = 1
    elif kind == "multi_label":
        extra["candidates"] = items
        label = ["alpha", "gamma"]
    else:
        label = True
    return Record.model_validate(
        {
            "id": name,
            "task_id": "teaching",
            "input": {"query": "route request", "context": {"state": "customer needs help"}},
            "question": question,
            "supervision": {"kind": kind, "labels": label, "has_hard_label": True},
            **extra,
        }
    )


def close(actual, expected, *, atol: float = ATOL) -> float:
    a = torch.as_tensor(actual, dtype=torch.float64)
    b = torch.as_tensor(expected, dtype=torch.float64)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=atol)
    return float((a - b).abs().max())


def manual_head(model: DecisionModel, hidden: torch.Tensor) -> torch.Tensor:
    first, _, _, last = model.head.net
    middle = F.gelu(F.linear(hidden.float(), first.weight, first.bias))
    return F.linear(middle, last.weight, last.bias).squeeze(-1)


def run_checks(language: str = "en") -> dict:
    torch.set_num_threads(1)
    tok = tiny_tokenizer()
    checks: list[dict] = []

    def passed(check_id, en, zh, **evidence):
        checks.append({"id": check_id, "status": "pass", "en": en, "zh": zh, **evidence})
        print(f"PASS {check_id}: {en if language == 'en' else zh}", flush=True)

    arithmetic = probabilities("choice", [0.0, 1.0, 2.0])
    close(arithmetic, [0.0900305732, 0.2447284711, 0.6652409558], atol=1e-9)
    expected_level = sum(i * p for i, p in enumerate(arithmetic))
    close(expected_level, 1.5752103826, atol=1e-9)
    close(probabilities("noul", [2.0]), [0.8807970780], atol=1e-9)
    close(
        probabilities("choice", [0.0, 1.0, 2.0], 2.0),
        [0.1863237232, 0.3071958857, 0.5064803911],
        atol=1e-9,
    )
    close((0.6 - 1 / 3) / (1 - 1 / 3), 0.4)
    score_example = [0.2, 0.5, 0.3]
    uniform_mad = sum(abs(i - 1) for i in range(3)) / 3
    concentration = max(
        0, 1 - sum(p * abs(i - 1) for i, p in enumerate(score_example)) / uniform_mad
    )
    close(concentration, 0.25)
    close(sum(i * p for i, p in enumerate(score_example)), 1.1)
    passed(
        "arithmetic",
        "Constructed categorical, binary and expected-level arithmetic agrees.",
        "构造的分类、二元概率与等级期望计算一致。",
        probabilities=arithmetic,
        expected_level=expected_level,
    )

    rows = [record("choice", 2), record("choice", 3), record("noul"), record("score", 3)]
    answer = tiny_model(tok, "answer_token")
    col = Collator(tok, answer.readout, 256)
    batch = col(rows)
    with torch.no_grad():
        hidden = answer.hidden(batch["input_ids"], batch["attention_mask"], batch["position_ids"])
        vocab = F.linear(hidden[:, -1], answer.backbone.lm_head.weight).float()
        gathered = vocab.gather(1, batch["codes"])
        gathered[2, 0] -= gathered[2, 1]
        error = close(answer(batch), gathered[:, : batch["mask"].shape[1]])
    assert batch["mask"].tolist() == [
        [True, True, False],
        [True, True, True],
        [True, False, False],
        [True, True, True],
    ]
    assert (batch["attention_mask"] == 0).any()
    assert torch.equal(batch["position_ids"], (batch["attention_mask"].cumsum(-1) - 1).clamp(min=0))
    for row, values in zip(rows, gathered, strict=True):
        p = probabilities(row.question.type, values[: row.target_size()].tolist())
        assert all(math.isfinite(value) for value in p)
        if row.question.type in {"choice", "score"}:
            close(sum(p), 1.0)
    passed(
        "answer-gather-ragged",
        "Vocabulary gather, yes-minus-no and separate ragged masks agree.",
        "词表 gather、yes 减 no 及两类变长掩码计算一致。",
        input_shape=list(batch["input_ids"].shape),
        hidden_shape=list(hidden.shape),
        output_shape=list(gathered[:, :3].shape),
        max_error=error,
    )

    scalar_rows = rows + [record("multi_label", 4)]
    for layout in ("list", "pairs"):
        model = tiny_model(tok, "candidate_scalar", layout)
        collator = Collator(tok, model.readout, 256)
        data = collator(scalar_rows)
        with torch.no_grad():
            h = model.hidden(data["input_ids"], data["attention_mask"], data["position_ids"])
            if layout == "list":
                idx = data["positions"].unsqueeze(-1).expand(-1, -1, h.size(-1))
                manual = manual_head(model, h.gather(1, idx))
            else:
                per_pair = manual_head(model, h[:, -1])
                manual = per_pair.new_zeros(data["mask"].shape)
                manual[data["pair_row"], data["pair_col"]] = per_pair
                assert len(per_pair) == sum(r.target_size() for r in scalar_rows)
            error = close(model(data), manual)
            loss, count = batch_loss(manual, data["mask"], data["targets"], data["kinds"])
        assert count == len(scalar_rows) and torch.isfinite(loss)
        # Invalid columns cannot affect the masked objective.
        poisoned = manual.masked_fill(~data["mask"], 123.0)
        close(batch_loss(poisoned, data["mask"], data["targets"], data["kinds"])[0], loss)
        passed(
            f"scalar-{layout}",
            f"Manual {layout} gather/MLP/scatter and masked four-type loss agree.",
            f"{layout} 的手工 gather／MLP／scatter 及四类型掩码损失一致。",
            input_shape=list(data["input_ids"].shape),
            hidden_shape=list(h.shape),
            score_shape=list(manual.shape),
            max_error=error,
        )

    listed = tiny_model(tok, "candidate_scalar", "list")
    list_col = Collator(tok, listed.readout, 256)
    original = record("choice", 3)
    original.question.options[0].text = "red"
    original.question.options[2].text = "blue"
    later_changed = original.model_copy(deep=True)
    later_changed.question.options[2].text = "gold"
    earlier_changed = original.model_copy(deep=True)
    earlier_changed.question.options[0].text = "green"
    with torch.no_grad():
        base_data = list_col([original])
        later_data = list_col([later_changed])
        earlier_data = list_col([earlier_changed])
        assert base_data["input_ids"].shape == later_data["input_ids"].shape
        assert base_data["input_ids"].shape == earlier_data["input_ids"].shape
        assert torch.equal(base_data["positions"], earlier_data["positions"])
        end_first = int(base_data["positions"][0, 0]) + 1
        assert torch.equal(
            base_data["input_ids"][:, :end_first], later_data["input_ids"][:, :end_first]
        )
        base_h = listed.hidden(
            base_data["input_ids"], base_data["attention_mask"], base_data["position_ids"]
        )
        later_h = listed.hidden(
            later_data["input_ids"], later_data["attention_mask"], later_data["position_ids"]
        )
        earlier_h = listed.hidden(
            earlier_data["input_ids"], earlier_data["attention_mask"], earlier_data["position_ids"]
        )
        close(base_h[:, :end_first], later_h[:, :end_first])
        last = int(base_data["positions"][0, 2])
        influence = float((base_h[:, last] - earlier_h[:, last]).abs().max())
    assert influence > 1e-7
    passed(
        "list-causality",
        "Later candidates preserve a fixed earlier prefix; earlier edits affect later states.",
        "后面的候选不改变固定的前缀；前面候选的变化能够影响后面的表示。",
        later_state_change=influence,
    )

    paired = tiny_model(tok, "candidate_scalar", "pairs")
    pair_col = Collator(tok, paired.readout, 256)
    with torch.no_grad():
        two = paired(pair_col([record("choice", 2)]))[0]
        three = paired(pair_col([record("choice", 3)]))[0]
    error = close(two, three[:2])
    p2, p3 = probabilities("choice", two.tolist()), probabilities("choice", three.tolist())
    assert p3[0] < p2[0] and p3[1] < p2[1]
    passed(
        "pairs-denominator",
        "Adding a pair preserves previous scores but changes categorical normalization.",
        "新增独立候选行保留原分数，但改变分类归一化分母。",
        max_score_error=error,
        two_probabilities=p2,
        three_probabilities=p3,
    )

    completion_col = Collator(tok, answer.readout, 256, completion=True)
    completion_data = completion_col(rows)
    with torch.no_grad():
        logits = answer(completion_data)
        h = answer.hidden(
            completion_data["input_ids"],
            completion_data["attention_mask"],
            completion_data["position_ids"],
        )
        close(logits, answer.backbone.lm_head(h[:, -2:]).float())
        close(logits[:, 0], vocab)
        loss, count = completion_loss(logits, completion_data["answer_ids"])
        manual_loss = (
            F.cross_entropy(
                logits.flatten(0, 1), completion_data["answer_ids"].flatten(), reduction="none"
            )
            .view(-1, 2)
            .mean(-1)
            .sum()
        )
        close(loss, manual_loss)
    assert count == len(rows) and logits.shape == (4, 2, len(tok))
    assert (completion_data["answer_ids"][:, 1] == tok.eos_token_id).all()
    passed(
        "completion",
        "Teacher-forced code/EOS logits and full-vocabulary loss agree without answer leakage.",
        "教师强制的代码／EOS logits 与全词表损失一致，首个预测不读取答案代码。",
        logits_shape=list(logits.shape),
        targets_shape=list(completion_data["answer_ids"].shape),
        question_count=count,
    )

    fitted_rows = []
    for i in range(24):
        scores = [0.0, 0.5 + i / 12, -0.5]
        fitted_rows.append({"scores": scores, "target": probabilities("choice", scores, 2.0)})
    fitted = fit_temperature("choice", fitted_rows)
    assert abs(fitted - 2.0) < 0.02
    calibration = {
        "method": "temperature",
        "temperatures": {
            "routing|choice": 0.5,
            "urgency|noul": 2.0,
            "rating|score": 1.5,
            "*|choice": 3.0,
        },
        "rows": {},
    }
    assert calibration_for(calibration, "unknown", "choice")["group"] == "*|choice"
    assert calibration_for(calibration, "unknown", "score")["fallback_reason"] == (
        "kind_temperature_unavailable"
    )
    passed(
        "temperature",
        "A known synthetic temperature is recovered; task/kind/unit fallbacks are explicit.",
        "拟合能够恢复已知的构造温度，任务／类型／单位温度回退均可检查。",
        constructed_temperature=2.0,
        fitted_temperature=fitted,
    )

    request = {
        "state": "customer needs help",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "route request",
                "criteria": {"alpha": "alpha", "beta": "beta", "gamma": "gamma"},
            },
            "urgent": {"type": "noul", "instructions": "request is urgent"},
            "rating": {
                "type": "score",
                "instructions": "select level",
                "criteria": ["low", "medium", "high"],
            },
        },
    }
    task_map = {"route": "routing", "urgent": "urgency", "rating": "rating"}
    meta = {
        "task": "teaching",
        "model": {"max_length": 256},
        "task_ids": list(task_map.values()),
        "serve": {"task_map": task_map},
    }
    predictor = Predictor(answer, tok, meta, calibration, torch.device("cpu"))
    native_rows = request_to_records(request, task_id="teaching", request_id="request")
    for r in native_rows:
        r.task_id = task_map[r.id.split(":", 1)[1]]
    native = predictor.predict_records(native_rows)

    async def post_request():
        transport = ASGITransport(app=create_app(predictor))
        async with AsyncClient(transport=transport, base_url="http://teaching.local") as client:
            return await client.post("/v1/systemone", json=request)

    response = asyncio.run(post_request())
    assert response.status_code == 200, response.text
    answers = response.json()["answers"]
    for row, result in zip(native_rows, native, strict=True):
        name = row.id.split(":", 1)[1]
        http = answers[name]
        p = result["probabilities"]
        temperature = calibration["temperatures"][f"{row.task_id}|{row.question.type}"]
        corrected = torch.tensor(result["raw_scores"], dtype=torch.float64) / temperature
        expected = corrected.sigmoid() if row.question.type == "noul" else corrected.softmax(0)
        close(p, expected)
        if row.question.type == "noul":
            close(http["noul"], p[0])
        else:
            close(list(http["probabilities"].values()), p)
            close(http["confidence"], max(p))
            if row.question.type == "score":
                close(http["score"], sum(i * value for i, value in enumerate(p)))
        assert result["calibration"]["group"] == f"{row.task_id}|{row.question.type}"
    passed(
        "http-calibration",
        "In-process HTTP and Record inference agree with three distinct task temperatures.",
        "进程内 HTTP 与 Record 推理在三个不同任务温度下输出一致。",
        task_temperatures={"routing": 0.5, "urgency": 2.0, "rating": 1.5},
        status_code=response.status_code,
        probability_only=True,
    )

    return {
        "status": "pass",
        "device": "cpu",
        "seed": 20261109,
        "model": "locally initialized Qwen3ForCausalLM; 2 layers, hidden width 32",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "scope_en": "Operator and interface checks only; no task-quality or speed claim.",
        "scope_zh": "仅验证算子和接口行为；不代表任务效果或性能测量。",
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON report / 可选 JSON 报告路径")
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    args = parser.parse_args()
    report = run_checks(args.language)
    print(f"{len(report['checks'])} checks passed / 项检查通过; CPU; random model / 随机模型")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(args.output)


if __name__ == "__main__":
    main()
