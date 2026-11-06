"""Create synthetic data and a random CPU model locally; no downloads or GPU use."""

from __future__ import annotations

import json
import string
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from metiscraft.data.store import write_jsonl, write_sha256sums


def main() -> None:
    root = Path.cwd()
    inputs = root / "input"
    base = root / "models" / "tiny-qwen3" / "v1"
    if inputs.exists() or base.exists():
        raise FileExistsError("input/ or models/tiny-qwen3/v1 already exists; use a new project")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    words = (
        ["[PAD]", "[UNK]", "[EOS]"]
        + list(string.ascii_uppercase)
        + [str(i) for i in range(200)]
        + [
            "Input",
            "Question",
            "Options",
            "Answer",
            "with",
            "the",
            "code",
            "of",
            "one",
            "option",
            "Which",
            "team",
            "should",
            "handle",
            "this",
            "message",
            "card",
            "arrival",
            "lost",
            "stolen",
            "billing",
            "tech",
            "Charges",
            "Bugs",
            "case",
            ":",
            "?",
            ".",
            "{",
            "}",
        ]
    )
    vocab = {word: i for i, word in enumerate(dict.fromkeys(words))}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    fast.padding_side = "left"
    config = Qwen3Config(
        vocab_size=len(fast),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    model = Qwen3ForCausalLM(config).float().cpu()
    model.save_pretrained(base)
    fast.save_pretrained(base)
    (base / "SOURCE.json").write_text(
        json.dumps(
            {
                "channel": "synthetic",
                "repo": "metiscraft/quickstart-tiny-qwen3",
                "revision": "random-seed-42-v1",
                "seed": 42,
                "purpose": "CPU engineering smoke test; not a pretrained model",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_sha256sums(base)

    index = 0
    for role, count in (("train", 48), ("calib", 24), ("dev", 12), ("test", 12)):
        rows = []
        for _ in range(count):
            label = "billing" if index % 2 == 0 else "tech"
            state = "card arrival" if label == "billing" else "lost stolen"
            rows.append(
                {
                    "id": f"case-{index}",
                    "task_id": "quickstart",
                    "input": {
                        "query": "Which team should handle this message?",
                        "context": {"state": f"{state} case {index}"},
                    },
                    "question": {
                        "type": "choice",
                        "options": [
                            {"id": "billing", "text": "Charges"},
                            {"id": "tech", "text": "Bugs"},
                        ],
                    },
                    "group": {"id": f"case-{index}"},
                    "supervision": {"kind": "choice", "labels": label},
                }
            )
            index += 1
        write_jsonl(inputs / f"{role}.jsonl", rows)
    print(f"Created synthetic inputs in {inputs}")
    print(f"Created a sealed random fp32 model in {base}")


if __name__ == "__main__":
    main()
