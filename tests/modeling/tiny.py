"""Tiny offline tokenizer and models (two architectures) for CPU tests."""

from __future__ import annotations

import string
from itertools import product

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from metiscraft.config import ReadoutConfig
from metiscraft.modeling.model import DecisionModel

WORDS = """Input Question Options Answer with the code of one option yes no or Levels number level
Candidates Candidate None above Is it red blue green big small item ticket billing tech refund
low medium high Which team should handle this message card arrival activate my lost stolen
very negative neutral positive review comment admiration anger state fresh Next step How
urgent Needs a human reply escalate if true false number utterance weather timer intent""".split()


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    upper = list(string.ascii_uppercase)
    vocab_words = (
        ["[PAD]", "[UNK]", "[EOS]"]
        + upper
        + list(string.ascii_lowercase)
        + list(string.digits)
        + ["".join(p) for p in product(upper, repeat=2)]
        + [":", "?", ".", ",", "-", "_", "!", "{", "}", '"']
        + WORDS
    )
    vocab = {}
    for w in vocab_words:
        vocab.setdefault(w, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.WordPiece(prefix="##")
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )
    fast.padding_side = "left"
    return fast


def tiny_backbone(arch: str, vocab_size: int, seed: int = 0):
    torch.manual_seed(seed)
    common = dict(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        tie_word_embeddings=False,
    )
    if arch == "qwen3":
        return Qwen3ForCausalLM(Qwen3Config(head_dim=8, **common)).float()
    if arch == "llama":
        return LlamaForCausalLM(LlamaConfig(**common)).float()
    raise ValueError(arch)


def tiny_decision_model(arch: str, readout: dict, tokenizer=None, seed: int = 0):
    tokenizer = tokenizer or tiny_tokenizer()
    backbone = tiny_backbone(arch, len(tokenizer), seed)
    model = DecisionModel(backbone, ReadoutConfig(**readout), backbone.config.hidden_size)
    return model, tokenizer


def choice_row(i: int, label: str = "billing", state: str = "ticket item red") -> dict:
    return {
        "id": f"c{i}",
        "task_id": "t",
        "input": {"query": "Which team should handle this ?", "context": {"state": state}},
        "question": {
            "type": "choice",
            "options": [{"id": "billing"}, {"id": "tech"}, {"id": "refund"}],
        },
        "supervision": {"kind": "choice", "labels": label},
        "provenance": {
            "dataset": "d",
            "converter": "x",
            "converter_version": "1",
            "input_hash": f"h{i}",
        },
    }


def noul_row(i: int, label: bool = True) -> dict:
    return {
        "id": f"n{i}",
        "task_id": "t",
        "input": {"query": "Is it red ?", "context": {"state": "item big red"}},
        "question": {"type": "noul"},
        "supervision": {"kind": "noul", "labels": label},
        "provenance": {
            "dataset": "d",
            "converter": "x",
            "converter_version": "1",
            "input_hash": f"n{i}",
        },
    }


def multilabel_row(i: int) -> dict:
    return {
        "id": f"m{i}",
        "task_id": "t",
        "input": {"query": "Which ?", "context": {"state": "review comment"}},
        "question": {"type": "multi_label"},
        "candidates": [{"id": "admiration"}, {"id": "anger"}, {"id": "neutral"}],
        "supervision": {
            "kind": "multi_label",
            "labels": ["anger"] if i % 2 else ["admiration", "neutral"],
        },
        "provenance": {
            "dataset": "d",
            "converter": "x",
            "converter_version": "1",
            "input_hash": f"m{i}",
        },
    }


def score_row(i: int) -> dict:
    return {
        "id": f"s{i}",
        "task_id": "t",
        "input": {"query": "How urgent ?", "context": {"state": "ticket"}},
        "question": {
            "type": "score",
            "levels": [
                {"id": "0", "text": "low"},
                {"id": "1", "text": "medium"},
                {"id": "2", "text": "high"},
            ],
        },
        "supervision": {"kind": "score", "labels": i % 3},
        "provenance": {
            "dataset": "d",
            "converter": "x",
            "converter_version": "1",
            "input_hash": f"s{i}",
        },
    }
