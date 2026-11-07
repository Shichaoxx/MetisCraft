"""Load a base causal LM and its tokenizer, and apply the fine-tuning mode."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from metiscraft.config import FinetuneConfig, ModelConfig, Paths

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def resolve_base(base: str, paths: Paths) -> str:
    """An absolute path, a path under ``models_root``, or (last resort) a Hub id."""
    candidate = Path(base).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    if paths.models_root is not None and (Path(paths.models_root) / base).exists():
        return str(Path(paths.models_root) / base)
    if candidate.exists():
        return str(candidate.resolve())
    return base


def base_identity(path: str) -> dict[str, Any]:
    from metiscraft.modeling.identity import capture_identity

    root = Path(path).expanduser().absolute()
    info: dict[str, Any] = {"path": str(root), "identity": capture_identity(root)}
    source = root / "SOURCE.json"
    if source.is_file():
        info["source"] = json.loads(source.read_text(encoding="utf-8"))
    config = root / "config.json"
    if config.is_file():
        info["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    license_file = next((p for p in (root / "LICENSE", root / "LICENSE.txt") if p.is_file()), None)
    if license_file is not None:
        info["license_file"] = license_file.name
    return info


def apply_finetune(model: nn.Module, ft: FinetuneConfig) -> nn.Module:
    if ft.kind == "lora":
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=ft.r,
            lora_alpha=ft.alpha,
            lora_dropout=ft.dropout,
            target_modules=ft.targets,
            task_type="CAUSAL_LM",
        )
        return get_peft_model(model, config)
    if ft.kind == "frozen":
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_backbone(cfg: ModelConfig, paths: Paths) -> tuple[nn.Module, Any, dict[str, Any]]:
    from transformers import AutoModelForCausalLM

    path = resolve_base(cfg.base, paths)
    identity = base_identity(path)
    tokenizer = load_tokenizer(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=DTYPES[cfg.dtype])
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if cfg.finetune.kind == "lora":
            model.enable_input_require_grads()
    model = apply_finetune(model, cfg.finetune)
    return model, tokenizer, identity
