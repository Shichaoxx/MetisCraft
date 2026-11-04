"""A small, explicit training loop (design §7).

The loss of one optimizer step is the sum of per-question losses divided by the number
of questions in the whole step, across gradient-accumulation micro-batches and ranks.
Each micro-batch therefore back-propagates ``loss_sum * world_size / n_global``; DDP
averages gradients over ranks, which turns that into the global mean.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from metiscraft.config import ObjectiveConfig, TrainConfig
from metiscraft.judgments import batch_loss
from metiscraft.train.collate import Collator, to_device


def dist_info() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def param_groups(model: torch.nn.Module, cfg: TrainConfig) -> list[dict[str, Any]]:
    head = getattr(model, "head", None)
    head_params = [p for p in head.parameters() if p.requires_grad] if head is not None else []
    head_ids = {id(p) for p in head_params}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": cfg.lr})
    if head_params:
        groups.append({"params": head_params, "lr": cfg.head_lr})
    if not groups:
        raise ValueError("nothing to train: no parameter requires grad")
    return groups


def pos_weight_for(
    rows: Sequence[dict[str, Any]], objective: ObjectiveConfig
) -> torch.Tensor | None:
    if objective.pos_weight == "none":
        return None
    if objective.pos_weight != "auto":
        return torch.tensor(float(objective.pos_weight))
    pos = neg = 0
    for row in rows:
        if row["question"]["type"] != "multi_label":
            continue
        k = len(row.get("candidates") or [])
        p = len(row["supervision"].get("labels") or [])
        pos, neg = pos + p, neg + k - p
    if pos == 0:
        return None
    return torch.tensor(min(max(neg / pos, 1.0), 100.0))


def accumulate(
    net: torch.nn.Module,
    micro: Sequence[dict[str, Any]],
    objective: ObjectiveConfig,
    *,
    device: torch.device,
    world: int = 1,
    use_bf16: bool = False,
    pos_weight: torch.Tensor | None = None,
) -> tuple[float, int]:
    """Forward and backward for one optimizer step; gradients equal those of the global mean."""
    n_local = sum(len(b["kinds"]) for b in micro)
    n_global = n_local
    if world > 1:
        count = torch.tensor(n_local, device=device)
        dist.all_reduce(count)
        n_global = int(count)
    loss_sum_total = 0.0
    for i, batch in enumerate(micro):
        batch = to_device(batch, device)
        sync = nullcontext() if (world == 1 or i == len(micro) - 1) else net.no_sync()
        with sync:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                scores = net(batch)
            loss_sum, _ = batch_loss(
                scores.float(),
                batch["mask"],
                batch["targets"],
                batch["kinds"],
                brier_weight=objective.brier_weight,
                pos_weight=pos_weight,
            )
            (loss_sum * world / n_global).backward()
        loss_sum_total += float(loss_sum.detach())
    return loss_sum_total, n_local


def train_loop(
    model: torch.nn.Module,
    rows: Sequence[dict[str, Any]],
    collator: Collator,
    cfg: TrainConfig,
    objective: ObjectiveConfig,
    *,
    device: torch.device,
    net: torch.nn.Module | None = None,
    log_path: Path | None = None,
) -> dict[str, Any]:
    """Train ``model`` in place; ``net`` is the DDP wrapper when running distributed."""
    rank, world = dist_info()
    net = net or model
    if world > 1:
        sampler = DistributedSampler(
            rows, num_replicas=world, rank=rank, shuffle=True, seed=cfg.seed
        )
    else:
        sampler = RandomSampler(rows, generator=torch.Generator().manual_seed(cfg.seed))
    loader = DataLoader(rows, batch_size=cfg.batch_size, sampler=sampler, collate_fn=collator)
    steps_per_epoch = max(1, math.ceil(len(loader) / cfg.grad_accum))
    total = cfg.max_steps or max(1, math.ceil(steps_per_epoch * cfg.epochs))

    from transformers import get_scheduler

    params = [p for g in param_groups(model, cfg) for p in g["params"]]
    optimizer = torch.optim.AdamW(param_groups(model, cfg), weight_decay=cfg.weight_decay)
    scheduler = get_scheduler(
        "linear",
        optimizer,
        num_warmup_steps=int(cfg.warmup_ratio * total),
        num_training_steps=total,
    )
    pos_weight = pos_weight_for(rows, objective)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    use_bf16 = cfg.bf16 and device.type == "cuda"
    log = open(log_path, "a", encoding="utf-8") if (log_path and rank == 0) else None

    model.train()
    epoch, step = 0, 0
    batches = iter(loader)
    last: dict[str, Any] = {}
    while step < total:
        micro = []
        while len(micro) < cfg.grad_accum:
            try:
                micro.append(next(batches))
            except StopIteration:
                epoch += 1
                if isinstance(sampler, DistributedSampler):
                    sampler.set_epoch(epoch)
                batches = iter(loader)
        loss_sum_total, n_local = accumulate(
            net,
            micro,
            objective,
            device=device,
            world=world,
            use_bf16=use_bf16,
            pos_weight=pos_weight,
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        last = {
            "step": step,
            "epoch": epoch,
            "loss": loss_sum_total / max(n_local, 1),
            "grad_norm": float(grad_norm),
            "lr": scheduler.get_last_lr()[0],
        }
        if log is not None and (step % cfg.logging_steps == 0 or step == total):
            log.write(json.dumps(last) + "\n")
            log.flush()
    if log is not None:
        log.close()
    return {"steps": total, "epochs_seen": epoch + 1, "last": last}
