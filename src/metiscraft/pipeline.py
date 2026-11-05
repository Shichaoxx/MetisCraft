"""One recipe, one run: train -> seal artifact -> calibrate -> evaluate -> report.

Launch with ``metiscraft train`` on one device, or ``torchrun --nproc-per-node N -m
metiscraft train ...`` for DDP; only rank 0 writes the artifact, calibration and report.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml

from metiscraft import __version__
from metiscraft.calibrate import fit_calibration
from metiscraft.config import Recipe, require_path, resolve_paths
from metiscraft.data.contract import Record
from metiscraft.data.store import DatasetVersion, resolve_version, sha256_file, write_jsonl
from metiscraft.evaluate.infer import predict
from metiscraft.evaluate.metrics import evaluate, evaluate_prior, prior_probabilities
from metiscraft.evaluate.report import render_report
from metiscraft.judgments import prior_shift, target_vector
from metiscraft.modeling.backbone import load_backbone
from metiscraft.modeling.model import DecisionModel
from metiscraft.modeling.prompts import choices_of
from metiscraft.trace import (
    commercial_use,
    env_info,
    git_info,
    new_run_dir,
    read_artifact_meta,
    save_artifact,
    save_calibration,
    write_json,
)
from metiscraft.train.collate import Collator
from metiscraft.train.loop import train_loop


def init_distributed() -> tuple[int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank() if world > 1 else 0
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device = torch.device("cuda", local)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return rank, world, device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _shared_run_dir(rank: int, world: int, make) -> Path:
    holder = [str(make()) if rank == 0 else None]
    if world > 1:
        dist.broadcast_object_list(holder, src=0)
    return Path(holder[0])


def _pseudo_preds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        rec = Record.model_validate(row)
        out.append(
            {
                "task_id": rec.task_id,
                "kind": rec.question.type,
                "choice_ids": [c.id for c in choices_of(rec)],
                "target": target_vector(rec),
            }
        )
    return out


def _manifest_hashes(dv: DatasetVersion) -> dict[str, str]:
    return {p.name: sha256_file(p) for p in sorted(dv.manifests_dir.glob("*.json"))}


def build_model(recipe: Recipe, paths) -> tuple[DecisionModel, Any, dict[str, Any]]:
    backbone, tokenizer, base = load_backbone(recipe.model, paths)
    return DecisionModel(backbone, recipe.readout, backbone.config.hidden_size), tokenizer, base


def run_recipe(recipe: Recipe, *, model_factory=build_model) -> Path:
    paths = resolve_paths(recipe.paths)
    data_root = require_path(paths, "data_root")
    runs_root = require_path(paths, "runs_root")
    rank, world, device = init_distributed()

    dv = resolve_version(data_root, recipe.data.dataset, recipe.data.version)
    dv.verify()
    data_meta = dv.meta()
    run = _shared_run_dir(rank, world, lambda: new_run_dir(runs_root, recipe.task, recipe.sha256()))
    if rank == 0:
        resolved = recipe.model_copy(update={"paths": paths})
        (run / "config.yaml").write_text(
            yaml.safe_dump(resolved.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        (run / "config.sha256").write_text(recipe.sha256() + "\n", encoding="utf-8")
        git, diff = git_info()
        write_json(run / "git.json", git)
        if diff:
            (run / "git.diff").write_text(diff, encoding="utf-8")
        write_json(run / "env.json", env_info())
        write_json(
            run / "data.json",
            {
                "name": dv.name,
                "version": dv.version,
                "root": str(dv.root),
                "manifests": _manifest_hashes(dv),
            },
        )

    seed_everything(recipe.train.seed)
    model, tokenizer, base = model_factory(recipe, paths)
    model.to(device)
    if rank == 0:
        write_json(run / "model.json", base)

    ordinal = recipe.objective.ordinal_sigma if recipe.objective.kind == "ordinal_soft" else None
    train_rows = (
        dv.load(recipe.data.train_split, recipe.data.max_train)
        if dv.has_split(recipe.data.train_split)
        else []
    )
    trainable = any(p.requires_grad for p in model.parameters())
    summary: dict[str, Any] = {"steps": 0}
    if train_rows and trainable:
        collator = Collator(
            tokenizer,
            recipe.readout,
            recipe.model.max_length,
            train=True,
            seed=recipe.train.seed,
            ordinal_sigma=ordinal,
            completion=recipe.objective.kind == "completion",
        )
        net = None
        if world > 1:
            net = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[device.index] if device.type == "cuda" else None
            )
        summary = train_loop(
            model,
            train_rows,
            collator,
            recipe.train,
            recipe.objective,
            device=device,
            net=net,
            log_path=run / "train_log.jsonl",
        )

    if rank == 0:
        terms = list(data_meta.get("terms") or [])
        meta = {
            "metiscraft_version": __version__,
            "task": recipe.task,
            "readout": recipe.readout.model_dump(mode="json"),
            "finetune": recipe.model.finetune.model_dump(mode="json"),
            "model": {
                "base": recipe.model.base,
                "dtype": recipe.model.dtype,
                "max_length": recipe.model.max_length,
            },
            "base": base,
            "dataset": {
                "name": dv.name,
                "version": dv.version,
                "license": data_meta.get("license"),
                "terms": terms,
            },
            "recipe_sha256": recipe.sha256(),
            "train": summary,
            "commercial_use": commercial_use(terms),
        }
        artifact = save_artifact(model, run / "artifact", meta)
        evaluate_run(recipe, dv, model, tokenizer, run, artifact, device, train_rows)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return run


def evaluate_run(
    recipe: Recipe,
    dv: DatasetVersion,
    model: DecisionModel,
    tokenizer,
    run: Path,
    artifact: Path,
    device: torch.device,
    train_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    collator = Collator(tokenizer, recipe.readout, recipe.model.max_length, train=False)
    batch = max(1, recipe.train.batch_size)
    offsets = prior_shift(read_artifact_meta(artifact).get("train"))

    def scores_for(split: str) -> list[dict[str, Any]]:
        return predict(
            model,
            dv.load(split, recipe.data.max_eval),
            collator,
            batch_size=batch,
            device=device,
            bf16=recipe.train.bf16,
            offsets=offsets,
        )

    calibration = None
    split = recipe.calibrate.split
    if recipe.calibrate.method == "temperature" and dv.has_split(split):
        calib_preds = scores_for(split)
        calibration = fit_calibration(calib_preds)
        save_calibration(calibration, run / "calibration", artifact, split)
    prior_rows = train_rows or (dv.load(split, recipe.data.max_eval) if dv.has_split(split) else [])
    priors = prior_probabilities(_pseudo_preds(prior_rows))

    results: dict[str, Any] = {}
    for eval_split in recipe.eval.splits:
        if not dv.has_split(eval_split):
            continue
        preds = scores_for(eval_split)
        if not preds:
            continue
        write_jsonl(run / "eval" / f"{eval_split}.predictions.jsonl", preds)
        methods = {"model": evaluate(preds)}
        if calibration is not None:
            methods["model + temperature"] = evaluate(preds, calibration)
        if "prior" in recipe.eval.baselines:
            methods["prior"] = evaluate_prior(preds, priors)
        results[eval_split] = methods
    write_json(run / "eval" / "metrics.json", results)
    header = {
        "task": recipe.task,
        "run": run.name,
        "dataset": f"{dv.name}@{dv.version}",
        "base": recipe.model.base,
        "readout": f"{recipe.readout.kind} ({recipe.readout.layout})",
        "finetune": recipe.model.finetune.kind,
        "recipe_sha256": recipe.sha256(),
        "calibration": None if calibration is None else calibration["temperatures"],
    }
    git = run / "git.json"
    if git.is_file():
        info = json.loads(git.read_text())
        header["git"] = f"{info.get('commit')} ({'dirty' if info.get('dirty') else 'clean'})"
    (run / "report.md").write_text(render_report(header, results), encoding="utf-8")
    return results


def evaluate_artifact(
    artifact: Path,
    dataset: str,
    *,
    version: str | None = None,
    splits: Sequence[str] = ("dev",),
    calibration_dir: Path | None = None,
    max_eval: int | None = None,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Evaluate a sealed artifact on any prepared dataset; temperatures come from its run."""
    from metiscraft.serve.predictor import Predictor

    predictor = Predictor.load(artifact, calibration_dir=calibration_dir)
    dv = resolve_version(require_path(resolve_paths(), "data_root"), dataset, version)
    dv.verify()
    results: dict[str, Any] = {}
    for split in splits:
        preds = predict(
            predictor.model,
            dv.load(split, max_eval),
            predictor.collator,
            batch_size=batch_size,
            device=predictor.device,
            offsets=predictor.offsets,
        )
        methods = {"model": evaluate(preds)}
        if predictor.calibration is not None:
            methods["model + temperature"] = evaluate(preds, predictor.calibration)
        results[split] = methods
    return results
