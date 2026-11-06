"""Run directories and sealed artifacts (design §9)."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from metiscraft import __version__
from metiscraft.data.store import IntegrityError, sha256_file, verify_sha256sums, write_sha256sums

ARTIFACT_FORMAT = "metiscraft-artifact/1"
NON_COMMERCIAL_TERMS = {
    "non_commercial",
    "research_only",
    "license_unspecified",
    "teacher_unknown",
    "eval_only_no_train",
}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def checkout_of(module: Path) -> Path | None:
    """The MetisCraft source checkout ``module`` (``src/metiscraft/<file>``) runs from.

    None for an installed package: its parent directories may belong to the user's own
    repository (a virtual environment inside a project), whose commit is not ours.
    """
    package = Path(module).resolve().parent
    root = package.parents[1]
    if package.parent.name != "src" or not (root / "pyproject.toml").is_file():
        return None
    if 'name = "metiscraft"' not in (root / "pyproject.toml").read_text(encoding="utf-8"):
        return None
    top = _git(root, "rev-parse", "--show-toplevel")
    return root if top is not None and Path(top.strip()).resolve() == root else None


def git_info() -> tuple[dict[str, Any], str]:
    root = checkout_of(Path(__file__))
    commit = _git(root, "rev-parse", "HEAD") if root is not None else None
    if commit is None:
        return {"commit": None, "source": "installed package", "version": __version__}, ""
    status = _git(root, "status", "--porcelain") or ""
    diff = _git(root, "diff", "HEAD") or ""
    return {"commit": commit.strip(), "dirty": bool(status.strip())}, diff


def env_info() -> dict[str, Any]:
    import peft
    import transformers

    info: dict[str, Any] = {
        "metiscraft": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "cuda_available": torch.cuda.is_available(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
    }
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def new_run_dir(runs_root: Path, task: str, recipe_sha: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run = Path(runs_root) / task / f"{stamp}-{recipe_sha[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    return run


def commercial_use(terms: list[str]) -> bool:
    return not (set(terms) & NON_COMMERCIAL_TERMS)


def notice_text(meta: dict[str, Any]) -> str:
    base = meta.get("base", {})
    source = base.get("source") or {}
    data = meta.get("dataset", {})
    lines = [
        f"This artifact was trained with MetisCraft {meta.get('metiscraft_version')}.",
        "",
        f"Base model: {source.get('repo') or base.get('path')} "
        f"(revision {source.get('revision', 'unknown')}).",
        "The base model's own licence applies to the base weights and to derived weights.",
        "",
        f"Training data: {data.get('name')}@{data.get('version')}",
        f"  licence: {data.get('license')}",
        f"  terms: {', '.join(data.get('terms') or []) or 'none'}",
        "",
    ]
    if meta.get("commercial_use"):
        lines.append("The training data terms do not restrict commercial use.")
    else:
        lines.append("Commercial use is NOT permitted: the training data terms restrict it.")
    return "\n".join(lines) + "\n"


def save_artifact(model: torch.nn.Module, artifact_dir: Path, meta: dict[str, Any]) -> Path:
    from safetensors.torch import save_file

    artifact_dir.mkdir(parents=True, exist_ok=False)
    backbone = model.backbone
    if hasattr(backbone, "peft_config"):
        backbone.save_pretrained(artifact_dir / "adapter")
    elif meta.get("finetune", {}).get("kind") == "full":
        backbone.save_pretrained(artifact_dir / "model", safe_serialization=True)
    if model.head is not None:
        state = {k: v.detach().cpu().contiguous() for k, v in model.head.state_dict().items()}
        save_file(state, str(artifact_dir / "head.safetensors"))
    meta = {"format": ARTIFACT_FORMAT, **meta}
    write_json(artifact_dir / "artifact.json", meta)
    (artifact_dir / "NOTICE").write_text(notice_text(meta), encoding="utf-8")
    write_sha256sums(artifact_dir)
    return artifact_dir


def artifact_sha(artifact_dir: Path) -> str:
    return sha256_file(artifact_dir / "SHA256SUMS")


def read_artifact_meta(artifact_dir: Path) -> dict[str, Any]:
    verify_sha256sums(artifact_dir)
    meta = json.loads((artifact_dir / "artifact.json").read_text(encoding="utf-8"))
    if meta.get("format") != ARTIFACT_FORMAT:
        raise IntegrityError(f"{artifact_dir}: unknown artifact format {meta.get('format')!r}")
    return meta


def save_calibration(calibration: dict[str, Any], out_dir: Path, parent: Path, split: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=False)
    payload = {**calibration, "fitted_on": split, "parent_artifact_sha256": artifact_sha(parent)}
    write_json(out_dir / "calibration.json", payload)
    write_sha256sums(out_dir)
    return out_dir


def load_calibration(calibration_dir: Path, parent: Path) -> dict[str, Any]:
    verify_sha256sums(calibration_dir)
    payload = json.loads((calibration_dir / "calibration.json").read_text(encoding="utf-8"))
    if payload.get("parent_artifact_sha256") != artifact_sha(parent):
        raise IntegrityError(f"{calibration_dir} was not fitted for {parent}")
    return payload
