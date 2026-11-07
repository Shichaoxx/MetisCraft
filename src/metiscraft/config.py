"""Recipe configuration: strict pydantic models, YAML loading, dotted overrides, paths.

A recipe is one YAML file with the sections below. Unknown keys are errors, so a
typo cannot silently fall back to a default. Defaults that affect results live
here, in one place, and every resolved recipe is written into its run directory.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Paths(_Strict):
    data_root: Path | None = None
    models_root: Path | None = None
    runs_root: Path | None = None


class DataConfig(_Strict):
    dataset: str
    version: str | None = None
    train_split: str = "train"
    calib_split: str = "calib"
    dev_split: str = "dev"
    test_split: str = "test"
    max_train: int | None = None
    max_eval: int | None = None


class FinetuneConfig(_Strict):
    kind: Literal["lora", "full", "frozen"] = "lora"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    targets: str | list[str] = "all-linear"


class ModelConfig(_Strict):
    base: str
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    max_length: int = 1024
    finetune: FinetuneConfig = Field(default_factory=FinetuneConfig)
    gradient_checkpointing: bool = False


class HeadConfig(_Strict):
    hidden: int = 256
    dropout: float = 0.0


class PromptConfig(_Strict):
    # plain: MetisCraft's own text layout; chat_json: a JSON payload in the tokenizer's chat
    # template with thinking off and lettered options (answer-token readout only)
    style: Literal["plain", "chat_json"] = "plain"
    system: str | None = None  # chat_json: the system message, if any


class ReadoutConfig(_Strict):
    kind: Literal["answer_token", "candidate_scalar"]
    layout: Literal["list", "pairs"] = "list"
    head: HeadConfig = Field(default_factory=HeadConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    shuffle_train: bool = True
    infer_permutations: Literal[1] = 1

    @model_validator(mode="after")
    def _prompt_fits(self) -> ReadoutConfig:
        if self.prompt.style != "plain" and self.kind != "answer_token":
            raise ValueError(f"prompt.style={self.prompt.style} needs kind=answer_token")
        return self


class ObjectiveConfig(_Strict):
    # completion: next-token cross-entropy over the whole vocabulary on the answer code and
    # the end-of-sequence token, as supervised fine-tuning does (answer-token readout only)
    kind: Literal["auto", "bce", "cross_entropy", "ordinal_soft", "completion"] = "auto"
    brier_weight: float = 0.0
    pos_weight: Literal["none", "auto"] | float = "none"
    ordinal_sigma: float = 1.0


class TrainConfig(_Strict):
    epochs: float = 1.0
    max_steps: int | None = None
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 2e-4
    head_lr: float = 1e-3
    weight_decay: float = 0.0
    scheduler: Literal["linear", "cosine"] = "linear"
    warmup_ratio: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 42
    bf16: bool = True
    logging_steps: int = 10


class CalibrateConfig(_Strict):
    method: Literal["temperature", "none"] = "temperature"
    split: str = "calib"


class EvalConfig(_Strict):
    splits: list[str] = Field(default_factory=lambda: ["dev"])
    baselines: list[str] = Field(default_factory=lambda: ["prior"])


class ServeConfig(_Strict):
    # Request question names map to the same task IDs used during calibration.
    task_map: dict[str, str] = Field(default_factory=dict)


class Recipe(_Strict):
    task: str
    data: DataConfig
    model: ModelConfig
    readout: ReadoutConfig
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    calibrate: CalibrateConfig = Field(default_factory=CalibrateConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)
    paths: Paths = Field(default_factory=Paths)

    @model_validator(mode="after")
    def _objective_fits(self) -> Recipe:
        if self.objective.kind == "completion" and self.readout.kind != "answer_token":
            raise ValueError("objective.kind=completion needs readout.kind=answer_token")
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def canonical_sha256(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def apply_overrides(data: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``a.b=c`` overrides; values are parsed as YAML (``3``, ``true``, ``[1, 2]``)."""
    result = json.loads(json.dumps(data, default=str))
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError(f"override {item!r} is not of the form a.b=value")
        value = yaml.safe_load(raw) if raw else ""
        node = result
        parts = key.split(".")
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"override {item!r}: {part!r} is not a section")
            node = child
        node[parts[-1]] = value
    return result


def load_recipe(path: str | os.PathLike[str], overrides: Iterable[str] = ()) -> Recipe:
    return Recipe.model_validate(apply_overrides(load_yaml(path), overrides))


PATH_ENV = {
    "data_root": "METISCRAFT_DATA_ROOT",
    "models_root": "METISCRAFT_MODELS_ROOT",
    "runs_root": "METISCRAFT_RUNS_ROOT",
}
USER_PATHS_FILE = Path("~/.config/metiscraft/paths.yaml")


def resolve_paths(explicit: Paths | None = None, user_file: Path | None = None) -> Paths:
    """Fill unset paths from ``METISCRAFT_*`` variables, then from the user paths file."""
    resolved = dict((explicit or Paths()).model_dump())
    file_path = (user_file or USER_PATHS_FILE).expanduser()
    from_file = load_yaml(file_path) if file_path.is_file() else {}
    for field, env_name in PATH_ENV.items():
        if resolved.get(field) is None:
            value = os.environ.get(env_name) or from_file.get(field)
            if value:
                resolved[field] = Path(value).expanduser()
    return Paths.model_validate(resolved)


def require_path(paths: Paths, field: str) -> Path:
    value = getattr(paths, field)
    if value is None:
        raise ValueError(
            f"paths.{field} is not set; give it in the recipe, set {PATH_ENV[field]}, "
            f"or add it to {USER_PATHS_FILE}"
        )
    return Path(value)
