# Recipe field reference

Recipes are YAML documents matching `metiscraft.config.Recipe`; unknown fields are rejected. Override fields with `a.b=c`, for example `metiscraft train <recipe> readout.kind=answer_token train.lr=1e-4`. Path precedence is recipe `paths` > environment variables such as `METISCRAFT_DATA_ROOT` > `~/.config/metiscraft/paths.yaml`.

Generated from the same schema as the Chinese reference by [`recipe_reference.py`](../cookbook/src/recipe_reference.py). Types and defaults come from the code. See [your own task](cookbook/own-task.md) for a complete example.

## Task

| Field | Type | Default | Description |
|---|---|---|---|
| `task` | str | required | Run/model label recorded in recipe metadata; it does not override `Record.task_id`. Artifact `task_ids` come from the training records, and temperature groups use each record's task and judgment type. |

## Data

| Field | Type | Default | Description |
|---|---|---|---|
| `data.dataset` | str | required | Name of the prepared dataset, including a local manifest import. |
| `data.version` | str \| `null` | `null` | Dataset version directory; optional when only one version exists. |
| `data.train_split` | str | `train` | Split used for training. |
| `data.calib_split` | str | `calib` | Independent split used to fit temperature. |
| `data.dev_split` | str | `dev` | Evaluation split used for model selection. |
| `data.test_split` | str | `test` | Test split evaluated after the artifact is sealed. |
| `data.max_train` | Annotated \| `null` | `null` | Optional training-record limit for debugging; null uses the whole split. |
| `data.max_eval` | Annotated \| `null` | `null` | Optional record limit applied to each evaluation split. |

## Base and fine-tuning

| Field | Type | Default | Description |
|---|---|---|---|
| `model.base` | str | required | Base directory relative to `models_root`; requires `SOURCE.json` and `SHA256SUMS`. |
| `model.dtype` | `bf16` \| `fp16` \| `fp32` | `bf16` | Precision used to load model weights. |
| `model.max_length` | int | `1024` | Maximum input tokens. Completion training reserves one token for the answer code; EOS is a target only. The plain prompt truncates state while preserving the question and candidates. `chat_json` rejects oversized inputs instead of truncating. |
| `model.finetune.kind` | `lora` \| `full` \| `frozen` | `lora` | `lora` trains adapters; `full` tunes the base; `frozen` freezes it. A frozen scalar readout can train its new MLP head; a frozen answer-token readout has no new head and skips training. |
| `model.finetune.r` | int | `16` | LoRA rank. |
| `model.finetune.alpha` | int | `32` | LoRA scaling factor. |
| `model.finetune.dropout` | float | `0.0` | LoRA dropout probability. |
| `model.finetune.targets` | str \| list[str] | `all-linear` | LoRA target modules: `all-linear` or a list of module names. |
| `model.gradient_checkpointing` | bool | `False` | Recompute activations to reduce training memory. |

## Readout

| Field | Type | Default | Description |
|---|---|---|---|
| `readout.kind` | `answer_token` \| `candidate_scalar` | required | `answer_token` gathers answer-code logits; `candidate_scalar` applies a scalar MLP to candidate representations. |
| `readout.layout` | `list` \| `pairs` | `list` | Scalar layout: `list` (default) places all candidates in one causal sequence; `pairs` places each candidate in a separate sequence. |
| `readout.head.hidden` | int | `256` | Scalar MLP hidden width. |
| `readout.head.dropout` | float | `0.0` | Scalar MLP dropout probability. |
| `readout.prompt.style` | `plain` \| `chat_json` | `plain` | Answer-token prompt: `plain` uses the text layout; `chat_json` puts JSON in the tokenizer chat template with thinking disabled and letter-coded choices or levels, as in the Tev1 recipe. `chat_json` accepts only choice and score questions. |
| `readout.prompt.system` | str \| `null` | `null` | Optional system message for `chat_json`; null omits that turn. |
| `readout.shuffle_train` | bool | `True` | Shuffle the scalar `list` prompt's candidate order during training to reduce position bias. It does not affect `pairs`. |
| `readout.infer_permutations` | `1` | `1` | Only 1 is implemented. Other values fail validation rather than silently doing nothing. |

## Objective

| Field | Type | Default | Description |
|---|---|---|---|
| `objective.kind` | `auto` \| `bce` \| `cross_entropy` \| `ordinal_soft` \| `completion` | `auto` | The judgment type determines the loss; `auto`, `bce` and `cross_entropy` are equivalent annotations in v0.1. `ordinal_soft` uses a Gaussian soft target around the gold level. Answer-token-only `completion` predicts the answer code and EOS over the full vocabulary. |
| `objective.brier_weight` | float | `0.0` | Additional Brier-loss weight for choice and score objectives. |
| `objective.pos_weight` | `none` \| `auto` \| Annotated | `none` | Positive multi-label weight: `none`, `auto` (training negatives/positives clipped to 1–100), or a number. Inference subtracts ln(weight) from those logits. |
| `objective.ordinal_sigma` | float | `1.0` | Gaussian width of ordinal soft targets, in level units. |

## Training

| Field | Type | Default | Description |
|---|---|---|---|
| `train.epochs` | float | `1.0` | Training epochs; fractional values are accepted. |
| `train.max_steps` | Annotated \| `null` | `null` | Maximum optimizer steps; overrides `epochs` when provided. |
| `train.batch_size` | int | `8` | Questions per microbatch per device. |
| `train.grad_accum` | int | `1` | Gradient accumulation microbatches. Effective batch size is `batch_size × grad_accum × devices`; loss is normalized by the global question count. |
| `train.lr` | float | `0.0002` | Learning rate for LoRA or full-base parameters. |
| `train.head_lr` | float | `0.001` | Learning rate for the readout head. |
| `train.weight_decay` | float | `0.0` | AdamW weight decay. |
| `train.scheduler` | `linear` \| `cosine` | `linear` | `linear` or `cosine` decay to zero after warmup. |
| `train.warmup_ratio` | float | `0.0` | Warmup steps as a fraction of total optimizer steps. |
| `train.max_grad_norm` | float | `1.0` | Gradient-clipping norm threshold. |
| `train.seed` | int | `42` | Seed for initialization and data ordering. |
| `train.bf16` | bool | `True` | Use bfloat16 autocast on CUDA. |
| `train.logging_steps` | int | `10` | Optimizer-step interval between `train_log.jsonl` entries. |

## Calibration

| Field | Type | Default | Description |
|---|---|---|---|
| `calibrate.method` | `temperature` \| `none` | `temperature` | `temperature` fits task/type temperatures; `none` disables fitting. |
| `calibrate.split` | str | `calib` | Independent split used for temperature fitting. |

## Evaluation

| Field | Type | Default | Description |
|---|---|---|---|
| `eval.splits` | list[str] | `['dev']` | Splits to evaluate. |
| `eval.baselines` | list[str] | `['prior']` | Baselines reported alongside model metrics; v0.1 supports `prior`. |

## Serving

| Field | Type | Default | Description |
|---|---|---|---|
| `serve.task_map` | dict | `{}` | HTTP question-name to artifact `task_id` mapping. Multi-task artifacts need an explicit mapping; single-task artifacts can select automatically. |

## Paths

| Field | Type | Default | Description |
|---|---|---|---|
| `paths.data_root` | Path \| `null` | `null` | Prepared-dataset root directory. |
| `paths.models_root` | Path \| `null` | `null` | Versioned base-model root directory. |
| `paths.runs_root` | Path \| `null` | `null` | Training and evaluation output root directory. |
