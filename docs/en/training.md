# Train, calibrate and evaluate

A recipe selects the dataset version, local base, readout, fine-tuning method,
loss and evaluation splits. `metiscraft train` runs the sequence and records its
configuration and artifacts. Start with the exported [quickstart](quickstart.md)
recipe before adapting it to a real task.

## Prepare a local pretrained base

Training requires a local directory containing weights, configuration, tokenizer,
`SOURCE.json` and `SHA256SUMS`. A Hub model ID alone is not sufficient. Select a
model compatible with your Transformers version, and pin its full repository
commit before downloading it into a new directory.

The following example must be edited with your selected repository and commit.
It downloads weights; it is not part of the offline quickstart.

```python
import json
import re
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from metiscraft.data.store import write_sha256sums
from metiscraft.modeling.identity import capture_identity

repo = "REPLACE_WITH_MODEL_REPOSITORY"
revision = "REPLACE_WITH_FULL_40_CHARACTER_COMMIT"
if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
    raise ValueError("Set revision to the model repository's full commit hash")

destination = Path("models") / "my-base" / revision
if destination.exists():
    raise FileExistsError(f"Use a new model version directory: {destination}")
snapshot_download(repo_id=repo, revision=revision, local_dir=str(destination))
download_metadata = destination / ".cache" / "huggingface"
if download_metadata.is_dir():
    shutil.rmtree(download_metadata)
(destination / "SOURCE.json").write_text(
    json.dumps({"channel": "huggingface", "repo": repo, "revision": revision}, indent=2)
    + "\n",
    encoding="utf-8",
)
write_sha256sums(destination)
print(capture_identity(destination)["sha256"])
```

The example removes only Hugging Face's local download metadata inside the
freshly created destination, before sealing it. Apply that step only to this
new download directory. Do not rerun downloads or remove files inside a sealed
model version; prepare a new version instead.

Locally generated hashes detect later changes; they do not independently prove
that downloaded bytes are authentic upstream releases. Compare against upstream
checksums or attestations when available. Preserve the actual source and license
information, and finish all preparation before sealing a model directory.

The loader checks actual weights, tokenizer and configuration bytes against the
manifest. It also rejects model files omitted from that manifest and symbolic
links. New artifacts bind this content identity: rewriting `SHA256SUMS` after
changing a base does not make it match an existing artifact. Identical files may
be moved to a new directory; use `--base-path` or `Predictor.load(..., base_path=...)`
to select the new location. See [base relocation](serving.md#move-a-base-directory).

## Configure the recipe

For the dataset from [the data guide](data.md), these are the sections to adapt
in the exported recipe; replace the commit placeholder with the same pinned value:

```yaml
task: support
data:
  dataset: support
  version: labeled-v1
model:
  base: my-base/REPLACE_WITH_FULL_40_CHARACTER_COMMIT
  dtype: fp32
  max_length: 512
  finetune:
    kind: lora
    r: 8
    alpha: 16
readout:
  kind: candidate_scalar
train:
  epochs: 1
  batch_size: 4
  lr: 0.00005
  bf16: false
calibrate:
  method: temperature
  split: calib
eval:
  splits: [dev, test]
paths:
  data_root: data
  models_root: models
  runs_root: runs
```

These settings illustrate the interface, not a tuned recipe. When adapting the
quickstart, remove its `train.max_steps: 2` for a full training run. Choose memory,
precision and training settings appropriate to your base and hardware. Relative
paths are resolved from the working directory.

| Readout | Suitable judgments | Mechanism |
|---|---|---|
| `answer_token` | `noul`, `choice`, `score` | Reads option-code logits at the answer position |
| `candidate_scalar` | All four types | Scores candidate representations with a learned head |

The `chat_json` prompt and full-vocabulary `completion` loss require
`answer_token`. Training supports linear and cosine learning-rate schedules.
`model.max_length` bounds the complete model input. Completion training reserves
one token for the appended answer code; EOS is a target only. Plain prompts shorten
the state to fit, while `chat_json` rejects prompts that exceed the remaining budget.
See the [recipe reference](recipe-reference.md) for the full schema.

```bash
metiscraft config show recipe.yaml
metiscraft train recipe.yaml
```

Dotted overrides are accepted, for example `metiscraft train recipe.yaml train.lr=1e-4`.
The model fits only `train`; temperature fits only `calib`; evaluation uses
`dev`, `test` or `eval_only`. Calibration does not guarantee reliable probabilities
after a distribution shift.

Epochs count complete micro-batches, including a rounded-up fractional epoch. The last accumulation step uses the remaining micro-batches rather than borrowing from the next epoch. `max_steps` overrides epochs and counts optimizer updates. The artifact records consumed micro-batches and sample presentations, including any distributed-sampler padding. Input and resolved configurations are saved separately; `recipe_sha256` remains the input identity.

## Inspect the run

| Output | What it records |
|---|---|
| `config.yaml`, `config.sha256` | Resolved recipe and configuration identity |
| `git.json`, optional `git.diff`, `env.json` | Source checkout or package version, changes and environment |
| `data.json`, `model.json` | Dataset version, manifests and base content identity |
| `train_log.jsonl` | Training steps and losses |
| `artifact/` | Adapter or model, readout, metadata, checksums and license notice |
| `calibration/` | Temperatures linked to this artifact |
| `eval/`, `report.md` | Per-question predictions, aggregate metrics and readable report |

Reload the result through [Predictor or the HTTP service](serving.md).

[中文](../zh/training.md)
