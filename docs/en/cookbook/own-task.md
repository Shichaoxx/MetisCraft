# Bring your own task

**Goal:** turn your labeled JSONL into a versioned dataset, train with a prepared pretrained base, and call the resulting artifact. This guide uses a two-way support-routing question; replace its labels and definitions with the task you actually want to measure.

## Before you start

You need an installed MetisCraft environment, labeled examples you are entitled to use, and hardware suitable for your chosen base model. The HTTP step also needs the `serve` extra. If you have not used the commands before, complete the [CPU workflow](first-model.md).

Create a new workspace from the installed starter:

```bash
metiscraft init my-support
cd my-support
```

The starter's `prepare.py` belongs to the synthetic CPU example. For this guide, supply your own `input/` files and prepared model, then replace `dataset.yaml` and `recipe.yaml` with the versions below.

## 1. Write one supervised question per line

Choose stable answer IDs and define their boundaries. This tutorial uses `billing` for charges, invoices and refunds, and `tech` for technical faults. Write Record JSONL: one complete JSON object per line.

This row illustrates the format; it is not a training dataset:

```json
{"id":"ticket-1001","task_id":"routing","input":{"query":"Which team should handle this message?","context":{"state":"I was charged twice for this month's plan."}},"question":{"type":"choice","options":[{"id":"billing","text":"Charges, invoices and refunds."},{"id":"tech","text":"Login failures, errors and technical faults."}]},"group":{"id":"conversation-1001","split_key":"conversation-1001"},"supervision":{"kind":"choice","labels":"billing"}}
```

- `input.query` is the question; `input.context.state` is the material to judge.
- `question.options` defines answer IDs and descriptions. `supervision.labels` is the correct ID.
- `task_id: routing` identifies this task for prediction and calibration.
- `group` keeps questions from the same conversation together. Related records need different record IDs but a shared group.

Prepare four files from your own examples:

| File | Use |
|---|---|
| `input/train.jsonl` | Fit model parameters |
| `input/calib.jsonl` | Fit temperatures after training |
| `input/dev.jsonl` | Compare settings and inspect errors |
| `input/test.jsonl` | Evaluate the selected setup on held-out examples |

Keep record IDs unique across all splits. Do not share inputs, conversations or groups across training, calibration and evaluation. Use representative labels and class coverage in each split; one example per file is not a useful evaluation.

For yes/no, ordered levels or multiple labels, follow the [Record format reference](../data.md#write-record-jsonl).

## 2. Describe and seal the dataset

Save this as `dataset.yaml`. Replace the example source, license and terms with the ones that apply to your data.

```yaml
name: support
version: labeled-v1
source: your-organization/support-labels
license: LicenseRef-Internal-Data
terms: [research_only]
splits:
  train: input/train.jsonl
  calib: input/calib.jsonl
  dev: input/dev.jsonl
  test: input/test.jsonl
```

Each split path is relative to this manifest and must remain inside its directory. Import does not need a built-in registry entry or the `reproduce` extra.

```bash
metiscraft data import dataset.yaml --data-root ./data
metiscraft data verify support --version labeled-v1 --data-root ./data
metiscraft data show support --version labeled-v1 --split train -n 2 --data-root ./data
```

You should now have `data/support/labeled-v1/`, with copied inputs, converted records, purpose manifests, `dataset.yaml` and `SHA256SUMS`. The importer adds source provenance and an input hash.

An unchanged import verifies the existing version. If you change a label, input or manifest, choose a new version instead of editing sealed files.

## 3. Prepare a pretrained base

An available starting point is **Qwen/Qwen3-0.6B-Base**, pinned to `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. This base was used in the recorded Banking77-derived example. That verifies a toolkit/base combination, not the quality of a new task.

Run the following Python code from the workspace to prepare a fresh local version. It downloads model files into the configured Hub cache, copies their contents into the working directory, then records the source and checksums. It does not load weights directly from the cache.

```python
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from metiscraft.data.store import write_sha256sums
from metiscraft.modeling.identity import capture_identity

repo = "Qwen/Qwen3-0.6B-Base"
revision = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
destination = Path("models") / "qwen3-0.6b-base" / revision
if destination.exists():
    raise FileExistsError(f"Use a new model version directory: {destination}")

snapshot = snapshot_download(repo_id=repo, revision=revision)
shutil.copytree(snapshot, destination, symlinks=False)
(destination / "SOURCE.json").write_text(
    json.dumps({"channel": "huggingface", "repo": repo, "revision": revision}, indent=2)
    + "\n",
    encoding="utf-8",
)
write_sha256sums(destination)
print(capture_identity(destination)["sha256"])
```

The resulting directory is `models/qwen3-0.6b-base/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/`. Keep its weights, tokenizer, configuration, `SOURCE.json` and `SHA256SUMS` together. Review the base model's license before use. For another model, follow [base preparation](../training.md#prepare-a-local-pretrained-base) and select a compatible pinned revision.

The recipe below uses fp32 on CPU for an explicit starting configuration. Training this pretrained model is substantially more work than the tiny CPU tutorial; available memory and the size of your data determine feasibility.

## 4. Connect the data, model and HTTP question

Replace `recipe.yaml` with the following recipe. Its base path matches the directory just prepared:

```yaml
task: support
data:
  dataset: support
  version: labeled-v1
model:
  base: qwen3-0.6b-base/da87bfb608c14b7cf20ba1ce41287e8de496c0cd
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
  baselines: [prior]
serve:
  task_map:
    team: routing
paths:
  data_root: data
  models_root: models
  runs_root: runs
```

This is a starting configuration, not a tuned result. It uses a complete training epoch rather than the starter's two-step check. Adjust batch size, sequence length and precision for your model and data.

For a compatible GPU, choose the device through your usual environment and override `model.dtype=bf16 train.bf16=true` when supported. Omit the CPU-only environment prefix shown below. The recorded customer-routing run's hardware and runtime describe that run's recipe and data, not an estimate for yours.

The names have separate roles: `support` names the dataset and run; `routing` is the records' task ID; `team` will be the HTTP question name. `serve.task_map` connects `team` to `routing` so serving selects the same task as training and calibration.

## 5. Train and inspect held-out results

Run these commands from the workspace root; recipe paths are relative to the working directory.

```bash
metiscraft config show recipe.yaml
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 metiscraft train recipe.yaml
```

The first command prints the resolved configuration. The second trains on `train`, fits temperature on `calib`, then evaluates `dev` and `test`. Copy its printed run path into `RUN`:

```bash
RUN='runs/support/replace-with-the-run-directory'
METISCRAFT_DATA_ROOT="$PWD/data" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  metiscraft eval "$RUN/artifact" support \
  --version labeled-v1 --split test
```

Read `report.md` in that run directory. Compare model accuracy with the prior baseline, then compare uncalibrated and calibrated NLL, Brier and ECE. Inspect incorrect examples, not just the average. Keep test results separate from the development decisions you make using `dev`.

You should have an `artifact/` and its sibling `calibration/`, plus evaluation outputs. The imported data and base identities are recorded with the run. Useful quality is a result to measure on your task, not a consequence of the commands finishing.

## 6. Call the artifact

Save this constructed example as `request.json`. Its option IDs, definitions and instructions match the training rows above:

```json
{
  "state": "The app closes whenever I try to sign in.",
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this message?",
      "criteria": {
        "billing": "Charges, invoices and refunds.",
        "tech": "Login failures, errors and technical faults."
      }
    }
  }
}
```

Start the service:

```bash
CUDA_VISIBLE_DEVICES='' metiscraft serve "$RUN/artifact" --host 127.0.0.1 --port 8000
```

From another terminal in the same workspace:

```bash
curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @request.json
```

Expect a `team` answer with a choice and probabilities for `billing` and `tech`. This guide does not prescribe its numerical output. Stop the foreground server with **Ctrl-C** after checking the response.

After stopping the server, run batch prediction or load the same artifact in Python:

```bash
CUDA_VISIBLE_DEVICES='' metiscraft predict "$RUN/artifact" input/test.jsonl --output predictions.jsonl
CUDA_VISIBLE_DEVICES='' python - "$RUN/artifact" <<'PYCODE'
import json
import sys
from pathlib import Path
from metiscraft.serve.predictor import Predictor

predictor = Predictor.load(sys.argv[1])
request = json.loads(Path("request.json").read_text())
print(json.dumps(predictor.answer_request(request), indent=2))
PYCODE
```

For batch Record prediction or Python loading, use the [serving guide](../serving.md).

## When a step fails

| Symptom | Action |
|---|---|
| Duplicate IDs or cross-split input/group conflicts | Correct the original JSONL assignments. Related cases belong together; changing only an ID does not remove leakage. |
| An existing version has different content | Keep the old sealed version and import the changed data under a new version. |
| The base or its identity files cannot be found | Complete base preparation, then check `model.base` and `paths.models_root`. |
| The model runs out of memory | Revisit base size, batch size, sequence length and supported precision before repeating the run. |
| HTTP reports a missing task mapping | Check that `team: routing` matches the request name and the actual Record task IDs. |
| Calibration makes a metric worse | Inspect the calibration split and the evaluation distribution. Treat the reported comparison as evidence; do not assume temperature fitting must help. |

## Next step

Read [best practices](../best-practices.md) to plan comparisons and calibration checks. The [recorded customer-routing example](decision.md) shows how to present a task result with its baseline and evidence boundary.

[中文](../../zh/cookbook/own-task.md)
