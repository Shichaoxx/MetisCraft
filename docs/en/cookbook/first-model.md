# Your first model on CPU

**Goal:** import data, train and calibrate a model, inspect its outputs, and make an HTTP request from a fresh workspace.

This tutorial uses a tiny, randomly initialized Qwen3 and synthetic labels. You will get a working artifact and service; the answers demonstrate the software workflow, not useful business accuracy.

## Before you start

- Complete [installation](../installation.md) with the `serve` extra.
- Use the installed environment in a POSIX shell. The package has been checked with Python 3.10 and 3.11.
- Choose a new workspace outside the source checkout. The generated example needs no model download or GPU.

## Know what the example contains

| File | Purpose |
|---|---|
| `prepare.py` | Creates the small random base and synthetic labeled data locally |
| `dataset.yaml` | Names the dataset version and its four splits |
| `recipe.yaml` | Connects the data, model, two training steps, calibration and evaluation |
| `records.jsonl` | An input for the Record prediction interface |
| `request.jsonl` / `request.json` | Inputs for request-format batch prediction and HTTP |

## Run the workflow

Follow the commands below in order. These commands are generated from the [packaged quickstart](../../../src/metiscraft/templates/quickstart/README.md). Keep the first terminal open while the HTTP server runs, and use a second terminal for the request.

<!-- metiscraft:quickstart -->

Export a fresh workspace, enter it, and create the example inputs:

```bash
metiscraft init my-first-judge
cd my-first-judge
python prepare.py
metiscraft data import dataset.yaml --data-root ./data
metiscraft data verify quickstart --version synthetic-v1 --data-root ./data
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 metiscraft train recipe.yaml
```

`prepare.py` writes only `input/` and `models/tiny-qwen3/v1/` under the current
directory, and refuses existing outputs. The model includes its tokenizer,
`SOURCE.json` and `SHA256SUMS`. The importer writes `data/quickstart/synthetic-v1/`.
Training fits a two-step LoRA adapter, fits temperature on `calib`, and evaluates
`dev` and `test`. Outputs are under `runs/quickstart/`.

Copy the run path printed by the training command into `RUN`:

```bash
RUN='runs/quickstart/replace-with-the-run-directory'
METISCRAFT_DATA_ROOT="$PWD/data" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  metiscraft eval "$RUN/artifact" quickstart --version synthetic-v1 --split test
CUDA_VISIBLE_DEVICES='' metiscraft predict "$RUN/artifact" records.jsonl
CUDA_VISIBLE_DEVICES='' metiscraft predict "$RUN/artifact" request.jsonl
CUDA_VISIBLE_DEVICES='' metiscraft serve "$RUN/artifact" --host 127.0.0.1 --port 8000
```

In another terminal in this workspace, call the running HTTP server:

```bash
curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @request.json
```

Stop the server with Ctrl-C. Serving requires the `serve` extra. The official
endpoint accepts `noul`, `choice` and `score`; `multi_label` is available through
the Record prediction API. This example has one task ID, `quickstart`, so HTTP
and Record requests select the same calibration group.

<!-- metiscraft:quickstart:end -->

## What you should have

Under the path printed by training, the important outputs are:

```text
runs/quickstart/<run-id>/
├── config.yaml
├── data.json
├── model.json
├── train_log.jsonl
├── artifact/
├── calibration/
├── eval/
│   └── metrics.json
└── report.md
```

`artifact/` contains the learned parameters and metadata; `calibration/` stays beside it so the loader can find its temperatures. `report.md` is the first place to inspect the run, while `eval/metrics.json` contains machine-readable metrics.

The HTTP response should include `answers.team.choice`, either `billing` or `tech`, and probabilities for those same options. A valid response and the generated files establish that the workflow ran. A high score on this constructed dataset is not evidence for a customer-support task.

If you inspect `/health` while the service is running, `status: "ok"`, `calibrated: true` and `base_verification: "verified"` describe the loaded state. They do not measure model quality.

## When a step fails

| Symptom | Check |
|---|---|
| `metiscraft` is not found | Activate the environment where you installed the package. |
| The workspace or generated files already exist | Choose a fresh directory; the example deliberately refuses to overwrite them. |
| `RUN` cannot be found | Replace the placeholder with the exact path printed by training, and remain in the example workspace. |
| The HTTP dependencies are missing | Follow the installation guide to install the `serve` extra from the same source or wheel. |
| The server port is busy | Select another `--port` and use that port in the request URL. |
| The answer is implausible | This is expected from a random tiny model trained for two steps. Check the files and response shape, not its judgment quality. |

## Next step

[Bring your own task](own-task.md) replaces the synthetic labels and random base with your data and a versioned pretrained model. To see what a trained task can look like first, read the [recorded customer-routing decision](decision.md).

[中文](../../zh/cookbook/first-model.md)
