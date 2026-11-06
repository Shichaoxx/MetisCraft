# A complete CPU workflow

After [installation](installation.md), export the example from the installed
package. No source checkout, model download or GPU is needed for these commands.
The tiny Qwen3 is randomly initialized and the data is synthetic: the resulting
scores validate the software workflow, not model accuracy or calibration quality.

## Import and train

Run this in a directory where `my-first-judge` does not already exist:

```bash
metiscraft init my-first-judge
cd my-first-judge
python prepare.py
metiscraft data import dataset.yaml --data-root ./data
metiscraft data verify quickstart --version synthetic-v1 --data-root ./data
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 metiscraft train recipe.yaml
```

`prepare.py` creates `input/` and `models/tiny-qwen3/v1/`, including the tokenizer,
`SOURCE.json` and `SHA256SUMS`. It refuses to overwrite existing outputs. Import
creates `data/quickstart/synthetic-v1/`. Training fits a two-step LoRA adapter,
fits temperature on `calib`, and evaluates `dev` and `test`.

## Evaluate and predict

Copy the printed run path into `RUN`:

```bash
RUN='runs/quickstart/replace-with-the-run-directory'
METISCRAFT_DATA_ROOT="$PWD/data" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  metiscraft eval "$RUN/artifact" quickstart --version synthetic-v1 --split test
CUDA_VISIBLE_DEVICES='' metiscraft predict "$RUN/artifact" records.jsonl
CUDA_VISIBLE_DEVICES='' metiscraft predict "$RUN/artifact" request.jsonl
CUDA_VISIBLE_DEVICES='' metiscraft serve "$RUN/artifact" --host 127.0.0.1 --port 8000
```

The two prediction files demonstrate Record JSONL and `/v1/systemone` request
JSONL. The server command stays in the foreground.

## Call the HTTP service

In another terminal in this workspace:

```bash
curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @request.json
```

Stop the server with **Ctrl-C**. Serving requires the `serve` extra. This example
has one task ID, `quickstart`, so Record and HTTP requests select the same
calibration group automatically.

Inspect `report.md`, `eval/metrics.json`, `artifact/` and `calibration/` inside
the run directory. For useful model results, continue with [your data](data.md)
and a [prepared pretrained base](training.md). This example does not validate
real-model quality, GPU kernels, throughput or out-of-distribution calibration.

[中文](../zh/quickstart.md)
