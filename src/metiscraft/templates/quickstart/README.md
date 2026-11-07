# A complete local workflow

This example runs from the installed package, with no source checkout, model download,
or GPU required. The randomly initialized tiny Qwen3 and synthetic data validate the
software workflow; their accuracy and calibration scores are not useful model results.

<!-- workflow:start -->

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

<!-- workflow:end -->

## Use your own data

Replace the JSONL files referenced by `dataset.yaml` with supervised Record rows;
set a new dataset name/version, describe the source and license, and update the
recipe. Paths in the manifest are relative to that manifest. A minimal row is:

```json
{"id":"ticket-1","task_id":"routing","input":{"query":"Which team?","context":{"state":"My card has not arrived."}},"question":{"type":"choice","options":[{"id":"billing"},{"id":"support"}]},"supervision":{"kind":"choice","labels":"support"}}
```

The importer adds source provenance and a hash of the input. Keep inputs and
groups separate between training, calibration and evaluation; shared inputs or
groups cause an error. Use `group.id` or `group.split_key` to keep related questions
from one case, conversation or document together. Each record needs a unique ID.
An identical import verifies the existing version; changed inputs require a new
version. Explicit split assignments are retained, with no silent deduplication.

For meaningful model quality, replace the random model with a prepared base model
and use representative data. This CPU example does not validate real-model
accuracy, GPU kernels, throughput, or out-of-distribution calibration.
