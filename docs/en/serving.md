# Load and serve an artifact

Use the `artifact/` directory printed by training. Keep its associated
`calibration/` directory beside it; Predictor discovers it automatically. Use
`--calibration` when calibration is stored elsewhere. Calibration files are
checked against the artifact they were fitted for.

## Predict Record JSONL

With `RUN` set to your run directory:

```bash
CUDA_VISIBLE_DEVICES='' metiscraft predict "$RUN/artifact" records.jsonl
```

Record prediction supports `noul`, `choice`, `score` and `multi_label`, subject to
the artifact's readout. Records carry their own `task_id`. Outputs include the
decision, probabilities, raw scores and base-verification status. When calibration
is present, each result also reports its selected temperature, group and
`fallback_reason`.

The Python equivalent is:

```python
from metiscraft.serve.predictor import Predictor

predictor = Predictor.load("runs/quickstart/replace-with-the-run-directory/artifact")
results = predictor.predict_records(records)
```

Here `records` is a list of Record objects or dictionaries. To require CPU in a
Python process, pass `device=torch.device("cpu")` after importing `torch`.

## Serve `/v1/systemone`

Install the `serve` extra, then start the service in the foreground:

```bash
CUDA_VISIBLE_DEVICES='' metiscraft serve "$RUN/artifact" --host 127.0.0.1 --port 8000
```

In another terminal, send the quickstart's request:

```bash
curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @request.json
curl -sS http://127.0.0.1:8000/health
```

Stop the service with Ctrl-C. The HTTP endpoint accepts **`noul`, `choice` and
`score`**. Use Record prediction for `multi_label`. `/health` reports whether a
calibrator was loaded and the `base_verification` state; it does not measure model
quality. This service does not provide authentication or a production deployment
stack.

## Route tasks consistently

HTTP questions use a task automatically only when artifact metadata explicitly
declares exactly one `task_id` in `task_ids`. If multiple tasks are declared, each
request question name needs a mapping to one of those task IDs. Missing mappings
are errors, rather than silently selecting a different calibration task.

Artifacts without a declared `task_ids` inventory, including historical artifacts,
also require an explicit mapping for every HTTP question. A single saved
temperature group does not establish a single-task model: small tasks may have no
task-specific temperature. In this case, the mapping explicitly declares the
task IDs for the load; use the IDs from the original data, not the recipe name.

For example, if the artifact contains `support/routing` and `support/urgency`,
create `task-map.yaml`:

```yaml
team: support/routing
urgency: support/urgency
```

Use it with that multi-task run:

```bash
metiscraft serve "$RUN/artifact" --task-map task-map.yaml
metiscraft predict "$RUN/artifact" request.jsonl --task-map task-map.yaml
```

The same mapping can be saved in a training recipe under `serve.task_map`, or
passed to `Predictor.load(..., task_map=...)`. The CLI override replaces the saved
mapping. Record inputs continue to use their explicit `task_id`.

Calibration first selects `<task_id>|<kind>`, then `*|<kind>` with reason
`task_temperature_unavailable`, then temperature 1 with reason
`kind_temperature_unavailable`. With no calibrator, Record output has
`calibration: null`. HTTP keeps the official response shape; routing and
calibration details are logged by `metiscraft.serve.predictor` at INFO level.
`metiscraft serve` enables this logger by default. Applications that use
`Predictor.answer_request` or `create_app` directly should configure that logger
in their own logging setup.

## Move a base directory

Keep the complete sealed base directory, including `SOURCE.json` and `SHA256SUMS`.
Its recorded location is absolute, while its content identity uses relative file
names. After moving it, select the new location explicitly:

```bash
metiscraft serve "$RUN/artifact" --base-path ./models/relocated-base
metiscraft predict "$RUN/artifact" records.jsonl --base-path ./models/relocated-base
```

`eval` also accepts `--base-path`. In Python, use
`Predictor.load(artifact, base_path="./models/relocated-base")`. The explicit path
takes precedence and must still match the recorded identity. Without an override,
the loader uses the recorded path when it exists, then falls back to the recipe's
base setting and configured model root. Relocation does not require editing the
artifact or regenerating the base's checksums.

## Historical artifacts

New artifacts verify the local base against the content identity recorded during
training. A mismatch fails even when the checksum manifest was regenerated.

Historical `metiscraft-artifact/1` artifacts without that identity are refused by
default. To inspect one knowingly, add `--allow-legacy-artifact` to `predict`,
`serve` or `eval`, or pass `allow_legacy_artifact=True` to `Predictor.load`.
Loading emits a warning and reports `legacy_unverified` in Record results and
`/health`. This does not retroactively verify the original training run. Existing
but invalid identity metadata cannot be bypassed with this option.

When the historical artifact also lacks `task_ids`, HTTP requests need both the
legacy opt-in and an explicit task map:

```bash
metiscraft serve "$RUN/artifact" --allow-legacy-artifact --task-map task-map.yaml
metiscraft predict "$RUN/artifact" request.jsonl --allow-legacy-artifact --task-map task-map.yaml
```

Record prediction and dataset evaluation use the records' own `task_id` values;
they do not need this HTTP question-name mapping.

[中文](../zh/serving.md)
