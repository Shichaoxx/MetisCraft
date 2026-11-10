# Compatibility and support

MetisCraft `0.1.0rc1` is a preview candidate. This page defines its tested scope,
public entry points and loading rules. It does not establish compatibility with a
future release candidate or a stable `0.1.0` release.

## Runtime and dependency scope

| Area | Validated scope | Boundary |
|---|---|---|
| Python | 3.10 and 3.11 | The metadata minimum `>=3.10` is an installation constraint; later versions have not been validated. |
| CPU workflow | Linux, POSIX shell, PyTorch CPU, source and installed wheel | Windows, macOS and other operating-system combinations are not in the release test matrix. |
| Distributed CPU training | Two-rank Gloo regression tests | This is not a guarantee of recovery from arbitrary process, host or network failures. |
| GPU training | The individually recorded Qwen3 and Qwen3.5 recipes | Model and kernel support depends on the selected base, Transformers and PyTorch build. CPU tests do not validate every CUDA path. |
| Inference | Transformers with `Predictor`, Python/JSONL and local HTTP | vLLM, SGLang and llama.cpp export are not implemented. |

The [installed-package validation](https://github.com/Shichaoxx/MetisCraft/actions/workflows/release.yml)
builds one wheel and checks it on both Python versions, including a complete
workflow outside the source checkout. A manual run skips publication.

The following resolved environments passed the
[2026-11-10 validation](https://github.com/Shichaoxx/MetisCraft/actions/runs/42762311014).
They are concrete evidence, rather than a claim that every combination between
the declared dependency minima and these versions works:

| Python | PyTorch | Transformers | PEFT | Accelerate |
|---|---|---|---|---|
| 3.10 | `2.14.1+cpu` | `5.20.2` | `0.21.2` | `1.15.0` |
| 3.11 | `2.15.0+cpu` | `5.22.2` | `0.21.2` | `1.15.0` |

The authoritative installation requirements are in `pyproject.toml`. A model
family can require a newer Transformers release than the package minimum.
Preserve the run's `env.json` and exact distribution identity when reproducing a
result; an environment that merely satisfies lower bounds is not the same
environment. Installing MetisCraft does not download weights or datasets.

## Public interfaces

| Interface | Candidate contract |
|---|---|
| CLI | `metiscraft --help` and each command's help define the available options. Recipes use the validated schema; unknown fields fail validation. |
| Python | `Record`, recipe loading, `run_recipe`, `evaluate_artifact` and `Predictor.load` are the documented entry points. Internal helper functions are not an extension API. |
| Record / JSONL | `noul`, `choice`, `score` and `multi_label` are supported subject to the artifact's readout. Preserve task and candidate IDs. |
| `/v1/systemone` | `noul`, `choice` and `score` use the documented request and response fields. Multi-label prediction uses Record / JSONL. |

For this candidate, HTTP `confidence` is the maximum option/level probability;
it is not TypeSafe's normalized confidence statistic. Record `score` decisions
use the modal level index, while HTTP `score` returns the expected zero-based
level index. These semantics are retained in this update. See
[interfaces](interfaces.md) and [output semantics](tensor-analysis/training-calibration.md#probability-score-and-confidence).

## Artifact and calibration compatibility

- New training runs write `metiscraft-artifact/2`. Loading verifies the recorded
  base's weights, tokenizer, configuration and source identity. Changed content
  fails verification even if a local checksum manifest was regenerated.
- Keep `calibration/` beside `artifact/`, or pass an explicit calibration path.
  Calibration must match the artifact it was fitted for. Keep the original
  task IDs or configure an explicit HTTP task map.
- A sealed base can move: use `--base-path` to load identical bytes at a new
  location. An artifact is not a self-contained copy of every base model.
- Historical `metiscraft-artifact/1` without base identity requires explicit
  `--allow-legacy-artifact` and remains `legacy_unverified`. Unknown artifact
  formats are rejected. This opt-in does not bypass an invalid identity.
- Training artifacts do not define a supported optimizer/checkpoint resume
  protocol. Loading an artifact for prediction is different from resuming training.

See [loading, task routing and relocation](serving.md) for commands.

## Distributed lifecycle

`run_recipe` releases a process group it creates when initialization, training,
artifact saving, calibration or evaluation raises. Cleanup does not add a
barrier; if cleanup itself fails while handling another exception, it logs the
cleanup failure and preserves the original exception. A teardown failure after
an otherwise successful run is reported to the caller.

A group initialized by the caller remains caller-owned on both success and
failure. The caller must coordinate participating ranks and release that group.
This ownership policy does not repair a group initialized earlier with different
dependency-import ordering.

Two-rank CPU tests inject model, post-backward training and evaluation failures,
then check both ranks' exit, group release and Gloo-thread release after the
exception has been handled. A rank blocked in an in-flight collective can still
depend on its backend timeout or process launcher. There is no automatic retry,
elastic recovery or GPU fault-tolerance guarantee.

## Changes between candidates

Version `0.1.0rc1` remains a preview: later candidates may change APIs, recipe
fields or artifact formats. Changes and migration instructions belong in the
[changelog](../../CHANGELOG.md). Preserve the package version, source revision,
configuration, environment and sealed base alongside each run. Do not assume a
future package can silently migrate an artifact; unsupported formats fail
explicitly.

[中文](../zh/compatibility.md)
