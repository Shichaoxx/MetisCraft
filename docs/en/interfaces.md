# Interfaces and artifacts

This page defines the current public contract. See [data](data.md), [training](training.md) and [serving](serving.md) for operational steps.

## Typed judgments

| Type | Probabilities | Record decision | HTTP result |
|---|---|---|---|
| `noul` | Affirmative probability | Threshold at 0.5 | `noul` probability |
| `choice` | Distribution over valid options | Argmax option, including an allowed null | Selected option; `confidence` is maximum probability |
| `score` | Distribution over ordered levels | Argmax level index | Expected level index; `confidence` is maximum probability |
| `multi_label` | Independent candidate probabilities | Each probability ≥ 0.5 | Not supported by `/v1/systemone` |

Python and JSONL batch prediction support all four types. Answer-token readout supports the first three; the scalar head supports all four, with `list` as the default and `pairs` as an alternative. Readout selection does not change probability definitions.

The HTTP request shape follows the System One pattern. Matching field names do not establish identical confidence semantics: [System One confidence](https://docs.typesafe.ai/confidence) uses additional normalization for choice and a distance-based definition for score. See [training, calibration and output](tensor-analysis/training-calibration.md).

## Recipes and Python

Recipes are loaded with `metiscraft.config.load_recipe`; `metiscraft.pipeline.run_recipe` trains, calibrates and evaluates. `metiscraft.pipeline.evaluate_artifact` evaluates a sealed artifact. Load predictions through `metiscraft.serve.predictor.Predictor.load`; validate data through `metiscraft.data.contract.Record`.

Fitting and evaluation split roles are separate. Shared groups or inputs across those roles are rejected. `eval_only` labels are only for evaluation. Temperatures are fitted on calibration data and selected by task/type, with observable fallback information. Calibration does not guarantee improvement after distribution shift.

## Loading and relocation

Artifacts record base weights, configuration and tokenizer content identity. Relocate identical content and load it through `--base-path`; changed content must fail verification. Legacy artifacts without historical identity require explicit compatibility mode and remain unverified.

Recorded configuration, data provenance, environment, fitted parameters and metrics describe one run. Data, base models and code retain their own licenses; unknown terms do not authorize commercial use.

## Further reading

[Recipe fields](recipe-reference.md) · [Concepts & Tensors](tensor-analysis/index.md) · [Limitations](limitations.md)
