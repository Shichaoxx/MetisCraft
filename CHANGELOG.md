# Changelog

## 0.1.0rc1 — Unreleased

This first public preview provides a recipe-based workflow for preparing data,
training typed decision models, calibrating and evaluating probabilities, and
serving the resulting artifacts. The candidate is not yet published to PyPI;
its APIs and artifact formats may change.

### Added

- Recipe-based `train`, `eval`, `predict` and `serve` commands connect prepared
  data, model fitting, calibration, evaluation and deployment.
- Four Record judgment types: yes/no, choice, ordered score and multiple labels.
- Answer-token and candidate-scalar readouts, with LoRA, full or frozen
  backbones; candidate-scalar supports list and pairs layouts.
- A built-in dataset registry and preparation pipeline with pinned sources,
  license/terms metadata, explicit split roles and checksummed dataset versions.
- Task/type temperature calibration fitted on `calib`, with raw and calibrated
  NLL, Brier, ECE and task metrics on evaluation splits.
- Traceable run directories record configuration, dataset/base identities,
  source revision when available, environment and per-question predictions,
  alongside sealed artifacts, calibration files and license notices.
- A Python Predictor and JSONL batch prediction for all four types;
  `/v1/systemone` HTTP serving for yes/no, choice and ordered score.
- Recorded recipes cover BoolQ, Banking77, CLINC150, SST-5, GoEmotions, customer
  routing and Tev1, with evidence boundaries retained below.
- `metiscraft init` exports a self-contained CPU quickstart from package resources.
  It creates synthetic data and a random tiny model without downloading weights.
- `metiscraft data import` accepts local supervised Record JSONL through a manifest,
  without editing the registry. Identical imports verify the existing version;
  changed inputs require a new version.
- Paired English and Chinese public documentation, Cookbook and Concepts & Tensors.
- Editable bilingual tensor diagrams, vector SVG/PDF downloads, local mathematics rendering and same-page language switching.

### Changed

- Training validates split purposes and rejects shared fitting/evaluation inputs
  or groups before loading the model. Local imports preserve explicit assignments.
- New `metiscraft-artifact/2` artifacts bind the verified content of the local
  base, including weights, tokenizer, configuration and source metadata. Reloading
  checks those bytes even if the local checksum manifest has been regenerated.
- HTTP questions use a known single task automatically or an explicit
  question-to-task mapping for multi-task artifacts. Missing mappings fail.
- Record predictions expose calibration group and fallback information, plus
  base-verification status. The health endpoint also reports base-verification status.
- Choice and score soft targets must sum to one, and choice options cannot use
  the reserved `__null__` ID. Local import rejects these errors before sealing.
- Custom data terms remain uninterpreted (`commercial_use: null`) instead of
  being declared commercially unrestricted. Notices report flags and retain
  the requirement to review the actual licences.

### Migration and evidence

- Development-era `metiscraft-artifact/1` artifacts without base identity require
  `--allow-legacy-artifact` or `allow_legacy_artifact=True`. They emit a warning
  and remain `legacy_unverified`; current files do not retroactively prove a run.
  This option applies to MetisCraft artifacts; it does not convert checkpoints
  from the archived Metis prototype.
- The Qwen3.5-4B Tev1 run meets the predefined accuracy criteria: 87.3% on the main
  set and 99.33% on policy transfer, versus the published 88.0% / 100%.
  Per-question answer agreement is 94.6% / 99.33%. Calibration can still worsen
  NLL, and the historical training settings remain unavailable.
- The CPU quickstart provides no real-model quality evidence. The formal
  GoEmotions result still predates the inference correction, and the same-protocol
  encoder comparison remains unfinished.

See [limitations](docs/en/limitations.md) for details.

The earlier Metis 0.1.0 project is archived under the `metis-v0.1.0` tag; it is
distinct from the MetisCraft release line.
### Reliability and documentation

- Candidate-scalar DDP training excludes unused, untied LM output weights while
  preserving shared input embeddings. CPU regressions cover tied weights,
  gradient checkpointing and two optimizer steps with gradient accumulation.
- Completion inputs include the answer code within `model.max_length`; Jev score
  gold rejects boolean, fractional and null labels instead of silently coercing them.
- Bilingual Cookbook routes cover a recorded decision, a local CPU workflow and custom data. Tutorial commands are drawn from packaged English/Chinese templates.
- Epoch tails no longer consume the next epoch to fill gradient accumulation. Numeric recipe domains are checked before model loading.
- Hard-label presence is explicit for new imports; soft-only and ambiguous legacy records do not produce invented hard-label metrics. Undefined metrics serialize as null.
- Raw and effective recipe identities are recorded separately; batch predictions preserve input order.
- Public documentation and distribution contents use an explicit publication policy; development notes remain on dev.
