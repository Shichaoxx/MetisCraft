# Changelog

## 0.1.0rc1 — Unreleased

This release candidate has not been published to PyPI. It prepares the installed
package for a complete local workflow and strengthens the boundaries between
training evidence, calibration and deployment.

### Added

- `metiscraft init` exports a self-contained CPU quickstart from package resources.
  It creates synthetic data and a random tiny model without downloading weights.
- `metiscraft data import` accepts local supervised Record JSONL through a manifest,
  without editing the registry. Identical imports verify the existing version;
  changed inputs require a new version.
- Paired English and Chinese core documentation and documentation-site sources.

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

- Historical `/1` artifacts without base identity require
  `--allow-legacy-artifact` or `allow_legacy_artifact=True`. They emit a warning
  and remain `legacy_unverified`; current files do not retroactively prove a run.
- Tev1 reproduction remains partially successful. The CPU quickstart provides no
  new real-model quality evidence. The formal GoEmotions result still predates the
  inference correction, and the same-protocol encoder comparison remains unfinished.

See [limitations](docs/en/limitations.md) and [status](docs/status.md) for details.

The earlier Metis 0.1.0 project is archived under the `metis-v0.1.0` tag; it is
distinct from the MetisCraft release line.
