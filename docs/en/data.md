# Use your own data

Import supervised Record JSONL with a local manifest. This creates a versioned,
checksummed dataset without changing the built-in registry.

## Describe the files

Create `dataset.yaml` beside your `input/` directory:

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

Replace the example source, license and terms with those that actually apply to
your data. `name` and `version` are single path components starting with an
alphanumeric character. Split paths are relative to the manifest and must stay
inside its directory. A source file cannot supply more than one split.

Custom terms are preserved. The artifact's `commercial_use` field summarizes
recognized restriction flags: `false` for a recorded restriction, `null` for
uninterpreted terms, and `true` when no recognized restriction is recorded.
This field does not grant permission; review the actual data and base licences.

## Write Record JSONL

Each line is one supervised question. For example:

```json
{"id":"ticket-1","task_id":"routing","input":{"query":"Which team?","context":{"state":"My card has not arrived."}},"question":{"type":"choice","options":[{"id":"billing"},{"id":"support"}]},"group":{"id":"case-1","split_key":"conversation-1"},"supervision":{"kind":"choice","labels":"support"}}
```

Keep record IDs unique across all imported splits. `task_id` names the task used
for calibration; it need not equal the dataset name. The importer adds source
provenance and an input hash, retaining any supplied provenance as nested metadata.

| Judgment | Question fields | Hard label |
|---|---|---|
| `noul` | `type: noul` | Boolean |
| `choice` | `options` with unique IDs | Option ID; `null` only with `allow_null` |
| `score` | Ordered `levels` with unique IDs | Zero-based level index |
| `multi_label` | Top-level `candidates` with unique IDs | List of positive candidate IDs |

Set `supervision.kind` to the question type. Soft probabilities can be supplied
in `supervision.soft`, aligned with the question's targets. Import requires
supervision; Record prediction inputs can omit it.
For `choice` and `score`, soft probabilities must sum to 1 within `1e-6`;
`noul` and `multi_label` carry independent probabilities. The choice option ID
`__null__` is reserved: enable `allow_null` to add the null answer instead.

## Keep fitting and evaluation separate

Use `train` for model fitting, `calib` for temperature fitting, and `dev`, `test`
or `eval_only` for evaluation. Training validates these roles before loading a
model. It rejects shared inputs, `group.id` or `group.split_key` between training,
calibration and evaluation. The check includes held-out splits not selected for
that particular run.

Questions from one case, conversation or document should share a group so they
stay together. Evaluation splits may overlap each other, but overlapping results
are not independent evidence. Local import preserves your explicit assignments:
it reports conflicts instead of silently dropping or moving records.

## Import and verify

```bash
metiscraft data import dataset.yaml --data-root ./data
metiscraft data verify support --version labeled-v1 --data-root ./data
metiscraft data show support --version labeled-v1 --split train -n 2 --data-root ./data
```

The result is `data/support/labeled-v1/`, containing source copies, converted
splits, purpose manifests, `dataset.yaml` and `SHA256SUMS`. Reimporting an identical
manifest and identical input bytes verifies the existing version. Changed inputs
or manifest content require a new version; the importer will not overwrite one.
Do not hand-edit sealed dataset files.

Set `data.dataset: support` and `data.version: labeled-v1` in your
[training recipe](training.md). For built-in datasets, `metiscraft data list`
lists registry entries and `metiscraft data prepare NAME` uses their pinned
source revisions. Custom imports do not become registry entries.

[中文](../zh/data.md)
