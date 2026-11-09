# Current limitations and evidence

MetisCraft 0.1.0rc1 is a release candidate. Interfaces and artifact formats may
change. A successful CPU quickstart establishes that the packaged workflow runs;
its random model and synthetic data cannot establish model quality.

## Model evidence

The Qwen3.5-4B Tev1 run **meets the accuracy criteria recorded before evaluation**:

| Evaluation set | MetisCraft, Qwen3.5-4B | Published Tev1, Qwen3.5-4B | Status |
|---|---:|---:|---|
| `test`, 1,000 questions | 87.3% (873/1,000) | 88.0% | Within the stated 2 percentage points |
| `policy_transfer`, 300 questions | 99.33% (298/300) | 100% | Close to the stated 100% target |

The earlier 2B run scored 86.4% / 92.0%. The completed 4B comparison uses the same
question IDs, inputs, gold labels and option order as the saved upstream results;
answers agree on 94.6% of the main set and 99.33% of policy transfer.
The upstream scores use per-option regex-constrained generation at temperature
zero; MetisCraft selects the highest-scoring candidate letter. Unconstrained
whole-vocabulary first-token diagnostics remain untested and must not be confused
with the upstream reported protocol. Matching accuracy does not recover the
unknown historical training settings or establish identical weights.
Only one seed was run, and the upstream evaluation sets were used during
development; these results are not evidence of generalization to untouched tasks.
Details are in the
[Tev1 record](reproductions/tev1.md).

The published GoEmotions run used code from before the `pos_weight` inference
correction. Its offline corrected scores are not a completed training rerun with
the fix. See the [recipe results](../../recipes/go_emotions/README_en.md).

An encoder baseline under the same splits, metrics and calibration protocol has
not been completed. There is no established accuracy or cost advantage over an
encoder from these results.

## Software scope

- Two readouts are implemented; ranking is outside this release.
- The HTTP endpoint supports three judgment types. Multi-label prediction uses
  the Record API.
- Serving uses Transformers and Predictor. Export to vLLM, SGLang or llama.cpp is
  not implemented.
- There is no supported resume-from-checkpoint workflow yet.
- Supported model architectures depend on the installed Transformers version;
  not every model family or GPU kernel has been validated.
- The local HTTP service has no built-in authentication or production operations layer.

## Interpreting integrity and calibration

File identities detect changes relative to recorded bytes. They do not prove
upstream authenticity, recover missing historical evidence or validate data
labels. Imported license and source fields are declarations, not independent
verification of rights or provenance.

Temperature calibration can worsen performance after a distribution shift. In
the 4B Tev1 run, policy-transfer NLL increased from 0.0522 to 0.0990 after applying
the fitted policy temperature; the cause remains unresolved. Accuracy was unchanged.
Compare calibrated and raw
metrics on representative held-out data, and keep their split roles separate.

The [support matrix](support-matrix.md) records ecosystem coverage and the evidence available for each project.

[中文](../zh/limitations.md)
