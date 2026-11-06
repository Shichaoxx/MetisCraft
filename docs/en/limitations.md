# Current limitations and evidence

MetisCraft 0.1.0rc1 is a release candidate. Interfaces and artifact formats may
change. A successful CPU quickstart establishes that the packaged workflow runs;
its random model and synthetic data cannot establish model quality.

## Model evidence

The Tev1 reproduction is **partially successful** under the criteria recorded
before evaluation:

| Evaluation set | MetisCraft, Qwen3.5-2B | Published Tev1, Qwen3.5-4B | Status |
|---|---:|---:|---|
| `test`, 1,000 questions | 86.4% | 88.0% | Within the stated 2B criterion |
| `policy_transfer`, 300 questions | 92.0% | 100% | Below the stated criterion |

These compare different base sizes. The current MetisCraft numbers select the
highest-scoring option token. Whole-vocabulary first-token evaluation and
per-question agreement with Tev1 remain unfinished, as does a 4B reproduction.
The upstream evaluation sets were used during development; these results are
not evidence of generalization to untouched tasks. Details are in the
[Tev1 record](../reproductions/tev1.md).

The published GoEmotions run used code from before the `pos_weight` inference
correction. Its offline corrected scores are not a completed training rerun with
the fix. See the [recipe results](../../recipes/go_emotions/README.md).

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
the existing Tev1 run, policy-transfer NLL increased after applying the fitted
policy temperature; the cause remains unresolved. Compare calibrated and raw
metrics on representative held-out data, and keep their split roles separate.

The broader [support matrix](../support-matrix.md) and [project status](../status.md)
track experiments and future work separately from this installation guide.

[中文](../zh/limitations.md)
