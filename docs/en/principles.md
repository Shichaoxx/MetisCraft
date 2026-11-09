# How decision models compute

MetisCraft is a toolkit for JevLike decision models. Inputs contain a situation, a question and an answer space. The model produces candidate scores, then converts them into typed probabilities and structured judgments. See [Concepts & Tensors](tensor-analysis/index.md) for the complete mechanism and [interfaces](interfaces.md) for the contract.

## From requests to tensors

A request may contain multiple questions; each question becomes one `Record`. Batch size $B$ therefore counts questions, not HTTP requests. $K_b$ counts the candidates for question $b$; $K_{\max}$ is the padded candidate width. Input padding and candidate-valid masks have different meanings; see [tensors, axes and indices](tensor-analysis/notation.md).

Provenance and grouping belong to the computation contract. Questions derived from the same material share a `group`, fitting and evaluation roles must not overlap, and `provenance` retains the source revision and row identity.

## Two readouts, three implemented paths

| Path | Core computation | Candidate visibility |
|---|---|---|
| `answer_token` | Final hidden → vocabulary logits → answer-code gather | Causal order within the prompt; the readout position sees all candidates |
| `candidate_scalar` / `list` (default) | Candidate-position gather → shared MLP | Later candidates can see earlier candidates |
| `candidate_scalar` / `pairs` | One row per candidate → final-position MLP → scatter | Separate candidate rows do not read each other |

See [answer tokens](tensor-analysis/answer-token.md) and [scalar heads](tensor-analysis/scalar-heads.md) for equations and numerical examples. Pairs can be batched; sequence counts do not equal sequential model calls. Shared-prefix tree packing is a research mechanism that the current package does not implement.

## From scores to probabilities

| Type | Probabilities | Record decision |
|---|---|---|
| `noul` | Sigmoid of one affirmative logit | Threshold at 0.5 |
| `choice` | Softmax over valid candidates | Argmax option |
| `score` | Softmax over ordered levels | Argmax level index |
| `multi_label` | Candidate-wise sigmoid | Threshold each column at 0.5 |

HTTP `score` is the probability-weighted expected level index; choice/score `confidence` is maximum candidate probability. These are distinct from the Record decision and from the official System One confidence definitions.

## Training and calibration

Candidate-space objectives differ from completion's full-vocabulary objective. Completion predicts the answer code and EOS at two aligned positions. Training losses are normalized by the global question count in an optimizer step. Gradient equivalence comparisons require controlled randomness, padding and sample ordering; they do not imply that arbitrary training configurations produce identical models.

Inference first applies the recorded positive-weight logit correction, then divides by temperature $\tau$. A separate `calib` split fits temperatures, selected by task/type. Evaluation retains raw and calibrated metrics. Positive temperatures preserve ranking but can change expected scores; calibration can also worsen NLL on the target distribution. See [training, calibration and output](tensor-analysis/training-calibration.md).

## Evidence and boundaries

The matched [Banking77](../../recipes/banking77/README_en.md) runs report accuracy 0.928 for answer tokens and 0.927 for scalar-list; this is not a universal ranking. [Tev1](reproductions/tev1.md) retains public-development-set, single-seed and calibration limitations. The historical [GoEmotions](../../recipes/go_emotions/README_en.md) training predates the prior correction; offline correction does not replace retraining.

Mechanisms, measurements and sources are checked separately. Unimplemented paths in the [seven-readout comparison](tensor-analysis/readout-comparison.md) are not package capabilities.
