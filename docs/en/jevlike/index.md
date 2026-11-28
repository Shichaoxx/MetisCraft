# JevLike decision models

A model can route a support message to a team and estimate whether it is urgent. We use JevLike for models that make a judgment given an input, a question and a defined set of possible answers.

## Tasks and answer spaces

The [introduction to decision models](../tensor-analysis/introduction.md) shows how one input can carry several named questions. The [notation guide](../tensor-analysis/notation.md) identifies the axes for questions, candidates and sequence length. A [customer support example](../cookbook/decision.md) pairs a complete request with the two answers the model returned.

## Methods and open-source implementations

The same multiple-choice question can be scored in different ways. Each study follows one implementation to its raw scores.

| Implementation | Where the score comes from | Evidence |
| --- | --- | --- |
| [Tev1-4B](tev1.md) | Vocabulary scores for the answer letters at the end of the full prompt | Upstream per-item answers and the archived MetisCraft reproduction |
| [JevForge](jevforge.md) | A shared scalar head at the last position of each separate candidate row | Pinned encoder, model and released training script |
| [Kev-4B](kev.md) | A decision vector paired with each option-end vector for the same question | Pinned pointer head, inference code and training configuration |

The [architecture atlas](../tensor-analysis/architectures.md) traces inputs through to candidate scores. Its implementation studies cover three groups.

- [Answer tokens and fixed slots](../tensor-analysis/architectures-token.md) compares the positions that supply scores in models such as jebadiah and d3.
- [Scalar heads, pointers and joint schemas](../tensor-analysis/architectures-structured.md) follows the scoring of candidates or fields in JevForge, Kev, Clef and JevAny.
- [Encoder classifiers and pairwise rerankers](../tensor-analysis/architectures-encoders.md) compares how models such as GLiClass encode text, labels or candidate pairs.

## Comparing methods and checking results

Work through two questions with different numbers of candidates to check padding, grouped probabilities and independent labels. [Candidate scores and probabilities](candidate-scores.md)

Change `green` to `red` in the second option. Can that change the score for the first option? [Switch between three readouts](../tensor-analysis/readout-comparison.md#candidate-edit) to see how much of the question each one has read when it scores the first option.

For a particular model, the [support matrix](../support-matrix.md) records the versions examined, toolkit support and verification completed. The [Tev1 reproduction](../reproductions/tev1.md) reports results on matched evaluation items and explains the limits of the experiment.
