# Choosing and diagnosing recipes

These observations come from recorded single-GPU Qwen3-0.6B recipes. Each recipe defines its conditions and original metrics. Use them to design comparisons and diagnose failures; they are not cross-task performance guarantees.

## Choose a readout

- Evaluate answer tokens for fixed answer spaces; every answer code must be a single token. [Banking77](../../recipes/banking77/README_en.md) reports accuracy 0.928 / 0.927 for answer-token / scalar-list.
- Use scalar heads for multiple labels or changing candidate spaces. Default `list` and independent-candidate `pairs` have different visibility; see [tensor analysis](tensor-analysis/scalar-heads.md). Hold other data and training settings fixed when comparing them.
- Shuffling candidate order helps control positional cues; it does not prove that positional bias is eliminated. An untrained random scalar head is not a useful zero-shot classifier.

## Objectives and data

- `objective.pos_weight` changes the prior scale of multi-label logits. Current inference subtracts recorded $\ln w$ before temperature scaling; [GoEmotions](../../recipes/go_emotions/README_en.md) distinguishes historical training from offline correction.
- `ordinal_soft` can represent adjacent-level structure, but the recorded SST-5 recipe has no matched objective comparison.
- Microbatches and gradient accumulation are normalized by the global question count; changing batches may still change randomness and the optimization trajectory.
- Pin source versions; retain licenses, terms, `group` and source rows. Fit temperatures on `calib`, select models on dev and evaluate on independent test data.

## Calibration and evaluation

| Observation | Check |
|---|---|
| High accuracy with unclear gain | Compare the training prior: customer urgency prior is 0.870, model 0.986 |
| ECE improves while the application objective worsens | Inspect NLL, Brier and task metrics together: SST-5 calibrated NLL 1.380→1.014, expected-score MAE 0.506→0.559 |
| Multi-label mean probability is too high | Check positive-weight correction and distribution differences; temperature alone cannot correct an offset |
| Out-of-scope requests are missed | Separate null precision / recall from in-scope accuracy: CLINC150 reports 0.971 / 0.765 / 0.959 |
| Calibration worsens after transfer | Retain raw probabilities and independent in-distribution evaluation; [Tev1](reproductions/tev1.md) includes an NLL deterioration example |

ECE uses 15 equal-width bins: top-1 probability for choice/score and affirmative probability for noul/multi_label. Low binned ECE does not establish reliability for every subgroup or after distribution shift.

## Traceable delivery

Run records preserve configuration, source revision and dirty state, environment, dataset/base identity, per-question outputs and calibration files. Associate claims with run IDs and state unverified parts. Random CPU tutorial scores are not model-quality evidence. Follow [your own task](cookbook/own-task.md) for the workflow and [Concepts & Tensors](tensor-analysis/index.md) for the computation.
