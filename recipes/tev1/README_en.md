# Tev1

Tev1 decision data combines five public datasets with synthetic policy decisions, priority routing and research classification. Questions have lettered options and a one-letter answer. These recipes follow the public recommendation; the [reproduction record](../../docs/en/reproductions/tev1.md) distinguishes that recommendation from the unavailable historical hosted run.

| Item | Details |
|---|---|
| Registry name | `tev1` |
| Upstream | [togethercomputer/tev1@1dde778](https://github.com/togethercomputer/tev1/tree/1dde7782382c9f49d627153759b8d1deab426ce0); rebuilt data matches all 21 public hashes |
| License | Build scripts MIT; rows retain dataset terms: MultiNLI mixed, BoolQ CC BY-SA 3.0, Banking77 CC BY 4.0, AG News non-commercial, SST-5 undeclared. Trained artifacts are not marked commercially usable |
| Types | `choice`; `score` for SST-5 |
| Splits | train 37,840 (new-v1 train); calib 2,448 and dev 2,120 (new-v1 dev split by group hash); test 1,000 (v1 main, original 88.0%); eval_only 300 (v1 policy_transfer, original 100%) |

## Recipes

| Recipe | Readout | Prompt | Objective | Tuning | Questions per step | Epochs | Learning rate |
|---|---|---|---|---|---|---|---|
| [2B](qwen3.5-2b_chat-completion.yaml) | Answer token | `chat_json`, Tev1 system prompt | Full-vocabulary `completion`: letter + EOS | All-linear LoRA r=8, alpha=16 | 36 (batch 4 × accumulation 9) | 1 | 5e-5, cosine, warmup 0.03 |
| [4B](qwen3.5-4b_chat-completion.yaml) | Answer token | Same | Same | Same | 36 (batch 4 × accumulation 9) | 1 | Same |

LoRA settings, recommended epochs/rate/scheduler, gradient clip 1, length 2,048 and answer-only loss follow the public recipe. All 43,708 prompts/answers were checked token by token against upstream exports.

Differences: MetisCraft does not pack samples; upstream batch 8 refers to packed 2,048-token sequences with approximately 36 questions per step. Batch 4 × accumulation 9 aligns question count, not packed computation. The 4B base revision is `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Actual hosted hyperparameters were not retained upstream.

## Run

```bash
pip install -e ".[reproduce]" flash-linear-attention
metiscraft data prepare tev1
metiscraft train recipes/tev1/qwen3.5-2b_chat-completion.yaml
```

Qwen3.5 linear-attention layers use `flash-linear-attention`. In the recorded 2B smoke configuration, the PyTorch reference path at batch 4 without gradient checkpointing reached 23.5 GB and did not finish its first step within 1.5 minutes. With the optimized kernel and checkpointing, peak memory was approximately 6 GB and each 36-question step took approximately 26 seconds; first-use kernel compilation added approximately 10 minutes. The original 1,052-step estimate was approximately 7.5 hours. These are configuration-specific observations, not a general hardware guarantee or a 4B memory claim.

## Recorded results

The 2026-11-06 2B run reports test 86.4% and policy transfer 92.0%; only its main check passed. The 2026-11-07 4B run reports **873/1,000 (87.3%) and 298/300 (99.33%)**, 0.70 / 0.67 percentage points below original 88.0% / 100%, meeting the original accuracy criteria.

Answer agreement is 94.6% / 99.33%. This does not recover the historical hosted job. Calibration can worsen NLL. The historical MetisCraft 4B run also includes 32 extra sample presentations in its tail under the old loop; current software fixes that budget behavior without rewriting historical results. See the [reproduction record](../../docs/en/reproductions/tev1.md) for run IDs, calibration metrics, readout protocols and limitations.
