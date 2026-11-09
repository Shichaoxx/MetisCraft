# BoolQ

Boolean reading comprehension: given a Wikipedia passage and a question, predict whether the answer is affirmative.

| Item | Details |
|---|---|
| Registry name | `boolq` |
| Upstream | [`google/boolq`](https://huggingface.co/datasets/google/boolq) @ `35b264d0` |
| License | CC-BY-SA-3.0; share-alike terms; trained artifacts retain applicable upstream terms |
| Judgment type | Yes/no noul |
| Splits | train 8,425 (upstream train); calib 483 (5% hash split from upstream train); dev 519 (5% hash split from upstream train); test 3,270 (upstream validation) |

## Recipe

| Recipe | Readout | Tuning | Effective batch | Epochs | Learning rate (LoRA / head) |
|---|---|---|---|---|---|
| [`qwen3-0.6b-base_answer-token.yaml`](qwen3-0.6b-base_answer-token.yaml) | ① Answer token | LoRA r=16 | 16 | 2 | 2e-4 / 1e-3 |

## Run

```bash
metiscraft data prepare boolq
metiscraft train recipes/boolq/qwen3-0.6b-base_answer-token.yaml
```

## Recorded results

### ① Answer token: `qwen3-0.6b-base_answer-token.yaml`

Run `20261105-110024-3a392e61`; commit `244ae64`; data `rev-35b264d0`; 1× RTX 3090; 1,054 steps, 38 minutes; noul temperature 2.05.

**test**, Yes/no noul (n=3,270)

| Method | Accuracy | AUC | NLL | Brier | ECE |
|---|---|---|---|---|---|
| Model | 0.851 | 0.914 | 0.469 | 0.123 | 0.094 |
| Model + temperature | 0.851 | 0.914 | 0.360 | 0.111 | 0.024 |
| Prior | 0.622 | 0.500 | 0.663 | 0.235 | 0.001 |

**dev**, Yes/no noul (n=519)

| Method | Accuracy | AUC | NLL | Brier | ECE |
|---|---|---|---|---|---|
| Model | 0.861 | 0.924 | 0.433 | 0.116 | 0.100 |
| Model + temperature | 0.861 | 0.924 | 0.340 | 0.105 | 0.045 |
| Prior | 0.630 | 0.500 | 0.659 | 0.233 | 0.010 |

ECE uses 15 equal-width bins: top-1 probability for choice/score and affirmative probability for each noul/multi_label candidate. Temperatures are fitted only on calib. The prior baseline uses training-set label frequencies without reading the input.

## External comparisons

External numbers use different subsets and protocols; they provide context rather than a matched comparison.

- jebadiah 4B-v2: accuracy 0.847 on a 300-question subset, from per-question records released with the weights; see the [ecosystem matrix](../../docs/en/support-matrix.md).
