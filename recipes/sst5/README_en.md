# SST-5

Five ordered movie-review sentiment levels, from very negative to very positive. Reports include accuracy, expected-score MAE and calibration metrics.

| Item | Details |
|---|---|
| Registry name | `sst5` |
| Upstream | [`SetFit/sst5`](https://huggingface.co/datasets/SetFit/sst5) @ `e51bdcd8` |
| License | Not declared upstream; the registry treats it as research-only; trained artifacts are not marked commercially usable, and source terms require review |
| Judgment type | Ordered score |
| Splits | train 8,540 (upstream train); calib 549 (first hash half of upstream dev); dev 552 (second hash half of upstream dev); test 2,210 (upstream test) |

## Recipe

| Recipe | Readout | Tuning | Effective batch | Epochs | Learning rate (LoRA / head) |
|---|---|---|---|---|---|
| [`qwen3-0.6b-base_answer-token.yaml`](qwen3-0.6b-base_answer-token.yaml) | ① Answer token | LoRA r=16 | 16 | 3 | 2e-4 / 1e-3 |

## Run

```bash
metiscraft data prepare sst5
metiscraft train recipes/sst5/qwen3-0.6b-base_answer-token.yaml
```

## Recorded results

### ① Answer token: `qwen3-0.6b-base_answer-token.yaml`

Run `20261105-114305-a49998c0`; commit `4299224`; data `rev-e51bdcd8`; 1× RTX 3090; 1,602 steps, 27 minutes; score temperature 2.45.

**test**, Ordered score (n=2,210)

| Method | Accuracy | Expected-score MAE | NLL | Brier | ECE |
|---|---|---|---|---|---|
| Model | 0.555 | 0.506 | 1.380 | 0.682 | 0.270 |
| Model + temperature | 0.555 | 0.559 | 1.014 | 0.569 | 0.040 |
| Prior | 0.231 | 1.141 | 1.580 | 0.789 | 0.041 |

**dev**, Ordered score (n=552)

| Method | Accuracy | Expected-score MAE | NLL | Brier | ECE |
|---|---|---|---|---|---|
| Model | 0.540 | 0.537 | 1.463 | 0.701 | 0.279 |
| Model + temperature | 0.540 | 0.574 | 1.053 | 0.584 | 0.063 |
| Prior | 0.257 | 1.082 | 1.576 | 0.787 | 0.014 |

ECE uses 15 equal-width bins: top-1 probability for choice/score and affirmative probability for each noul/multi_label candidate. Temperatures are fitted only on calib. The prior baseline uses training-set label frequencies without reading the input.

## External comparisons

External numbers use different subsets and protocols; they provide context rather than a matched comparison.
