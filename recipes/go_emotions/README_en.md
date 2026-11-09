# GoEmotions (simplified)

Multi-label emotion classification for Reddit comments: 28 candidates (27 emotions plus neutral), each with an independent probability. Most comments have one label; inspect macro-F1 and candidate calibration alongside micro-F1.

| Item | Details |
|---|---|
| Registry name | `go_emotions` |
| Upstream | [`google-research-datasets/go_emotions`](https://huggingface.co/datasets/google-research-datasets/go_emotions) @ `add49224`, configuration `simplified` |
| License | Apache-2.0; may contain personal information; trained artifacts retain applicable upstream terms |
| Judgment type | Multiple labels |
| Splits | train 43,252 (upstream train); calib 2,730 (first hash half of upstream validation); dev 2,696 (second hash half of upstream validation); test 5,427 (upstream test) |

## Recipe

| Recipe | Readout | Tuning | Effective batch | Epochs | Learning rate (LoRA / head) |
|---|---|---|---|---|---|
| [`qwen3-0.6b-base_candidate-scalar.yaml`](qwen3-0.6b-base_candidate-scalar.yaml) | ③ Scalar head (list layout) | LoRA r=16 | 16 | 1 | 2e-4 / 1e-3 |

## Run

```bash
metiscraft data prepare go_emotions
metiscraft train recipes/go_emotions/qwen3-0.6b-base_candidate-scalar.yaml
```

## Recorded results

### ③ Scalar head: `qwen3-0.6b-base_candidate-scalar.yaml`

Run `20261105-085531-38a1bf6f`; commit `85fd8ac` (untracked files present; tracked code matched the commit); data `rev-add49224`; 1× RTX 3090; 2,704 steps, 65 minutes; multi_label temperature 1.25.

**test**, Multiple labels (n=5,427)

| Method | micro-F1 | macro-F1 | Exact match | NLL | Brier | ECE |
|---|---|---|---|---|---|---|
| Model | 0.429 | 0.409 | 0.065 | 0.221 | 0.070 | 0.109 |
| Model + temperature | 0.429 | 0.409 | 0.065 | 0.215 | 0.068 | 0.118 |
| Prior | 0.000 | 0.000 | 0.000 | 0.148 | 0.036 | 0.000 |

**dev**, Multiple labels (n=2,696)

| Method | micro-F1 | macro-F1 | Exact match | NLL | Brier | ECE |
|---|---|---|---|---|---|---|
| Model | 0.428 | 0.417 | 0.056 | 0.222 | 0.070 | 0.109 |
| Model + temperature | 0.428 | 0.417 | 0.056 | 0.216 | 0.068 | 0.118 |
| Prior | 0.000 | 0.000 | 0.000 | 0.149 | 0.037 | 0.000 |

**This run predates the inference correction.** Training used pos_weight=22.8 without subtracting ln w at inference, inflating probabilities; temperature scaling did not correct the offset (ECE worsened above). Current inference subtracts ln w before calibration. Below, this rule is applied offline to saved dev/test scores. Calib scores were not saved, so temperature is 1. Corrected-code retraining has not been performed.

**test**, Offline correction (n=5,427)

| Method | micro-F1 | macro-F1 | Exact match | NLL | Brier | ECE |
|---|---|---|---|---|---|---|
| Subtract ln w (offline, temperature 1) | 0.587 | 0.460 | 0.466 | 0.076 | 0.022 | 0.002 |

**dev**, Offline correction (n=2,696)

| Method | micro-F1 | macro-F1 | Exact match | NLL | Brier | ECE |
|---|---|---|---|---|---|---|
| Subtract ln w (offline, temperature 1) | 0.575 | 0.438 | 0.457 | 0.078 | 0.022 | 0.003 |

ECE uses 15 equal-width bins: top-1 probability for choice/score and affirmative probability for each noul/multi_label candidate. Temperatures are fitted only on calib. The prior baseline uses training-set label frequencies without reading the input.

## External comparisons

External numbers use different subsets and protocols; they provide context rather than a matched comparison.
