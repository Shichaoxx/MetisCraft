# Banking77

Online banking intent classification: select one of 77 intents for a customer message. The recipes share data splits and evaluation definitions, comparing answer tokens with scalar-list. Their max_length values differ: 1,024 for answer tokens, 768 for scalar-list; this is not a strict readout-only ablation.

| Item | Details |
|---|---|
| Registry name | `banking77` |
| Upstream | [`PolyAI/banking77`](https://huggingface.co/datasets/PolyAI/banking77) @ `18469841` |
| License | CC-BY-4.0; attribution required; trained artifacts retain applicable upstream terms |
| Judgment type | Choice |
| Splits | train 8,988 (upstream train); calib 473 (5% hash split from upstream train); dev 537 (5% hash split from upstream train); test 3,080 (upstream test) |

## Recipe

| Recipe | Readout | Tuning | Effective batch | Epochs | Learning rate (LoRA / head) |
|---|---|---|---|---|---|
| [`qwen3-0.6b-base_answer-token.yaml`](qwen3-0.6b-base_answer-token.yaml) | ① Answer token | LoRA r=16 | 16 | 2 | 2e-4 / 1e-3 |
| [`qwen3-0.6b-base_candidate-scalar.yaml`](qwen3-0.6b-base_candidate-scalar.yaml) | ③ Scalar head (list layout) | LoRA r=16 | 16 | 2 | 2e-4 / 1e-3 |

## Run

```bash
metiscraft data prepare banking77
metiscraft train recipes/banking77/qwen3-0.6b-base_answer-token.yaml
metiscraft train recipes/banking77/qwen3-0.6b-base_candidate-scalar.yaml
```

## Recorded results

### ① Answer token: `qwen3-0.6b-base_answer-token.yaml`

Run `20261105-074726-8a84f136`; commit `85fd8ac` (untracked files present; tracked code matched the commit); data `rev-18469841`; 1× RTX 3090; 1,124 steps, 59 minutes; choice temperature 1.17.

**test**, Choice (n=3,080)

| Method | Accuracy | macro-F1 | NLL | Brier | ECE |
|---|---|---|---|---|---|
| Model | 0.928 | 0.928 | 0.307 | 0.114 | 0.024 |
| Model + temperature | 0.928 | 0.928 | 0.298 | 0.113 | 0.009 |
| Prior | 0.013 | 0.000 | 4.384 | 0.988 | 0.006 |

**dev**, Choice (n=537)

| Method | Accuracy | macro-F1 | NLL | Brier | ECE |
|---|---|---|---|---|---|
| Model | 0.896 | 0.879 | 0.453 | 0.163 | 0.044 |
| Model + temperature | 0.896 | 0.879 | 0.427 | 0.160 | 0.033 |
| Prior | 0.019 | 0.000 | 4.303 | 0.986 | 0.001 |

### ③ Scalar head: `qwen3-0.6b-base_candidate-scalar.yaml`

Run `20261105-043310-e809a40e`; commit `42b013c`; data `rev-18469841`; 1× RTX 3090; 1,124 steps, 38 minutes; choice temperature 1.33.

**test**, Choice (n=3,080)

| Method | Accuracy | macro-F1 | NLL | Brier | ECE |
|---|---|---|---|---|---|
| Model | 0.927 | 0.927 | 0.302 | 0.115 | 0.025 |
| Model + temperature | 0.927 | 0.927 | 0.280 | 0.114 | 0.011 |
| Prior | 0.013 | 0.000 | 4.384 | 0.988 | 0.006 |

**dev**, Choice (n=537)

| Method | Accuracy | macro-F1 | NLL | Brier | ECE |
|---|---|---|---|---|---|
| Model | 0.899 | 0.892 | 0.391 | 0.155 | 0.047 |
| Model + temperature | 0.899 | 0.892 | 0.354 | 0.151 | 0.034 |
| Prior | 0.019 | 0.000 | 4.303 | 0.986 | 0.001 |

ECE uses 15 equal-width bins: top-1 probability for choice/score and affirmative probability for each noul/multi_label candidate. Temperatures are fitted only on calib. The prior baseline uses training-set label frequencies without reading the input.

## External comparisons

External numbers use different subsets and protocols; they provide context rather than a matched comparison.

- jebadiah 4B-v2: accuracy 0.700 and ECE 0.103 on a 300-question subset; see the [ecosystem matrix](../../docs/en/support-matrix.md).
