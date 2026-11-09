# CLINC150

Intent classification with a larger candidate set: 150 intents plus a null answer for out-of-scope requests (`oos` upstream), giving 151 candidates. It complements Banking77 for examining readout and calibration behavior at a different candidate count.

| Item | Details |
|---|---|
| Registry name | `clinc150` |
| Upstream | [`clinc/clinc_oos`](https://huggingface.co/datasets/clinc/clinc_oos) @ `155b9c71`, configuration `plus` |
| License | CC-BY-3.0; attribution required; trained artifacts retain applicable upstream terms |
| Judgment type | Choice |
| Splits | train 15,245 (upstream train); calib 1,592 (first hash half of upstream validation); dev 1,508 (second hash half of upstream validation); test 5,500 (upstream test) |

## Recipe

| Recipe | Readout | Tuning | Effective batch | Epochs | Learning rate (LoRA / head) |
|---|---|---|---|---|---|
| [`qwen3-0.6b-base_candidate-scalar.yaml`](qwen3-0.6b-base_candidate-scalar.yaml) | ③ Scalar head (list layout) | LoRA r=16 | 16 | 1 | 2e-4 / 1e-3 |

## Run

```bash
metiscraft data prepare clinc150
metiscraft train recipes/clinc150/qwen3-0.6b-base_candidate-scalar.yaml
```

## Recorded results

### ③ Scalar head: `qwen3-0.6b-base_candidate-scalar.yaml`

Run `20261105-060357-53ec012a`; commit `85fd8ac`; data `rev-155b9c71`; 1× RTX 3090; 953 steps, 77 minutes; choice temperature 1.36.

**test**, Choice (n=5,500)

| Method | Accuracy | macro-F1 | NLL | Brier | ECE |
|---|---|---|---|---|---|
| Model | 0.923 | 0.938 | 0.341 | 0.125 | 0.045 |
| Model + temperature | 0.923 | 0.938 | 0.294 | 0.121 | 0.019 |
| Prior | 0.182 | 0.002 | 4.861 | 0.990 | 0.165 |

**dev**, Choice (n=1,508)

| Method | Accuracy | macro-F1 | NLL | Brier | ECE |
|---|---|---|---|---|---|
| Model | 0.961 | 0.961 | 0.170 | 0.059 | 0.020 |
| Model + temperature | 0.961 | 0.961 | 0.147 | 0.056 | 0.009 |
| Prior | 0.031 | 0.000 | 4.999 | 0.993 | 0.015 |

Argmax test breakdown: in-scope accuracy 0.959 on 4,500 examples; null recall 0.765 on 1,000 out-of-scope examples; null precision 0.971 among 788 null predictions.

ECE uses 15 equal-width bins: top-1 probability for choice/score and affirmative probability for each noul/multi_label candidate. Temperatures are fitted only on calib. The prior baseline uses training-set label frequencies without reading the input.

## External comparisons

External numbers use different subsets and protocols; they provide context rather than a matched comparison.
