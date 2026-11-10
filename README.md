<p align="center">
  <a href="https://shichaoxx.github.io/MetisCraft/en/">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/visuals/figures/wordmark-dark.svg">
    <img src="docs/visuals/figures/wordmark-light.svg" alt="MetisCraft" width="320">
  </picture>
  </a>
  <br><b>A toolkit for JevLike decision models.</b>
</p>

<p align="center">
  <a href="https://github.com/Shichaoxx/MetisCraft/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Shichaoxx/MetisCraft/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <a href="LICENSE"><img alt="Apache 2.0" src="https://img.shields.io/badge/license-Apache--2.0-087f78"></a>
  <a href="https://shichaoxx.github.io/MetisCraft/en/installation/"><img alt="Python 3.10 and 3.11" src="https://img.shields.io/badge/python-3.10%20%7C%203.11-243438"></a>
  <a href="https://shichaoxx.github.io/MetisCraft/en/changelog/"><img alt="0.1.0rc1 public preview" src="https://img.shields.io/badge/preview-0.1.0rc1-bd8c36"></a>
</p>

<p align="center">
  README: <b>English</b> · <a href="README_zh.md">简体中文</a><br>
  <a href="https://shichaoxx.github.io/MetisCraft/en/tensor-analysis/introduction/">Intro</a> ·
  <a href="https://shichaoxx.github.io/MetisCraft/en/cookbook/first-model/">Get started</a> ·
  <a href="https://shichaoxx.github.io/MetisCraft/en/cookbook/">Cookbook</a> ·
  <a href="https://shichaoxx.github.io/MetisCraft/en/tensor-analysis/">Concepts & Tensors</a> ·
  <a href="https://shichaoxx.github.io/MetisCraft/en/">Documentation</a> ·
  <a href="https://shichaoxx.github.io/MetisCraft/en/contributing/">Contribute</a>
</p>

**Train, reproduce and deploy JevLike models.** One toolkit connects training, calibration, evaluation and serving. Reproduce an open recipe or build a decision model for your own labeled task.

## Tev1 4B, reproduced with MetisCraft

| Evaluation set | MetisCraft 4B | Original Tev1 4B | Difference |
|---|---:|---:|---:|
| Main decisions · 1,000 questions | **87.3% · 873/1,000** | 88.0% | −0.70 pp |
| Policy transfer · 300 questions | **99.33% · 298/300** | 100% | −0.67 pp |

**Both predeclared accuracy checks passed.** Single seed, the same public development questions; the original training history was not recovered, and calibration can worsen NLL. [Reproduction record and limits](https://shichaoxx.github.io/MetisCraft/en/reproductions/tev1/) · [Training recipe](https://shichaoxx.github.io/MetisCraft/en/recipes/tev1/) · [Model architecture atlas](https://shichaoxx.github.io/MetisCraft/en/tensor-analysis/architectures/).

## One message. Two decisions.

> “My card was stolen this morning and there are two payments on it that I did not make.”

An existing customer-routing model selects **security** and answers whether the message is urgent. This is an actual response excerpt from run `20261105-101057-30757bf7`, on a constructed message. A probability is not an accuracy score.

```json
{
  "queue": {"choice": "security", "confidence": 0.9999995054983101},
  "urgent": {"noul": 0.9999314182034214}
}
```

[Full request and response](https://shichaoxx.github.io/MetisCraft/en/cookbook/decision/) · [Recipe and test results](https://shichaoxx.github.io/MetisCraft/en/recipes/banking77_support/). On the same run, queue accuracy is **97.8%**, versus a **24.7%** prior baseline (n=3,080). This is a teaching scenario, not a production deployment.

## Four kinds of judgment. One workflow.

| Judgment | A question you can ask | Python / batch | HTTP |
|---|---|:---:|:---:|
| Yes/no `noul` | Is this message urgent? | ✓ | ✓ |
| Choice `choice` | Which team should handle it? | ✓ | ✓ |
| Ordered score `score` | What severity level applies? | ✓ | ✓ |
| Multiple labels `multi_label` | Which topics are present? | ✓ | — |

Choose answer-token logits or a candidate scalar head in your recipe. Train with LoRA, full or frozen backbones. Artifacts record configuration, data versions, base identity and evaluation results. Calibration uses held-out data; its benefit must be measured on your target distribution.

## Inside a decision model

One recorded customer-routing configuration combines **Qwen3-0.6B-Base + LoRA** with a shared **1024 → 256 → 1 scalar head**. In `list` layout, candidate-end hidden states are gathered from a causal sequence, scored and normalized over the defined answer space.

<a href="https://shichaoxx.github.io/MetisCraft/en/tensor-analysis/architectures/">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/tensor-analysis/figures/metiscraft-architecture.en.dark.svg">
    <img src="docs/tensor-analysis/figures/metiscraft-architecture.en.svg" alt="MetisCraft scalar-list tensor architecture: tokens, hidden-state gather, shared MLP and probabilities" width="100%">
  </picture>
</a>

[Explore the architecture](https://shichaoxx.github.io/MetisCraft/en/tensor-analysis/architectures/) for tensor shapes, candidate visibility, numerical examples and source references.

<details>
<summary><b>Tev1 4B: the answer-token reproduction architecture</b></summary>

Tev1 uses the existing **Qwen3.5-4B vocabulary head**. The MetisCraft reproduction trains LoRA with full-vocabulary letter/EOS completion targets, then selects answer-code logits at inference.

<a href="https://shichaoxx.github.io/MetisCraft/en/reproductions/tev1/">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/tensor-analysis/figures/tev1-architecture.en.dark.svg">
    <img src="docs/tensor-analysis/figures/tev1-architecture.en.svg" alt="Tev1-4B hybrid backbone, vocabulary projection, answer-code gather and MetisCraft completion training" width="100%">
  </picture>
</a>

[Reproduction protocol and limits](https://shichaoxx.github.io/MetisCraft/en/reproductions/tev1/) distinguish the upstream constrained-generation protocol from direct-logit evaluation.

</details>

## Get started

[Compatibility and support](https://shichaoxx.github.io/MetisCraft/en/compatibility/) defines the tested dependency environments, interfaces and artifact-loading rules for `0.1.0rc1`.

Use Python 3.10 or 3.11 and install a suitable PyTorch build for your hardware. Source installation is available now:

```bash
git clone --branch release/0.1.0rc1 https://github.com/Shichaoxx/MetisCraft.git
cd MetisCraft
python -m pip install '.[serve]'
```

In a workspace outside the checkout:

```bash
metiscraft init my-first-judge
cd my-first-judge
python prepare.py
metiscraft data import dataset.yaml --data-root ./data
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 metiscraft train recipe.yaml
```

[Run your first model](https://shichaoxx.github.io/MetisCraft/en/cookbook/first-model/) includes installation, expected outputs, evaluation and HTTP calls. It uses a tiny random model and synthetic data: **no GPU or weight download required**. Then [bring your own task](https://shichaoxx.github.io/MetisCraft/en/cookbook/own-task/).

## Recipes and results

| Recipe | Recorded evidence | Boundary |
|---|---|---|
| [Customer routing](https://shichaoxx.github.io/MetisCraft/en/recipes/banking77_support/) | One model answers choice and yes/no questions | A teaching scenario derived from Banking77 |
| [Banking77](https://shichaoxx.github.io/MetisCraft/en/recipes/banking77/) | Two readouts under one protocol | No matched encoder comparison yet |
| [Tev1 4B](https://shichaoxx.github.io/MetisCraft/en/reproductions/tev1/) | 87.3% vs original 88.0%; transfer 99.33% vs 100% | Accuracy targets met; training-history and calibration limits remain |
| [BoolQ](https://shichaoxx.github.io/MetisCraft/en/recipes/boolq/) · [SST-5](https://shichaoxx.github.io/MetisCraft/en/recipes/sst5/) · [GoEmotions](https://shichaoxx.github.io/MetisCraft/en/recipes/go_emotions/) | Recipes across judgment types | GoEmotions results predate the prior-shift fix; see the recipe |

Every number links to a recipe or reproduction record. The [ecosystem matrix](https://shichaoxx.github.io/MetisCraft/en/support-matrix/) covers JevAny, ms-swift and other projects; [limitations](https://shichaoxx.github.io/MetisCraft/en/limitations/) distinguish software checks from model-quality evidence.

## Documentation and community

[Cookbook](https://shichaoxx.github.io/MetisCraft/en/cookbook/) · [Concepts & Tensors](https://shichaoxx.github.io/MetisCraft/en/tensor-analysis/) · [Five-minute walkthrough](https://shichaoxx.github.io/MetisCraft/en/tensor-analysis/walkthrough/) · [Data format](https://shichaoxx.github.io/MetisCraft/en/data/) · [Training](https://shichaoxx.github.io/MetisCraft/en/training/) · [Serving](https://shichaoxx.github.io/MetisCraft/en/serving/) · [Changelog](https://shichaoxx.github.io/MetisCraft/en/changelog/)

[Contribute](https://shichaoxx.github.io/MetisCraft/en/contributing/), [report a bug](https://github.com/Shichaoxx/MetisCraft/issues/new?template=bug_report.yml) or [suggest a feature](https://github.com/Shichaoxx/MetisCraft/issues/new?template=feature_request.yml).

[JevLike introduction: task, data and tensor architecture](https://shichaoxx.github.io/MetisCraft/en/tensor-analysis/introduction/)

## License

Code is [Apache-2.0](LICENSE). Datasets and base models retain their own terms; check metadata and the artifact `NOTICE` before distributing trained weights.
