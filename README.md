<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/visuals/figures/wordmark-dark.svg">
    <img src="docs/visuals/figures/wordmark-light.svg" alt="MetisCraft" width="320">
  </picture>
  <br><b>Train. Calibrate. Decide.</b>
</p>

<p align="center">
  <a href="https://github.com/Shichaoxx/MetisCraft/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Shichaoxx/MetisCraft/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <a href="LICENSE"><img alt="Apache 2.0" src="https://img.shields.io/badge/license-Apache--2.0-087f78"></a>
  <a href="docs/en/installation.md"><img alt="Python 3.10 and 3.11" src="https://img.shields.io/badge/python-3.10%20%7C%203.11-243438"></a>
  <a href="CHANGELOG.md"><img alt="0.1.0rc1 public preview" src="https://img.shields.io/badge/preview-0.1.0rc1-bd8c36"></a>
</p>

<p align="center">
  <b>English</b> · <a href="README_zh.md">简体中文</a><br>
  <a href="docs/en/cookbook/first-model.md">Get started</a> ·
  <a href="docs/en/cookbook/index.md">Cookbook</a> ·
  <a href="https://shichaoxx.github.io/MetisCraft/en/">Documentation</a> ·
  <a href="CONTRIBUTING.md">Contribute</a>
</p>

**Decision models, built for your tasks.** Turn your task into a decision model. Bring labeled data and a local base model; one recipe takes you from training and calibration to evaluation and serving.

## One message. Two decisions.

> “My card was stolen this morning and there are two payments on it that I did not make.”

An existing customer-routing model selects **security** and answers whether the message is urgent. This is an actual response excerpt from run `20261105-101057-30757bf7`, on a constructed message. A probability is not an accuracy score.

```json
{
  "queue": {"choice": "security", "confidence": 0.9999995054983101},
  "urgent": {"noul": 0.9999314182034214}
}
```

[Full request and response](docs/en/cookbook/decision.md) · [Recipe and test results](recipes/banking77_support/README.md). On the same run, queue accuracy is **97.8%**, versus a **24.7%** prior baseline (n=3,080). This is a teaching scenario, not a production deployment.

## Four kinds of judgment. One workflow.

| Judgment | A question you can ask | Python / batch | HTTP |
|---|---|:---:|:---:|
| Yes/no `noul` | Is this message urgent? | ✓ | ✓ |
| Choice `choice` | Which team should handle it? | ✓ | ✓ |
| Ordered score `score` | What severity level applies? | ✓ | ✓ |
| Multiple labels `multi_label` | Which topics are present? | ✓ | — |

Choose answer-token logits or a candidate scalar head in your recipe. Train with LoRA, full or frozen backbones. Artifacts record configuration, data versions, base identity and evaluation results. Calibration uses held-out data; its benefit must be measured on your target distribution.

<picture>
  <source media="(max-width: 600px)" srcset="docs/visuals/figures/pipeline-mobile.svg">
  <img src="docs/visuals/figures/pipeline.svg" alt="Data, train, calibrate, evaluate, serve">
</picture>

## Get started

Use Python 3.10 or 3.11 and install a suitable PyTorch build for your hardware. Source installation is available now:

```bash
git clone --branch main https://github.com/Shichaoxx/MetisCraft.git
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

[Run your first model](docs/en/cookbook/first-model.md) includes installation, expected outputs, evaluation and HTTP calls. It uses a tiny random model and synthetic data: **no GPU or weight download required**. Then [bring your own task](docs/en/cookbook/own-task.md).

## Recipes and results

| Recipe | Recorded evidence | Boundary |
|---|---|---|
| [Customer routing](recipes/banking77_support/README.md) | One model answers choice and yes/no questions | A teaching scenario derived from Banking77 |
| [Banking77](recipes/banking77/README.md) | Two readouts under one protocol | No matched encoder comparison yet |
| [Tev1 4B](docs/reproductions/tev1.md) | 87.3% vs original 88.0%; transfer 99.33% vs 100% | Accuracy targets met; training-history and calibration limits remain |
| [BoolQ](recipes/boolq/README.md) · [SST-5](recipes/sst5/README.md) · [GoEmotions](recipes/go_emotions/README.md) | Recipes across judgment types | GoEmotions results predate the prior-shift fix; see the recipe |

Every number links to a recipe or reproduction record. The [ecosystem matrix](docs/support-matrix.md) covers JevAny, ms-swift and other projects; [limitations](docs/en/limitations.md) distinguish software checks from model-quality evidence.

## Documentation and community

[Cookbook](docs/en/cookbook/index.md) · [Data format](docs/en/data.md) · [Training](docs/en/training.md) · [Serving](docs/en/serving.md) · [Changelog](CHANGELOG.md)

[Contribute](CONTRIBUTING.md), [report a bug](https://github.com/Shichaoxx/MetisCraft/issues/new?template=bug_report.yml) or [suggest a feature](https://github.com/Shichaoxx/MetisCraft/issues/new?template=feature_request.yml).

## License

Code is [Apache-2.0](LICENSE). Datasets and base models retain their own terms; check metadata and the artifact `NOTICE` before distributing trained weights.
