<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/visuals/figures/hero-dark.svg">
    <img src="docs/visuals/figures/hero-light.svg" alt="MetisCraft — 训练、校准、决策" width="100%">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/Shichaoxx/MetisCraft/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Shichaoxx/MetisCraft/actions/workflows/ci.yml/badge.svg?branch=toolkit-rewrite"></a>
  <a href="LICENSE"><img alt="Apache 2.0" src="https://img.shields.io/badge/license-Apache--2.0-087f78"></a>
  <a href="docs/zh/installation.md"><img alt="Python 3.10 和 3.11" src="https://img.shields.io/badge/python-3.10%20%7C%203.11-243438"></a>
  <a href="CHANGELOG.md"><img alt="候选预览版" src="https://img.shields.io/badge/preview-0.1.0rc1-bd8c36"></a>
</p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b><br>
  <a href="docs/zh/quickstart.md">快速上手</a> ·
  <a href="docs/zh/index.md">文档</a> ·
  <a href="docs/support-matrix.md">生态与支持矩阵</a> ·
  <a href="CONTRIBUTING.md">参与贡献</a>
</p>

**把自己的业务做成决策模型。** MetisCraft 是训练 Jev 类 System One 模型的工具包：一次前向回答有明确类型的问题，用一份配方连接数据、基座、读出、训练、校准、评测与部署。

> **公开预览版准备中。** 当前为 `0.1.0rc1` 候选版，尚未发布到 PyPI，接口可能调整。已验证的范围与限制见[能力边界](docs/zh/limitations.md)，开发进展见[项目状态](docs/status.md)。

## 一次决策是什么样的？

提供一段情境和一道答案范围明确的问题，得到结构化判断与可评测、可校准的概率。

```json
{
  "state": "My card was stolen and I see two payments I did not make.",
  "questions": {
    "queue": {
      "type": "choice",
      "instructions": "Which team should handle this message?",
      "criteria": {"security": "Lost cards and unrecognised payments", "support": "General account help"}
    }
  }
}
```

响应包含 `answers.queue.choice`、`confidence` 和各选项的概率。题目定义要做的判断，数据教会模型怎样回答。一个有真实运行记录的例子见[银行客服分流](recipes/banking77_support/README.md)。

<picture>
  <source media="(max-width: 600px)" srcset="docs/visuals/figures/judgments-mobile.svg">
  <img src="docs/visuals/figures/judgments.svg" alt="四种判断类型；图中概率仅为示意">
</picture>

## 可以做什么？

| 能力 | 预览版范围 |
|---|---|
| 类型化判断 | Record API 支持是非、选择、等级打分与多标签 |
| 可组合读出 | 答案 token logits、逐候选标量头；LoRA、全参和冻结基座 |
| 自有数据 | 导入本地 Record JSONL，显式指定训练、校准和评测划分 |
| 校准与评测 | 按任务／类型拟合温度；NLL、Brier、ECE 与任务指标；记录校准回退情况 |
| 运行记录 | 配置、数据版本、可获得的源码提交、环境与经过校验的基座身份 |
| 部署 | Python／批量预测支持四类型；`/v1/systemone` 支持是非、选择与等级 |

<picture>
  <source media="(max-width: 600px)" srcset="docs/visuals/figures/pipeline-mobile.svg">
  <img alt="数据、训练、校准、评测、部署全流程" src="docs/visuals/figures/pipeline.svg">
</picture>

温度在独立校准集上拟合。是否改善概率，需要在目标评测分布上验证；分布变化后的置信度不因此获得保证。

## 安装

使用 Python 3.10 或 3.11，先安装适合硬件的 PyTorch，再从源码安装：

```bash
git clone --branch toolkit-rewrite https://github.com/Shichaoxx/MetisCraft.git
cd MetisCraft
python -m pip install '.[serve]'
```

开发安装使用 `python -m pip install -e '.[dev,serve,reproduce]'`。本地构建的发行包可以通过 `python -m pip install './dist/metiscraft-0.1.0rc1-py3-none-any.whl[serve]'` 安装。完整前置条件与模型准备方式见[安装说明](docs/zh/installation.md)。

## 跑通第一个闭环

以下示例使用**微型随机模型与构造数据**，在 CPU 上检查软件流程，无需下载模型；其中的分数不代表模型效果。在源码仓库外的新目录运行：

```bash
metiscraft init my-first-judge
cd my-first-judge
python prepare.py
metiscraft data import dataset.yaml --data-root ./data
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 metiscraft train recipe.yaml
```

训练会打印运行目录，用该目录下的产物启动服务：

```bash
CUDA_VISIBLE_DEVICES='' metiscraft serve <运行目录>/artifact
# 在同一工作目录下的另一个终端调用：
curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @request.json
```

[完整快速上手](docs/zh/quickstart.md)还包括完整性校验、评测、批量预测和停止服务。随后可[导入自有数据](docs/zh/data.md)，将随机模型替换为有版本记录的预训练基座。

## 配方与证据

| 配方 | 展示内容 | 证据边界 |
|---|---|---|
| [Banking77](recipes/banking77/README.md) | 同一任务下比较两种读出 | Qwen3-0.6B 配方结果；尚无 encoder 同协议对照 |
| [客服分流](recipes/banking77_support/README.md) | 一个模型回答多个业务问题 | 从 Banking77 派生的公开示例，不代表生产部署验证 |
| [BoolQ](recipes/boolq/README.md) · [SST-5](recipes/sst5/README.md) | 是非与等级判断 | 分别遵循各数据集条款 |
| [GoEmotions](recipes/go_emotions/README.md) | 多标签的独立概率 | 已报告训练发生在先验偏移修复前；修复后代码重训待完成 |
| [Tev1](docs/reproductions/tev1.md) | 重建公开数据并复现配方 | **4B 准确率标准达标**：test 87.3%，原版 88.0%；transfer 99.33%，原版 100%。答案一致率 94.6%／99.33%；校准与历史训练过程仍有限制 |

MetisCraft 是决策模型生态中的一部分。[支持矩阵](docs/support-matrix.md)区分兼容性、实现与复现证据，收录 JevAny、ms-swift、Unsloth 等项目，不宣称首个或唯一的统一工具包。

## 学习与参与

- **使用**：[安装](docs/zh/installation.md) · [自有数据](docs/zh/data.md) · [训练](docs/zh/training.md) · [服务](docs/zh/serving.md)。
- **理解原理**：[Cookbook](docs/cookbook/README.md) · [读出与张量图](docs/readouts/README.md)。
- **参与改进**：[贡献指南](CONTRIBUTING.md) · [报告问题](https://github.com/Shichaoxx/MetisCraft/issues/new?template=bug_report.yml) · [建议功能](https://github.com/Shichaoxx/MetisCraft/issues/new?template=feature_request.yml)。
- **关注进展**：[版本变化](CHANGELOG.md) · [状态与下一步](docs/status.md) · [已知限制](docs/zh/limitations.md)。

## 许可证

代码使用 [Apache-2.0](LICENSE)。数据集与基座权重各有独立条款；使用或分发训练产物前，请查看数据元信息与产物 `NOTICE`。

MetisCraft 的名字来自 *mētis*（实践中的智巧）与 *craft*（手艺、打造）。猫头鹰标识借用雅典娜与智慧的联系。由 [shichao](https://github.com/Shichaoxx) 维护。早期 Metis 0.1.0 保留在 [`metis-v0.1.0`](https://github.com/Shichaoxx/MetisCraft/tree/metis-v0.1.0)。
