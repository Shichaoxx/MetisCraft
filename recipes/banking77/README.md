# Banking77

网银客服意图识别：一条用户消息，从 77 个意图里选一个。两份配方只差读出方式，用来在同一协议下对照 ① 答案 token 和 ③ 逐候选标量头。

| 项目 | 内容 |
|---|---|
| 注册名 | `banking77` |
| 上游 | [`PolyAI/banking77`](https://huggingface.co/datasets/PolyAI/banking77) @ `18469841` |
| 许可证 | CC-BY-4.0；条款：需署名；用它训练的产物：按上游许可证 |
| 判断类型 | 选择 choice |
| 划分 | train 8,988（上游 train）；calib 473（从上游 train 按哈希切出 5%）；dev 537（从上游 train 按哈希切出 5%）；test 3,080（上游 test） |

## 配方

| 配方 | 读出 | 微调 | 有效批量 | 轮数 | 学习率（LoRA / 头） |
|---|---|---|---|---|---|
| [`qwen3-0.6b-base_answer-token.yaml`](qwen3-0.6b-base_answer-token.yaml) | ① 答案 token | LoRA r=16 | 16 | 2 | 2e-4 / 1e-3 |
| [`qwen3-0.6b-base_candidate-scalar.yaml`](qwen3-0.6b-base_candidate-scalar.yaml) | ③ 逐候选标量头（list 布局） | LoRA r=16 | 16 | 2 | 2e-4 / 1e-3 |

## 运行

```bash
metiscraft data prepare banking77
metiscraft train recipes/banking77/qwen3-0.6b-base_answer-token.yaml
metiscraft train recipes/banking77/qwen3-0.6b-base_candidate-scalar.yaml
```

## 验收结果

### ① 答案 token：`qwen3-0.6b-base_answer-token.yaml`

运行 `20261105-074726-8a84f136`；提交 `85fd8ac`（工作区有未跟踪文件，已跟踪的代码与提交一致）；数据 `rev-18469841`；1× RTX 3090；1,124 步，训练 59 分钟；温度：choice 1.17。

**test**，选择 choice（n=3,080）

| 方法 | 准确率 | macro-F1 | NLL | Brier | ECE |
|---|---|---|---|---|---|
| 模型 | 0.928 | 0.928 | 0.307 | 0.114 | 0.024 |
| 模型 + 温度 | 0.928 | 0.928 | 0.298 | 0.113 | 0.009 |
| 先验 | 0.013 | 0.000 | 4.384 | 0.988 | 0.006 |

**dev**，选择 choice（n=537）

| 方法 | 准确率 | macro-F1 | NLL | Brier | ECE |
|---|---|---|---|---|---|
| 模型 | 0.896 | 0.879 | 0.453 | 0.163 | 0.044 |
| 模型 + 温度 | 0.896 | 0.879 | 0.427 | 0.160 | 0.033 |
| 先验 | 0.019 | 0.000 | 4.303 | 0.986 | 0.001 |

### ③ 逐候选标量头：`qwen3-0.6b-base_candidate-scalar.yaml`

运行 `20261105-043310-e809a40e`；提交 `42b013c`；数据 `rev-18469841`；1× RTX 3090；1,124 步，训练 38 分钟；温度：choice 1.33。

**test**，选择 choice（n=3,080）

| 方法 | 准确率 | macro-F1 | NLL | Brier | ECE |
|---|---|---|---|---|---|
| 模型 | 0.927 | 0.927 | 0.302 | 0.115 | 0.025 |
| 模型 + 温度 | 0.927 | 0.927 | 0.280 | 0.114 | 0.011 |
| 先验 | 0.013 | 0.000 | 4.384 | 0.988 | 0.006 |

**dev**，选择 choice（n=537）

| 方法 | 准确率 | macro-F1 | NLL | Brier | ECE |
|---|---|---|---|---|---|
| 模型 | 0.899 | 0.892 | 0.391 | 0.155 | 0.047 |
| 模型 + 温度 | 0.899 | 0.892 | 0.354 | 0.151 | 0.034 |
| 先验 | 0.019 | 0.000 | 4.303 | 0.986 | 0.001 |

ECE 用 15 个等宽分箱：choice 和 score 取 top-1 概率，noul 和 multi_label 的每个候选取正类概率。“模型 + 温度”只用 calib 划分拟合温度；“先验”按训练集的类别频率给概率，是不看输入的基线。

## 参照

下面的数字来自公开报告，评测子集和协议与这里不同，只用来定位量级。

- encoder 微调强基线：官方 test 准确率约 0.937–0.944（作者自报，见[调研](../../docs/research/decision-model-sota.md)）。
- jebadiah 4B-v2：300 题子集准确率 0.700，ECE 0.103（见[支持矩阵](../../docs/support-matrix.md)）。
- LLM-as-Jev：Qwen3-0.6B + LoRA（选项似然读出）准确率 0.590（作者自报，见[调研](../../docs/research/decision-model-sota.md)）。
- 零样本 Jev：1,000 条子集准确率约 0.82（作者自报，见[场景调研](../../docs/research/scenarios.md)）。
