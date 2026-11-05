# BoolQ

阅读理解是非题：给一段维基百科段落和一个问题，判断答案是不是“是”。

| 项目 | 内容 |
|---|---|
| 注册名 | `boolq` |
| 上游 | [`google/boolq`](https://huggingface.co/datasets/google/boolq) @ `35b264d0` |
| 许可证 | CC-BY-SA-3.0；条款：相同方式共享；用它训练的产物：按上游许可证 |
| 判断类型 | 是非 noul |
| 划分 | train 8,425（上游 train）；calib 483（从上游 train 按哈希切出 5%）；dev 519（从上游 train 按哈希切出 5%）；test 3,270（上游 validation） |

## 配方

| 配方 | 读出 | 微调 | 有效批量 | 轮数 | 学习率（LoRA / 头） |
|---|---|---|---|---|---|
| [`qwen3-0.6b-base_answer-token.yaml`](qwen3-0.6b-base_answer-token.yaml) | ① 答案 token | LoRA r=16 | 16 | 2 | 2e-4 / 1e-3 |

## 运行

```bash
metiscraft data prepare boolq
metiscraft train recipes/boolq/qwen3-0.6b-base_answer-token.yaml
```

## 验收结果

### ① 答案 token：`qwen3-0.6b-base_answer-token.yaml`

待验收。

ECE 用 15 个等宽分箱：choice 和 score 取 top-1 概率，noul 和 multi_label 的每个候选取正类概率。“模型 + 温度”只用 calib 划分拟合温度；“先验”按训练集的类别频率给概率，是不看输入的基线。

## 参照

下面的数字来自公开报告，评测子集和协议与这里不同，只用来定位量级。

- encoder 微调强基线：官方 validation 准确率 0.884（作者自报，见[调研](../../docs/research/decision-model-sota.md)）。这里的 test 就是官方 validation，可以直接对照。
- jebadiah 4B-v2：300 题子集准确率 0.847（随权重公开的逐题记录，见[支持矩阵](../../docs/support-matrix.md)）。
