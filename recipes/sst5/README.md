# SST-5

电影评论句子的五级情感：very negative 到 very positive，按等级打分。除准确率外报告期望分的 MAE 和校准指标。

| 项目 | 内容 |
|---|---|
| 注册名 | `sst5` |
| 上游 | [`SetFit/sst5`](https://huggingface.co/datasets/SetFit/sst5) @ `e51bdcd8` |
| 许可证 | 上游未声明；条款：上游未声明许可证、仅限研究；用它训练的产物：不可商用 |
| 判断类型 | 等级 score |
| 划分 | train 8,540（上游 train）；calib 549（上游 dev 按哈希分半的前一半）；dev 552（上游 dev 按哈希分半的后一半）；test 2,210（上游 test） |

## 配方

| 配方 | 读出 | 微调 | 有效批量 | 轮数 | 学习率（LoRA / 头） |
|---|---|---|---|---|---|
| [`qwen3-0.6b-base_answer-token.yaml`](qwen3-0.6b-base_answer-token.yaml) | ① 答案 token | LoRA r=16 | 16 | 3 | 2e-4 / 1e-3 |

## 运行

```bash
metiscraft data prepare sst5
metiscraft train recipes/sst5/qwen3-0.6b-base_answer-token.yaml
```

## 验收结果

### ① 答案 token：`qwen3-0.6b-base_answer-token.yaml`

运行 `20261105-114305-a49998c0`；提交 `4299224`；数据 `rev-e51bdcd8`；1× RTX 3090；1,602 步，训练 27 分钟；温度：score 2.45。

**test**，等级 score（n=2,210）

| 方法 | 准确率 | 期望分 MAE | NLL | Brier | ECE |
|---|---|---|---|---|---|
| 模型 | 0.555 | 0.506 | 1.380 | 0.682 | 0.270 |
| 模型 + 温度 | 0.555 | 0.559 | 1.014 | 0.569 | 0.040 |
| 先验 | 0.231 | 1.141 | 1.580 | 0.789 | 0.041 |

**dev**，等级 score（n=552）

| 方法 | 准确率 | 期望分 MAE | NLL | Brier | ECE |
|---|---|---|---|---|---|
| 模型 | 0.540 | 0.537 | 1.463 | 0.701 | 0.279 |
| 模型 + 温度 | 0.540 | 0.574 | 1.053 | 0.584 | 0.063 |
| 先验 | 0.257 | 1.082 | 1.576 | 0.787 | 0.014 |

ECE 用 15 个等宽分箱：choice 和 score 取 top-1 概率，noul 和 multi_label 的每个候选取正类概率。“模型 + 温度”只用 calib 划分拟合温度；“先验”按训练集的类别频率给概率，是不看输入的基线。

## 参照

下面的数字来自公开报告，评测子集和协议与这里不同，只用来定位量级。

- encoder 微调强基线：test 准确率 0.596–0.623（作者自报，见[调研](../../docs/research/decision-model-sota.md)）。
