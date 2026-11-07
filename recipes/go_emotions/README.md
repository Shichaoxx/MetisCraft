# GoEmotions（simplified）

Reddit 评论的情绪多标签：28 个候选（27 种情绪加 neutral），每个候选独立给出概率。大多数评论只有一个标签，所以除 micro-F1 外还要看 macro-F1 和逐候选校准。

| 项目 | 内容 |
|---|---|
| 注册名 | `go_emotions` |
| 上游 | [`google-research-datasets/go_emotions`](https://huggingface.co/datasets/google-research-datasets/go_emotions) @ `add49224`，配置 `simplified` |
| 许可证 | Apache-2.0；条款：可能含个人信息；用它训练的产物：按上游许可证 |
| 判断类型 | 多标签 multi_label |
| 划分 | train 43,252（上游 train）；calib 2,730（上游 validation 按哈希分半的前一半）；dev 2,696（上游 validation 按哈希分半的后一半）；test 5,427（上游 test） |

## 配方

| 配方 | 读出 | 微调 | 有效批量 | 轮数 | 学习率（LoRA / 头） |
|---|---|---|---|---|---|
| [`qwen3-0.6b-base_candidate-scalar.yaml`](qwen3-0.6b-base_candidate-scalar.yaml) | ③ 逐候选标量头（list 布局） | LoRA r=16 | 16 | 1 | 2e-4 / 1e-3 |

## 运行

```bash
metiscraft data prepare go_emotions
metiscraft train recipes/go_emotions/qwen3-0.6b-base_candidate-scalar.yaml
```

## 验收结果

### ③ 逐候选标量头：`qwen3-0.6b-base_candidate-scalar.yaml`

运行 `20261105-085531-38a1bf6f`；提交 `85fd8ac`（工作区有未跟踪文件，已跟踪的代码与提交一致）；数据 `rev-add49224`；1× RTX 3090；2,704 步，训练 65 分钟；温度：multi_label 1.25。

**test**，多标签 multi_label（n=5,427）

| 方法 | micro-F1 | macro-F1 | 完全匹配 | NLL | Brier | ECE |
|---|---|---|---|---|---|---|
| 模型 | 0.429 | 0.409 | 0.065 | 0.221 | 0.070 | 0.109 |
| 模型 + 温度 | 0.429 | 0.409 | 0.065 | 0.215 | 0.068 | 0.118 |
| 先验 | 0.000 | 0.000 | 0.000 | 0.148 | 0.036 | 0.000 |

**dev**，多标签 multi_label（n=2,696）

| 方法 | micro-F1 | macro-F1 | 完全匹配 | NLL | Brier | ECE |
|---|---|---|---|---|---|---|
| 模型 | 0.428 | 0.417 | 0.056 | 0.222 | 0.070 | 0.109 |
| 模型 + 温度 | 0.428 | 0.417 | 0.056 | 0.216 | 0.068 | 0.118 |
| 先验 | 0.000 | 0.000 | 0.000 | 0.149 | 0.037 | 0.000 |

**这次运行用的是修复前的代码**：训练时正样本加权 pos_weight=22.8，推理时没有扣除 ln w，概率整体偏高，温度缩放也修不回来（上表 ECE 不降反升）。修复后的规则是先扣除 ln w 再校准。下表把它用在已保存的 dev/test 打分上；calib 的打分当时没有保存，所以温度取 1。正式结果待有空卡时用修复后的代码重跑。

**test**，离线修正（n=5,427）

| 方法 | micro-F1 | macro-F1 | 完全匹配 | NLL | Brier | ECE |
|---|---|---|---|---|---|---|
| 扣除 ln w（离线，温度 1） | 0.587 | 0.460 | 0.466 | 0.076 | 0.022 | 0.002 |

**dev**，离线修正（n=2,696）

| 方法 | micro-F1 | macro-F1 | 完全匹配 | NLL | Brier | ECE |
|---|---|---|---|---|---|---|
| 扣除 ln w（离线，温度 1） | 0.575 | 0.438 | 0.457 | 0.078 | 0.022 | 0.003 |

ECE 用 15 个等宽分箱：choice 和 score 取 top-1 概率，noul 和 multi_label 的每个候选取正类概率。“模型 + 温度”只用 calib 划分拟合温度；“先验”按训练集的类别频率给概率，是不看输入的基线。

## 参照

下面的数字来自公开报告，评测子集和协议与这里不同，只用来定位量级。
