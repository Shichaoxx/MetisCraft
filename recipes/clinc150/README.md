# CLINC150

意图识别，候选更多：150 个意图，范围外的请求（上游标为 `oos`）应答 null，一共 151 个候选。用来和 Banking77 对照，看候选数翻倍后读出和校准的表现。

| 项目 | 内容 |
|---|---|
| 注册名 | `clinc150` |
| 上游 | [`clinc/clinc_oos`](https://huggingface.co/datasets/clinc/clinc_oos) @ `155b9c71`，配置 `plus` |
| 许可证 | CC-BY-3.0；条款：需署名；用它训练的产物：按上游许可证 |
| 判断类型 | 选择 choice |
| 划分 | train 15,245（上游 train）；calib 1,592（上游 validation 按哈希分半的前一半）；dev 1,508（上游 validation 按哈希分半的后一半）；test 5,500（上游 test） |

## 配方

| 配方 | 读出 | 微调 | 有效批量 | 轮数 | 学习率（LoRA / 头） |
|---|---|---|---|---|---|
| [`qwen3-0.6b-base_candidate-scalar.yaml`](qwen3-0.6b-base_candidate-scalar.yaml) | ③ 逐候选标量头（list 布局） | LoRA r=16 | 16 | 1 | 2e-4 / 1e-3 |

## 运行

```bash
metiscraft data prepare clinc150
metiscraft train recipes/clinc150/qwen3-0.6b-base_candidate-scalar.yaml
```

## 验收结果

### ③ 逐候选标量头：`qwen3-0.6b-base_candidate-scalar.yaml`

运行 `20261105-060357-53ec012a`；提交 `85fd8ac`；数据 `rev-155b9c71`；1× RTX 3090；953 步，训练 77 分钟；温度：choice 1.36。

**test**，选择 choice（n=5,500）

| 方法 | 准确率 | macro-F1 | NLL | Brier | ECE |
|---|---|---|---|---|---|
| 模型 | 0.923 | 0.938 | 0.341 | 0.125 | 0.045 |
| 模型 + 温度 | 0.923 | 0.938 | 0.294 | 0.121 | 0.019 |
| 先验 | 0.182 | 0.002 | 4.861 | 0.990 | 0.165 |

**dev**，选择 choice（n=1,508）

| 方法 | 准确率 | macro-F1 | NLL | Brier | ECE |
|---|---|---|---|---|---|
| 模型 | 0.961 | 0.961 | 0.170 | 0.059 | 0.020 |
| 模型 + 温度 | 0.961 | 0.961 | 0.147 | 0.056 | 0.009 |
| 先验 | 0.031 | 0.000 | 4.999 | 0.993 | 0.015 |

test 拆开看（按 argmax）：范围内 4,500 条准确率 0.959；范围外（应答 null）1,000 条召回 0.765；判为 null 的 788 条精确率 0.971。

ECE 用 15 个等宽分箱：choice 和 score 取 top-1 概率，noul 和 multi_label 的每个候选取正类概率。“模型 + 温度”只用 calib 划分拟合温度；“先验”按训练集的类别频率给概率，是不看输入的基线。

## 参照

下面的数字来自公开报告，评测子集和协议与这里不同，只用来定位量级。
