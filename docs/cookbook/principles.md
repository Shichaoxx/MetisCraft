# 决策模型的计算原理

MetisCraft 是 JevLike 判断模型工具包。输入由材料、问题和答案空间组成；模型产生候选分数，再按题型转换成概率与结构化判断。完整机制见[原理与张量解析](../zh/tensor-analysis/index.md)，接口定义见[接口与产物](../zh/interfaces.md)。

## 从请求到张量

同一请求可以包含多个问题，每个问题成为一个 `Record`。因此批大小 $B$ 指问题数，不是 HTTP 请求数。$K_b$ 是第 $b$ 题的候选数，$K_{\max}$ 是批内候选填充宽度。输入 padding mask 与候选有效 mask 含义不同，详见[张量、轴与索引](../zh/tensor-analysis/notation.md)。

来源与分组记录是计算契约的一部分：同一材料派生的问题共享 `group`，拟合与评测用途不得交叉；`provenance` 保留来源版本和原始行号。

## 两类读出，三条实现路径

| 路径 | 核心计算 | 候选可见性 |
|---|---|---|
| `answer_token` | 末位置 hidden → 词表 logits → 答案码 gather | 提示内按因果方向可见；读出位置看到全部候选 |
| `candidate_scalar` / `list`（默认） | 候选位置 gather → 共享 MLP | 后候选可以看到前候选 |
| `candidate_scalar` / `pairs` | 每候选一行 → 末位置 MLP → scatter | 不同候选行不互相读取 |

详细公式和数值例见[答案 token](../zh/tensor-analysis/answer-token.md)及[标量头](../zh/tensor-analysis/scalar-heads.md)。pairs 可以批量执行；序列数不等于串行模型调用次数。共享前缀树是研究机制，当前包没有实现。

## 从分数到概率

| 类型 | 概率 | Record 决策 |
|---|---|---|
| `noul` | 单个正类 logit 的 sigmoid | 0.5 阈值 |
| `choice` | 有效候选内 softmax | argmax 选项 |
| `score` | 有序等级内 softmax | argmax 等级索引 |
| `multi_label` | 各候选 sigmoid | 各列以 0.5 为阈值 |

HTTP 的 `score` 是等级索引的概率期望；choice/score 的 `confidence` 是最大候选概率。它们与 Record 决策以及 System One 官方 confidence 定义分别说明，不能混用。

## 训练与校准

候选空间损失与 completion 全词表损失不同。completion 的两个预测位置分别预测答案码和 EOS。训练损失按优化步中的全局问题数归一；梯度等价性比较需要控制随机性、padding、样本顺序等条件，不意味着不同训练配置必然产生相同模型。

推理先应用已记录的正类加权 logit 修正，再除以校准温度 $\tau$。温度由独立 `calib` 拟合，并按任务／类型选择；模型评测同时保留原始与校准结果。正温度保持排序，但可改变等级期望；校准也可能恶化目标分布上的 NLL。见[训练、校准与输出](../zh/tensor-analysis/training-calibration.md)。

## 证据与边界

[Banking77](../../recipes/banking77/README.md)同协议准确率为答案 token 0.928、scalar-list 0.927；这不是普遍优劣结论。[Tev1](../reproductions/tev1.md)保留公开开发题、单种子和校准限制。[GoEmotions](../../recipes/go_emotions/README.md)的历史训练早于先验修正，离线修正不能替代重训。

机制、指标和来源分别核对；[七类读出对照](../zh/tensor-analysis/readout-comparison.md)中未实现的路径不构成安装包能力。
