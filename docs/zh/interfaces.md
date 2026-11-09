# 接口与产物约定

本页定义当前公共契约。操作步骤见[数据](data.md)、[训练](training.md)和[服务](serving.md)。

## 类型化判断

| 类型 | 概率 | Record 决策 | HTTP 结果 |
|---|---|---|---|
| `noul` | 正类概率 | 以 0.5 为阈值 | `noul` 概率 |
| `choice` | 有效选项上的分布 | argmax 选项，可含允许的 null | 选中选项；`confidence` 为最大概率 |
| `score` | 有序等级上的分布 | argmax 等级索引 | 等级索引期望；`confidence` 为最大概率 |
| `multi_label` | 各候选独立概率 | 每列概率 ≥ 0.5 | `/v1/systemone` 不支持 |

Python 和 JSONL 批量入口支持四种类型。答案 token 支持前三种；标量头支持全部类型，默认 `list`，可选择 `pairs`。读出选择不会改变概率定义。

HTTP 请求形状沿用 System One 模式；字段同名不代表 confidence 语义一致。[System One 概率定义](https://docs.typesafe.ai/confidence)对 choice 另作归一化，对 score 采用等级距离定义，详见[训练、校准与输出](tensor-analysis/training-calibration.md)。

## 配方与 Python

配方由 `metiscraft.config.load_recipe` 读取；`metiscraft.pipeline.run_recipe` 执行训练、校准和评测；`metiscraft.pipeline.evaluate_artifact` 评测封存产物。加载预测使用 `metiscraft.serve.predictor.Predictor.load`，数据验证使用 `metiscraft.data.contract.Record`。

训练、校准和评测用途独立，拟合与评测之间的组或输入重叠会被拒绝。`eval_only` 真值只能评测。温度只在校准集拟合，按任务和类型选择，回退信息可检查。校准不保证在分布变化后改善指标。

## 加载与迁移

产物记录基座权重、配置和 tokenizer 的内容身份。同一内容迁移后可用 `--base-path` 加载；内容改变须被检出。历史身份缺失的旧产物需要显式兼容模式，并保持未验证状态。

配置、数据来源、环境、拟合参数和指标用于解释一次运行。数据、基座和代码保留各自许可证；未知条款不构成商业使用授权。

## 进一步阅读

[配方字段](../cookbook/recipe-reference.md) · [原理与张量解析](tensor-analysis/index.md) · [能力边界](limitations.md)
