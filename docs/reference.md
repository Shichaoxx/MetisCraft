# 接口与产物约定

本页说明当前可用的接口。数据格式、训练和服务的操作分别见[数据](zh/data.md)、[训练](zh/training.md)和[服务](zh/serving.md)。

## 类型化判断

- `noul` 返回正类概率；`choice` 返回选项上的归一化分布。
- `score` 返回有序等级上的归一化分布；`multi_label` 为每个候选独立返回概率。
- Python 与批量入口支持四种类型，HTTP `/v1/systemone` 支持前三种。
- `answer_token` 读取答案码的 logits；`candidate_scalar` 使用逐候选标量头。选择读出不会改变问题类型的概率含义。

## 配方与 Python

配方经 `metiscraft.config.load_recipe` 读取；`metiscraft.pipeline.run_recipe` 执行训练、校准和评测；`metiscraft.pipeline.evaluate_artifact` 评测已封存产物。加载预测使用 `metiscraft.serve.predictor.Predictor.load`，数据验证使用 `metiscraft.data.contract.Record`。

训练、校准、评测划分独立，组间或输入重叠会被拒绝。`eval_only` 真值只能评测。温度仅在校准集拟合，按任务和类型选择；回退原因可检查。校准不保证在分布变化后改善可靠性。

## 加载与迁移

产物保存基座权重、配置和 tokenizer 内容身份；同一内容迁移后可通过 `--base-path` 加载。内容变化必须被发现。缺少历史身份的旧产物需要显式兼容开关，保持未验证状态。

产物中的配置、数据来源、环境、微调参数和评测结果用于解释一次运行。数据集、基座与代码各自保留许可证；未知条款不构成商业使用授权。

## 进一步阅读

[字段参考](cookbook/recipe-reference.md) · [原理](cookbook/principles.md) · [能力边界](zh/limitations.md)
