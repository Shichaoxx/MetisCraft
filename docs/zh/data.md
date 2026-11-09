# 接入自己的数据

用本地清单导入带监督信息的 Record JSONL，即可生成有版本和校验和的数据集，无需修改
内置注册表。

## 声明输入文件

在 `input/` 旁创建 `dataset.yaml`：

```yaml
name: support
version: labeled-v1
source: your-organization/support-labels
license: LicenseRef-Internal-Data
terms: [research_only]
splits:
  train: input/train.jsonl
  calib: input/calib.jsonl
  dev: input/dev.jsonl
  test: input/test.jsonl
```

将示例中的来源、许可证和条款替换成数据实际适用的值。`name` 和 `version` 必须是以
字母或数字开头的单个路径段。划分文件路径相对于清单，不能越出清单所在目录；同一源文件
不能同时供给多个划分。

自定义条款会原样保留。产物的 `commercial_use` 字段只汇总可识别的限制标记：
已声明限制时为 `false`，有未解释的条款时为 `null`，未记录可识别限制时为 `true`。
该字段不授予使用许可；仍需核对实际数据条款与基座许可证。

## 编写 Record JSONL

每行是一道带监督信息的问题，例如：

```json
{"id":"ticket-1","task_id":"routing","input":{"query":"Which team?","context":{"state":"My card has not arrived."}},"question":{"type":"choice","options":[{"id":"billing"},{"id":"support"}]},"group":{"id":"case-1","split_key":"conversation-1"},"supervision":{"kind":"choice","labels":"support"}}
```

记录 ID 在全部导入划分中必须唯一。`task_id` 是校准使用的任务名，不必与数据集名称相同。
导入器补充来源信息和输入哈希；原有 provenance 会保留在嵌套元数据中。

| 判断类型 | 问题字段 | 硬标签 |
|---|---|---|
| `noul` | `type: noul` | 布尔值 |
| `choice` | ID 唯一的 `options` | 选项 ID；仅启用 `allow_null` 时可为 `null` |
| `score` | ID 唯一且有顺序的 `levels` | 从零开始的等级下标 |
| `multi_label` | 顶层 ID 唯一的 `candidates` | 正例候选 ID 列表 |

`supervision.kind` 必须与问题类型相同。也可在 `supervision.soft` 中提供与目标顺序对齐的
软概率。数据导入要求监督信息；用于预测的 Record 可以省略监督信息。
`choice` 和 `score` 的软概率之和须在 `1e-6` 容差内等于 1；`noul` 与
`multi_label` 使用独立概率。选项 ID `__null__` 为保留值，需空答案时使用 `allow_null`。

新导入会记录 `supervision.has_hard_label`，区分缺失标签与合法 null、`false`、空正例集合；仅有软监督时也可显式设为 `false`。提供软目标时训练和校准沿用软目标，硬准确率／F1 只在已知硬标签上计算。报告给出 `n_hard`、`n_soft` 和 `n_hard_unknown`；含义不明的历史 null＋soft 记录只算软指标。未定义指标以 JSON `null` 和原因表示。

## 分开拟合与评测用途

`train` 用于模型拟合，`calib` 用于温度拟合，`dev`、`test` 和 `eval_only` 用于评测。
训练会在加载模型前验证这些用途，拒绝训练、校准和评测之间共享输入、`group.id` 或
`group.split_key`；检查也包括本次运行未选中的留出划分。

同一案例、对话或文档的问题应共享 group，保证整体划分。不同评测集之间允许重叠，
但重叠结果不能当作相互独立的证据。本地导入保留显式划分：发现冲突时报错，不静默删除
或移动记录。

## 导入并校验

```bash
metiscraft data import dataset.yaml --data-root ./data
metiscraft data verify support --version labeled-v1 --data-root ./data
metiscraft data show support --version labeled-v1 --split train -n 2 --data-root ./data
```

结果位于 `data/support/labeled-v1/`，包括源文件副本、转换后的划分、用途清单、
`dataset.yaml` 和 `SHA256SUMS`。清单与输入字节都相同的重复导入会验证已有版本；
输入或清单内容改变时必须使用新版本，导入器不会覆盖已有版本。不要手工修改已封存的数据文件。

在[训练配方](training.md)中设置 `data.dataset: support` 和 `data.version: labeled-v1`。
内置数据集可用 `metiscraft data list` 查看，以 `metiscraft data prepare NAME` 按固定来源
版本准备。自定义导入不会变成注册表条目。

[English](../en/data.md)
