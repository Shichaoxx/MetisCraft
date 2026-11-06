# 加载与部署产物

使用训练输出的 `artifact/` 目录，并将对应的 `calibration/` 留在旁边，Predictor 会自动
发现它。校准目录在其他位置时用 `--calibration` 指定；加载时会核验校准文件是否属于
这份产物。

## 预测 Record JSONL

将 `RUN` 设置为运行目录后执行：

```bash
CUDA_VISIBLE_DEVICES='' metiscraft predict "$RUN/artifact" records.jsonl
```

Record 预测支持 `noul`、`choice`、`score` 和 `multi_label`，具体取决于产物采用的读出。
Record 自带 `task_id`。输出包括决策、概率、原始分数与基座验证状态；加载了校准参数时，
还会报告所选温度、分组和 `fallback_reason`。

Python 中的对应用法：

```python
from metiscraft.serve.predictor import Predictor

predictor = Predictor.load("runs/quickstart/replace-with-the-run-directory/artifact")
results = predictor.predict_records(records)
```

其中 `records` 是 Record 对象或字典的列表。在 Python 进程中指定 CPU 时，先导入
`torch`，再传入 `device=torch.device("cpu")`。

## 提供 `/v1/systemone` 服务

安装 `serve` 依赖组后，在前台启动服务：

```bash
CUDA_VISIBLE_DEVICES='' metiscraft serve "$RUN/artifact" --host 127.0.0.1 --port 8000
```

另开终端，发送快速上手中的请求：

```bash
curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @request.json
curl -sS http://127.0.0.1:8000/health
```

用 Ctrl-C 停止服务。HTTP 端点支持 **`noul`、`choice` 和 `score`**；`multi_label` 使用
Record 预测。`/health` 报告是否加载了校准器以及 `base_verification` 状态，不评估模型效果。
这个服务不包含身份认证或完整的生产部署设施。

## 保持任务路由一致

只有产物元数据的 `task_ids` 明确声明了一个 task ID 时，HTTP 问题才自动使用该任务。
声明了多个任务时，必须将每个请求问题名映射到其中一个 task ID；缺少映射会报错，
不会静默选用另一个校准任务。

没有声明 `task_ids` 的产物，包括历史产物，也必须为每个 HTTP 问题提供显式映射。
仅有一个已保存温度分组不能证明模型只有一个任务：小样本任务可能没有单独的温度。
此时映射为本次加载明确声明 task ID，应使用原始数据中的任务 ID，而不是配方名称。

例如，产物包含 `support/routing` 和 `support/urgency` 时，创建 `task-map.yaml`：

```yaml
team: support/routing
urgency: support/urgency
```

对这个多任务运行使用：

```bash
metiscraft serve "$RUN/artifact" --task-map task-map.yaml
metiscraft predict "$RUN/artifact" request.jsonl --task-map task-map.yaml
```

也可以在训练配方的 `serve.task_map` 中保存映射，或传给 `Predictor.load(..., task_map=...)`。
CLI 传入的映射会替换产物中保存的映射。Record 输入仍使用它自己明确给出的 `task_id`。

校准先找 `<task_id>|<kind>`；找不到则使用 `*|<kind>`，并记录原因
`task_temperature_unavailable`；再找不到则使用温度 1，原因是
`kind_temperature_unavailable`。未加载校准器时，Record 输出的 `calibration` 为 `null`。
HTTP 保持官方响应结构；路由和校准详情由 `metiscraft.serve.predictor` logger 以 INFO
级别记录。`metiscraft serve` 默认启用这个 logger；直接使用 `Predictor.answer_request`
或 `create_app` 的应用，需要在自己的日志配置中设置它。

## 迁移基座目录

保留完整的已封存基座目录，包括 `SOURCE.json` 和 `SHA256SUMS`。产物记录的原位置是
绝对路径，内容身份使用相对文件名。迁移后可以明确指定新位置：

```bash
metiscraft serve "$RUN/artifact" --base-path ./models/relocated-base
metiscraft predict "$RUN/artifact" records.jsonl --base-path ./models/relocated-base
```

`eval` 也接受 `--base-path`。Python 中使用
`Predictor.load(artifact, base_path="./models/relocated-base")`。显式路径优先，内容仍必须
匹配记录的身份。未覆盖时，加载器先使用仍然存在的原路径，再回退到配方的基座设置和
配置的模型根目录。迁移不需要修改产物，也不需要重新生成基座校验和。

## 历史产物

新产物会将当前本地基座与训练时记录的内容身份核对。内容不符时，即使重新生成校验清单
也会加载失败。

缺少内容身份的历史 `metiscraft-artifact/1` 产物默认拒绝加载。明确需要检查时，可在
`predict`、`serve` 或 `eval` 命令加上 `--allow-legacy-artifact`，或向 `Predictor.load`
传入 `allow_legacy_artifact=True`。加载会发出 warning，并在 Record 结果和 `/health`
中报告 `legacy_unverified`。这不能追溯证明原训练运行使用的字节；已有但损坏的身份信息
也不能用这个开关绕过。

历史产物还缺少 `task_ids` 时，HTTP 请求同时需要 legacy 开关和显式任务映射：

```bash
metiscraft serve "$RUN/artifact" --allow-legacy-artifact --task-map task-map.yaml
metiscraft predict "$RUN/artifact" request.jsonl --allow-legacy-artifact --task-map task-map.yaml
```

Record 预测与数据集评测使用记录本身的 `task_id`，不需要这种 HTTP 问题名映射。

[English](../en/serving.md)
