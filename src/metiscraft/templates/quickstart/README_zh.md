# 完整的本地流程

这个示例从已安装的包运行，不需要源码目录、模型下载或 GPU。
随机初始化的小型 Qwen3 和构造数据用于验证软件流程；
它们的准确率与校准分数不代表有用的模型效果。

导出一个新工作目录，进入目录，再创建示例输入：

<!-- workflow:start -->

```bash
metiscraft init my-first-judge
cd my-first-judge
python prepare.py
metiscraft data import dataset.yaml --data-root ./data
metiscraft data verify quickstart --version synthetic-v1 --data-root ./data
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 metiscraft train recipe.yaml
```

`prepare.py` 只在当前目录下写入 `input/` 和 `models/tiny-qwen3/v1/`，
并拒绝覆盖已有输出。模型目录包含分词器、`SOURCE.json` 与 `SHA256SUMS`。
导入器写入 `data/quickstart/synthetic-v1/`。
训练执行两步 LoRA 更新，在 `calib` 上拟合温度，并评测 `dev` 与 `test`。
输出位于 `runs/quickstart/` 下。

把训练命令打印的运行路径填入 `RUN`：

```bash
RUN='runs/quickstart/replace-with-the-run-directory'
METISCRAFT_DATA_ROOT="$PWD/data" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  metiscraft eval "$RUN/artifact" quickstart --version synthetic-v1 --split test
CUDA_VISIBLE_DEVICES='' metiscraft predict "$RUN/artifact" records.jsonl
CUDA_VISIBLE_DEVICES='' metiscraft predict "$RUN/artifact" request.jsonl
CUDA_VISIBLE_DEVICES='' metiscraft serve "$RUN/artifact" --host 127.0.0.1 --port 8000
```

在同一工作目录的另一个终端中，调用正在运行的 HTTP 服务：

```bash
curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @request.json
```

完成后用 Ctrl-C 停止服务。HTTP 服务需要 `serve` 扩展。
官方端点接受 `noul`、`choice` 和 `score`；
`multi_label` 通过 Record 预测 API 使用。
本例只有一个任务 ID `quickstart`，因此 HTTP 与 Record 请求会选择相同的校准分组。

<!-- workflow:end -->

## 使用自己的数据

将 `dataset.yaml` 引用的 JSONL 文件替换为带标注的 Record 行；
设置新的数据集名称与版本，描述来源和许可证，并更新配方。
清单中的路径相对于清单文件。最小记录如下：

```json
{"id":"ticket-1","task_id":"routing","input":{"query":"Which team?","context":{"state":"My card has not arrived."}},"question":{"type":"choice","options":[{"id":"billing"},{"id":"support"}]},"supervision":{"kind":"choice","labels":"support"}}
```

导入器会补充来源记录与输入哈希。
训练、校准和评测之间应隔离输入和组；共享输入或组会报错。
使用 `group.id` 或 `group.split_key` 把同一案例、会话或文档派生的问题放在一起。
每条记录需要唯一 ID。
相同内容的导入会核验已有版本；输入改变时需要新版本。
显式划分会原样保留，不会静默去重。

要获得有意义的模型效果，请换用准备好的基座和有代表性的数据。
本 CPU 示例不验证真实模型准确率、GPU 内核、吞吐量或分布外校准效果。
