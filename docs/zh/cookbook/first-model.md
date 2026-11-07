# 在 CPU 上训练第一个模型

**目标：**在一个新工作目录中完成数据导入、训练和校准，检查产物，再发出一次 HTTP 请求。

教程使用随机初始化的小型 Qwen3 和构造标签。你会得到能加载的产物与服务；这些答案用于验证软件流程，不代表业务准确率。

## 开始前

- 按[安装说明](../installation.md)安装包含 `serve` 扩展的包。
- 在已安装环境的 POSIX shell 中执行命令。包已在 Python 3.10 和 3.11 上检查。
- 在源码目录之外选择一个新工作目录。生成的示例不需要下载模型，也不需要 GPU。

## 先认识示例文件

| 文件 | 用途 |
|---|---|
| `prepare.py` | 在本地创建随机小基座和构造标注数据 |
| `dataset.yaml` | 声明数据版本与四个划分 |
| `recipe.yaml` | 连接数据、模型、两步训练、校准和评测 |
| `records.jsonl` | Record 预测接口的输入 |
| `request.jsonl` / `request.json` | 请求格式的批量预测输入与 HTTP 输入 |

## 运行完整流程

按顺序执行下面的命令。这些命令由[随包提供的命令源](../../../src/metiscraft/templates/quickstart/README_zh.md)同步生成。HTTP 服务运行时保持第一个终端打开，在第二个终端发送请求。

<!-- metiscraft:quickstart -->

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

<!-- metiscraft:quickstart:end -->

## 完成后应该看到什么

训练命令打印的运行目录中，主要文件如下：

```text
runs/quickstart/<run-id>/
├── config.yaml
├── data.json
├── model.json
├── train_log.jsonl
├── artifact/
├── calibration/
├── eval/
│   └── metrics.json
└── report.md
```

`artifact/` 保存学习到的参数与元数据；`calibration/` 放在它旁边，供加载器找到温度参数。先读 `report.md` 了解运行情况，程序可读取 `eval/metrics.json` 中的指标。

HTTP 响应应包含 `answers.team.choice`，值为 `billing` 或 `tech`，并给出这两个选项的概率。有效响应与完整产物说明流程已运行；构造数据上的高分不能证明客服场景有效。

如果在服务运行时检查 `/health`，`status: "ok"`、`calibrated: true` 和 `base_verification: "verified"` 描述的是加载状态，不是模型质量。

## 遇到问题时

| 现象 | 检查方法 |
|---|---|
| 找不到 `metiscraft` 命令 | 激活实际安装包的环境。 |
| 工作目录或生成文件已存在 | 换一个新目录；示例有意拒绝覆盖已有文件。 |
| 找不到 `RUN` | 用训练打印的准确路径替换占位值，并留在示例工作目录执行。 |
| 缺少 HTTP 依赖 | 按安装说明，从相同源码或 wheel 安装 `serve` 扩展。 |
| 服务端口已被占用 | 改用另一个 `--port`，并同步修改请求 URL。 |
| 答案不合理 | 随机小模型只训练两步，出现这种情况正常。这里检查产物和响应结构，不检查业务判断质量。 |

## 下一步

[把自己的任务接进来](own-task.md)，将构造标签与随机基座换成你的数据和有版本的预训练模型。也可以先看[客服分流的真实记录](decision.md)，了解一个已训练任务的输出与评测。

[English](../../en/cookbook/first-model.md)
