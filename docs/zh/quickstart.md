# 在 CPU 上跑通完整流程

[安装](installation.md)后，从已安装的包导出示例。以下命令不需要源码检出、模型下载或
GPU。示例中的小型 Qwen3 随机初始化，数据也由程序构造；生成的分数只用于验证软件流程，
不代表模型准确率或校准质量。

## 导入与训练

在尚不存在 `my-first-judge` 的目录中执行：

```bash
metiscraft init my-first-judge
cd my-first-judge
python prepare.py
metiscraft data import dataset.yaml --data-root ./data
metiscraft data verify quickstart --version synthetic-v1 --data-root ./data
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 metiscraft train recipe.yaml
```

`prepare.py` 创建 `input/` 和 `models/tiny-qwen3/v1/`，模型目录包括分词器、`SOURCE.json`
和 `SHA256SUMS`；已有输出时会拒绝覆盖。导入产物位于 `data/quickstart/synthetic-v1/`。
训练运行两步 LoRA，在 `calib` 上拟合温度，再评测 `dev` 和 `test`。

## 评测与预测

将训练命令打印的运行路径填入 `RUN`：

```bash
RUN='runs/quickstart/replace-with-the-run-directory'
METISCRAFT_DATA_ROOT="$PWD/data" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  metiscraft eval "$RUN/artifact" quickstart --version synthetic-v1 --split test
CUDA_VISIBLE_DEVICES='' metiscraft predict "$RUN/artifact" records.jsonl
CUDA_VISIBLE_DEVICES='' metiscraft predict "$RUN/artifact" request.jsonl
CUDA_VISIBLE_DEVICES='' metiscraft serve "$RUN/artifact" --host 127.0.0.1 --port 8000
```

两个预测输入分别演示 Record JSONL 和 `/v1/systemone` 请求 JSONL。最后一条命令在前台
运行服务。

## 调用 HTTP 服务

另开终端，进入这个工作目录：

```bash
curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @request.json
```

用 **Ctrl-C** 停止服务。服务需要安装 `serve` 依赖组。这个示例只有 `quickstart` 一个
task ID，因此 Record 和 HTTP 请求会自动选择相同的校准分组。

可以查看运行目录中的 `report.md`、`eval/metrics.json`、`artifact/` 和 `calibration/`。
要获得有意义的模型结果，下一步接入[自己的数据](data.md)和[准备好的预训练基座](training.md)。
该示例不验证真实模型效果、GPU 内核、吞吐量或分布外校准。

[English](../en/quickstart.md)
