# 把自己的任务接进来

**目标：**把标注 JSONL 导入为有版本的数据集，使用准备好的预训练基座训练，再调用产物。本篇以两类客服分流为例；请换成你实际想评测的标签与定义。

## 开始前

你需要已安装的 MetisCraft 环境、可使用的标注数据，以及适合所选基座的硬件。HTTP 步骤还需要 `serve` 扩展。如果不熟悉命令，先完成 [CPU 流程](first-model.md)。

从已安装的包导出一个新工作目录：

```bash
metiscraft init my-support
cd my-support
```

模板中的 `prepare.py` 属于构造数据的 CPU 示例。本篇由你提供 `input/` 文件和准备好的模型，再把 `dataset.yaml` 与 `recipe.yaml` 替换为下文的内容。

## 1. 每行写一道带答案的问题

为答案选择稳定的 ID，并写清类别边界。本篇用 `billing` 表示收费、账单与退款问题，用 `tech` 表示技术故障。数据采用 Record JSONL：每行一个完整 JSON 对象。

下面一行用于说明格式，本身不是训练数据集：

```json
{"id":"ticket-1001","task_id":"routing","input":{"query":"Which team should handle this message?","context":{"state":"I was charged twice for this month's plan."}},"question":{"type":"choice","options":[{"id":"billing","text":"Charges, invoices and refunds."},{"id":"tech","text":"Login failures, errors and technical faults."}]},"group":{"id":"conversation-1001","split_key":"conversation-1001"},"supervision":{"kind":"choice","labels":"billing"}}
```

- `input.query` 是问题；`input.context.state` 是需要判断的材料。
- `question.options` 定义答案 ID 与描述；`supervision.labels` 是正确答案的 ID。
- `task_id: routing` 标识用于预测和校准的任务。
- `group` 把同一会话派生的问题放在一起。相关记录使用不同记录 ID，但共用组。

把自己的样本准备为四个文件：

| 文件 | 用途 |
|---|---|
| `input/train.jsonl` | 拟合模型参数 |
| `input/calib.jsonl` | 训练后拟合温度 |
| `input/dev.jsonl` | 比较配置、检查错误 |
| `input/test.jsonl` | 对选定方案做留出评测 |

记录 ID 在所有划分中保持唯一。训练、校准和评测之间不能共享输入、会话或组。各划分应覆盖有代表性的标签和类别；每个文件只放一条记录不能构成有意义的评测。

是非、有序等级和多标签的写法见 [Record 格式参考](../data.md)。

## 2. 描述并封存数据集

将下面的内容保存为 `dataset.yaml`，把示例来源、许可证与条款替换成实际适用于数据的内容。

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

划分路径相对于这份清单，且必须位于它的目录内。自有数据导入无需内置注册表条目，也不需要 `reproduce` 扩展。

```bash
metiscraft data import dataset.yaml --data-root ./data
metiscraft data verify support --version labeled-v1 --data-root ./data
metiscraft data show support --version labeled-v1 --split train -n 2 --data-root ./data
```

完成后应得到 `data/support/labeled-v1/`，包含输入副本、转换后的记录、用途清单、`dataset.yaml` 和 `SHA256SUMS`。导入器会补充来源记录与输入哈希。

不变的导入会核验现有版本。修改标签、输入或清单时，使用新版本号，不直接修改已封存文件。

## 3. 准备预训练基座

一个可选起点是 **Qwen/Qwen3-0.6B-Base**，固定到 `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`。有记录的 Banking77 衍生示例使用了这个基座；这验证的是工具包与基座的组合，不是新任务的效果。

在工作目录中运行下面的 Python 代码，准备一个新的本地版本。它先将模型文件下载到已配置的 Hub 缓存，再把实际内容复制到工作目录，记录来源和校验和；训练不直接从缓存加载权重。

```python
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from metiscraft.data.store import write_sha256sums
from metiscraft.modeling.identity import capture_identity

repo = "Qwen/Qwen3-0.6B-Base"
revision = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
destination = Path("models") / "qwen3-0.6b-base" / revision
if destination.exists():
    raise FileExistsError(f"Use a new model version directory: {destination}")

snapshot = snapshot_download(repo_id=repo, revision=revision)
shutil.copytree(snapshot, destination, symlinks=False)
(destination / "SOURCE.json").write_text(
    json.dumps({"channel": "huggingface", "repo": repo, "revision": revision}, indent=2)
    + "\n",
    encoding="utf-8",
)
write_sha256sums(destination)
print(capture_identity(destination)["sha256"])
```

得到的目录是 `models/qwen3-0.6b-base/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/`。保留其中的权重、分词器、配置、`SOURCE.json` 与 `SHA256SUMS`，使用前检查基座许可证。选择其他模型时，按[基座准备说明](../training.md)选取兼容模型并固定版本。

下方以 CPU fp32 作为明确的起步配置。训练这个预训练模型的工作量远大于随机小模型教程，能否承担取决于可用内存和自己的数据规模。

## 4. 连接数据、模型与 HTTP 问题

用下面的配方替换 `recipe.yaml`，其中基座路径与刚准备的目录一致：

```yaml
task: support
data:
  dataset: support
  version: labeled-v1
model:
  base: qwen3-0.6b-base/da87bfb608c14b7cf20ba1ce41287e8de496c0cd
  dtype: fp32
  max_length: 512
  finetune:
    kind: lora
    r: 8
    alpha: 16
readout:
  kind: candidate_scalar
train:
  epochs: 1
  batch_size: 4
  lr: 0.00005
  bf16: false
calibrate:
  method: temperature
  split: calib
eval:
  splits: [dev, test]
  baselines: [prior]
serve:
  task_map:
    team: routing
paths:
  data_root: data
  models_root: models
  runs_root: runs
```

这是起步配置，不是已调优的结果。它训练一个完整轮次，不保留模板的两步检查限制。按模型与数据调整批量、序列长度和精度。

使用兼容 GPU 时，按自己的环境选择设备；支持 BF16 时可覆盖 `model.dtype=bf16 train.bf16=true`，并去掉下方仅用 CPU 的环境前缀。客服分流记录中的硬件和耗时只描述那次运行的配方与数据，不是对你的任务的估时。

这里几个名称各有用途：`support` 是数据集和运行名称；`routing` 是记录中的任务 ID；`team` 将作为 HTTP 问题名。`serve.task_map` 将 `team` 连接到 `routing`，使服务选择与训练、校准一致的任务。

<aside class="metis-reading-note" data-metis-theme="light" data-metis-light-src="../../visuals/figures/metis-guide-chibi.webp" data-metis-light-full-src="../../visuals/figures/metis-guide-chibi.png" data-metis-dark-src="../../visuals/figures/metis-guide-night.webp" data-metis-dark-full-src="../../visuals/figures/metis-guide-night.png" data-metis-light-alt="穿着学院装、抱着星图书的小墨" data-metis-dark-alt="穿着夜间衣装、抱着星图书的小墨" aria-label="小墨阅读提示" markdown="1">

<figure class="metis-reading-avatar"><a data-metis-full href="../../visuals/figures/metis-guide-chibi.png" target="_blank" rel="noopener" aria-label="在新标签页查看小墨插图"><img data-metis-portrait loading="lazy" decoding="async" src="../../visuals/figures/metis-guide-chibi.webp" alt="穿着学院装、抱着星图书的小墨" width="120" height="120"></a></figure>

<div class="metis-reading-copy" markdown="1">

<p class="metis-reading-title"><strong>小墨提示 · 对齐任务名称</strong></p>

训练前，核对 Record 任务 ID `routing`、HTTP 问题名 `team` 与映射 `team: routing`。标注数据和请求中的选项 ID 与定义也应保持一致。

</div>
</aside>

## 5. 训练并检查留出结果

在工作目录根部执行；配方中的相对路径以当前目录为准。

```bash
metiscraft config show recipe.yaml
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 metiscraft train recipe.yaml
```

第一条命令打印解析后的配置。第二条在 `train` 上训练，在 `calib` 上拟合温度，再评测 `dev` 和 `test`。把打印的运行路径填入 `RUN`：

```bash
RUN='runs/support/replace-with-the-run-directory'
METISCRAFT_DATA_ROOT="$PWD/data" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  metiscraft eval "$RUN/artifact" support \
  --version labeled-v1 --split test
```

打开该目录中的 `report.md`。先将模型准确率与先验基线比较，再比较校准前后的 NLL、Brier 和 ECE。检查错误样本，而不只看均值；使用 `dev` 作开发决策时，保留 `test` 作为独立评测。

完成后应有 `artifact/`、同级的 `calibration/` 与评测输出。运行同时记录了导入数据和基座的身份。业务效果需要在自己的任务上测量，不能由命令顺利结束推出。

## 6. 调用产物

将下面这条构造示例保存为 `request.json`。选项 ID、定义和问题说明与上面的训练记录一致：

```json
{
  "state": "The app closes whenever I try to sign in.",
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this message?",
      "criteria": {
        "billing": "Charges, invoices and refunds.",
        "tech": "Login failures, errors and technical faults."
      }
    }
  }
}
```

启动服务：

```bash
CUDA_VISIBLE_DEVICES='' metiscraft serve "$RUN/artifact" --host 127.0.0.1 --port 8000
```

在同一工作目录的另一个终端执行：

```bash
curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @request.json
```

响应应包含 `team` 的选择结果，以及 `billing`、`tech` 两项的概率。本篇不预设模型会返回怎样的具体数值。检查完成后，用 **Ctrl-C** 停止前台服务。

批量 Record 预测与 Python 加载方法见[服务指南](../serving.md)。

停止服务后，可以批量预测，或在 Python 中加载同一产物：

```bash
CUDA_VISIBLE_DEVICES='' metiscraft predict "$RUN/artifact" input/test.jsonl --output predictions.jsonl
CUDA_VISIBLE_DEVICES='' python - "$RUN/artifact" <<'PYCODE'
import json
import sys
from pathlib import Path
from metiscraft.serve.predictor import Predictor

predictor = Predictor.load(sys.argv[1])
request = json.loads(Path("request.json").read_text())
print(json.dumps(predictor.answer_request(request), indent=2))
PYCODE
```

## 遇到问题时

| 现象 | 处理方法 |
|---|---|
| ID 重复，或输入、组跨划分冲突 | 修正原始 JSONL 的分配。相关案例应留在一起，只改 ID 不能消除泄漏。 |
| 已有版本与导入内容不同 | 保留旧的封存版本，将修改后的数据导入新版本。 |
| 找不到基座或身份文件 | 完成基座准备，再检查 `model.base` 与 `paths.models_root`。 |
| 训练内存或显存不足 | 重新检查基座大小、批量、序列长度和支持的精度，再决定是否重跑。 |
| HTTP 提示缺少任务映射 | 核对 `team: routing` 是否对应请求问题名与真实 Record 任务 ID。 |
| 校准后某个指标更差 | 检查校准集和评测分布，以实际对比为准，不预设温度拟合一定有效。 |

## 下一步

阅读[最佳实践](../../cookbook/best-practices.md)，设计对照与校准检查。[客服分流的真实记录](decision.md)展示了如何同时交代任务结果、基线与证据边界。

[English](../../en/cookbook/own-task.md)
