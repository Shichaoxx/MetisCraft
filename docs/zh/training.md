# 训练、校准与评测

一份配方选择数据版本、本地基座、读出、微调方式、损失和评测划分。`metiscraft train`
依次完成这些步骤，并记录配置和产物。建议从导出的[快速上手](quickstart.md)配方开始，
再适配真实任务。

## 准备本地预训练基座

训练要求本地目录包含权重、配置、分词器、`SOURCE.json` 和 `SHA256SUMS`；仅提供 Hub
模型 ID 不够。先选用与当前 Transformers 版本兼容的模型，固定其仓库完整提交号，再下载
到新的模型目录。

下面的示例需要替换成你选择的仓库和提交号。它会下载权重，不属于离线快速上手流程。

```python
import json
import re
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from metiscraft.data.store import write_sha256sums
from metiscraft.modeling.identity import capture_identity

repo = "REPLACE_WITH_MODEL_REPOSITORY"
revision = "REPLACE_WITH_FULL_40_CHARACTER_COMMIT"
if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
    raise ValueError("Set revision to the model repository's full commit hash")

destination = Path("models") / "my-base" / revision
if destination.exists():
    raise FileExistsError(f"Use a new model version directory: {destination}")
snapshot_download(repo_id=repo, revision=revision, local_dir=str(destination))
download_metadata = destination / ".cache" / "huggingface"
if download_metadata.is_dir():
    shutil.rmtree(download_metadata)
(destination / "SOURCE.json").write_text(
    json.dumps({"channel": "huggingface", "repo": repo, "revision": revision}, indent=2)
    + "\n",
    encoding="utf-8",
)
write_sha256sums(destination)
print(capture_identity(destination)["sha256"])
```

示例只在封存前清理新建目标目录内部的 Hugging Face 下载元数据。这一步仅用于本次新
下载的目录。已封存的模型版本不要再次下载或删除内部文件，需要变化时准备新版本。

本地生成的哈希用于发现后续变化，不能独立证明下载的内容就是可信的上游发行物。
上游提供校验和或证明时应额外核对。保留真实来源和许可证信息，完成全部准备后再封存
模型目录。

加载器核验实际权重、分词器和配置字节是否与清单一致，也会拒绝清单漏列的模型文件和
符号链接。新产物绑定这份内容身份：修改基座后重新生成 `SHA256SUMS`，仍不能匹配原产物。
字节完全相同的文件可以迁移目录，用 `--base-path` 或 `Predictor.load(..., base_path=...)`
选择新位置，详见[基座目录迁移](serving.md#迁移基座目录)。

## 配置配方

使用[数据指南](data.md)中的数据集时，可按下例调整导出配方；将提交号占位符替换为前面
固定的同一提交号：

```yaml
task: support
data:
  dataset: support
  version: labeled-v1
model:
  base: my-base/REPLACE_WITH_FULL_40_CHARACTER_COMMIT
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
paths:
  data_root: data
  models_root: models
  runs_root: runs
```

这些设置用于演示接口，不是调优后的配方。从快速上手改为完整训练时，要去掉原来的
`train.max_steps: 2`。根据基座和硬件选择显存、精度与训练参数。相对路径从当前工作目录解析。

| 读出 | 适用判断 | 机制 |
|---|---|---|
| `answer_token` | `noul`、`choice`、`score` | 读取答案位置上选项码的 logits |
| `candidate_scalar` | 四种全部 | 用学习到的头为候选表示打分 |

`chat_json` 提示和全词表 `completion` 损失要求使用 `answer_token`。学习率调度支持
linear 与 cosine。完整字段见[配方参考](../cookbook/recipe-reference.md)。

```bash
metiscraft config show recipe.yaml
metiscraft train recipe.yaml
```

命令支持点号覆盖，例如 `metiscraft train recipe.yaml train.lr=1e-4`。模型只拟合 `train`，
温度只拟合 `calib`，评测使用 `dev`、`test` 或 `eval_only`。校准不能保证分布变化后的
概率仍然可靠。

## 查看运行记录

| 输出 | 记录内容 |
|---|---|
| `config.yaml`、`config.sha256` | 解析后的配方与配置身份 |
| `git.json`、可选 `git.diff`、`env.json` | 源码提交或安装包版本、改动与环境 |
| `data.json`、`model.json` | 数据版本、清单与基座内容身份 |
| `train_log.jsonl` | 训练步数和损失 |
| `artifact/` | 适配器或模型、读出、元数据、校验和与许可证说明 |
| `calibration/` | 绑定到该产物的温度参数 |
| `eval/`、`report.md` | 逐题预测、汇总指标与可读报告 |

训练结果可通过 [Predictor 或 HTTP 服务](serving.md)重新加载。

[English](../en/training.md)

轮数按微批预算计算，小数轮向上取整；尾部梯度累积使用实际剩余微批，不从下一轮补齐。`max_steps` 覆盖轮数并按优化器更新次数计数。产物记录微批与样本呈现次数，包括分布式采样的填充。输入配方与解析后的配置分别保存；`recipe_sha256` 保留输入身份含义。
