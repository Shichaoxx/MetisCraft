# 复现 Tev1-4B-experimental

目标：用 MetisCraft 把 Tev1 的训练过程端到端重做一遍，补上它没有公开的部分，让每一步都能追溯。

依据：[`togethercomputer/tev1`](https://github.com/togethercomputer/tev1/tree/1dde7782382c9f49d627153759b8d1deab426ce0) @ `1dde778`（2026-11-05 读取）。

## Tev1 公开了什么、缺什么

| 部分 | 公开情况 |
|---|---|
| 数据 | 公开。构建脚本锁定了 5 个公开数据集的 revision，合成部分由确定性生成器产出（规则树和模板都在代码里，不用教师模型，也不用 Jev 输出）。作者从干净目录重建后，全部导出文件的哈希与原清单一致。 |
| 训练 | **不公开**。训练在 Together 的托管微调服务上完成，仓库里只有提交任务的脚本。任务 ID、上传文件的哈希和实际超参都没有留存（取回时接口返回 403），仓库给的是"建议配方"，作者注明不是历史任务的设置。 |
| 权重 | 公开（Hugging Face）。 |
| 评测 | 部分公开。有 1,000 道主决策题和 300 道政策迁移题的逐题结果（88.0% 与 100%），含 top-5 logprob；但这两组是开发中反复用过的题，不是未接触的测试，也不报校准。 |

### 数据构成（new v1）

训练 37,840 条、开发 4,568 条；训练集共 16,929,529 个 token，最长 1,526 个 token。

| 来源 | 训练条数 | 锁定版本 |
|---|---:|---|
| MultiNLI | 5,000 | `nyu-mll/multi_nli` @ `da70db2a` |
| BoolQ | 3,000 | `google/boolq` @ `35b264d0` |
| Banking77 | 3,000 | `legacy-datasets/banking77` @ `f5412156` |
| AG News | 1,500 | `fancyzhx/ag_news` @ `eb185aad` |
| SST-5 | 2,000 | `SetFit/sst5` @ `e51bdcd8` |
| 合成政策判断 | 13,500 | 生成器 |
| 优先级路由 | 6,000 | 生成器 |
| 合成研究分类 | 3,840 | 生成器加分类表 |

提示格式用 `Qwen/Qwen3.5-2B` @ `15852e8c` 的分词器与非思考模式的对话模板渲染；答案是一个选项字母加 EOS。

### 建议配方

基座 `Qwen/Qwen3.5-4B`；LoRA 作用于全部线性层，r 8、alpha 16、dropout 0；1 个 epoch，批量 8，学习率 5e-5，cosine 调度，warmup 0.03，梯度裁剪 1；序列长度 2,048，开启 packing；只在答案上算损失。

## 用 MetisCraft 复现

1. **数据**：完成，见下节。
2. **训练**：配方 [`recipes/tev1/`](../../recipes/tev1/README.md) 复刻建议配方。工具包为此补上的三项已完成，CPU 上用微型模型测过：
   - 提示 `readout.prompt.style: chat_json`：系统提示 + JSON 载荷，套用非思考模式的对话模板。用官方分词器渲染，43,708 条的提示 token 与 Tev1 导出的完全相同，答案都是"字母 + EOS"；
   - 损失 `objective.kind: completion`：在整个词表上训练字母和 EOS 两个 token，与 ① 原有的选项内 softmax 并列；
   - 调度 `train.scheduler: cosine`。

   Tev1 开启了 packing，每步约 36 道题；MetisCraft 不拼接样本，配方用批量 4 × 梯度累积 9 对齐每步题数。

   基座先用 Qwen3.5-2B（与锁定的分词器同一版本），再用 Qwen3.5-4B。Qwen3.5 是图文模型的文本部分，24 层里每 4 层只有 1 层全注意力，其余是线性注意力。训练前要确认的三点：
   - 只加载文本部分：可以。实际加载 Qwen3.5-2B 时没有缺失的参数（输出层与词嵌入共享），视觉塔和多 token 预测头不加载；
   - 左填充：不污染线性注意力的状态。在真实模型上取 8 道不同长度的题，批量左填充与逐条前向的答案 logits 在 fp32（PyTorch 参考实现）下最多差 6.3e-5；在 bf16 下用 `flash-linear-attention` 的内核最多差 0.375，与参考实现之间也最多差 0.375，都是舍入误差，选出的字母全部一致；
   - 打包后的状态重置：MetisCraft 不拼接样本，不涉及。

   GPU 冒烟（2026-11-06，单张 RTX 3090）：线性注意力要用 `flash-linear-attention`，否则走 PyTorch 参考实现，批量 4 时显存涨到 23.5 GB、第一步 1.5 分钟没走完。装上它并开梯度检查点后，显存峰值约 6 GB，每步（36 题）约 26 秒，10 步内损失从 0.65 降到 0.48，校准和评测都跑通。照此估算，完整训练单卡约 7.5 小时。还差正式训练。
3. **评测**：
   - 在同一批 1,000 + 300 道开发题上逐题对拍；
   - 按 MetisCraft 的口径，在 5 个公开数据集的全量 test 上报准确率、NLL、Brier、ECE 与先验基线，补上 Tev1 没有的校准结果。
4. **记录**：运行目录、产物与逐题结果全部可追溯；结果写回本页和[支持矩阵](../support-matrix.md)。

## 数据

```bash
pip install -e ".[reproduce]"   # Tev1 的构建脚本要用 datasets 和 jinja2
metiscraft data prepare tev1
```

`data prepare` 依次：

1. 按提交号检出 Tev1 并核对代码树；
2. 准备 5 个输入：注册表里同一 revision 的 `multi_nli`、`boolq`、`banking77_legacy`、`ag_news`、`sst5`；分词器取自 `Qwen/Qwen3.5-2B@15852e8c` 的权重目录，照 Tev1 锁定的 transformers 5.17.0 那样以 `Qwen2Tokenizer` 加载（原因见下）；
3. 原样运行 Tev1 的 4 个构建脚本和 3 个校验脚本。运行时关闭 Hub，由 MetisCraft 的包装层回答脚本读数据和分词器的调用，只给上面声明过的输入；
4. 对比 Tev1 公开的哈希（结果见下）；
5. 把记录转成 MetisCraft 的格式。选项保持 Tev1 的顺序，状态对象保持原来的键顺序，这样以后能用 MetisCraft 重新渲染出逐字节相同的提示。

机制见 [`design.md`](../design.md) §4.2 的"外部构建"。

**核对结果**（2026-11-06）：Tev1 在该提交里公开的 21 个哈希全部一致：
- new-v1 的 7 个最终文件（`runs/new-v1/reproduction.json`）；
- 构建它们用到的 12 个 v1、v2.1 文件（`runs/new-v1/dataset-manifest.json`）；
- 两个评测集，即 1,000 道 test 和 300 道 policy_transfer（`runs/new-v1/evaluation.json`）。

构建环境是 Python 3.11、datasets 5.1.0、transformers 5.22.1，与 Tev1 锁定的 Python 3.12、datasets 4.8.5、transformers 5.17.0 不同。版本差别只在一处影响产物：

- Qwen3.5 在 Hub 上的配置把分词器类写成 `Qwen2Tokenizer`。transformers 5.17.0 照此加载，预分词用的是 Qwen2 的正则；较新的版本把 Qwen3.5 列入"Hub 上分词器类写错的模型"，改用 `Qwen3_5Tokenizer`，与模型自带的 `tokenizer.json` 一致。
- 两个正则只在组合附加符（Unicode 类别 M）附近切分不同。Tev1 的数据里只有 2 条 BoolQ 段落含这类字符（天城文和泰米尔文的元音符号），按新类计数分别比 Tev1 的记录少 2 个和 8 个 token。提示文本完全相同，只有三个 `records/train.jsonl` 里的 `token_count` 不同，哈希因此对不上。
- 注册表给分词器注明 `class: Qwen2Tokenizer` 后，21 个哈希全部一致。这只影响记录里的 token 计数：MetisCraft 训练时用基座自己的分词器重新切分。

另外两项核对：
- 转换后的 43,708 条记录都能逐条还原成 Tev1 的原始消息；
- 训练集与 calib、dev、test、eval_only 之间没有同输入或同组的行，MetisCraft 的去重一条也没删，训练集与 Tev1 相同。

| MetisCraft 划分 | 来自 Tev1 | 条数 |
|---|---|---:|
| train | new-v1 的 train | 37,840 |
| calib、dev | new-v1 的 dev，按组哈希对半分 | 2,448 + 2,120 |
| test | v1 的 test，即 Tev1 报 88.0% 的 1,000 道主决策题 | 1,000 |
| eval_only | v1 的 policy_transfer，即报 100% 的 300 道政策迁移题 | 300 |

转换规则：
- 等级题（SST-5）记为 score，等级按原顺序；
- 其余记为 choice。BoolQ 在 Tev1 里本来就是两个带字母的选项，所以也记为 choice，并在 `provenance` 里注明原类型；
- `task_id` 是 `tev1/<来源>`，所以温度、先验和报告都按来源分开。

Tev1 自己还留了 v2 的 test 和 transfer，作者标为"未接触的最终评测"。需要真正没见过的测试集时可以加进来。

## 状态

- 2026-11-05：读完公开材料，确定方案。
- 2026-11-06：数据部分接入 MetisCraft（外部构建），重建后 21 个公开哈希全部一致；训练要用的提示格式、全词表答案损失和 cosine 调度已补上，配方写好；GPU 冒烟跑通（见上），正式训练待排。
