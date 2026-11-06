# Tev1

Tev1 的决策数据：5 个公开数据集改写成的选择题和等级题，加上政策判断、优先级路由、研究分类三类合成题；每题给出带字母的选项，只答一个字母。这份配方照 Tev1 公开的建议配方训练，用来复现 Tev1-4B-experimental 的训练过程，来由与进度见[复现记录](../../docs/reproductions/tev1.md)。

| 项目 | 内容 |
|---|---|
| 注册名 | `tev1` |
| 上游 | [`togethercomputer/tev1`](https://github.com/togethercomputer/tev1/tree/1dde7782382c9f49d627153759b8d1deab426ce0) @ `1dde778`；数据由它的构建脚本重建，公开的 21 个哈希全部一致 |
| 许可证 | 构建脚本 MIT；公开数据集的行沿用各自条款（MultiNLI 混合、BoolQ CC BY-SA 3.0、Banking77 CC BY 4.0、AG News 非商用、SST-5 未声明）；用它训练的产物不标可商用 |
| 判断类型 | 选择 choice；等级 score（SST-5） |
| 划分 | train 37,840（new-v1 的 train）；calib 2,448、dev 2,120（new-v1 的 dev 按组哈希对半分）；test 1,000（v1 的 test，Tev1 报 88.0% 的主决策题）；eval_only 300（v1 的 policy_transfer，报 100% 的政策迁移题） |

## 配方

| 配方 | 读出 | 提示 | 损失 | 微调 | 每步题数 | 轮数 | 学习率 |
|---|---|---|---|---|---|---|---|
| [`qwen3.5-2b_chat-completion.yaml`](qwen3.5-2b_chat-completion.yaml) | ① 答案 token | `chat_json`，Tev1 的系统提示 | `completion`：字母 + EOS，整个词表 | LoRA r=8、alpha 16，全部线性层 | 36（批量 4 × 累积 9） | 1 | 5e-5，cosine，warmup 0.03 |

与 Tev1 建议配方相同的部分：LoRA 设置、轮数、学习率与调度、梯度裁剪 1、序列长度 2,048、只在答案上算损失；提示和答案逐 token 与 Tev1 导出的一致（43,708 条全部核对）。

不同的部分：
- 基座先用 Qwen3.5-2B，与 Tev1 锁定的分词器同一版本；Tev1 用的是 4B。
- Tev1 开启 packing，"批量 8"指 8 条拼满 2,048 token 的序列，每步约 36 道题。MetisCraft 不拼接样本，用批量 4 × 梯度累积 9 保持每步 36 题；显存不够时可以改这两个数，乘积保持 36。
- Tev1 在 Together 的托管服务上训练，实际用的超参没有留存，这里照的是仓库给的建议配方。

## 运行

```bash
pip install -e ".[reproduce]" flash-linear-attention
metiscraft data prepare tev1
metiscraft train recipes/tev1/qwen3.5-2b_chat-completion.yaml
```

Qwen3.5 的线性注意力层要装 `flash-linear-attention`，否则走 transformers 的 PyTorch 参考实现：批量 4 不开梯度检查点时显存涨到 23.5 GB，第一步 1.5 分钟还没走完。装上它并开梯度检查点后，单张 RTX 3090 上显存峰值约 6 GB，每步（36 题）约 26 秒；第一次运行还要额外花约 10 分钟编译内核，之后有缓存。照此估算，一个 epoch（1,052 步）单卡约 7.5 小时。

## 验收结果

还没有完整运行。2026-11-06 在 GPU 上跑了 10 步冒烟（含校准和 dev 抽样评测），全流程跑通，速度与显存见上。
