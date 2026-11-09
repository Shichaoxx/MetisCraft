# Tev1

Tev1 的决策数据包含五个公开数据集，以及政策判断、优先级路由、研究分类三类合成题；每题给出带字母的选项，只答一个字母。配方遵循公开建议；[复现记录](../../docs/reproductions/tev1.md)区分建议配置与无法取得的历史托管训练作业。

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
| [`qwen3.5-4b_chat-completion.yaml`](qwen3.5-4b_chat-completion.yaml) | ① 答案 token | 同上 | 同上 | 同上 | 36（批量 4 × 累积 9） | 1 | 同上 |

与 Tev1 建议配方相同的部分：LoRA 设置、轮数、学习率与调度、梯度裁剪 1、序列长度 2,048、只在答案上算损失；提示和答案逐 token 与 Tev1 导出的一致（43,708 条全部核对）。

不同的部分：
- 先验证 Qwen3.5-2B，后补 Qwen3.5-4B；4B 的权重版本锁定为 `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。
- Tev1 开启 packing，"批量 8"指 8 条拼满 2,048 token 的序列，每步约 36 道题。MetisCraft 不拼接样本，用批量 4 × 梯度累积 9 保持每步 36 题；显存不够时可以改这两个数，乘积保持 36。
- Tev1 在 Together 的托管服务上训练，实际用的超参没有留存，这里照的是仓库给的建议配方。

## 运行

```bash
pip install -e ".[reproduce]" flash-linear-attention
metiscraft data prepare tev1
metiscraft train recipes/tev1/qwen3.5-2b_chat-completion.yaml
```

Qwen3.5 的线性注意力层使用 `flash-linear-attention`。已记录的 2B 冒烟配置中，PyTorch 参考实现以批量 4、不启用梯度检查点运行时，显存达到 23.5 GB，第一步在 1.5 分钟内未完成。使用优化内核和检查点后，单张 RTX 3090 峰值约 6 GB，每步（36 题）约 26 秒；首次内核编译额外约 10 分钟。原先对 1,052 步的估计为 7.5 小时。这些是特定配置观测，不是通用资源保证，也不是 4B 显存测量。

## 验收结果

2026-11-06 的 2B 结果为 test 86.4%、policy_transfer 92.0%，当时仅主集达标。
2026-11-07 的 4B 结果为 **873／1,000（87.3%）和 298／300（99.33%）**，
相对原版 88.0%／100% 分别低 0.7／0.67 个百分点，达到原定准确率标准。
主集逐题答案一致率 94.6%，迁移集 99.33%；这不等于恢复原作者的历史训练作业。
温度校准仍可能恶化 NLL。运行 ID、校准前后指标、原版读出协议与限制见
[复现记录](../../docs/reproductions/tev1.md)。

旧循环的该次运行在尾步额外呈现 32 次样本；当前版本已修复预算行为，历史结果不回写，详见复现记录。
