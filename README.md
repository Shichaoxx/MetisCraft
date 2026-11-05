# MetisCraft

**训练 Jev 类（System One）决策模型的工具包：一次前向给出类型化判断和校准过的概率。**

A toolkit for training Jev-style System One decision models: typed judgments with calibrated probabilities in one forward pass.

- **通用 judge**：是非（noul）、选择（choice）、等级打分（score）、多标签（multi_label）四种判断类型，各有明确的输出语义、损失、指标和校准方式。
- **垂直场景做成自己的 Jev**：给定场景的数据和判断类型，用一份配方完成训练、校准、评测和部署。
- **部件可换**：基座、读出（答案 token / 逐候选标量头）、微调方式（LoRA / 全参）都由配置决定，同协议对照只改一行。
- **知道自己有多准**：按类型的温度校准，报告固定 ECE 口径并同时给出 NLL、Brier 和先验基线。
- **有据可查**：每次运行记录配置、git 提交、环境、数据与基座身份；产物带 `SHA256SUMS` 与许可证 `NOTICE`。
- **兼容官方请求格式**：`metiscraft serve` 提供 `POST /v1/systemone`。

> 开发版（v0.1.0.dev0），接口还会变。设计见 [`docs/design.md`](docs/design.md)，进度见 [`docs/status.md`](docs/status.md)。

## 安装

```bash
pip install -e ".[serve]"        # 开发时用 ".[dev,serve]"
```

## 快速上手

```bash
# 数据与权重的位置（也可以写进 ~/.config/metiscraft/paths.yaml）
export METISCRAFT_DATA_ROOT=/path/to/data METISCRAFT_MODELS_ROOT=/path/to/models METISCRAFT_RUNS_ROOT=/path/to/runs

metiscraft data list                    # 注册表里的数据集
metiscraft data prepare banking77       # 按固定 revision 下载、转换、去重、写版本目录
metiscraft train recipes/banking77/qwen3-0.6b-base_candidate-scalar.yaml   # 训练 → 校准 → 评测 → 报告
metiscraft serve <运行目录>/artifact     # POST /v1/systemone
```

配方可以在命令行覆盖：`metiscraft train <配方> readout.kind=answer_token train.lr=1e-4`。

## 读出

| 读出 | 做法 | 适用类型 |
|---|---|---|
| ① 答案 token | 在答案位置读选项码 token 的 logits；不加新层，保留零样本能力 | noul、choice、score |
| ③ 逐候选标量头 | 每个候选末 token 的隐藏状态经 MLP 得到分数（list / pairs 布局） | 四种全部 |

张量图与公式见 [`docs/readouts/`](docs/readouts/)。当前较好的 Jev 类项目与 MetisCraft 的支持情况见 [`docs/support-matrix.md`](docs/support-matrix.md)。

## 许可证

代码以 [Apache-2.0](LICENSE) 发布。数据集与基座权重各有许可证和条款，注册表与产物的 `NOTICE` 会逐项写明；用非商用数据训练的产物会标为不可商用。

早期版本 Metis 0.1.0 已归档为 tag [`metis-v0.1.0`](https://github.com/Shichaoxx/MetisCraft/tree/metis-v0.1.0)。由 [shichao](https://github.com/Shichaoxx) 维护。
