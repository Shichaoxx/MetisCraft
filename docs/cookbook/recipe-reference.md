# 配方字段参考

配方是 YAML，结构与 `metiscraft.config.Recipe` 一致；未知字段会直接报错。命令行可以用 `a.b=c` 覆盖任意字段，例如 `metiscraft train <配方> readout.kind=answer_token train.lr=1e-4`。路径的优先级是：配方里的 `paths` > 环境变量 `METISCRAFT_DATA_ROOT` 等 > `~/.config/metiscraft/paths.yaml`。

本页由 [`src/recipe_reference.py`](src/recipe_reference.py) 从代码生成，类型和默认值以代码为准。完整示例见[接入自己的任务](../zh/cookbook/own-task.md)。

## 任务

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `task` | str | 必填 | 任务名，写进每条记录的 `task_id`、产物和报告；温度按“任务\|类型”拟合。 |

## 数据

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `data.dataset` | str | 必填 | 注册表里的数据集名。 |
| `data.version` | str \| `null` | `null` | 数据集版本目录名；只有一个版本时可以不写。 |
| `data.train_split` | str | `train` | 训练用的划分。 |
| `data.calib_split` | str | `calib` | 拟合温度用的划分。 |
| `data.dev_split` | str | `dev` | 选择用的评测划分。 |
| `data.test_split` | str | `test` | 封存后才评的测试划分。 |
| `data.max_train` | Annotated \| `null` | `null` | 训练集最多取多少条，调试用；不写就是全部。 |
| `data.max_eval` | Annotated \| `null` | `null` | 每个评测划分最多取多少条。 |

## 基座与微调

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `model.base` | str | 必填 | 基座在 `models_root` 下的相对路径；目录里要有 `SOURCE.json` 和 `SHA256SUMS`。 |
| `model.dtype` | `bf16` \| `fp16` \| `fp32` | `bf16` | 加载权重的精度。 |
| `model.max_length` | int | `1024` | 模型输入的最大 token 数；completion 训练为追加的答案码预留 1 个 token（EOS 仅作目标）。超出时截短材料，问题和选项保留；`chat_json` 不截短，超出就报错。 |
| `model.finetune.kind` | `lora` \| `full` \| `frozen` | `lora` | `lora` 只训 LoRA；`full` 全参；`frozen` 冻结主干，只训读出头。 |
| `model.finetune.r` | int | `16` | LoRA 秩。 |
| `model.finetune.alpha` | int | `32` | LoRA 缩放系数。 |
| `model.finetune.dropout` | float | `0.0` | LoRA dropout。 |
| `model.finetune.targets` | str \| list[str] | `all-linear` | LoRA 作用的模块：`all-linear` 或模块名列表。 |
| `model.gradient_checkpointing` | bool | `False` | 用重算换显存。 |

## 读出

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `readout.kind` | `answer_token` \| `candidate_scalar` | 必填 | `answer_token` 是 ① 答案 token；`candidate_scalar` 是 ③ 逐候选标量头。 |
| `readout.layout` | `list` \| `pairs` | `list` | ③ 的布局：`list` 把候选写在同一条序列；`pairs` 每个候选单独一条，用作对照。 |
| `readout.head.hidden` | int | `256` | ③ 的 MLP 头隐藏层宽度。 |
| `readout.head.dropout` | float | `0.0` | ③ 的 MLP 头 dropout。 |
| `readout.prompt.style` | `plain` \| `chat_json` | `plain` | ① 的提示写法：`plain` 是 MetisCraft 的文本布局；`chat_json` 把题目写成 JSON 载荷放进分词器的对话模板（关闭思考），选项和等级都用字母，即 Tev1 的格式，只接受选择和等级题。 |
| `readout.prompt.system` | str \| `null` | `null` | `chat_json` 的系统提示；不写就没有系统回合。 |
| `readout.shuffle_train` | bool | `True` | 训练时打乱候选顺序（③），避免记住位置。 |
| `readout.infer_permutations` | `1` | `1` | 当前只接受 1；多次排列推理尚未实现，其他值会被拒绝。 |

## 目标

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `objective.kind` | `auto` \| `bce` \| `cross_entropy` \| `ordinal_soft` \| `completion` | `auto` | 损失由判断类型决定，`auto`、`bce`、`cross_entropy` 在 v0.1 效果相同，只作标注；`ordinal_soft` 让等级题改用以金标为中心的高斯软目标；`completion` 只用于 ①，像监督微调那样在整个词表上训练答案码和 EOS 两个 token。 |
| `objective.brier_weight` | float | `0.0` | 在交叉熵之外再加 Brier 项的权重（选择、等级）。 |
| `objective.pos_weight` | `none` \| `auto` \| Annotated | `none` | 多标签正样本权重：`none`、`auto`（训练集负正比，限制在 1 到 100）或一个数；推理时自动扣回 ln w。 |
| `objective.ordinal_sigma` | float | `1.0` | 序数软目标的高斯宽度，单位是等级。 |

## 训练

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `train.epochs` | float | `1.0` | 训练轮数，可以是小数。 |
| `train.max_steps` | Annotated \| `null` | `null` | 最多优化步数；写了就覆盖 `epochs`。 |
| `train.batch_size` | int | `8` | 每个微批的题数（每张卡）。 |
| `train.grad_accum` | int | `1` | 梯度累积步数；有效批量是 `batch_size × grad_accum × 卡数`，损失按全局题数归一。 |
| `train.lr` | float | `0.0002` | LoRA（或全参）的学习率。 |
| `train.head_lr` | float | `0.001` | 读出头的学习率。 |
| `train.weight_decay` | float | `0.0` | AdamW 的权重衰减。 |
| `train.scheduler` | `linear` \| `cosine` | `linear` | 学习率调度：`linear` 线性衰减到 0；`cosine` 按余弦衰减到 0。 |
| `train.warmup_ratio` | float | `0.0` | 预热步数占总步数的比例；之后按 `train.scheduler` 衰减。 |
| `train.max_grad_norm` | float | `1.0` | 梯度裁剪阈值。 |
| `train.seed` | int | `42` | 随机种子，决定数据顺序和初始化。 |
| `train.bf16` | bool | `True` | 在 CUDA 上用 bf16 autocast。 |
| `train.logging_steps` | int | `10` | 每多少步写一行 `train_log.jsonl`。 |

## 校准

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `calibrate.method` | `temperature` \| `none` | `temperature` | `temperature` 按“任务\|类型”拟合温度；`none` 不校准。 |
| `calibrate.split` | str | `calib` | 拟合温度用的划分。 |

## 评测

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `eval.splits` | list[str] | `['dev']` | 要评测的划分。 |
| `eval.baselines` | list[str] | `['prior']` | 和模型同表报告的基线；v0.1 支持 `prior`。 |

## 服务

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `serve.task_map` | dict | `{}` | HTTP 题名到产物 task_id 的映射；多任务请求需要明确映射，单任务可自动选择。 |

## 路径

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `paths.data_root` | Path \| `null` | `null` | 数据根目录。 |
| `paths.models_root` | Path \| `null` | `null` | 权重根目录。 |
| `paths.runs_root` | Path \| `null` | `null` | 运行目录的根。 |
