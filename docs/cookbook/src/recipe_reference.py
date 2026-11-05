"""Write ../recipe-reference.md from the recipe schema in metiscraft.config.

Types and defaults come from the pydantic models; the Chinese descriptions live below and the
script refuses to run if a field has none, so the reference cannot silently fall behind the code.

    python recipe_reference.py
"""

from __future__ import annotations

import types
import typing
from pathlib import Path

from pydantic import BaseModel

from metiscraft import config

OUT = Path(__file__).resolve().parent.parent / "recipe-reference.md"

DESCRIPTIONS = {
    "task": "任务名，写进每条记录的 `task_id`、产物和报告；温度按“任务|类型”拟合。",
    "data.dataset": "注册表里的数据集名。",
    "data.version": "数据集版本目录名；只有一个版本时可以不写。",
    "data.train_split": "训练用的划分。",
    "data.calib_split": "拟合温度用的划分。",
    "data.dev_split": "选择用的评测划分。",
    "data.test_split": "封存后才评的测试划分。",
    "data.max_train": "训练集最多取多少条，调试用；不写就是全部。",
    "data.max_eval": "每个评测划分最多取多少条。",
    "model.base": "基座在 `models_root` 下的相对路径；目录里要有 `SOURCE.json` 和 `SHA256SUMS`。",
    "model.dtype": "加载权重的精度。",
    "model.max_length": "提示的最大 token 数；超出时截短材料，问题和选项保留。",
    "model.finetune.kind": "`lora` 只训 LoRA；`full` 全参；`frozen` 冻结主干，只训读出头。",
    "model.finetune.r": "LoRA 秩。",
    "model.finetune.alpha": "LoRA 缩放系数。",
    "model.finetune.dropout": "LoRA dropout。",
    "model.finetune.targets": "LoRA 作用的模块：`all-linear` 或模块名列表。",
    "model.gradient_checkpointing": "用重算换显存。",
    "readout.kind": "`answer_token` 是 ① 答案 token；`candidate_scalar` 是 ③ 逐候选标量头。",
    "readout.layout": "③ 的布局：`list` 把候选写在同一条序列；`pairs` 每个候选单独一条，用作对照。",
    "readout.head.hidden": "③ 的 MLP 头隐藏层宽度。",
    "readout.head.dropout": "③ 的 MLP 头 dropout。",
    "readout.shuffle_train": "训练时打乱候选顺序（③），避免记住位置。",
    "readout.infer_permutations": "预留字段，v0.1 未实现，设了也不生效。",
    "objective.kind": "损失由判断类型决定，`auto`、`bce`、`cross_entropy` 在 v0.1 效果相同，只作标注；`ordinal_soft` 让等级题改用以金标为中心的高斯软目标。",
    "objective.brier_weight": "在交叉熵之外再加 Brier 项的权重（选择、等级）。",
    "objective.pos_weight": "多标签正样本权重：`none`、`auto`（训练集负正比，限制在 1 到 100）或一个数；推理时自动扣回 ln w。",
    "objective.ordinal_sigma": "序数软目标的高斯宽度，单位是等级。",
    "train.epochs": "训练轮数，可以是小数。",
    "train.max_steps": "最多优化步数；写了就覆盖 `epochs`。",
    "train.batch_size": "每个微批的题数（每张卡）。",
    "train.grad_accum": "梯度累积步数；有效批量是 `batch_size × grad_accum × 卡数`，损失按全局题数归一。",
    "train.lr": "LoRA（或全参）的学习率。",
    "train.head_lr": "读出头的学习率。",
    "train.weight_decay": "AdamW 的权重衰减。",
    "train.warmup_ratio": "预热步数占总步数的比例；之后线性衰减到 0。",
    "train.max_grad_norm": "梯度裁剪阈值。",
    "train.seed": "随机种子，决定数据顺序和初始化。",
    "train.bf16": "在 CUDA 上用 bf16 autocast。",
    "train.logging_steps": "每多少步写一行 `train_log.jsonl`。",
    "calibrate.method": "`temperature` 按“任务|类型”拟合温度；`none` 不校准。",
    "calibrate.split": "拟合温度用的划分。",
    "eval.splits": "要评测的划分。",
    "eval.baselines": "和模型同表报告的基线；v0.1 支持 `prior`。",
    "paths.data_root": "数据根目录。",
    "paths.models_root": "权重根目录。",
    "paths.runs_root": "运行目录的根。",
}

SECTION_TITLES = {
    "task": "任务",
    "data": "数据",
    "model": "基座与微调",
    "readout": "读出",
    "objective": "目标",
    "train": "训练",
    "calibrate": "校准",
    "eval": "评测",
    "paths": "路径",
}


def type_name(ann) -> str:
    origin = typing.get_origin(ann)
    if origin is typing.Literal:
        return " \\| ".join(f"`{a}`" for a in typing.get_args(ann))
    if origin in (typing.Union, types.UnionType):
        return " \\| ".join(type_name(a) for a in typing.get_args(ann))
    if origin is list:
        return f"list[{', '.join(type_name(a) for a in typing.get_args(ann))}]"
    if ann is type(None):
        return "`null`"
    return getattr(ann, "__name__", str(ann))


def default_text(field) -> str:
    if field.is_required():
        return "必填"
    value = field.default if field.default_factory is None else field.default_factory()
    return "`null`" if value is None else f"`{value}`"


def rows(model: type[BaseModel], prefix: str = "") -> list[tuple[str, str, str]]:
    out = []
    for name, field in model.model_fields.items():
        path = prefix + name
        ann = field.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            out += rows(ann, path + ".")
        else:
            out.append((path, type_name(ann), default_text(field)))
    return out


def main() -> None:
    table = rows(config.Recipe)
    missing = [p for p, _, _ in table if p not in DESCRIPTIONS]
    stale = [p for p in DESCRIPTIONS if p not in {r[0] for r in table}]
    if missing or stale:
        raise SystemExit(f"descriptions out of sync: missing {missing}, stale {stale}")
    lines = [
        "# 配方字段参考",
        "",
        "配方是 YAML，结构与 `metiscraft.config.Recipe` 一致；未知字段会直接报错。命令行可以用 `a.b=c` 覆盖任意字段，"
        "例如 `metiscraft train <配方> readout.kind=answer_token train.lr=1e-4`。"
        "路径的优先级是：配方里的 `paths` > 环境变量 `METISCRAFT_DATA_ROOT` 等 > `~/.config/metiscraft/paths.yaml`。",
        "",
        "本页由 [`src/recipe_reference.py`](src/recipe_reference.py) 从代码生成，类型和默认值以代码为准。完整示例见"
        "[把业务做成 Jev](your-own-jev.md#4-写配方)。",
        "",
    ]
    section = None
    for path, typ, default in table:
        top = path.split(".")[0]
        if top != section:
            section = top
            lines += [
                f"## {SECTION_TITLES[top]}",
                "",
                "| 字段 | 类型 | 默认 | 说明 |",
                "|---|---|---|---|",
            ]
        desc = DESCRIPTIONS[path].replace("|", "\\|")  # a bare pipe would split the table cell
        lines.append(f"| `{path}` | {typ} | {default} | {desc} |")
        if (
            path == table[-1][0]
            or table[[r[0] for r in table].index(path) + 1][0].split(".")[0] != top
        ):
            lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT, len(table), "fields")


if __name__ == "__main__":
    main()
