"""Write ../recipe-reference.md from the recipe schema in metiscraft.config.

Types and defaults come from one pydantic schema; both language dictionaries are checked
against that schema, so neither translation can silently omit a new field.

    python recipe_reference.py
"""

from __future__ import annotations

import argparse
import types
import typing
from pathlib import Path

from pydantic import BaseModel

from metiscraft import config

OUT = Path(__file__).resolve().parent.parent / "recipe-reference.md"
EN_OUT = OUT.parents[1] / "en/recipe-reference.md"

DESCRIPTIONS = {
    "task": "运行与模型的名称标签，写入配方元数据；不覆盖 `Record.task_id`。产物的 `task_ids` 取训练记录的任务集合，温度按记录的任务与判断类型分组。",
    "data.dataset": "已准备的数据集名称，包括通过本地 manifest 导入的数据集。",
    "data.version": "数据集版本目录名；只有一个版本时可以不写。",
    "data.train_split": "训练用的划分。",
    "data.calib_split": "拟合温度用的划分。",
    "data.dev_split": "选择用的评测划分。",
    "data.test_split": "封存后才评的测试划分。",
    "data.max_train": "训练集最多取多少条，调试用；不写就是全部。",
    "data.max_eval": "每个评测划分最多取多少条。",
    "model.base": "基座在 `models_root` 下的相对路径；目录里要有 `SOURCE.json` 和 `SHA256SUMS`。",
    "model.dtype": "加载权重的精度。",
    "model.max_length": "模型输入的最大 token 数；completion 训练为追加的答案码预留 1 个 token（EOS 仅作目标）。超出时截短材料，问题和选项保留；`chat_json` 不截短，超出就报错。",
    "model.finetune.kind": "`lora` 训练 LoRA 适配器；`full` 微调基座；`frozen` 冻结主干。冻结时只有标量读出的新 MLP 头可训练，答案 token 读出没有新头，会跳过训练。",
    "model.finetune.r": "LoRA 秩。",
    "model.finetune.alpha": "LoRA 缩放系数。",
    "model.finetune.dropout": "LoRA dropout。",
    "model.finetune.targets": "LoRA 作用的模块：`all-linear` 或模块名列表。",
    "model.gradient_checkpointing": "用重算换显存。",
    "readout.kind": "`answer_token` 选取答案码 logits；`candidate_scalar` 对候选表示应用标量 MLP。",
    "readout.layout": "标量布局：`list`（默认）将候选放在同一条因果序列；`pairs` 将每个候选放在独立序列。",
    "readout.head.hidden": "③ 的 MLP 头隐藏层宽度。",
    "readout.head.dropout": "③ 的 MLP 头 dropout。",
    "readout.prompt.style": "① 的提示写法：`plain` 是 MetisCraft 的文本布局；`chat_json` 把题目写成 JSON 载荷放进分词器的对话模板（关闭思考），选项和等级都用字母，即 Tev1 的格式，只接受选择和等级题。",
    "readout.prompt.system": "`chat_json` 的系统提示；不写就没有系统回合。",
    "readout.shuffle_train": "仅在标量 `list` 训练时打乱候选的提示排列，减少位置偏差；不作用于 `pairs`。",
    "readout.infer_permutations": "当前只接受 1；多次排列推理尚未实现，其他值会被拒绝。",
    "objective.kind": "损失由判断类型决定，`auto`、`bce`、`cross_entropy` 在 v0.1 效果相同，只作标注；`ordinal_soft` 让等级题改用以金标为中心的高斯软目标；`completion` 只用于 ①，像监督微调那样在整个词表上训练答案码和 EOS 两个 token。",
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
    "train.scheduler": "学习率调度：`linear` 线性衰减到 0；`cosine` 按余弦衰减到 0。",
    "train.warmup_ratio": "预热步数占总步数的比例；之后按 `train.scheduler` 衰减。",
    "train.max_grad_norm": "梯度裁剪阈值。",
    "train.seed": "随机种子，决定数据顺序和初始化。",
    "train.bf16": "在 CUDA 上用 bf16 autocast。",
    "train.logging_steps": "每多少步写一行 `train_log.jsonl`。",
    "calibrate.method": "`temperature` 按“任务|类型”拟合温度；`none` 不校准。",
    "calibrate.split": "拟合温度用的划分。",
    "eval.splits": "要评测的划分。",
    "eval.baselines": "和模型同表报告的基线；v0.1 支持 `prior`。",
    "serve.task_map": "HTTP 题名到产物 task_id 的映射；多任务请求需要明确映射，单任务可自动选择。",
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
    "serve": "服务",
    "paths": "路径",
}

EN_DESCRIPTIONS = {
    "task": "Run/model label recorded in recipe metadata; it does not override `Record.task_id`. Artifact `task_ids` come from the training records, and temperature groups use each record's task and judgment type.",
    "data.dataset": "Name of the prepared dataset, including a local manifest import.",
    "data.version": "Dataset version directory; optional when only one version exists.",
    "data.train_split": "Split used for training.",
    "data.calib_split": "Independent split used to fit temperature.",
    "data.dev_split": "Evaluation split used for model selection.",
    "data.test_split": "Test split evaluated after the artifact is sealed.",
    "data.max_train": "Optional training-record limit for debugging; null uses the whole split.",
    "data.max_eval": "Optional record limit applied to each evaluation split.",
    "model.base": "Base directory relative to `models_root`; requires `SOURCE.json` and `SHA256SUMS`.",
    "model.dtype": "Precision used to load model weights.",
    "model.max_length": "Maximum input tokens. Completion training reserves one token for the answer code; EOS is a target only. The plain prompt truncates state while preserving the question and candidates. `chat_json` rejects oversized inputs instead of truncating.",
    "model.finetune.kind": "`lora` trains adapters; `full` tunes the base; `frozen` freezes it. A frozen scalar readout can train its new MLP head; a frozen answer-token readout has no new head and skips training.",
    "model.finetune.r": "LoRA rank.",
    "model.finetune.alpha": "LoRA scaling factor.",
    "model.finetune.dropout": "LoRA dropout probability.",
    "model.finetune.targets": "LoRA target modules: `all-linear` or a list of module names.",
    "model.gradient_checkpointing": "Recompute activations to reduce training memory.",
    "readout.kind": "`answer_token` gathers answer-code logits; `candidate_scalar` applies a scalar MLP to candidate representations.",
    "readout.layout": "Scalar layout: `list` (default) places all candidates in one causal sequence; `pairs` places each candidate in a separate sequence.",
    "readout.head.hidden": "Scalar MLP hidden width.",
    "readout.head.dropout": "Scalar MLP dropout probability.",
    "readout.prompt.style": "Answer-token prompt: `plain` uses the text layout; `chat_json` puts JSON in the tokenizer chat template with thinking disabled and letter-coded choices or levels, as in the Tev1 recipe. `chat_json` accepts only choice and score questions.",
    "readout.prompt.system": "Optional system message for `chat_json`; null omits that turn.",
    "readout.shuffle_train": "Shuffle the scalar `list` prompt's candidate order during training to reduce position bias. It does not affect `pairs`.",
    "readout.infer_permutations": "Only 1 is implemented. Other values fail validation rather than silently doing nothing.",
    "objective.kind": "The judgment type determines the loss; `auto`, `bce` and `cross_entropy` are equivalent annotations in v0.1. `ordinal_soft` uses a Gaussian soft target around the gold level. Answer-token-only `completion` predicts the answer code and EOS over the full vocabulary.",
    "objective.brier_weight": "Additional Brier-loss weight for choice and score objectives.",
    "objective.pos_weight": "Positive multi-label weight: `none`, `auto` (training negatives/positives clipped to 1–100), or a number. Inference subtracts ln(weight) from those logits.",
    "objective.ordinal_sigma": "Gaussian width of ordinal soft targets, in level units.",
    "train.epochs": "Training epochs; fractional values are accepted.",
    "train.max_steps": "Maximum optimizer steps; overrides `epochs` when provided.",
    "train.batch_size": "Questions per microbatch per device.",
    "train.grad_accum": "Gradient accumulation microbatches. Effective batch size is `batch_size × grad_accum × devices`; loss is normalized by the global question count.",
    "train.lr": "Learning rate for LoRA or full-base parameters.",
    "train.head_lr": "Learning rate for the readout head.",
    "train.weight_decay": "AdamW weight decay.",
    "train.scheduler": "`linear` or `cosine` decay to zero after warmup.",
    "train.warmup_ratio": "Warmup steps as a fraction of total optimizer steps.",
    "train.max_grad_norm": "Gradient-clipping norm threshold.",
    "train.seed": "Seed for initialization and data ordering.",
    "train.bf16": "Use bfloat16 autocast on CUDA.",
    "train.logging_steps": "Optimizer-step interval between `train_log.jsonl` entries.",
    "calibrate.method": "`temperature` fits task/type temperatures; `none` disables fitting.",
    "calibrate.split": "Independent split used for temperature fitting.",
    "eval.splits": "Splits to evaluate.",
    "eval.baselines": "Baselines reported alongside model metrics; v0.1 supports `prior`.",
    "serve.task_map": "HTTP question-name to artifact `task_id` mapping. Multi-task artifacts need an explicit mapping; single-task artifacts can select automatically.",
    "paths.data_root": "Prepared-dataset root directory.",
    "paths.models_root": "Versioned base-model root directory.",
    "paths.runs_root": "Training and evaluation output root directory.",
}

EN_SECTION_TITLES = {
    "task": "Task",
    "data": "Data",
    "model": "Base and fine-tuning",
    "readout": "Readout",
    "objective": "Objective",
    "train": "Training",
    "calibrate": "Calibration",
    "eval": "Evaluation",
    "serve": "Serving",
    "paths": "Paths",
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


def render(lang: str) -> str:
    """Render both languages from the same schema walk and stable field identifiers."""
    table = rows(config.Recipe)
    descriptions = DESCRIPTIONS if lang == "zh" else EN_DESCRIPTIONS
    titles = SECTION_TITLES if lang == "zh" else EN_SECTION_TITLES
    missing = [p for p, _, _ in table if p not in descriptions]
    stale = [p for p in descriptions if p not in {r[0] for r in table}]
    if missing or stale:
        raise SystemExit(f"descriptions out of sync: missing {missing}, stale {stale}")
    lines = (
        [
            "# 配方字段参考",
            "",
            "配方是 YAML，结构与 `metiscraft.config.Recipe` 一致；未知字段会直接报错。命令行可以用 `a.b=c` 覆盖任意字段，"
            "例如 `metiscraft train <配方> readout.kind=answer_token train.lr=1e-4`。"
            "路径的优先级是：配方里的 `paths` > 环境变量 `METISCRAFT_DATA_ROOT` 等 > `~/.config/metiscraft/paths.yaml`。",
            "",
            "本页由 [`src/recipe_reference.py`](src/recipe_reference.py) 从代码生成，类型和默认值以代码为准。完整示例见"
            "[接入自己的任务](../zh/cookbook/own-task.md)。",
            "",
        ]
        if lang == "zh"
        else [
            "# Recipe field reference",
            "",
            "Recipes are YAML documents matching `metiscraft.config.Recipe`; unknown fields are rejected. "
            "Override fields with `a.b=c`, for example `metiscraft train <recipe> readout.kind=answer_token train.lr=1e-4`. "
            "Path precedence is recipe `paths` > environment variables such as `METISCRAFT_DATA_ROOT` > "
            "`~/.config/metiscraft/paths.yaml`.",
            "",
            "Generated from the same schema as the Chinese reference by "
            "[`recipe_reference.py`](../cookbook/src/recipe_reference.py). Types and defaults come from the code. "
            "See [your own task](cookbook/own-task.md) for a complete example.",
            "",
        ]
    )
    section = None
    for path, typ, default in table:
        top = path.split(".")[0]
        if top != section:
            section = top
            lines += [
                f"## {titles[top]}",
                "",
                "| 字段 | 类型 | 默认 | 说明 |"
                if lang == "zh"
                else "| Field | Type | Default | Description |",
                "|---|---|---|---|",
            ]
        desc = descriptions[path].replace("|", "\\|")  # a bare pipe would split the table cell
        if lang == "en" and default == "必填":
            default = "required"
        lines.append(f"| `{path}` | {typ} | {default} | {desc} |")
        if (
            path == table[-1][0]
            or table[[r[0] for r in table].index(path) + 1][0].split(".")[0] != top
        ):
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail on drift without writing")
    parser.add_argument("--lang", choices=("en", "zh", "all"), default="all")
    args = parser.parse_args()
    for lang in ("en", "zh") if args.lang == "all" else (args.lang,):
        output = OUT if lang == "zh" else EN_OUT
        content = render(lang)
        if args.check:
            if not output.is_file() or output.read_text(encoding="utf-8") != content:
                raise SystemExit(f"{lang} recipe reference differs; run recipe_reference.py")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(content, encoding="utf-8")
        print(output, len(rows(config.Recipe)), "fields")


if __name__ == "__main__":
    main()
