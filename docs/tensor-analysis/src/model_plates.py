"""Flat model plates: token layouts, indexed hidden rows and real matrix products.

This is an independent TikZ implementation of the user's reference grammar.
The internship figures are references, never imported, modified or embedded.
All dimensions are symbolic illustrations; equal axes keep identical edges.
"""

# Keep mathematical formulas and bilingual captions intact for source editing.
# Normal Python formatting and every other lint rule still apply to this file.
# ruff: noqa: E501

from __future__ import annotations

import json
import re
from pathlib import Path

AXES = {
    "T": 9,
    "d": 8,
    "K": 3,
    "m": 2,
    "V": 12,
    "p": 4,
    "w": 4,
    "R": 3,
    "N": 3,
    "J": 6,
    "Q": 2,
    "f": 16,
}
EDGES = {
    "T": 0.42,
    "d": 0.24,
    "K": 0.42,
    "m": 0.24,
    "V": 0.28,
    "p": 0.32,
    "w": 0.32,
    "R": 0.42,
    "N": 0.42,
    "J": 0.42,
    "Q": 0.42,
    "f": 0.32,
    "1": 0.24,
}


def tr(lang, en, zh):
    return zh if lang == "zh" else en


def prose(text):
    """Catalogue prose has literal configuration names, not raw TeX identifiers."""
    return re.sub(r"(?<!\\)_", r"\\_", text)


class Plate:
    """Emit self-contained geometry with an inspectable object ledger."""

    def __init__(self, lang, width=22, axes=None):
        self.lang, self.width = lang, width
        self.axes = AXES | (axes or {})
        self.lines = [
            r"\begin{document}",
            r"\begin{tikzpicture}[x=1cm,y=1cm,text=Ink]",
        ]
        self.bounds = []

    def text(self, x, y, text, width=None, size="small", anchor="center", color="Ink"):
        opts = f"font=\\{size},text={color},anchor={anchor},align=center,inner sep=0pt"
        if width:
            opts += f",text width={width}cm"
        self.lines.append(rf"\node[{opts}] at ({x:.3f},{y:.3f}) {{{text}}};")

    def stage(self, x, y, text, width=5):
        self.text(x, y, r"\textbf{" + text + "}", width, color="Muted")

    def formula(self, text):
        self.text(self.width / 2, 4.0, "$\\displaystyle " + text + "$", size="large")

    def arrow(self, x1, y1, x2, y2, color="Ink", dashed=False):
        style = f"-{{Latex[length=1.8mm]}},draw={color}!80!Canvas,line width=.7pt"
        if dashed:
            style += ",dashed"
        self.lines += [
            r"\begin{scope}[on background layer]",
            rf"\draw[{style}] ({x1:.3f},{y1:.3f}) -- ({x2:.3f},{y2:.3f});",
            r"\end{scope}",
        ]

    def tensor(
        self,
        name,
        x,
        y,
        row,
        col,
        symbol,
        shape,
        role="Hid",
        selected=None,
        support=None,
        labels=True,
    ):
        nr, nc = (self.axes.get(a, int(a) if a.isdigit() else 0) for a in (row, col))
        if not nr or not nc:
            raise ValueError((row, col))
        rh, cw = EDGES[row], EDGES[col]
        width, height = nc * cw, nr * rh
        top, bottom = y + height / 2, y - height / 2
        obj = {
            "name": name,
            "kind": "tensor",
            "axes": [row, col],
            "rows": nr,
            "cols": nc,
            "x": x,
            "y": bottom,
            "width": width,
            "height": height,
            "symbol": symbol,
            "shape": shape,
        }
        self.bounds.append(obj)
        for r in range(nr):
            for c in range(nc):
                on = support(r, c) if support else True
                cell_role = role if selected is None or r in selected else "Ctx"
                level = (30, 55, 80)[(r * 17 + c * 11 + r * c * 3 + len(name)) % 11 % 3]
                if not on:
                    continue
                xx, yy = x + c * cw + 0.015, top - (r + 1) * rh + 0.015
                self.lines.append(
                    rf"\path[fill={cell_role}!{level}!Canvas,rounded corners=.6pt] "
                    rf"({xx:.3f},{yy:.3f}) rectangle ++({cw - 0.03:.3f},{rh - 0.03:.3f});"
                )
        self.lines.append(
            rf"\draw[draw=Ink!55!Canvas,line width=.45pt] "
            rf"({x - 0.045:.3f},{top:.3f}) -- ++(-.06,0) -- ++(0,{-height:.3f}) -- ++(.06,0);"
        )
        self.lines.append(
            rf"\draw[draw=Ink!55!Canvas,line width=.45pt] "
            rf"({x + width + 0.045:.3f},{top:.3f}) -- ++(.06,0) -- ++(0,{-height:.3f}) -- ++(-.06,0);"
        )
        if labels:
            self.text(x + width / 2, bottom - 0.32, "$" + symbol + "$")
            self.text(
                x + width / 2, bottom - 0.70, "$" + shape + "$", size="scriptsize", color="Muted"
            )
        return obj

    def row_y(self, obj, row):
        return obj["y"] + obj["height"] - (row + 0.5) * EDGES[obj["axes"][0]]

    def gather(self, source, target, rows, color="Hid"):
        if len(rows) != target["rows"] or source["cols"] != target["cols"]:
            raise ValueError("Invalid row gather")
        for i, row in enumerate(rows):
            self.arrow(
                source["x"] + source["width"] + 0.13,
                self.row_y(source, row),
                target["x"] - 0.13,
                self.row_y(target, i),
                color,
            )

    def op(self, x, y, width, text, height=0.85):
        self.lines.append(
            rf"\node[draw=Ink!28!Canvas,fill=Ink!3!Canvas,rounded corners=2pt,"
            rf"font=\small,text width={width - 0.22:.3f}cm,align=center,inner sep=3pt,"
            rf"minimum height={height:.3f}cm] at ({x + width / 2:.3f},{y:.3f}) {{{text}}};"
        )

    def tokens(self, x, y, labels, roles, symbol="x", indices=True):
        if len(labels) != self.axes["T"] or len(roles) != self.axes["T"]:
            raise ValueError("Token strip must share T with H")
        unit = EDGES["T"]
        for i, (label, role) in enumerate(zip(labels, roles, strict=True)):
            xx = x + i * unit
            self.lines.append(
                rf"\path[fill={role}!68!Canvas,rounded corners=1pt] "
                rf"({xx + 0.015:.3f},{y - unit / 2:.3f}) rectangle ++({unit - 0.03:.3f},{unit - 0.03:.3f});"
            )
            self.text(xx + unit / 2, y, label, size="scriptsize")
            if indices:
                self.text(xx + unit / 2, y + 0.5, str(i), size="scriptsize", color="Muted")
        self.text(x + self.axes["T"] * unit / 2, y - 0.7, "$" + symbol + "$", color="Ink")
        self.text(
            x + self.axes["T"] * unit / 2,
            y - 1.1,
            tr(self.lang, "$T$ integer token IDs", "$T$ 个整数 token ID"),
            size="scriptsize",
            color="Muted",
        )

    def indices(self, x, y, labels, target, symbol):
        for i, label in enumerate(labels):
            xx = x + i * EDGES["K"]
            self.lines.append(
                rf"\path[draw=Idx!70!Canvas,fill=Canvas,rounded corners=1pt] "
                rf"({xx:.3f},{y:.3f}) rectangle ++(.39,.42);"
            )
            self.text(xx + 0.195, y + 0.21, "$" + label + "$", size="scriptsize")
        self.text(x + 0.60, y + 0.75, "$" + symbol + "$", size="scriptsize", color="Muted")
        self.arrow(x + 0.60, y - 0.1, target[0], target[1], "Idx", True)

    def footer(self, y, axes, objects, mechanism, subject):
        rows = [
            (tr(self.lang, "Axes", "轴"), axes),
            (tr(self.lang, "Objects", "对象"), objects),
            (tr(self.lang, "Mechanism", "机制"), mechanism),
        ]
        text = r"\begin{tabular}{@{}p{2.2cm}@{\hspace{7pt}}p{" + str(self.width - 3.2) + "cm}@{}}"
        for i, (label, detail) in enumerate(rows):
            text += r"\textbf{" + label + r"} & \raggedright " + detail + r"\tabularnewline"
            if i < 2:
                text += "[6pt]"
        text += r"\end{tabular}"
        self.lines.append(
            rf"\node[anchor=north west,fill=Ink!3!Canvas,rounded corners=3pt,inner sep=8pt,"
            rf"font=\small,text width={self.width - 0.6:.3f}cm,align=left] (platefooter) at (0,{y:.3f}) {{{text}}};"
        )
        self.lines.append(
            r"\node[font=\scriptsize,text=Muted,anchor=north,inner sep=0pt] "
            r"at ($(platefooter.south)+(0,-.25)$) {" + subject + r" $\cdot$ shichao};"
        )

    def finish(self):
        self.lines += [r"\end{tikzpicture}", r"\end{document}"]
        return "\n".join(self.lines) + "\n", self.bounds


def metiscraft(lang):
    p = Plate(lang)
    p.formula(
        r"\widehat H=H[P,:],\quad U=\operatorname{GELU}(\widehat H W_1^{\mathsf T}+b_1),"
        r"\quad s=\operatorname{squeeze}(UW_2^{\mathsf T}+b_2)"
    )
    p.stage(1.9, 2.85, tr(lang, "1. One causal candidate list", "1. 同题候选的因果序列"), 4.2)
    p.stage(6.3, 2.85, "2. Qwen3-0.6B-Base", 3.8)
    p.stage(
        11.7,
        2.85,
        tr(lang, "3. Candidate-end gather", "3. 选择候选段末尾的 hidden 行"),
        5.5,
    )
    p.stage(18.2, 2.85, tr(lang, "Candidate visibility", "候选可见性"), 4)
    p.tokens(
        0,
        0,
        ["$s$", "$s$", "$q$", "$c_1$", "$c_1$", "$c_2$", "$c_2$", "$c_3$", "$c_3$"],
        ["Ctx", "Ctx", "Qst", "Opt", "Hid", "Opt", "Hid", "Opt", "Hid"],
    )
    p.op(
        4.5,
        0,
        3.5,
        tr(
            lang,
            r"28 causal decoder blocks\\RMSNorm $\to$ GQA\\RMSNorm $\to$ gated FFN\\residuals; LoRA $r=16$",
            r"28 层因果 decoder\\RMSNorm $\to$ GQA\\RMSNorm $\to$ 门控 FFN\\残差；LoRA $r=16$",
        ),
        2.65,
    )
    h = p.tensor("hidden", 8.8, 0, "T", "d", "H", r"T\times d", selected=[4, 6, 8])
    hh = p.tensor("gathered", 12.3, 0, "K", "d", r"\widehat H", r"K\times d")
    p.arrow(3.93, 0, 4.35, 0)
    p.arrow(8.15, 0, 8.65, 0)
    p.gather(h, hh, [4, 6, 8])
    p.indices(12.55, 1.55, ["4", "6", "8"], (13.25, 0.88), "P")
    mask = p.tensor(
        "causal_visibility",
        17.4,
        0,
        "K",
        "K",
        "M^{\\rm cand\\to cand}",
        r"K\times K",
        role="Msk",
        support=lambda r, c: c <= r,
    )
    p.text(
        18.0,
        -1.8,
        tr(
            lang,
            r"row = query; column = key\\later candidates see earlier ones",
            r"行是 query；列是 key\\后候选可读取前候选",
        ),
        4.3,
        size="scriptsize",
        color="Muted",
    )
    assert mask["width"] == mask["height"]
    y = -5.1
    p.stage(
        3.8,
        -3.6,
        tr(
            lang,
            "4. Shared scalar MLP: $1024 \\to 256 \\to 1$",
            "4. 共享标量 MLP：$1024 \\to 256 \\to 1$",
        ),
        8,
    )
    p.stage(
        17.8,
        -3.6,
        tr(lang, "5. Normalize over the valid answer space", "5. 在有效答案空间中归一化"),
        7,
    )
    p.tensor("gathered_again", 0.35, y, "K", "d", r"\widehat H", r"K\times d")
    p.text(2.9, y, "$\\times$", size="Large")
    p.tensor("weight1", 3.7, y, "d", "m", r"W_1^{\mathsf T}", r"d\times m", "Wgt")
    p.op(5, y, 1.6, r"$+b_1$; GELU")
    p.arrow(4.32, y, 4.85, y)
    p.arrow(6.75, y, 7.2, y)
    p.tensor("mlp_hidden", 7.35, y, "K", "m", "U", r"K\times m")
    p.text(8.5, y, "$\\times$", size="Large")
    p.tensor("weight2", 9.3, y, "m", "1", r"W_2^{\mathsf T}", r"m\times1", "Wgt")
    p.text(10.4, y, "$+b_2=$", size="small")
    p.tensor("pre_squeeze", 11.3, y, "K", "1", "v", r"K\times1", "Lgt")
    p.op(12.35, y, 1.85, "squeeze")
    p.arrow(11.7, y, 12.2, y)
    p.arrow(14.35, y, 14.9, y)
    p.tensor("scores", 15.05, y, "K", "1", "s", "K", "Lgt")
    p.op(16.45, y, 3.0, r"$/\tau$; softmax / sigmoid")
    p.arrow(15.45, y, 16.3, y)
    p.arrow(19.6, y, 20.2, y)
    p.tensor("probabilities", 20.35, y, "K", "1", "p", "K", "Lgt")
    p.footer(
        -7.35,
        tr(
            lang,
            r"One question shown; batch $B$ omitted. $T$: tokens; $K$: valid candidates; $d=1024$, $m=256$. Tiles illustrate axis structure, not the literal dimensions.",
            r"展示单题，省略 batch $B$。$T$：token 数；$K$：有效候选数；$d=1024$，$m=256$。格数仅示意轴结构，不是真实维度。",
        ),
        tr(
            lang,
            r"$x$ and $P$ are integer IDs/positions; $H,\widehat H,U$ are continuous states; $W_1,W_2$ are learned weights; $s$ are raw scores and $p$ probabilities. Biases appear in the formula.",
            r"$x,P$ 为整数 ID／位置；$H,\widehat H,U$ 为连续隐状态；$W_1,W_2$ 为可训练权重；$s$ 为原始分数，$p$ 为概率。偏置在公式中显式给出。",
        ),
        tr(
            lang,
            r"Choice example, \texttt{candidate\_scalar/list}: actual candidate-end tokens, no inserted markers. Noul instead selects the final Answer cue ($K=1$). Choice/score use masked softmax; noul/multi-label use sigmoid. No tree cache; LM head bypassed.",
            r"主图为 choice 的 \texttt{candidate\_scalar/list}：选中实际候选段末 token，不插入标记；noul 改取 Answer 提示末位（$K=1$）。choice／score 用有效候选 softmax；noul／多标签用 sigmoid。无 tree cache；绕过词表头。",
        ),
        "MetisCraft: scalar-list model architecture",
    )
    return p.finish()


def tev1(lang):
    p = Plate(lang)
    p.formula(
        r"h=H_{t^\star,:},\quad z=W_{\rm lm}h^{\mathsf T},"
        r"\quad s_k=z_{\ell_k},\quad p_k=\operatorname{softmax}_k(s/\tau)"
    )
    p.stage(
        1.9, 2.85, tr(lang, "1. State, question and answer choices", "1. 状态、问题与答案选项"), 4.2
    )
    p.stage(6.3, 2.85, "2. Qwen3.5-4B", 3.6)
    p.stage(11.7, 2.85, tr(lang, "3. Last prompt position", "3. 提示末位置读出"), 5.6)
    p.tokens(
        0,
        0,
        ["$s$", "$s$", "$q$", "A", "$c_1$", "B", "$c_2$", "C", "?"],
        ["Ctx", "Ctx", "Qst", "Opt", "Opt", "Opt", "Opt", "Opt", "Hid"],
    )
    p.op(
        4.5,
        0,
        3.5,
        tr(
            lang,
            r"32 hybrid blocks\\$[3\,\mathrm{DeltaNet}+1\,\mathrm{full}]\times8$\\dense gated FFNs\\replica: LoRA $r=8$",
            r"32 层混合主干\\$[3\,\mathrm{DeltaNet}+1\,\mathrm{full}]\times8$\\稠密门控 FFN\\复现：LoRA $r=8$",
        ),
        2.65,
    )
    h = p.tensor("hidden", 8.8, 0, "T", "d", "H", r"T\times d", selected=[8])
    hh = p.tensor("last_hidden", 12.3, 0, "1", "d", "h", r"1\times d")
    p.arrow(3.93, 0, 4.35, 0)
    p.arrow(8.15, 0, 8.65, 0)
    p.gather(h, hh, [8])
    p.text(13.25, 1.35, "$t^\\star=T-1$", color="Idx")
    p.arrow(13.25, 1, 13.25, 0.38, "Idx", True)
    p.text(
        18.1,
        0,
        tr(
            lang,
            r"$d=2560$, $V=248320$\\existing LM vocabulary head\\no added scalar MLP\\text branch; no rationale decoding",
            r"$d=2560$，$V=248320$\\沿用原始 LM 词表头\\不新增标量 MLP\\文本分支；不生成推理链",
        ),
        5.0,
    )
    y = -5.55
    p.stage(3.4, -3.4, tr(lang, "4. Full vocabulary projection", "4. 完整词表投影"), 6)
    p.stage(12.6, -3.4, tr(lang, "5. Answer-code gather", "5. 选择答案码 logits"), 4.8)
    p.stage(18.3, -3.4, tr(lang, "6. Candidate probabilities", "6. 候选概率"), 5.0)
    p.tensor("lm_weight", 0.5, y, "V", "d", r"W_{\rm lm}", r"V\times d", "Wgt")
    p.text(3.15, y, "$\\times$", size="Large")
    p.tensor("column_hidden", 4.05, y, "d", "1", r"h^{\mathsf T}", r"d\times1")
    p.text(5.2, y, "$=$", size="Large")
    z = p.tensor("vocabulary_logits", 6.15, y, "V", "1", "z", "V", "Lgt", selected=[2, 6, 10])
    s = p.tensor("candidate_scores", 10.2, y, "K", "1", "s", "K", "Lgt")
    p.gather(z, s, [2, 6, 10], "Lgt")
    p.indices(8.2, -3.85, [r"\ell_A", r"\ell_B", r"\ell_C"], (9.0, -4.95), r"\ell")
    p.op(12.15, y, 3.0, r"$/\tau$; softmax$_K$")
    p.arrow(10.6, y, 12, y)
    p.arrow(15.3, y, 16.1, y)
    p.tensor("probabilities", 16.25, y, "K", "1", "p", "K", "Lgt")
    p.text(
        19.55,
        y,
        tr(
            lang,
            r"$\ell_k$: token ID of letter $k$\\not a hidden-state position\\$W_{\rm lm}$ tied to embeddings",
            r"$\ell_k$：字母 $k$ 的 token ID\\不是 hidden 行位置\\$W_{\rm lm}$ 与 embedding 绑定",
        ),
        3.6,
        size="scriptsize",
        color="Muted",
    )
    p.footer(
        -8.25,
        tr(
            lang,
            r"Single question; $T$: prompt length, $d=2560$, $V=248320$, $K$: valid choices. Tile counts illustrate axes. $h$ is explicitly transposed before the GEMV.",
            r"展示单题。$T$：提示长度；$d=2560$，$V=248320$；$K$：有效选项数。格数示意轴结构；GEMV 前显式转置 $h$。",
        ),
        tr(
            lang,
            r"The replica computes all $V$ logits, then gathers integer code IDs $\ell$; no scalar head or selected-weight-only shortcut is implemented on this path. Softmax runs only over valid choices.",
            r"复现代码先计算全部 $V$ 个 logits，再按整数答案码 $\ell$ gather；此路径没有标量头，也未实现只投影部分词表权重的捷径。softmax 只覆盖有效选项。",
        ),
        tr(
            lang,
            r"Replica training: input prompt + gold letter; two full-vocabulary CE targets (letter, EOS). Upstream saved evaluation uses temperature-0 constrained generation; the replica reads logits directly. Accuracy reproduction does not recover historical weights/training.",
            r"复现训练：输入提示＋正确字母，两位置全词表 CE 目标为字母、EOS。上游保存评测使用 temperature-0 约束生成；复现直接读 logits。达到准确率判据不代表恢复历史权重／训练。",
        ),
        "Tev1-4B: upstream family and MetisCraft reproduction",
    )
    return p.finish()


def model_header(p, info, layout=None, roles=None, selected=None):
    """Shared context only; each readout below implements its own actual algebra."""
    lang = p.lang
    p.stage(1.9, 2.85, tr(lang, "1. Token layout", "1. Token 组织"), 4.1)
    title = {
        "jevforge": "JevForge",
        "kev": "Kev",
        "clef": "Clef",
        "jevany": "JevAny",
        "jevany-lm": "JevAny LM-token",
        "gliclass": "GLiClass",
        "gliner2": "GLiNER2",
        "open-jev": "Open-Jev",
        "setfit": "SetFit",
        "qwen3-reranker": "Qwen3 Reranker",
    }.get(info["id"], info["id"])
    p.stage(6.3, 2.85, "2. " + title, 3.6)
    p.stage(
        11.7, 2.85, tr(lang, "3. Contextual states and indices", "3. 上下文状态与读出索引"), 5.6
    )
    layout = layout or ["$s$", "$s$", "$q$", "A", "$c_1$", "B", "$c_2$", "C", "?"]
    roles = roles or ["Ctx", "Ctx", "Qst", "Opt", "Opt", "Opt", "Opt", "Opt", "Hid"]
    p.tokens(0, 0, layout, roles)
    p.op(4.5, 0, 3.5, r"\\".join(prose(x) for x in info["backbone"][lang]), 2.65)
    h = p.tensor("hidden", 8.8, 0, "T", "d", "H", r"T\times d", selected=selected or [8])
    p.arrow(3.93, 0, 4.35, 0)
    p.arrow(8.15, 0, 8.65, 0)
    return h


def model_footer(p, info, y, axes=None):
    axes = axes or tr(
        p.lang,
        "Single-question illustration; batch omitted. Tile counts encode axes, not literal sizes.",
        "展示单题并省略 batch；格数表示轴结构，不是真实大小。",
    )
    objects = tr(
        p.lang,
        r"Gray: input/context; violet: hidden states; orange: parameter matrices; red: logits/probabilities. Positions, code IDs and owner maps are integers, not scores. Tiles show roles, not measured magnitudes.",
        r"灰色：输入／上下文；紫色：隐状态；橙色：参数矩阵；红色：logits／概率。位置、答案码和 owner 为整数索引，不是分数。色块表示角色，不表示实测大小。",
    )
    p.footer(
        y,
        "$" + info["dimensions"] + "$; " + axes,
        objects,
        prose(info["mechanism"][p.lang]) + r"\\[3pt] " + prose(info["training"][p.lang]),
        info["id"] + tr(p.lang, ": inspected model path", "：模型路径核查"),
    )


def cache_fork(p):
    lang = p.lang
    p.op(
        16.1,
        0.8,
        3.6,
        tr(lang, r"prefix prefill\\KV / recurrent / conv", r"前缀预填充\\KV／循环／卷积状态"),
        1.1,
    )
    p.op(15.4, -1.1, 2.1, tr(lang, "suffix 1", "后缀 1"))
    p.op(18.3, -1.1, 2.1, tr(lang, "suffix 2", "后缀 2"))
    p.arrow(17.9, 0.1, 16.45, -0.55, "Msk")
    p.arrow(17.9, 0.1, 19.35, -0.55, "Msk")


def token_model(info, lang):
    binary = info.get("binary", False)
    p = Plate(lang, axes={"K": 2} if binary else None)
    selected_before = info["kind"] == "selected_lm"
    fixed = info["kind"] == "fixed_slots"
    formula = (
        r"h=H_{t^\star,:},\quad s_k=\langle W_{\ell_k,:},h\rangle,\quad p=\operatorname{softmax}_K(s/\tau)"
        if not fixed
        else r"h=H_{t^\star,:},\quad z=R h^{\mathsf T},\quad s=z_{0:K},\quad p=\operatorname{softmax}_K(s/\tau)"
    )
    if binary:
        formula = r"z=W_{\rm lm}h^{\mathsf T},\quad p_{yes}=\operatorname{softmax}(z_{no},z_{yes})_{yes}=\sigma(z_{yes}-z_{no})"
    p.formula(formula)
    layout = None if not binary else ["$q$", "$q$", "$d$", "$d$", "$d$", "$d$", "$a$", "$a$", "?"]
    binary_roles = (
        ["Qst", "Qst", "Opt", "Opt", "Opt", "Opt", "Gen", "Gen", "Hid"] if binary else None
    )
    h = model_header(p, info, layout=layout, roles=binary_roles)
    hh = p.tensor("last_hidden", 12.3, 0, "1", "d", "h", r"1\times d")
    p.gather(h, hh, [8])
    p.text(13.25, 1.35, "$t^\\star$", color="Idx")
    p.arrow(13.25, 1, 13.25, 0.38, "Idx", True)
    if info.get("cache"):
        cache_fork(p)
    else:
        p.text(18.0, 0, "$" + info["dimensions"] + "$", 5)
    y = -5.55
    if selected_before:
        native_lm = info["id"] == "jevany-lm"
        p.stage(
            3.0, -3.4, tr(lang, "4. Select existing LM weight rows", "4. 选择现有 LM 权重行"), 5.8
        )
        p.stage(
            12.7,
            -3.4,
            tr(lang, "5. LM-weight dtype projection", "5. 按 LM 权重精度投影")
            if native_lm
            else tr(lang, "5. FP32 selected-row projection", "5. FP32 部分行投影"),
            5.3,
        )
        w = p.tensor(
            "lm_weight", 0.3, y, "V", "d", r"W_{\rm lm}", r"V\times d", "Wgt", selected=[2, 6, 10]
        )
        wc = p.tensor("selected_weight", 4.4, y, "K", "d", r"W_{\ell,:}", r"K\times d", "Wgt")
        p.gather(w, wc, [2, 6, 10], "Wgt")
        p.indices(7.0, -3.85, [r"\ell_A", r"\ell_B", r"\ell_C"], (5.3, -4.65), r"\ell")
        p.text(7.2, y, "$\\times$", size="Large")
        p.tensor("hidden_column", 8.2, y, "d", "1", r"h^{\mathsf T}", r"d\times1")
        p.text(9.5, y, "$=$", size="Large")
        p.tensor("scores", 10.5, y, "K", "1", "s", "K", "Lgt")
        p.op(
            12.3,
            y,
            3.3,
            r"$/\tau$; softmax$_K$"
            if native_lm
            else tr(lang, r"mask; $/\tau$\\softmax$_K$", r"有效候选；$/\tau$\\softmax$_K$"),
            1.1,
        )
        p.arrow(10.9, y, 12.15, y)
        p.arrow(15.75, y, 16.6, y)
        p.tensor("probabilities", 16.75, y, "K", "1", "p", "K", "Lgt")
        p.text(
            19.9,
            y,
            tr(
                lang,
                r"LM weight dtype matmul\\logits converted to FP32",
                r"按 LM 权重 dtype 乘法\\logits 再转 FP32",
            )
            if native_lm
            else tr(
                lang,
                r"No new scalar head\\no full-$V$ inference logits",
                r"不新增标量头\\推理不产生完整 $V$ logits",
            ),
            3.5,
            size="scriptsize",
            color="Muted",
        )
    else:
        p.stage(3.4, -3.4, tr(lang, "4. Stored head projection", "4. 已存头参数的投影"), 6)
        p.stage(12.7, -3.4, tr(lang, "5. Valid score selection", "5. 选择有效分数"), 4.8)
        axis = "J" if fixed else "V"
        wname = "R" if fixed else r"W_{\rm lm}"
        p.tensor("readout_weight", 0.5, y, axis, "d", wname, axis + r"\times d", "Wgt")
        p.text(3.15, y, "$\\times$", size="Large")
        p.tensor("hidden_column", 4.05, y, "d", "1", r"h^{\mathsf T}", r"d\times1")
        p.text(5.2, y, "$=$", size="Large")
        rows = list(range(p.axes["K"])) if fixed else ([3, 9] if binary else [2, 6, 10])
        z = p.tensor("all_logits", 6.15, y, axis, "1", "z", axis, "Lgt", selected=rows)
        s = p.tensor("scores", 10.2, y, "K", "1", "s", "K", "Lgt")
        p.gather(z, s, rows, "Lgt")
        labs = (
            ["0", "1", "2"]
            if fixed
            else ([r"\ell_n", r"\ell_y"] if binary else [r"\ell_A", r"\ell_B", r"\ell_C"])
        )
        p.indices(8.2, -3.85, labs, (9, -4.95), "k" if fixed else r"\ell")
        p.op(12.15, y, 3, r"$/\tau$; softmax$_K$" if not binary else r"softmax$_{\{no,yes\}}$")
        p.arrow(10.6, y, 12, y)
        p.arrow(15.3, y, 16.1, y)
        p.tensor(
            "probabilities",
            16.25,
            y,
            "K",
            "1",
            "p_2" if binary else "p",
            "2" if binary else "K",
            "Lgt",
        )
        if binary:
            p.op(18.0, y, 1.85, "select yes")
            p.arrow(16.65, y, 17.85, y)
            p.arrow(20.0, y, 20.85, y)
            p.tensor("yes_probability", 21.0, y, "1", "1", "p_{yes}", "1", "Lgt")
        else:
            p.text(
                19.6,
                y,
                tr(
                    lang,
                    "independent slot weights" if fixed else "Existing LM vocabulary head",
                    "独立槽位权重" if fixed else "沿用原始 LM 词表头",
                ),
                3.7,
                size="scriptsize",
                color="Muted",
            )
    model_footer(p, info, -8.3)
    return p.finish()


def scalar_model(info, lang):
    marker = info["kind"] == "encoder_scalar"
    # GLiNER2's 2d expansion conserves the physical d edge, rather than using
    # the MetisCraft bottleneck geometry for a different head.
    p = Plate(lang, axes={"m": 16} if marker else {"m": 8})
    act = "ReLU" if marker else "GELU"
    selection = r"U=H[P,:]" if marker else r"U_r=H_r[t_r,:],\;t_r=\sum_t A^{pad}_{r,t}-1"
    p.formula(
        selection
        + r",\quad A=\operatorname{"
        + act
        + r"}(UW_1^{\mathsf T}+b_1),\quad s=\operatorname{squeeze}(AW_2^{\mathsf T}+b_2)"
    )
    if marker:
        h = model_header(
            p,
            info,
            ["[P]", "[L]", "$c_1$", "[L]", "$c_2$", "[L]", "$c_3$", "|", "$x$"],
            ["Qst", "Hid", "Opt", "Hid", "Opt", "Hid", "Opt", "Ctx", "Ctx"],
            [1, 3, 5],
        )
        u = p.tensor("gathered", 12.3, 0, "K", "d", "U", r"K\times d")
        p.gather(h, u, [1, 3, 5])
        p.text(
            18.1,
            0,
            tr(
                lang,
                r"Read [L], skip [P]\\no CLS pooling\\joint bidirectional encoding",
                r"读取 [L]、跳过 [P]\\不做 CLS 池化\\联合双向编码",
            ),
            4.5,
        )
    else:
        h = model_header(
            p,
            info,
            ["$s$", "$s$", "$q$", "$c$", "$c$", "$c$", "$a$", "$a$", "$a$"],
            ["Ctx", "Ctx", "Qst", "Opt", "Opt", "Opt", "Gen", "Gen", "Hid"],
        )
        p.text(
            1.9,
            -1.7,
            tr(
                lang,
                r"$R=\sum_q K_q$ independent rows\\only one candidate in each row",
                r"$R=\sum_q K_q$ 条独立序列\\每行仅含一个候选",
            ),
            4.0,
            size="scriptsize",
            color="Muted",
        )
        p.op(
            11.3,
            0,
            2.15,
            tr(lang, r"batched last\\valid-token gather", r"逐行取末个\\有效 token"),
            1.2,
        )
        p.arrow(10.85, 0, 11.15, 0)
        p.tensor("gathered", 14.1, 0, "R", "d", "U", r"R\times d")
        p.arrow(13.6, 0, 13.95, 0)
        p.text(
            18.9,
            0,
            tr(
                lang,
                r"$H_r[T_r,d]$ shown above\\each row has a separate causal context\\no cross-candidate attention",
                r"上方展示一行 $H_r[T_r,d]$\\各行具有独立因果上下文\\无跨候选注意力",
            ),
            4.6,
        )
    axis = "K" if marker else "R"
    if marker:
        y = -5.6
        p.stage(
            10.8,
            -3.5,
            tr(lang, "4. Shared d-to-2d expansion and ReLU", "4. 共享 d→2d 扩展及 ReLU"),
            15,
        )
        p.tensor("readout", 0.5, y, axis, "d", "U", axis + r"\times d")
        p.text(3.2, y, "$\\times$", size="Large")
        p.tensor("weight1", 4.15, y, "d", "m", r"W_1^{\mathsf T}", r"d\times m", "Wgt")
        p.op(9.1, y, 2.0, r"$+b_1$; ReLU")
        p.arrow(8.13, y, 8.95, y)
        p.arrow(11.25, y, 12.15, y)
        p.tensor("activation", 12.3, y, axis, "m", "A", axis + r"\times m")
        y = -10.4
        p.stage(
            10.8,
            -7.9,
            tr(
                lang,
                "5. Scalar contraction, squeeze and activation",
                "5. 标量收缩、squeeze 及概率激活",
            ),
            16,
        )
        p.tensor("activation_again", 0.5, y, axis, "m", "A", axis + r"\times m")
        p.text(5.1, y, "$\\times$", size="Large")
        p.tensor("weight2", 6.0, y, "m", "1", r"W_2^{\mathsf T}", r"m\times1", "Wgt")
        p.text(7.3, y, "$+b_2=$", size="small")
        p.tensor("scalar", 8.2, y, axis, "1", "v", axis + r"\times1", "Lgt")
        p.op(9.5, y, 2.1, "squeeze")
        p.arrow(8.6, y, 9.35, y)
        p.arrow(11.75, y, 12.4, y)
        p.tensor("score", 12.55, y, axis, "1", "s", axis, "Lgt")
        p.op(
            14.0,
            y,
            3.2,
            tr(
                lang,
                r"$/T_{temp}$\\single: softmax; multi: sigmoid",
                r"$/T_{temp}$\\单标签 softmax；多标签 sigmoid",
            ),
            1.5,
        )
        p.arrow(12.95, y, 13.85, y)
        p.arrow(17.35, y, 18.2, y)
        p.tensor("probability", 18.35, y, axis, "1", "p", axis, "Lgt")
        model_footer(p, info, -13.5)
    else:
        y = -5.1
        p.stage(
            10.8,
            -3.5,
            tr(
                lang,
                "4. Shared scalar head, then group by question owner",
                "4. 共享标量头，再按问题 owner 分组",
            ),
            18,
        )
        p.tensor("readout", 0.35, y, axis, "d", "U", r"R\times d")
        p.text(2.9, y, "$\\times$", size="Large")
        p.tensor("weight1", 3.65, y, "d", "m", r"W_1^{\mathsf T}", r"d\times m", "Wgt")
        p.op(6.25, y, 1.5, r"$+b_1$; GELU")
        p.arrow(5.72, y, 6.1, y)
        p.arrow(7.9, y, 8.3, y)
        p.tensor("activation", 8.45, y, axis, "m", "A", r"R\times m")
        p.text(11.05, y, "$\\times$", size="Large")
        p.tensor("weight2", 11.95, y, "m", "1", r"W_2^{\mathsf T}", r"m\times1", "Wgt")
        p.text(13.05, y, "$+b_2=$", size="small")
        p.tensor("scores", 13.9, y, axis, "1", "z", "R", "Lgt")
        p.op(
            15.2,
            y,
            4.0,
            tr(
                lang,
                r"squeeze; group by owner\\$/\tau_{type}$; softmax$_{K_q}$",
                r"squeeze；按 owner 分组\\$/\tau_{type}$；softmax$_{K_q}$",
            ),
            1.3,
        )
        p.arrow(14.3, y, 15.05, y)
        p.arrow(19.35, y, 20.1, y)
        p.tensor("probabilities", 20.25, y, "K", "1", "p_q", "K_q", "Lgt")
        model_footer(
            p,
            info,
            -7.6,
            tr(
                lang,
                r"$R$: independent candidate rows, not questions. The illustration shows one question with $R=K=3$. Equal d and m edges reflect this head's equal width.",
                r"$R$ 为独立候选行数，不是问题数。示例仅展示一题的 $R=K=3$ 行；d 与 m 等宽对应此头的等宽隐层。",
            ),
        )
    return p.finish()


def pointer_model(info, lang):
    p = Plate(lang)
    p.formula(
        r"E=H[P_{opt},:],\;D=H[P_{decide},:],\quad k=EW_k^{\mathsf T}+b_k,\;q=DW_q^{\mathsf T}+b_q,\quad s=kq^{\mathsf T}/\sqrt{d_p}"
    )
    h = model_header(
        p,
        info,
        ["$s$", "$q$", "$c_1$", ")", "$c_2$", ")", "$c_3$", ")", "D"],
        ["Ctx", "Qst", "Opt", "Hid", "Opt", "Hid", "Opt", "Hid", "Hid"],
        [3, 5, 7, 8],
    )
    e = p.tensor("options", 12.3, 0.6, "K", "d", "E", r"K\times d")
    p.gather(h, e, [3, 5, 7])
    d = p.tensor("decision", 12.3, -1.6, "1", "d", "D", r"1\times d")
    p.gather(h, d, [8])
    p.text(
        18.2,
        0,
        tr(
            lang,
            r"E: option-closing positions\\D: decision marker\\decision sees every option",
            r"E：候选闭合位置\\D：decision 标记\\decision 可见全部选项",
        ),
        4.8,
    )
    p.stage(4.3, -3.55, tr(lang, "4. Separate biased projections", "4. 两个独立带偏置投影"), 8)
    for y, name, axis, role, weight in [(-5.3, "E", "K", "Hid", "k"), (-8.7, "D", "1", "Qst", "q")]:
        p.tensor(name + "_again", 0.3, y, axis, "d", name, axis + r"\times d", role)
        p.text(2.85, y, "$\\times$", size="Large")
        p.tensor(
            "W" + weight, 3.65, y, "d", "p", "W_" + weight + r"^{\mathsf T}", r"d\times d_p", "Wgt"
        )
        p.text(5.6, y, "$+b_" + weight + "=$", size="small")
        p.tensor("projected_" + weight, 6.5, y, axis, "p", weight, axis + r"\times d_p", role)
    p.stage(16.2, -3.55, tr(lang, "5. Query-option dot product", "5. Query 与候选的点积"), 10)
    y = -6.9
    p.tensor("keys_again", 10.4, y, "K", "p", "k", r"K\times d_p")
    p.text(12.45, y, "$\\times$", size="Large")
    p.tensor("query_column", 13.35, y, "p", "1", r"q^{\mathsf T}", r"d_p\times1", "Qst")
    p.text(14.5, y, "$=$", size="Large")
    p.tensor("dot", 15.4, y, "K", "1", r"kq^{\mathsf T}", "K", "Lgt")
    p.op(16.75, y, 3.4, r"$/\sqrt{d_p}$; $/\tau$\\softmax$_K$", 1.2)
    p.arrow(15.8, y, 16.6, y)
    p.arrow(20.3, y, 20.75, y)
    p.tensor("probabilities", 20.9, y, "K", "1", "p", "K", "Lgt")
    model_footer(p, info, -10.45)
    return p.finish()


def fixed_class_model(info, lang):
    p = Plate(lang)
    pre = info["id"] == "open-jev"
    p.formula(
        r"e=\operatorname{mean}_{mask,T}(H),\quad r=\operatorname{LN}(\operatorname{GELU}(eW_{pre}^{\mathsf T})),\quad z=rW_{cls}^{\mathsf T}+b"
        if pre
        else r"e=\operatorname{mean}_{mask,T}(H),\quad z=eW_{cls}^{\mathsf T}+b,\quad p=\operatorname{softmax}_K(z)"
    )
    model_header(p, info, ["$x$"] * 9, ["Ctx"] * 9, list(range(9)))
    p.op(11.35, 0, 2.0, r"mask mean$_T$")
    p.arrow(10.85, 0, 11.2, 0)
    p.arrow(13.5, 0, 14.0, 0)
    p.tensor("pooled", 14.15, 0, "1", "d", "e", r"1\times d")
    p.text(
        18.9,
        0,
        tr(
            lang,
            r"Text-only student input\\class names do not enter the encoder\\each head column is a fixed class",
            r"学生仅输入文本\\类别名不进入编码器\\头的每一列对应固定类别",
        ),
        4.7,
    )
    y = -5.1
    p.stage(11, -3.5, tr(lang, "4. Fixed-class classifier", "4. 固定类别分类头"), 19)
    p.tensor("pooled_again", 0.35, y, "1", "d", "e", r"1\times d")
    if pre:
        p.op(3.25, y, 4.0, r"Linear $d\to d$\\GELU; LN; dropout $=0$", 1.3)
        p.arrow(2.4, y, 3.1, y)
        p.arrow(7.4, y, 7.9, y)
        p.tensor("classifier_feature", 8.05, y, "1", "d", "r", r"1\times d")
        p.text(10.7, y, "$\\times$", size="Large")
        p.tensor("class_weight", 11.55, y, "d", "K", r"W_{cls}^{\mathsf T}", r"d\times K", "Wgt")
        p.text(13.7, y, "$+b=$", size="small")
        p.tensor("class_logits", 14.55, y, "1", "K", "z", r"1\times K", "Lgt")
        p.op(16.7, y, 2.65, r"$/T_{cal}+b_{cal}$\\softmax$_K$", 1.1)
        p.arrow(15.95, y, 16.55, y)
        p.arrow(19.5, y, 20.1, y)
        p.tensor("class_probabilities", 20.25, y, "1", "K", "p", r"1\times K", "Lgt")
    else:
        p.text(3.2, y, "$\\times$", size="Large")
        p.tensor("class_weight", 4.15, y, "d", "K", r"W_{cls}^{\mathsf T}", r"d\times K", "Wgt")
        p.text(6.3, y, "$+b=$", size="small")
        p.tensor("class_logits", 7.2, y, "1", "K", "z", r"1\times K", "Lgt")
        p.op(9.8, y, 2.5, "softmax$_K$")
        p.arrow(8.6, y, 9.65, y)
        p.arrow(12.45, y, 13.1, y)
        p.tensor("class_probabilities", 13.25, y, "1", "K", "p", r"1\times K", "Lgt")
        p.text(
            18.5,
            y,
            tr(
                lang,
                r"Stage 1 trains the sentence body\\Stage 2 fits logistic regression\\no runtime candidate encoding",
                r"阶段一训练句向量 body\\阶段二拟合逻辑回归\\推理不编码候选文本",
            ),
            5.4,
        )
    model_footer(p, info, -7.5)
    return p.finish()


def encoder_dot_model(info, lang):
    p = Plate(lang, width=22.4)
    p.formula(
        r"q=f_{text}(H_{CLS,:}),\quad L=f_{label}(H[P_L,:]),\quad s=qL^{\mathsf T},\quad p=\sigma(s)"
    )
    h = model_header(
        p,
        info,
        ["CLS", "$x$", "$x$", "L", "$c_1$", "L", "$c_2$", "L", "$c_3$"],
        ["Hid", "Ctx", "Ctx", "Hid", "Opt", "Hid", "Opt", "Hid", "Opt"],
        [0, 3, 5, 7],
    )
    c = p.tensor("class_features", 12.3, 0.6, "K", "d", "C_0", r"K\times d")
    p.gather(h, c, [3, 5, 7])
    p.tensor("text_feature", 12.3, -1.65, "1", "d", "t", r"1\times d")
    # The CLS route occupies its own high lane; it never crosses another matrix.
    p.arrow(10.85, p.row_y(h, 0), 11.2, 1.95, "Hid")
    p.lines += [
        r"\begin{scope}[on background layer]",
        r"\draw[-{Latex[length=1.8mm]},draw=Hid!80!Canvas,line width=.7pt] (11.2,1.95) -- (14.8,1.95) -- (14.8,-1.65) -- (14.35,-1.65);",
        r"\end{scope}",
    ]
    p.text(
        18.2,
        0,
        tr(
            lang,
            r"Joint bidirectional input\\read CLS + label markers\\no feature L2; no temperature",
            r"联合双向输入\\读取 CLS 与标签标记\\不做特征 L2、无温度缩放",
        ),
        4.6,
    )
    p.stage(
        4.7,
        -3.55,
        tr(lang, "4. Distinct text and label projectors", "4. 文本与标签各自独立的投影器"),
        9,
    )
    for y, symbol, axis, output in [(-5.3, "C_0", "K", "L"), (-8.7, "t", "1", "q")]:
        p.tensor(symbol + "_again", 0.3, y, axis, "d", symbol, axis + r"\times d")
        p.op(3.2, y, 3.4, r"Linear $d\to d$\\GELU; Linear $d\to d$", 1.25)
        p.arrow(2.35, y, 3.05, y)
        p.arrow(6.75, y, 7.4, y)
        p.tensor(output + "_projected", 7.55, y, axis, "d", output, axis + r"\times d")
    p.stage(16.5, -3.55, tr(lang, "5. Text-label dot product", "5. 文本与标签点积"), 10.0)
    y = -6.9
    p.tensor("q_again", 10.4, y, "1", "d", "q", r"1\times d")
    p.text(13.0, y, "$\\times$", size="Large")
    p.tensor("label_transpose", 13.8, y, "d", "K", r"L^{\mathsf T}", r"d\times K")
    p.text(15.8, y, "$=$", size="Large")
    p.tensor("scores", 16.6, y, "1", "K", "s", r"1\times K", "Lgt")
    p.op(18.7, y, 1.45, "sigmoid")
    p.arrow(18.0, y, 18.55, y)
    p.arrow(20.3, y, 20.7, y)
    p.tensor("probabilities", 20.85, y, "1", "K", "p", r"1\times K", "Lgt")
    model_footer(p, info, -10.45)
    return p.finish()


def joint_model(info, lang):
    p = Plate(lang, width=24)
    p.formula(
        r"M=\operatorname{LN}(H)W_{mem}^{\mathsf T},\quad O=\operatorname{Route}^{2}(O^0;M),\quad F=\operatorname{Decoder}^{4}(F^0;M)"
    )
    model_header(
        p,
        info,
        ["$s$", "$q_1$", "$c_1$", "$c_2$", "$q_2$", "$c_3$", "$a$", "$a$", "$a$"],
        ["Ctx", "Qst", "Opt", "Opt", "Qst", "Opt", "Gen", "Gen", "Hid"],
        list(range(9)),
    )
    p.text(
        12,
        3.45,
        r"$O^0=CW_c^{\mathsf T}+LW_l^{\mathsf T}+(UW_{oq}^{\mathsf T})_{owner}$",
        size="small",
        color="Muted",
    )
    p.op(11.4, 0, 2.3, r"LN$_d$\\Linear $d\to w$", 1.3)
    p.arrow(10.85, 0, 11.25, 0)
    p.arrow(13.85, 0, 14.4, 0)
    p.tensor("memory", 14.55, 0, "T", "w", "M", r"T\times w")
    p.text(
        20.0,
        0,
        tr(
            lang,
            r"$H_c=LN_d(H)$; $g=(H_c)_{-1}$\\C,U: span means of $H_c$\\M: all valid token memory",
            r"$H_c=LN_d(H)$；$g=(H_c)_{-1}$\\C、U：$H_c$ 的 span mean\\M：全部有效 token memory",
        ),
        5.4,
    )
    y = -5.7
    p.stage(
        4.3,
        -3.55,
        tr(lang, "4. Context, lexical and question features", "4. 上下文、词面与问题特征"),
        8.8,
    )
    p.tensor("context_pool", 0.3, y, "N", "d", "C", r"N\times d")
    p.tensor("lexical_pool", 3.3, y, "N", "d", "L", r"N\times d", "Wgt")
    p.tensor("question_pool", 6.3, y, "Q", "d", "U", r"Q\times d", "Qst")
    p.text(
        1.25,
        -7.3,
        tr(lang, "mean option spans", "选项 span mean"),
        2.7,
        size="scriptsize",
        color="Muted",
    )
    p.text(
        4.25,
        -7.3,
        tr(lang, "mean LM output rows", "LM 输出权重行 mean"),
        2.8,
        size="scriptsize",
        color="Muted",
    )
    p.text(
        7.25,
        -7.3,
        tr(lang, "mean instruction spans", "问题 span mean"),
        2.8,
        size="scriptsize",
        color="Muted",
    )
    p.op(
        9.15,
        y,
        3.7,
        tr(
            lang,
            r"3 Linear $d\to w$\\owner broadcast; sum",
            r"3 路 Linear $d\to w$\\owner 广播；求和",
        ),
        1.25,
    )
    p.tensor("option_queries", 13.75, y, "N", "w", "O^0", r"N\times w")
    p.arrow(12.99, y, 13.6, y)
    p.op(
        16.0,
        y,
        4.7,
        tr(
            lang,
            r"$2\times$ evidence layer\\CA$(LN(O),LN(M))$\\residual; pre-LN FFN + residual",
            r"$2\times$ evidence 层\\CA$(LN(O),LN(M))$\\残差；pre-LN FFN＋残差",
        ),
        1.8,
    )
    p.arrow(15.18, y, 15.85, y)
    p.arrow(20.85, y, 21.4, y)
    p.tensor("routed_options", 21.55, y, "N", "w", "O", r"N\times w")
    p.stage(
        11.8,
        -8.55,
        tr(
            lang,
            "5. Candidate-weighted fields, then global field interaction",
            "5. 候选加权形成字段，再进行全局字段交互",
        ),
        22,
    )
    y = -10.8
    p.tensor("options_again", 0.3, y, "N", "w", "O", r"N\times w")
    p.op(
        2.65,
        y,
        6.0,
        r"$F_{base}=UW_{field}^{\mathsf T}$\\"
        r"$\alpha=\operatorname{softmax}_K(O\cdot F_{base}/\sqrt{w})$\\"
        r"$F^0=F_{base}+LN(\sum\alpha O)$\\$+W_g g+e_{type}$",
        2.15,
    )
    p.arrow(1.73, y, 2.5, y)
    p.arrow(8.8, y, 9.35, y)
    p.tensor("fields_initial", 9.5, y, "Q", "w", "F^0", r"Q\times w", "Qst")
    p.op(
        11.8,
        y,
        5.0,
        tr(
            lang,
            r"$4\times$ TransformerDecoder\\field self-attn; cross-attn to M\\FFN; residuals; final LN",
            r"$4\times$ TransformerDecoder\\字段 self-attn；对 M cross-attn\\FFN；残差；末层 LN",
        ),
        1.9,
    )
    p.arrow(10.93, y, 11.65, y)
    p.arrow(16.95, y, 17.55, y)
    p.tensor("fields", 17.7, y, "Q", "w", "F", r"Q\times w", "Qst")
    p.text(
        21.5,
        y,
        tr(
            lang,
            r"Global field visibility\\full token-memory access\\companion scorer plate follows",
            r"字段之间全可见\\可读全部 token memory\\配套图展开评分器",
        ),
        4.3,
    )
    model_footer(
        p,
        info,
        -13.0,
        tr(
            lang,
            r"One record, Q questions, N total options (N is not a fixed K per field). The illustration uses three option rows and two fields; w=1024.",
            r"一个 record 含 Q 道题、N 个总候选；N 不等于每题固定 K。示意三行候选、两个字段；w=1024。",
        ),
    )
    return p.finish()


def joint_scorer(info, lang):
    p = Plate(lang, width=24)
    p.formula(
        r"X=[F_g,V,F_g\odot V,|F_g-V|],\quad j=\rho\cos(F_g,V)+\operatorname{MLP}(X),\quad s=\pi+\sigma(\gamma)j"
    )
    p.stage(
        12,
        3.2,
        tr(
            lang,
            "Joint residual: 4096-to-1024-to-1 MLP, plus field-option cosine",
            "联合残差：4096→1024→1 MLP，加字段与候选 cosine",
        ),
        23,
    )
    p.tensor("concat", 0.3, 0, "N", "f", "X", r"N\times4w")
    p.text(6.1, 0, "$\\times$", size="Large")
    p.tensor("weight1", 7.0, 0, "f", "w", r"W_1^{\mathsf T}", r"4w\times w", "Wgt")
    p.op(9.25, 0, 1.6, r"$+b_1$; GELU")
    p.arrow(8.43, 0, 9.1, 0)
    p.arrow(11.0, 0, 11.55, 0)
    p.tensor("activation", 11.7, 0, "N", "w", "A", r"N\times w")
    p.text(13.75, 0, "$\\times$", size="Large")
    p.tensor("weight2", 14.65, 0, "w", "1", r"W_2^{\mathsf T}", r"w\times1", "Wgt")
    p.op(15.85, 0, 4.2, r"$+b_2$; squeeze\\$+\rho\cos(F_g,V)$", 1.15)
    p.arrow(15.04, 0, 15.7, 0)
    p.arrow(20.2, 0, 20.9, 0)
    p.tensor("joint", 21.05, 0, "N", "1", "j", "N", "Lgt")
    p.stage(
        12,
        -4.0,
        tr(
            lang,
            "Lexical prior plus gated residual, then field-local normalization",
            "词面先验加门控残差，再按字段局部归一化",
        ),
        23,
    )
    p.text(
        12,
        -4.7,
        r"$\pi_n=\rho_p\langle L_2(L_n),L_2(U_{owner(n)}+g)\rangle,\quad g=(LN_d(H))_{-1}$",
        size="small",
        color="Muted",
    )
    y = -6.1
    p.tensor("lexical_prior", 0.3, y, "N", "1", r"\pi", "N", "Lgt")
    p.text(1.7, y, "$+$", size="Large")
    p.op(
        2.65,
        y,
        5.0,
        tr(
            lang,
            r"broadcast scalar gate $\sigma(\gamma)$\\multiply joint residual j",
            r"广播标量门控 $\sigma(\gamma)$\\乘联合残差 j",
        ),
        1.2,
    )
    p.text(8.6, y, "$=$", size="Large")
    p.tensor("scores", 9.5, y, "N", "1", "s", "N", "Lgt")
    p.op(11.2, y, 4.1, tr(lang, "split by field; softmax", "按字段切分；softmax"))
    p.arrow(9.9, y, 11.05, y)
    p.arrow(15.45, y, 16.15, y)
    p.tensor("probabilities", 16.3, y, "N", "1", "p", "N", "Lgt")
    p.text(
        20.2,
        y,
        tr(
            lang,
            r"$F_g$: owner-indexed field\\$V=LN(O)$\\$L_2$: feature-wise normalization",
            r"$F_g$：按 owner 取字段\\$V=LN(O)$\\$L_2$：沿特征轴归一化",
        ),
        4.8,
        size="scriptsize",
    )
    model_footer(
        p,
        info,
        -8.0,
        tr(
            lang,
            r"N total options; w=1024. Four width-w feature parts concatenate to width 4w; both 4w contraction edges match. Bias terms are included in the MLP.",
            r"N 为总候选数；w=1024。四份宽 w 特征拼成 4w，两处 4w 收缩边一致。MLP 包含偏置。",
        ),
    )
    return p.finish()


def available():
    return ("metiscraft-architecture", "tev1-architecture", "architecture-clef-scorer") + tuple(
        "architecture-" + model["id"] for model in sources().get("models", [])
    )


def render(slug, lang):
    if slug == "metiscraft-architecture":
        return metiscraft(lang)
    if slug == "tev1-architecture":
        return tev1(lang)
    models = {"architecture-" + m["id"]: m for m in sources()["models"]}
    if slug == "architecture-clef-scorer":
        return joint_scorer(models["architecture-clef"], lang)
    info = models[slug]
    kind = info["kind"]
    if kind in {"full_lm", "selected_lm", "fixed_slots"}:
        return token_model(info, lang)
    if kind in {"scalar_pairs", "encoder_scalar"}:
        return scalar_model(info, lang)
    if kind == "pointer":
        return pointer_model(info, lang)
    if kind == "fixed_class":
        return fixed_class_model(info, lang)
    if kind == "encoder_dot":
        return encoder_dot_model(info, lang)
    if kind == "joint":
        return joint_model(info, lang)
    raise ValueError(slug)


def validate_ledger(bounds):
    """Recompute faces from symbolic axes; do not merely trust the renderer."""
    for obj in bounds:
        if obj.get("kind") != "tensor":
            continue
        row, col = obj["axes"]
        assert abs(obj["height"] - obj["rows"] * EDGES[row]) < 1e-9, obj
        assert abs(obj["width"] - obj["cols"] * EDGES[col]) < 1e-9, obj
        if row == col:
            assert obj["height"] == obj["width"], obj


def sources():
    path = Path(__file__).with_name("model-atlas.json")
    return json.loads(path.read_text()) if path.exists() else {}
