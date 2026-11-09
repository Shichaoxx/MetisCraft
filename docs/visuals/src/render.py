"""Render MetisCraft's editable SVG identity and explanatory figures (stdlib only)."""

from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "figures"
INK, TEAL = "#243438", "#087f78"


def text(x, y, value, size=20, color=INK, weight=400, **attrs):
    extra = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
        f'font-weight="{weight}" {extra}>{escape(value)}</text>'
    )


def monogram(color=TEAL):
    return (
        f'<g transform="translate(13 28) skewY(-14)" fill="{color}">'
        '<rect x="0" y="14" width="19" height="54" rx="3"/>'
        '<rect x="27" y="30" width="19" height="38" rx="3" opacity=".6"/>'
        '<rect x="54" y="0" width="19" height="68" rx="3" opacity=".85"/></g>'
    )


def save(name, width, height, title, body, bg="none"):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title">'
        f'<title id="title">{escape(title)}</title>'
        '<style>text {font-family: "Inter", "DejaVu Sans", Arial, sans-serif}</style>'
        f'<rect width="{width}" height="{height}" rx="18" fill="{bg}"/>{body}</svg>\n',
        encoding="utf-8",
    )


def main():
    save("mark-light", 100, 100, "MetisCraft three-plane mark, light theme", monogram())
    save("mark-dark", 100, 100, "MetisCraft three-plane mark, dark theme", monogram("#cbd5df"))
    save("mark-ink", 100, 100, "MetisCraft three-plane mark, monochrome", monogram(INK))
    adaptive = monogram().replace('fill="#087f78"', 'class="brand-fill"')
    save(
        "mark",
        100,
        100,
        "MetisCraft three-plane mark",
        "<style>.brand-fill { fill:#087f78 } "
        "@media(prefers-color-scheme:dark) { .brand-fill { fill:#cbd5df } }</style>" + adaptive,
    )
    for mode, bg, ink, muted, accent in (
        ("light", "#f6faf9", INK, "#546b6d", TEAL),
        ("dark", "#132527", "#edf5f3", "#b0c9c4", "#63d1bc"),
    ):
        mark_color = accent if mode == "light" else "#cbd5df"
        save(
            f"wordmark-{mode}",
            610,
            112,
            "MetisCraft",
            f'<g transform="translate(0 0) scale(1.05)">{monogram(mark_color)}</g>'
            + text(117, 82, "MetisCraft", 80, ink, 700, letter_spacing="-2"),
        )
        body = text(64, 68, "A TOOLKIT FOR JEVLIKE MODELS", 16, accent, 650, letter_spacing="2")
        body += text(58, 186, "MetisCraft", 104, ink, 700, letter_spacing="-4")
        body += text(64, 252, "Train. Calibrate. Decide.", 30, ink, 500)
        body += f'<path d="M64 294 H1136" stroke="{accent}" stroke-opacity=".2"/>'
        body += text(64, 348, "Reproduce open recipes. Build your own models.", 23, muted)
        save(f"hero-{mode}", 1200, 400, "MetisCraft — Train. Calibrate. Decide.", body, bg)
    body = f'<g transform="translate(80 64) scale(1.15)">{monogram()}</g>'
    body += text(214, 158, "MetisCraft", 88, INK, 700, letter_spacing="-3")
    body += text(92, 240, "Train. Calibrate. Decide.", 42, INK, 600)
    body += text(94, 310, "A toolkit for JevLike decision models", 28, "#546b6d")
    body += '<path d="M94 378 H1106" stroke="#c5dbd6"/>'
    body += text(94, 447, "Typed outputs  /  Composable readouts  /  Traceable runs", 25, TEAL)
    body += text(94, 553, "github.com/Shichaoxx/MetisCraft", 21, "#546b6d")
    save("social-preview", 1200, 630, "MetisCraft repository preview", body, "#f6faf9")

    body = text(32, 46, "One recipe, from data to decisions", 30, INK, 650)
    stages = [
        ("Data", "Validate splits"),
        ("Train", "Fit a readout"),
        ("Calibrate", "Fit temperature"),
        ("Evaluate", "Compare results"),
        ("Serve", "Load & answer"),
    ]
    for i, (name, detail) in enumerate(stages):
        x = 32 + i * 235
        body += f'<rect x="{x}" y="88" width="202" height="104" rx="12" '
        body += 'fill="#edf6f3" stroke="#b6d6cd"/>'
        body += text(x + 17, 127, name, 25, TEAL, 650)
        body += text(x + 17, 161, detail, 18, "#435e5b")
        if i < 4:
            body += f'<path d="M{x + 209} 140 H{x + 227} m-5 -5 l5 5 -5 5" '
            body += 'fill="none" stroke="#698883" stroke-width="2"/>'
    body += text(32, 242, "Recorded along the way", 16, TEAL, 650)
    body += text(270, 242, "Configuration · data version · model identity · metrics", 16)
    save(
        "pipeline",
        1200,
        276,
        "Data, training, calibration, evaluation and serving",
        body,
        "#ffffff",
    )
    body = text(32, 48, "From data to decisions", 28, INK, 650)
    for i, (name, detail) in enumerate(stages):
        y = 82 + i * 120
        body += f'<rect x="32" y="{y}" width="416" height="92" rx="12" '
        body += 'fill="#edf6f3" stroke="#b6d6cd"/>'
        body += text(52, y + 37, name, 28, TEAL, 650)
        body += text(52, y + 68, detail, 21, "#435e5b")
        if i < 4:
            body += f'<path d="M240 {y + 98} v16 m-5 -5 l5 5 5 -5" '
            body += 'fill="none" stroke="#698883" stroke-width="2"/>'
    body += text(32, 704, "Configuration · data · model · metrics", 20, TEAL)
    save("pipeline-mobile", 480, 738, "Data to decisions, vertical overview", body, "#ffffff")

    body = text(32, 46, "Four ways to ask a decision model", 30, INK, 650)
    cards = [
        ("Yes / no", "noul", "Does this need a human?", "P(yes) = 0.85"),
        ("Choice", "choice", "Which team should handle it?", "billing: 0.70 · support: 0.30"),
        ("Ordered score", "score", "How urgent is this request?", "P(level 1), …, P(level 5)"),
        ("Multiple labels", "multi_label", "Which topics are present?", "P(payments), P(cards), …"),
    ]
    for i, (name, kind, query, output) in enumerate(cards):
        x, y = 32 + (i % 2) * 590, 80 + (i // 2) * 176
        body += f'<rect x="{x}" y="{y}" width="558" height="152" rx="12" '
        body += 'fill="#f6faf9" stroke="#d2e3df"/>'
        body += text(x + 22, y + 35, name, 23, TEAL, 650)
        body += text(x + 350, y + 34, kind, 15, "#6a7577")
        body += text(x + 22, y + 77, query, 20)
        body += text(x + 22, y + 119, output, 20, "#76571f", 500)
    body += text(
        32,
        456,
        "Illustrative outputs, not benchmark results. Each type has its own probability semantics.",
        16,
        "#546b6d",
    )
    save("judgments", 1200, 486, "Four judgment types, with illustrative outputs", body, "#ffffff")
    body = text(28, 44, "Four types of judgment", 28, INK, 650)
    for i, (name, kind, query, output) in enumerate(cards):
        y = 76 + i * 186
        body += f'<rect x="24" y="{y}" width="432" height="166" rx="12" '
        body += 'fill="#f6faf9" stroke="#d2e3df"/>'
        body += text(44, y + 36, name, 25, TEAL, 650)
        body += text(44, y + 65, kind, 17, "#6a7577")
        body += text(44, y + 103, query, 21)
        body += text(44, y + 139, output, 21, "#76571f", 500)
    body += text(28, 846, "Illustrative outputs, not benchmark results.", 18, "#546b6d")
    body += text(28, 875, "Probability semantics differ by type.", 18, "#546b6d")
    save("judgments-mobile", 480, 904, "Four judgment types, vertical examples", body, "#ffffff")

    # Both languages share geometry and typography; only editorial labels differ.
    labels_zh = {
        "One recipe, from data to decisions": "一份配方，从数据到判断",
        "Data, training, calibration, evaluation and serving": "数据、训练、校准、评测与服务",
        "Data to decisions, vertical overview": "从数据到判断：纵向流程",
        "From data to decisions": "从数据到判断",
        "Recorded along the way": "全流程记录",
        "Configuration · data version · model identity · metrics": (
            "配置 · 数据版本 · 模型身份 · 指标"
        ),
        "Configuration · data · model · metrics": "配置 · 数据 · 模型 · 指标",
        "Validate splits": "校验数据划分",
        "Fit a readout": "训练读出",
        "Fit temperature": "拟合温度",
        "Compare results": "比较结果",
        "Load & answer": "加载与判断",
        "Data": "数据",
        "Train": "训练",
        "Calibrate": "校准",
        "Evaluate": "评测",
        "Serve": "服务",
        "Four judgment types, with illustrative outputs": "四种判断类型及构造示例",
        "Four judgment types, vertical examples": "四种判断类型：纵向示例",
        "Four ways to ask a decision model": "向判断模型提出四类问题",
        "Four types of judgment": "四种判断类型",
        "Yes / no": "是非判断",
        "Choice": "选择判断",
        "Ordered score": "等级判断",
        "Multiple labels": "多标签判断",
        "Does this need a human?": "需要人工处理吗？",
        "Which team should handle it?": "应该交给哪个团队？",
        "How urgent is this request?": "请求的紧急程度是几级？",
        "Which topics are present?": "包含哪些主题？",
        (
            "Illustrative outputs, not benchmark results. "
            "Each type has its own probability semantics."
        ): "构造输出示例，并非评测结果。不同类型采用各自的概率语义。",
        "Illustrative outputs, not benchmark results.": "构造输出示例，并非评测结果。",
        "Probability semantics differ by type.": "概率语义因类型而异。",
        "P(yes) = 0.85": "P(是) = 0.85",
        "billing: 0.70 · support: 0.30": "账单: 0.70 · 支持: 0.30",
        "P(level 1), …, P(level 5)": "P(等级 1), …, P(等级 5)",
        "P(payments), P(cards), …": "P(支付), P(卡片), …",
    }
    for name in ("pipeline", "pipeline-mobile", "judgments", "judgments-mobile"):
        svg = (OUT / f"{name}.svg").read_text(encoding="utf-8")
        for source, translated in sorted(labels_zh.items(), key=lambda item: -len(item[0])):
            svg = svg.replace(f">{escape(source)}<", f">{escape(translated)}<")
        svg = svg.replace('"DejaVu Sans", Arial', '"Noto Sans CJK SC", "WenQuanYi Zen Hei", Arial')
        (OUT / f"{name}-zh.svg").write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
