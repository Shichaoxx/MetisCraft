"""Render MetisCraft's editable SVG identity and explanatory figures (stdlib only)."""

from dataclasses import dataclass
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "figures"
INK, TEAL = "#243438", "#087f78"


def text(x, y, value, size=20, color=INK, weight=400, css_class=None, **attrs):
    if css_class:
        attrs["class"] = css_class
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

    for suffix, palette in PALETTES.items():
        for lang in ("en", "zh"):
            for mobile in (False, True):
                variant = ("-mobile" if mobile else "") + ("-zh" if lang == "zh" else "") + suffix
                draw_pipeline("pipeline" + variant, palette, lang, mobile)
                draw_judgments("judgments" + variant, palette, lang, mobile)


@dataclass(frozen=True)
class Palette:
    paper: str
    ink: str
    muted: str
    rule: str
    blue: str
    clay: str
    sage: str


PALETTES = {
    "": Palette("#FAF9F5", "#24231F", "#625F56", "#C4BEB1", "#375A75", "#924428", "#4F653A"),
    ".dark": Palette("#1C1C1A", "#FAF9F5", "#BCB9AE", "#655F54", "#9BC1DF", "#ECA388", "#B0C596"),
}

# A distinct glyph and a real run output identify each stage in grayscale, too.
STAGES = [
    ("Data", "数据", "Validate splits", "校验数据划分", "data.json"),
    ("Train", "训练", "Fit a readout", "训练读出", "artifact/"),
    ("Calibrate", "校准", "Fit temperature", "拟合温度", "calibration/"),
    ("Evaluate", "评测", "Compare results", "比较结果", "eval/metrics.json"),
    ("Serve", "服务", "Load and answer", "加载并作答", "/v1/systemone"),
]
GLYPHS = [
    "M-13 -11 H13 V11 H-13 Z M-13 -4 H13 M-13 3 H13 M-4 -11 V11",
    "M0 -13 L15 -6 0 1 -15 -6 Z M-15 0 L0 7 15 0 M-15 6 L0 13 15 6",
    "M-13 -8 H-7 M-1 -8 H13 M-7 -11 H-1 V-5 H-7 Z M-13 7 H3 M9 7 H13 M3 4 H9 V10 H3 Z",
    "M-13 -12 V12 H14 M-7 7 V1 H-3 V7 Z M0 7 V-4 H4 V7 Z M7 7 V-10 H11 V7 Z",
    "M-13 -10 H2 V10 H-13 Z M-6 0 H15 M10 -5 L15 0 10 5",
]
SERIF = 'Georgia, "Songti SC", "Noto Serif CJK SC", serif'
MONO = 'ui-monospace, "SFMono-Regular", Consolas, monospace'


def draw_pipeline(name, p, lang, mobile):
    zh = lang == "zh"
    width, height = (360, 504) if mobile else (950, 228)
    title = "一份配方，从数据到判断" if zh else "One recipe, from data to decisions"
    if mobile:
        title = "从数据到判断" if zh else "From data to decisions"
    body = f"<style>text.heading {{font-family: {SERIF}}} text.file {{font-family: {MONO}}}</style>"
    body += text(22, 40, title, 23 if mobile else 28, p.ink, 500, css_class="heading")
    colors = (p.blue, p.clay, p.sage, p.sage, p.clay)
    for i, (en, cn, detail, detail_zh, artifact) in enumerate(STAGES):
        x, y = (46, 90 + 90 * i) if mobile else (58 + 184 * i, 100)
        if i < len(STAGES) - 1:
            if mobile:
                arrow = f"M{x} {y + 24} v41 m-4 -5 4 5 4 -5"
            else:
                arrow = f"M{x + 29} {y} h126 m-6 -4 6 4 -6 4"
            body += f'<path d="{arrow}" fill="none" stroke="{p.muted}" stroke-width="1.5"/>'
        body += (
            f'<g transform="translate({x} {y})" fill="none" stroke="{colors[i]}" '
            f'stroke-linecap="round" stroke-linejoin="round">'
            f'<circle r="21" stroke-width="1"/><path d="{GLYPHS[i]}" stroke-width="1.8"/></g>'
        )
        left = x + 43 if mobile else x - 26
        top = y - 5 if mobile else y + 55
        body += text(left, top, cn if zh else en, 20, p.ink, 600)
        body += text(left, top + 25, detail_zh if zh else detail, 17, p.muted)
        body += text(left, top + 47, artifact, 15, colors[i], 500, css_class="file")
    save(name, width, height, title, body, p.paper)


JUDGMENTS = [
    (
        "Yes / no",
        "是非判断",
        "noul",
        "Does this need a human?",
        "需要人工处理吗？",
        "P(yes) = 0.85",
        "P(是) = 0.85",
    ),
    (
        "Choice",
        "选择判断",
        "choice",
        "Which team should handle it?",
        "应该交给哪个团队？",
        "billing 0.70 · support 0.30",
        "账单 0.70 · 支持 0.30",
    ),
    (
        "Ordered score",
        "等级判断",
        "score",
        "How urgent is this request?",
        "请求的紧急程度是几级？",
        "P(level 1), …, P(level 5)",
        "P(等级 1), …, P(等级 5)",
    ),
    (
        "Multiple labels",
        "多标签判断",
        "multi_label",
        "Which topics are present?",
        "包含哪些主题？",
        "P(payments), P(cards), …",
        "P(支付), P(卡片), …",
    ),
]


def draw_judgments(name, p, lang, mobile):
    zh = lang == "zh"
    width, height = (360, 556) if mobile else (950, 433)
    title = "向判断模型提出四类问题" if zh else "Four types of judgment"
    body = f"<style>text.heading {{font-family: {SERIF}}} text.kind {{font-family: {MONO}}}</style>"
    body += text(22, 40, title, 23 if mobile else 28, p.ink, 500, css_class="heading")
    if not mobile:
        for x, en, cn in [
            (24, "Type", "判断类型"),
            (262, "Question", "问题"),
            (595, "Example output", "构造输出示例"),
        ]:
            body += text(x, 84, cn if zh else en, 16, p.muted, 600)
    for i, (en, cn, kind, query, query_zh, output, output_zh) in enumerate(JUDGMENTS):
        y = 65 + i * 111 if mobile else 99 + i * 72
        body += f'<path d="M22 {y} H{width - 22}" stroke="{p.rule}"/>'
        body += text(24, y + 29, cn if zh else en, 20, p.ink, 600)
        body += text(
            width - 24 if mobile else 24,
            y + (28 if mobile else 54),
            kind,
            15,
            p.clay,
            css_class="kind",
            text_anchor="end" if mobile else "start",
        )
        body += text(
            24 if mobile else 262,
            y + (59 if mobile else 41),
            query_zh if zh else query,
            17,
            p.muted,
        )
        body += text(
            24 if mobile else 595,
            y + (86 if mobile else 41),
            output_zh if zh else output,
            18,
            p.blue,
            550,
        )
    note = "示例用于说明各类型的概率含义。" if zh else "Illustrative probabilities for each type."
    body += text(24, height - 20, note, 16, p.muted)
    save(name, width, height, title, body, p.paper)


if __name__ == "__main__":
    main()
