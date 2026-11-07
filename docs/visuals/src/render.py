"""Render MetisCraft's editable SVG identity and explanatory figures (stdlib only)."""

from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "figures"
INK, TEAL, GOLD = "#243438", "#087f78", "#bd8c36"


def text(x, y, value, size=20, color=INK, weight=400, **attrs):
    extra = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
        f'font-weight="{weight}" {extra}>{escape(value)}</text>'
    )


def owl(x=0, y=0, scale=1, color=TEAL):
    return f'''<g transform="translate({x} {y}) scale({scale})" fill="none"
        stroke="{color}" stroke-width="5" stroke-linejoin="round" stroke-linecap="round">
      <path d="M18 25 L18 8 L38 19 Q50 14 62 19 L82 8 L82 53
               Q80 78 50 91 Q20 78 18 53 Z"/>
      <path d="M20 31 Q33 18 50 37 Q67 18 80 31"/>
      <circle cx="35" cy="44" r="13"/><circle cx="65" cy="44" r="13"/>
      <circle cx="35" cy="44" r="4" fill="{color}" stroke="none"/>
      <circle cx="65" cy="44" r="4" fill="{color}" stroke="none"/>
      <path d="M45 61 L50 68 L55 61" stroke="{GOLD}"/>
    </g>'''


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
    save("mark", 100, 100, "MetisCraft owl", owl())
    for mode, bg, ink, muted, accent in (
        ("light", "#f6faf9", INK, "#546b6d", TEAL),
        ("dark", "#132527", "#edf5f3", "#b0c9c4", "#63d1bc"),
    ):
        save(
            f"wordmark-{mode}",
            620,
            116,
            "MetisCraft",
            owl(4, 6, 1, accent) + text(124, 78, "MetisCraft", 62, ink, 650),
        )
        body = text(55, 65, "A TOOLKIT FOR TYPED DECISIONS", 16, accent, 650, letter_spacing="2")
        body += text(50, 160, "MetisCraft", 82, ink, 700, letter_spacing="-3")
        body += text(55, 218, "Train. Calibrate. Decide.", 32, ink, 500)
        body += text(55, 262, "Build decision models for your own tasks.", 21, muted)
        body += f'<path d="M55 308 H745" stroke="{accent}" stroke-opacity=".2"/>'
        body += text(55, 348, "Typed outputs", 17, muted)
        body += text(265, 348, "Composable readouts", 17, muted)
        body += text(553, 348, "Traceable runs", 17, muted)
        body += owl(868, 68, 2.6, accent)
        save(f"hero-{mode}", 1200, 400, "MetisCraft — Train. Calibrate. Decide.", body, bg)
    body = owl(76, 60, 1.4) + text(250, 160, "MetisCraft", 78, INK, 700)
    body += text(92, 290, "Train. Calibrate. Decide.", 55, INK, 600)
    body += text(94, 353, "A toolkit for typed decision models", 30, "#546b6d")
    body += '<path d="M94 410 H1106" stroke="#c5dbd6"/>'
    body += text(94, 473, "Your data  /  Your tasks  /  Traceable runs", 25, TEAL)
    body += text(94, 559, "github.com/Shichaoxx/MetisCraft", 21, "#546b6d")
    save("social-preview", 1200, 630, "MetisCraft repository preview", body, "#f6faf9")

    body = text(32, 46, "One recipe, from data to decisions", 30, INK, 650)
    stages = [
        ("Data", "Validate & split"),
        ("Train", "Choose a readout"),
        ("Calibrate", "Fit on held-out data"),
        ("Evaluate", "Measure & compare"),
        ("Serve", "Load & answer"),
    ]
    for i, (name, detail) in enumerate(stages):
        x = 32 + i * 235
        body += f'<rect x="{x}" y="88" width="202" height="104" rx="12" '
        body += 'fill="#edf6f3" stroke="#b6d6cd"/>'
        body += text(x + 17, 127, name, 25, TEAL, 650)
        body += text(x + 17, 161, detail, 14, "#435e5b")
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


if __name__ == "__main__":
    main()
