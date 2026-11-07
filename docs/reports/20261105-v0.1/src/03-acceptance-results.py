"""Figure 3: headline metric against the prior, and ECE before and after temperature (test split).

Reads src/data/acceptance.json; run through the visualization runner so vizkit's font and the
sea-salt coral palette are importable:  python3 <shared>/run.py python src/03-acceptance-results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scientific_colors import series_styles  # noqa: E402
from vizkit.style import pick_font  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
RECIPES = [
    ("boolq/qwen3-0.6b-base_answer-token", "BoolQ ①"),
    ("banking77/qwen3-0.6b-base_answer-token", "Banking77 ①"),
    ("banking77/qwen3-0.6b-base_candidate-scalar", "Banking77 ③"),
    ("clinc150/qwen3-0.6b-base_candidate-scalar", "CLINC150 ③"),
    ("sst5/qwen3-0.6b-base_answer-token", "SST-5 ①"),
    ("go_emotions/qwen3-0.6b-base_candidate-scalar", "GoEmotions ③"),
    ("banking77_support/qwen3-0.6b-base_candidate-scalar", "客服分流 ③"),
]
KIND = {"noul": "是非", "choice": "选择", "score": "等级", "multi_label": "多标签"}
HEADLINE = {
    "noul": ("accuracy", "准确率"),
    "choice": ("accuracy", "准确率"),
    "score": ("accuracy", "准确率"),
    "multi_label": ("macro_f1", "macro-F1"),
}
INK, MUTED, LINE = "#273444", "#657080", "#DADDE3"


def rows() -> list[dict]:
    results = json.loads((HERE / "data" / "acceptance.json").read_text(encoding="utf-8"))
    out = []
    for key, name in RECIPES:
        r = results.get(key)
        if r is None:
            continue
        test = r["metrics"]["test"]
        for kind in [k for k in KIND if k in test["model"]]:
            metric, label = HEADLINE[kind]
            tuned = test.get("model + temperature", {}).get(kind, test["model"][kind])
            tag = "（修复前）" if r.get("offline_prior_shift") else ""
            out.append(
                {
                    "label": f"{name}{tag} · {KIND[kind]}",
                    "metric": label,
                    "model": tuned[metric],
                    "prior": test["prior"][kind][metric],
                    "ece_before": test["model"][kind]["ece"],
                    "ece_after": tuned["ece"],
                }
            )
    return out


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": pick_font(),
            "font.size": 10,
            "axes.edgecolor": LINE,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "svg.fonttype": "none",
        }
    )
    blue, coral, green, _ = series_styles("sea-salt-coral", count=4)
    data = rows()
    n = len(data)
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(10.5, 1.0 + 0.48 * n),
        sharey=True,
        gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.08},
    )
    ys = list(range(n))[::-1]
    for y, d in zip(ys, data, strict=True):
        ax1.plot([d["prior"], d["model"]], [y, y], color=LINE, lw=2, zorder=1)
        ax1.scatter(d["prior"], y, s=46, facecolor="white", edgecolor=MUTED, lw=1.3, zorder=2)
        ax1.scatter(d["model"], y, s=54, color=blue["stroke"], zorder=3)
        ax1.annotate(
            f"{d['model']:.3f}",
            (d["model"], y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=blue["stroke"],
        )
        ax2.plot([d["ece_after"], d["ece_before"]], [y, y], color=LINE, lw=2, zorder=1)
        ax2.scatter(d["ece_before"], y, s=50, marker="s", color=coral["stroke"], zorder=2)
        ax2.scatter(d["ece_after"], y, s=54, color=green["stroke"], zorder=3)
        ax2.annotate(
            f"{d['ece_before']:.3f} → {d['ece_after']:.3f}",
            (max(d["ece_before"], d["ece_after"]), y),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK,
        )
    ax1.set_yticks(ys, [f"{d['label']}（{d['metric']}）" for d in data])
    ax1.set_xlim(0, 1.08)
    ax1.set_title("主指标：模型 + 温度（实心）对先验（空心）", fontsize=10.5, color=INK, loc="left")
    top = max(max(d["ece_before"], d["ece_after"]) for d in data)
    ax2.set_xlim(0, top * 1.6 + 0.01)
    ax2.set_title("ECE：校准前（方块）→ 校准后（圆点）", fontsize=10.5, color=INK, loc="left")
    for ax in (ax1, ax2):
        ax.grid(axis="x", color=LINE, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        ax.set_ylim(-0.7, n - 0.3)
    fig.text(
        0.5,
        0.005,
        "MetisCraft v0.1：test 划分上的验收结果@Shichao",
        ha="center",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(
        left=0.34, right=0.97, top=1 - 0.42 / (1.0 + 0.48 * n), bottom=0.5 / (1.0 + 0.48 * n)
    )
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"03-acceptance-results.{ext}", dpi=200)
    print(OUT / "03-acceptance-results.png", f"{n} rows")


if __name__ == "__main__":
    main()
