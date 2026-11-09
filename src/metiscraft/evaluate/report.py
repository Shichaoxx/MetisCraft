"""Markdown reports. Every number in a report comes from the run directory it lives in."""

from __future__ import annotations

from typing import Any

from metiscraft.evaluate.metrics import ECE_NOTE

METRIC_ORDER = [
    "n",
    "n_hard",
    "n_soft",
    "n_hard_unknown",
    "accuracy",
    "macro_f1",
    "micro_f1",
    "exact_match",
    "auc",
    "mae_expected",
    "nll",
    "brier",
    "brier_soft",
    "nll_soft",
    "ece",
]


def _fmt(name: str, value: Any) -> str:
    if value is None:
        return "-"
    if name in {"n", "n_hard", "n_soft", "n_hard_unknown"}:
        return str(int(value))
    if isinstance(value, float) and value != value:  # nan
        return "-"
    return f"{value:.4f}"


def render_report(header: dict[str, Any], results: dict[str, dict[str, dict[str, Any]]]) -> str:
    lines = [f"# MetisCraft report: {header['task']}", ""]
    for key in (
        "run",
        "dataset",
        "base",
        "readout",
        "finetune",
        "recipe_sha256",
        "resolved_config_sha256",
        "git",
        "calibration",
    ):
        if header.get(key) is not None:
            lines.append(f"- {key}: {header[key]}")
    lines += [f"- {ECE_NOTE}", ""]
    for split, methods in results.items():
        lines += [f"## {split}", ""]
        kinds = sorted({k for m in methods.values() for k in m})
        for kind in kinds:
            cols = [c for c in METRIC_ORDER if any(c in methods[m].get(kind, {}) for m in methods)]
            lines.append(f"**{kind}**")
            lines.append("")
            lines.append("| method | " + " | ".join(cols) + " |")
            lines.append("|---|" + "---|" * len(cols))
            for method, by_kind in methods.items():
                m = by_kind.get(kind)
                if m is None:
                    continue
                lines.append(f"| {method} | " + " | ".join(_fmt(c, m.get(c)) for c in cols) + " |")
            lines.append("")
            for method, by_kind in methods.items():
                reasons = by_kind.get(kind, {}).get("undefined_reasons", {})
                if reasons:
                    detail = "; ".join(f"{key}: {value}" for key, value in reasons.items())
                    lines.append(f"- {method}: {detail}")
            lines.append("")
    return "\n".join(lines)
