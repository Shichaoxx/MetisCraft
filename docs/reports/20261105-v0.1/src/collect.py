"""Collect the numbers and real snippets the v0.1 report shows into src/data/.

Runs where the acceptance runs live. Run directories and prepared datasets are found through
MetisCraft's path config, so no machine paths end up in this file or in the collected data.
The runs to show are listed in src/data/runs.json as {"<dataset>/<recipe>": "<run id>"}.

    python collect.py [--api-example PATH]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from metiscraft.config import resolve_paths

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SCENARIO = "banking77_support/qwen3-0.6b-base_candidate-scalar"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def tree(root: Path, depth: int = 2, prefix: str = "") -> list[str]:
    """Names only, like `tree -L depth`; directories end with a slash."""
    lines = []
    entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name))
    for i, entry in enumerate(entries):
        last = i == len(entries) - 1
        lines.append(f"{prefix}{'└── ' if last else '├── '}{entry.name}{'/' if entry.is_dir() else ''}")
        if entry.is_dir() and depth > 1:
            lines += tree(entry, depth - 1, prefix + ("    " if last else "│   "))
    return lines


def null_breakdown(run: Path, split: str = "test") -> dict | None:
    """Choice questions with a null option: in-scope accuracy, null recall and null precision (argmax)."""
    path = run / "eval" / f"{split}.predictions.jsonl"
    c = dict(ins=0, ins_ok=0, null=0, null_ok=0, pred_null=0, pred_null_ok=0)
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("kind") != "choice" or "__null__" not in r["choice_ids"]:
            continue
        gold = r["choice_ids"][max(range(len(r["target"])), key=r["target"].__getitem__)]
        pred = r["choice_ids"][max(range(len(r["scores"])), key=r["scores"].__getitem__)]
        key = "null" if gold == "__null__" else "ins"
        c[key] += 1
        c[key + "_ok"] += pred == gold
        if pred == "__null__":
            c["pred_null"] += 1
            c["pred_null_ok"] += gold == "__null__"
    return c if c["null"] else None


def run_summary(run: Path) -> dict:
    log = [json.loads(x) for x in (run / "train_log.jsonl").read_text().splitlines()]
    env = read_json(run / "env.json")
    artifact = read_json(run / "artifact" / "artifact.json")
    return {
        "run": run.name,
        "commit": read_json(run / "git.json")["commit"][:7],
        "dirty": read_json(run / "git.json").get("dirty", False),
        "tracked_changes": (run / "git.diff").is_file(),
        "data_version": read_json(run / "data.json").get("version"),
        "device": (env.get("devices") or ["CPU"])[0].replace("NVIDIA GeForce ", ""),
        "versions": {k: env.get(k) for k in ("torch", "transformers", "peft")},
        "steps": log[-1]["step"],
        "train_minutes": round(log[-1]["elapsed_s"] / 60, 1),
        "loss_first": log[0]["loss"],
        "loss_last": log[-1]["loss"],
        "temperatures": read_json(run / "calibration" / "calibration.json")["temperatures"],
        "readout": artifact["readout"],
        "finetune": artifact["finetune"],
        "commercial_use": artifact["commercial_use"],
        "dataset": artifact["dataset"],
        "metrics": read_json(run / "eval" / "metrics.json"),
        "null_breakdown": null_breakdown(run),
    }


def main() -> None:
    paths = resolve_paths()
    runs = read_json(DATA / "runs.json")
    results = {}
    for recipe, run_id in runs.items():
        results[recipe] = run_summary(paths.runs_root / recipe.split("/")[0] / run_id)
    (DATA / "acceptance.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")

    snippets: dict = {}
    if SCENARIO in runs:
        run = paths.runs_root / SCENARIO.split("/")[0] / runs[SCENARIO]
        snippets["run_tree"] = [run.name + "/"] + tree(run, depth=2)
        snippets["notice"] = (run / "artifact" / "NOTICE").read_text(encoding="utf-8")
        snippets["recipe"] = (HERE.parents[3] / "recipes" / f"{SCENARIO}.yaml").read_text(encoding="utf-8")
        version = results[SCENARIO]["data_version"]
        dataset_dir = paths.data_root / SCENARIO.split("/")[0] / version
        snippets["dataset_tree"] = [f"{SCENARIO.split('/')[0]}/{version}/"] + tree(dataset_dir, depth=2)
        first = (dataset_dir / "derived" / "train.jsonl").open(encoding="utf-8")
        snippets["records"] = [json.loads(next(first)) for _ in range(2)]
    if "--api-example" in sys.argv:
        snippets["api_example"] = read_json(Path(sys.argv[sys.argv.index("--api-example") + 1]))
    (DATA / "snippets.json").write_text(json.dumps(snippets, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(results)} runs; snippets: {sorted(snippets)}")


if __name__ == "__main__":
    main()
