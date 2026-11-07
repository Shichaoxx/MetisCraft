"""Tiny stand-ins for upstream raw files, shaped like the real repositories."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def write_parquet(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path.as_posix()


def write_card(raw: Path, dataset_info) -> None:
    front = yaml.safe_dump({"dataset_info": dataset_info}, sort_keys=False)
    (raw / "README.md").write_text(f"---\n{front}---\n# card\n", encoding="utf-8")


def class_feature(name: str, names: list[str], sequence: bool = False) -> dict:
    label = {"class_label": {"names": {str(i): n for i, n in enumerate(names)}}}
    return {"name": name, "sequence": label} if sequence else {"name": name, "dtype": label}


def shard(raw: Path, subdir: str, split: str, rows: list[dict]) -> None:
    write_parquet(raw / subdir / f"{split}-00000-of-00001.parquet", rows)


def boolq(raw: Path) -> None:
    (raw / "README.md").write_text("# boolq\n", encoding="utf-8")
    rows = [
        {"question": f"is item {i} red", "answer": i % 2 == 0, "passage": f"Passage {i}."}
        for i in range(60)
    ]
    shard(raw, "data", "train", rows)
    shard(
        raw,
        "data",
        "validation",
        rows[:5] + [{"question": "q new", "answer": True, "passage": "p"}],
    )


def banking77(raw: Path) -> None:
    names = ["activate_my_card", "card_arrival", "lost_or_stolen_card"]
    write_card(
        raw, {"features": [{"name": "text", "dtype": "string"}, class_feature("label", names)]}
    )
    train = [{"text": f"message number {i}", "label": i % 3} for i in range(80)]
    test = [{"text": "message number 1", "label": 1}, {"text": "fresh message", "label": 2}]
    shard(raw, "data", "train", train)
    shard(raw, "data", "test", test)


def clinc150(raw: Path) -> None:
    names = ["oos", "weather", "timer"]
    info = [
        {"config_name": "small", "features": [class_feature("intent", ["x"])]},
        {
            "config_name": "plus",
            "features": [{"name": "text", "dtype": "string"}, class_feature("intent", names)],
        },
    ]
    write_card(raw, info)
    for split, n in (("train", 30), ("validation", 10), ("test", 10)):
        shard(
            raw,
            "plus",
            split,
            [{"text": f"{split} utterance {i}", "intent": i % 3} for i in range(n)],
        )


def sst5(raw: Path) -> None:
    (raw / "README.md").write_text("sst5\n", encoding="utf-8")
    for split, n in (("train", 20), ("dev", 20), ("test", 6)):
        lines = [
            json.dumps({"text": f"{split} review {i}", "label": i % 5, "label_text": "x"})
            for i in range(n)
        ]
        (raw / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def go_emotions(raw: Path) -> None:
    names = ["admiration", "anger", "neutral"]
    info = [
        {"config_name": "raw", "features": [{"name": "text", "dtype": "string"}]},
        {
            "config_name": "simplified",
            "features": [
                {"name": "text", "dtype": "string"},
                class_feature("labels", names, sequence=True),
                {"name": "id", "dtype": "string"},
            ],
        },
    ]
    write_card(raw, info)
    for split, n in (("train", 20), ("validation", 8), ("test", 8)):
        rows = [
            {
                "text": f"{split} comment {i}",
                "labels": [i % 3] if i % 4 else [0, 1],
                "id": f"{split}{i}",
            }
            for i in range(n)
        ]
        shard(raw, "simplified", split, rows)


def typed_decisions(raw: Path) -> None:
    (raw / "README.md").write_text("ll\n", encoding="utf-8")
    questions = {
        "action": {
            "type": "choice",
            "instructions": "Next step?",
            "criteria": {"reply": "Answer it.", "escalate": "Hand off."},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent?",
            "criteria": ["low", "medium", "high"],
        },
        "needs_human": {"type": "noul", "instructions": "Needs a human?"},
    }

    def row(i, split):
        gold = {
            "action": {
                "type": "choice",
                "label": "reply",
                "probabilities": {"reply": 0.7, "escalate": 0.3},
            },
            "urgency": {
                "type": "score",
                "label": "2",
                "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
            },
            "needs_human": {"type": "noul", "label": "false", "probability_true": 0.2},
        }
        return {
            "id": f"{split}-{i}",
            "workflow": "customer_service",
            "split": split,
            "state": json.dumps({"msg": f"{split} case {i}"}),
            "questions": json.dumps(questions),
            "gold": json.dumps(gold),
            "factors": "{}",
            "label_agreement": "{}",
            "n_questions": 3,
        }

    shard(raw, "all", "train", [row(i, "train") for i in range(6)])
    shard(raw, "all", "test", [row(i, "test") for i in range(4)])


def _flat_rows(split: str, n: int, described: bool, noul_pair: bool) -> list[dict]:
    rows = []
    for i in range(n):
        gid = f"{split}:{i // 2}"
        if i % 3 == 0:
            options = ["billing: Charges.", "tech: Bugs."] if described else ["billing", "tech"]
            row = {"kind": "choice", "options": options, "target": [0.0, 1.0]}
        elif i % 3 == 1:
            row = {"kind": "score", "options": ["low", "high"], "target": [0.5, 0.5]}
        else:
            row = {
                "kind": "noul",
                "options": ["no", "yes"] if noul_pair else [],
                "target": [0.0, 1.0] if noul_pair else [1.0],
            }
        row.update(
            {
                "id": f"{split}-{i}",
                "group_id": gid,
                "question": f"Question {i}?",
                "source": "src",
                "split": split,
            }
        )
        rows.append(row)
    return rows


def open_jev(raw: Path) -> None:
    (raw / "README.md").write_text("oj\n", encoding="utf-8")
    (raw / "LICENSE-DATA").write_text("CC0\n", encoding="utf-8")
    sub = "data/release-v2-redistributable"
    for split, n in (("train", 12), ("calibration", 4), ("validation", 4), ("test", 4), ("ood", 4)):
        rows = _flat_rows(split, n, described=True, noul_pair=True)
        for i, r in enumerate(rows):
            r.update(
                {
                    "state_json": json.dumps(f"{split} state {i // 2}"),
                    "metadata_json": "{}",
                    "record_json": "{}",
                    "original_line_number": i,
                }
            )
        shard(raw, sub, split, rows)


def tasksource_jev(raw: Path) -> None:
    (raw / "README.md").write_text("ts\n", encoding="utf-8")
    for split, n in (("train", 12), ("validation", 6), ("test", 6)):
        rows = _flat_rows(split, n, described=False, noul_pair=False)
        for i, r in enumerate(rows):
            r.update(
                {
                    "state": f"{split} state {i // 2}",
                    "variant": "direct" if i % 4 else "paraphrase",
                    "source_index": f"src:{split}:{i}",
                    "example_id": f"e{i}",
                    "license": "mit",
                    "license_use": "commercial" if i % 5 else "non-commercial",
                }
            )
        shard(raw, "data", split, rows)


def multi_nli(raw: Path) -> None:
    names = ["entailment", "neutral", "contradiction"]
    write_card(
        raw, {"features": [{"name": "premise", "dtype": "string"}, class_feature("label", names)]}
    )
    for split, n in (("train", 30), ("validation_matched", 6), ("validation_mismatched", 6)):
        rows = [
            {
                "premise": f"{split} premise {i}",
                "hypothesis": f"hyp {i}",
                "label": i % 3 if i else -1,
                "genre": "fiction",
            }
            for i in range(n)
        ]
        shard(raw, "data", split, rows)


def banking77_support(raw: Path) -> None:
    """Banking77-shaped raw files whose card lists every intent the scenario mapping names."""
    from metiscraft.data.registry import get_entry

    questions = get_entry("banking77_support").params["questions"]
    names = sorted({s for sources in questions["queue"]["map"].values() for s in sources})
    write_card(
        raw, {"features": [{"name": "text", "dtype": "string"}, class_feature("label", names)]}
    )
    used = [names.index(n) for n in ("activate_my_card", "card_arrival", "lost_or_stolen_card")]
    train = [{"text": f"message number {i}", "label": used[i % 3]} for i in range(60)]
    test = [{"text": f"another message {i}", "label": used[i % 3]} for i in range(9)]
    shard(raw, "data", "train", train)
    shard(raw, "data", "test", test)


def tev1(raw: Path) -> None:
    """Tev1-shaped build outputs: lettered options in a shuffled order, SST-5 levels in order,
    BoolQ as two options and a policy row with an object state."""

    def row(source: str, split: str, i: int) -> dict:
        kind, keys = {
            "mnli": ("choice", ["neutral", "contradicted", "entailed"]),
            "boolq": ("noul", ["no", "yes"]),
            "sst5": ("score", ["0", "1", "2", "3", "4"]),
            "policy": ("choice", ["unknown", "eligible", "ineligible"]),
        }[source]
        gold = keys[i % len(keys)]
        state = f"{split} passage {i} for {source}"
        if source == "policy":
            record = {"order_value_usd": i, "item_unopened": None}
            state = {"record": record, "policy": f"Eligible if unopened ({split})."}
        options = [
            {"label": "ABCDE"[j], "key": k, "description": f"{k}."} for j, k in enumerate(keys)
        ]
        return {
            "id": f"{source}:{split}:{i}",
            "source": source,
            "split": split,
            "kind": kind,
            "group_id": f"{source}:{split}:{i // 2}",
            "state": state,
            "question": f"Question about {source}?",
            "options": options,
            "answer": "ABCDE"[keys.index(gold)],
            "answer_key": gold,
            "provenance": {"generator": "fixture"},
            "token_count": 40 + i,
        }

    layout = {
        ("new-v1", "train"): ["mnli", "boolq", "sst5", "policy"],
        ("new-v1", "dev"): ["mnli", "boolq", "sst5", "policy"],
        ("v1", "test"): ["mnli", "boolq", "sst5"],
        ("v1", "policy_transfer"): ["policy"],
    }
    for (build, split), sources in layout.items():
        n = 12 if split == "train" else 4
        path = raw / "data" / build / "records" / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [row(source, split, i) for source in sources for i in range(n)]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


BUILDERS = {
    "boolq": boolq,
    "banking77": banking77,
    "clinc150": clinc150,
    "sst5": sst5,
    "go_emotions": go_emotions,
    "typed_decisions": typed_decisions,
    "open_jev": open_jev,
    "tasksource_jev": tasksource_jev,
    "multi_nli": multi_nli,
    "banking77_support": banking77_support,
    "tev1": tev1,
}


def fake_fetcher(entry, data_root, raw_dir):
    BUILDERS[entry.name](raw_dir)
    return sorted(p.relative_to(raw_dir).as_posix() for p in raw_dir.rglob("*") if p.is_file())
