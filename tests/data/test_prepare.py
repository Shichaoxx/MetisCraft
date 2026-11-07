import json
import os

import pytest

from metiscraft.data.contract import Record
from metiscraft.data.dedup import eval_first_filter
from metiscraft.data.prepare import prepare_dataset
from metiscraft.data.registry import load_registry
from metiscraft.data.store import IntegrityError, verify_sha256sums, write_sha256sums
from tests.data.fixtures import BUILDERS, fake_fetcher


@pytest.fixture
def data_root(tmp_path):
    return tmp_path / "data"


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_prepare_every_converter(name, data_root, tmp_path):
    dv = prepare_dataset(name, data_root, fetcher=fake_fetcher, staging_parent=tmp_path)
    meta = dv.meta()
    assert meta["status"] == "ready" and meta["eval_frozen"] is True
    assert dv.verify() > 0
    ids = set()
    for role, info in meta["splits"].items():
        rows = dv.load(role)
        assert len(rows) == info["rows"]
        for row in rows:
            Record.model_validate(row)
            assert row["id"] not in ids, "a row appears in two splits"
            ids.add(row["id"])
    assert all(not os.access(p, os.W_OK) for p in dv.raw_dir.rglob("*") if p.is_file())
    # Preparing again verifies the existing version instead of rewriting it.
    assert prepare_dataset(name, data_root, fetcher=None).root == dv.root


def test_roles_follow_registry(data_root, tmp_path):
    def roles(name):
        return set(
            prepare_dataset(name, data_root, fetcher=fake_fetcher, staging_parent=tmp_path).meta()[
                "splits"
            ]
        )

    assert roles("banking77") == {"train", "calib", "dev", "test"}  # calib/dev carved from train
    assert roles("clinc150") == {"train", "calib", "dev", "test"}  # calib/dev halve validation
    assert roles("typed_decisions") == {"calib", "test"}  # teacher labels: never trained on
    assert roles("open_jev") == {"train", "calib", "dev", "test", "eval_only"}


def test_eval_first_dedup_in_pipeline(data_root, tmp_path):
    dv = prepare_dataset("banking77", data_root, fetcher=fake_fetcher, staging_parent=tmp_path)
    assert "message number 1" not in {r["input"]["context"]["state"] for r in dv.load("train")}
    assert (dv.manifests_dir / "dedup_report.json").is_file()
    assert dv.meta()["splits"]["test"]["rows"] == 2


def _row(rid, digest, group=None):
    row = {"id": rid, "provenance": {"input_hash": digest}}
    if group:
        row["group"] = {"id": group}
    return row


def test_eval_first_filter_rules():
    train = [
        _row("t1", "h-eval"),  # same input as an eval row
        _row("t2", "h2", group="g1"),  # same group as t3, whose input is in eval
        _row("t3", "h-eval2", group="g1"),
        _row("t4", "h4", group="g-eval"),  # group used by an eval row
        _row("t5", "h5"),
    ]
    evals = {
        "test": [_row("e1", "h-eval"), _row("e2", "h-eval2"), _row("e3", "hx", group="g-eval")]
    }
    kept, report = eval_first_filter(train, evals)
    assert [r["id"] for r in kept] == ["t5"]
    assert report["hits_same_input"] == 2 and report["hits_same_group"] == 1
    assert report["dropped"] == 4


def test_null_label_and_filters(data_root, tmp_path):
    clinc = prepare_dataset("clinc150", data_root, fetcher=fake_fetcher, staging_parent=tmp_path)
    rows = clinc.load("train")
    assert all(r["question"]["allow_null"] for r in rows)
    assert all("oos" not in [o["id"] for o in r["question"]["options"]] for r in rows)
    assert any("labels" not in r["supervision"] for r in rows)  # oos -> null answer

    ts = prepare_dataset("tasksource_jev", data_root, fetcher=fake_fetcher, staging_parent=tmp_path)
    assert all(r["provenance"]["extra"]["license_use"] == "commercial" for r in ts.load("train"))
    assert all(r["provenance"]["extra"]["variant"] == "direct" for r in ts.load("test"))
    assert ts.meta()["conversion_stats"]


def test_max_rows_makes_a_separate_version(data_root, tmp_path):
    full = prepare_dataset("sst5", data_root, fetcher=fake_fetcher, staging_parent=tmp_path)
    small = prepare_dataset(
        "sst5", data_root, max_rows=3, fetcher=fake_fetcher, staging_parent=tmp_path
    )
    assert small.root != full.root and small.version.endswith("-max3")
    assert small.meta()["splits"]["test"]["rows"] == 3


def test_tampering_is_detected(data_root, tmp_path):
    dv = prepare_dataset("boolq", data_root, fetcher=fake_fetcher, staging_parent=tmp_path)
    path = dv.split_path("dev")
    path.chmod(0o644)
    path.write_text(path.read_text() + "\n{}\n")
    with pytest.raises(IntegrityError, match="changed"):
        dv.verify()


def test_checksums_refuse_symlinks_and_extras(tmp_path):
    root = tmp_path / "v"
    root.mkdir()
    (root / "a.txt").write_text("a")
    write_sha256sums(root)
    (root / "b.txt").write_text("b")
    with pytest.raises(IntegrityError, match="unlisted"):
        verify_sha256sums(root)
    (root / "b.txt").unlink()
    (root / "link").symlink_to(root / "a.txt")
    with pytest.raises(IntegrityError, match="symlink"):
        verify_sha256sums(root)


def test_registry_entries_are_consistent():
    registry = load_registry()
    for name, entry in registry.items():
        assert entry.hf or entry.git, name
        assert len(entry.revision) == 40, name
        if entry.converter:
            assert entry.files and entry.splits, name
    assert {e.tier for e in registry.values()} == {"must", "optional", "eval_only"}


def test_vertical_scenario_mapping(data_root, tmp_path):
    dv = prepare_dataset(
        "banking77_support", data_root, fetcher=fake_fetcher, staging_parent=tmp_path
    )
    rows = dv.load("train") + dv.load("test")
    by_group = {}
    for row in rows:
        by_group.setdefault(row["group"]["id"], {})[row["id"].rsplit(":", 1)[1]] = row
    assert all(set(g) == {"queue", "urgent"} for g in by_group.values())
    for group in by_group.values():
        source = group["queue"]["provenance"]["extra"]["source_label"]
        queue = group["queue"]["supervision"]["labels"]
        urgent = group["urgent"]["supervision"]["labels"]
        assert queue == ("security" if source == "lost_or_stolen_card" else "cards")
        assert urgent is (source == "lost_or_stolen_card")
    roles = {}
    for role in dv.meta()["splits"]:
        for row in dv.load(role):
            roles.setdefault(row["group"]["id"], set()).add(role)
    assert all(len(r) == 1 for r in roles.values()), "a request was split across roles"


def test_vertical_scenario_rejects_unmapped_labels(data_root, tmp_path):
    from tests.data.fixtures import class_feature, shard, write_card

    def fetch_with_unknown(entry, data_root, raw_dir):
        names = ["activate_my_card", "brand_new_intent"]
        write_card(
            raw_dir,
            {"features": [{"name": "text", "dtype": "string"}, class_feature("label", names)]},
        )
        shard(raw_dir, "data", "train", [{"text": "hi", "label": 1}])
        shard(raw_dir, "data", "test", [{"text": "yo", "label": 0}])
        return ["README.md"]

    with pytest.raises(ValueError, match="brand_new_intent"):
        prepare_dataset(
            "banking77_support", data_root, fetcher=fetch_with_unknown, staging_parent=tmp_path
        )


def test_tev1_records_keep_the_upstream_order(data_root, tmp_path):
    dv = prepare_dataset("tev1", data_root, fetcher=fake_fetcher, staging_parent=tmp_path)
    meta = dv.meta()
    assert {role: info["from"] for role, info in meta["splits"].items()}["eval_only"] == (
        "v1/policy_transfer"
    )
    assert set(meta["splits"]) == {"train", "calib", "dev", "test", "eval_only"}
    dedup = json.loads((dv.manifests_dir / "dedup_report.json").read_text(encoding="utf-8"))
    assert dedup["dropped"] == 0
    rows = {r["id"]: r for role in meta["splits"] for r in dv.load(role)}
    mnli = rows["tev1:mnli:train:1"]
    assert [o["id"] for o in mnli["question"]["options"]] == ["neutral", "contradicted", "entailed"]
    assert mnli["supervision"]["labels"] == "contradicted"
    assert mnli["task_id"] == "tev1/mnli"
    assert mnli["provenance"]["extra"]["answer_letter"] == "B"
    sst = rows["tev1:sst5:train:3"]
    assert sst["question"]["type"] == "score" and sst["supervision"]["labels"] == 3
    boolq = rows["tev1:boolq:train:0"]
    assert boolq["question"]["type"] == "choice"
    assert boolq["provenance"]["extra"]["tev1_kind"] == "noul"
    # Object states keep the upstream key order (not sorted), nulls included.
    policy = rows["tev1:policy:train:0"]["input"]["context"]["state"]
    assert list(policy) == ["record", "policy"]
    assert list(policy["record"].items()) == [("order_value_usd", 0), ("item_unopened", None)]
