"""External builds: a local git repository stands in for a public project."""

import hashlib
import json
import subprocess

import pytest

from metiscraft.data.external import BuildError, build, checkout, fetch_build, git_source
from metiscraft.data.prepare import prepare_dataset
from metiscraft.data.registry import DatasetEntry, get_entry
from metiscraft.data.store import sha256_file
from tests.data.fixtures import fake_fetcher

# Counts BoolQ's training rows through the Hub API, as an upstream builder would.
BUILDER = """
import sys
from pathlib import Path

from datasets import load_dataset

rows = load_dataset(sys.argv[1], revision=sys.argv[2])["train"]
Path("out").mkdir()
Path("out/count.txt").write_text(f"{len(rows)}\\n")
"""

# Records which tokenizer class the builder was given.
TOKENIZER_BUILDER = """
import sys
from pathlib import Path

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], revision=sys.argv[2])
Path("out").mkdir()
Path("out/tokenizer.txt").write_text(type(tokenizer).__name__ + "\\n")
"""


def git(cwd, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def upstream(
    tmp_path, published: str, builder: str = BUILDER, output: str = "count.txt"
) -> tuple[str, str]:
    repo = tmp_path / "project"
    repo.mkdir()
    (repo / "build.py").write_text(builder, encoding="utf-8")
    # "history" stands for hashes a project records but does not claim to reproduce.
    tables = {"files": {output: published}, "history": {output: "0" * 64}}
    (repo / "published.json").write_text(json.dumps(tables), encoding="utf-8")
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "builder")
    return f"file://{repo}", git(repo, "rev-parse", "HEAD")


def entry(url: str, commit: str, repo: str = "google/boolq", declared: bool = True):
    boolq = get_entry("boolq")
    return DatasetEntry(
        name="toy_build",
        tier="optional",
        git=f"{url}@{commit}",
        files=["out/*"],
        judgment="noul",
        license="mit",
        params={
            "build": {
                "inputs": {"datasets": {"google/boolq": "boolq"} if declared else {}},
                "steps": [["build.py", repo, boolq.revision]],
                "verify": [
                    {"manifest": "published.json", "key": "files", "root": "out"},
                    {
                        "manifest": "published.json",
                        "key": "history",
                        "root": "out",
                        "required": False,
                    },
                ],
            }
        },
    )


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "data"
    prepare_dataset("boolq", root, fetcher=fake_fetcher, staging_parent=tmp_path)
    return root


def raw_dir(tmp_path):
    path = tmp_path / "stage" / "toy_build" / "v" / "raw"
    path.mkdir(parents=True)
    return path


def test_build_is_kept_only_when_the_published_hashes_match(data_root, tmp_path):
    url, commit = upstream(tmp_path, hashlib.sha256(b"60\n").hexdigest())
    result = fetch_build(entry(url, commit), data_root, raw_dir(tmp_path))
    assert result.files == ["out/count.txt"]
    assert (tmp_path / "stage/toy_build/v/raw/out/count.txt").read_text() == "60\n"
    build = result.source["build"]
    assert build["verified"] == {
        "files": 1,
        "manifests": ["published.json"],
        "reference": {"files": 1, "differ": ["out/count.txt"]},
    }
    assert build["inputs"]["datasets"]["google/boolq"]["dataset"] == "boolq"
    assert build["steps"] == [f"build.py google/boolq {get_entry('boolq').revision}"]
    assert result.manifests["build_verification.json"]["files"][0]["built"] == (
        hashlib.sha256(b"60\n").hexdigest()
    )
    # The checkout is cached by commit and stays clean; the scratch tree is gone.
    cached = checkout(entry(url, commit), data_root / "_downloads/metiscraft/git")
    assert not (cached / "out").exists()
    assert not list((tmp_path / "stage/toy_build").glob(".build-*"))


def test_a_different_output_is_refused(data_root, tmp_path):
    url, commit = upstream(tmp_path, hashlib.sha256(b"59\n").hexdigest())
    with pytest.raises(BuildError, match="differ from the published hashes: out/count.txt"):
        fetch_build(entry(url, commit), data_root, raw_dir(tmp_path))


def test_undeclared_inputs_are_refused(data_root, tmp_path):
    url, commit = upstream(tmp_path, hashlib.sha256(b"60\n").hexdigest())
    with pytest.raises(BuildError, match="not declared in params.build.inputs"):
        fetch_build(entry(url, commit, declared=False), data_root, raw_dir(tmp_path))


@pytest.mark.parametrize("declared", [None, "Qwen2Tokenizer"])
def test_tokenizers_load_as_the_class_the_project_used(tmp_path, declared):
    from transformers import AutoTokenizer

    from tests.modeling.tiny import tiny_tokenizer

    weights = tmp_path / "models/tiny/tok"
    tiny_tokenizer().save_pretrained(weights)
    sums = [f"{sha256_file(path)}  {path.name}" for path in sorted(weights.iterdir())]
    (weights / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    pin = {"repo": "org/tiny", "revision": "a" * 40}
    (weights / "SOURCE.json").write_text(json.dumps(pin), encoding="utf-8")
    expected = declared or type(AutoTokenizer.from_pretrained(weights)).__name__
    url, commit = upstream(
        tmp_path,
        hashlib.sha256(f"{expected}\n".encode()).hexdigest(),
        TOKENIZER_BUILDER,
        "tokenizer.txt",
    )
    spec = {"weights": "tiny/tok", **({"class": declared} if declared else {})}
    params = {
        "inputs": {"tokenizers": {"org/tiny": spec if declared else "tiny/tok"}},
        "steps": [["build.py", "org/tiny", "a" * 40]],
        "verify": [{"manifest": "published.json", "key": "files", "root": "out"}],
    }
    toy = entry(url, commit).model_copy(update={"params": {"build": params}})
    result = build(toy, tmp_path / "data", tmp_path / "models", tmp_path / "work")
    assert result["inputs"]["tokenizers"]["org/tiny"] == {**spec, "revision": "a" * 40}
    assert (tmp_path / "work/tree/out/tokenizer.txt").read_text() == f"{expected}\n"


def test_git_sources_pin_a_full_commit():
    toy = entry("file:///nowhere", "0" * 40)
    assert git_source(toy) == ("file:///nowhere", "0" * 40)
    with pytest.raises(ValueError, match="40-character"):
        git_source(toy.model_copy(update={"git": "owner/repo@1dde778"}))
    assert git_source(toy.model_copy(update={"git": "owner/repo@" + "a" * 40}))[0] == (
        "https://github.com/owner/repo.git"
    )
