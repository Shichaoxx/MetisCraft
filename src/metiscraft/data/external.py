"""Build a dataset by running a public project's own data builders at a pinned commit.

Some projects publish deterministic builders and the hashes of what they produce rather
than the data itself. For a registry entry with ``git`` and ``params.build``:

1. the project is checked out at the pinned commit under
   ``<data_root>/_downloads/metiscraft/git/``, verified by the commit id, and copied to a
   scratch tree, so the builders never write into the checkout;
2. each of ``params.build.steps`` runs unmodified in this Python environment with the
   Hugging Face Hub switched off. A wrapper (this module, run as
   ``python -I -m metiscraft.data.external <inputs.json> <script> [args...]``) answers the
   builders' ``datasets.load_dataset`` calls from prepared MetisCraft versions pinned at
   the same revisions and their ``AutoTokenizer.from_pretrained`` calls from verified
   weight directories, and refuses anything ``params.build.inputs`` does not declare;
3. the outputs are accepted only when every hash published in ``params.build.verify``
   matches;
4. the built files matching the entry's ``files`` patterns are copied into ``raw/`` under
   their repository paths.
"""

from __future__ import annotations

import json
import os
import platform
import runpy
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

from metiscraft.data.registry import DatasetEntry
from metiscraft.data.store import sha256_file

OFFLINE = {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx", ".h5", ".msgpack")
RECORDED_PACKAGES = ("datasets", "transformers", "tokenizers", "jinja2", "pyarrow")


class BuildError(RuntimeError):
    """An external build failed, or its outputs differ from the published hashes."""


def git_source(entry: DatasetEntry) -> tuple[str, str]:
    """Clone URL and commit of ``git: <owner>/<repo>@<commit>`` (GitHub) or ``<url>@<commit>``."""
    where, _, commit = (entry.git or "").rpartition("@")
    if not where or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError(f"{entry.name}: git must be <repo>@<full 40-character commit id>")
    return (where if "://" in where else f"https://github.com/{where}.git"), commit


def _git(cwd: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return done.stdout.strip()


def checkout(entry: DatasetEntry, cache: Path) -> Path:
    """The pinned commit, checked out once under ``cache`` and never written to afterwards."""
    url, commit = git_source(entry)
    owner, repo = url.rstrip("/").removesuffix(".git").split("/")[-2:]
    dest = Path(cache) / f"{owner}__{repo}@{commit}"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".checkout-", dir=dest.parent))
        try:
            _git(tmp, "init", "-q")
            for options in ([], ["-c", "http.version=HTTP/1.1"]):  # retry once over HTTP/1.1
                try:
                    _git(tmp, *options, "fetch", "-q", "--depth", "1", url, commit)
                    break
                except subprocess.CalledProcessError as exc:
                    failure = exc.stderr.strip()
            else:
                raise BuildError(f"cannot fetch {url} at {commit}: {failure}")
            _git(tmp, "checkout", "-q", "--detach", "FETCH_HEAD")
            tmp.rename(dest)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if _git(dest, "rev-parse", "HEAD") != commit or _git(dest, "status", "--porcelain"):
        raise BuildError(f"{dest} is not a clean checkout of {commit}")
    return dest


def _copy_tokenizer(root: Path, dest: Path) -> None:
    """Copy a weight directory except the weights, checking each file against SHA256SUMS."""
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, _, rel = line.partition("  ")
        if not rel or rel.endswith(WEIGHT_SUFFIXES):
            continue
        if sha256_file(root / rel) != digest:
            raise BuildError(f"{root / rel} does not match its SHA256SUMS")
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, dest / rel)


def serve_inputs(
    entry: DatasetEntry, data_root: Path, models_root: Path | None, work: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The wrapper's configuration, and what to record about each declared input."""
    from metiscraft.data.prepare import prepare_dataset
    from metiscraft.data.registry import get_entry

    inputs = entry.params["build"].get("inputs", {})
    config: dict[str, Any] = {
        "datasets": {},
        "tokenizers": {},
        "cache_dir": str(work / "datasets-cache"),
    }
    record: dict[str, Any] = {"datasets": {}, "tokenizers": {}}
    for repo, name in inputs.get("datasets", {}).items():
        dep = get_entry(name)
        if dep.repo != repo:
            raise BuildError(f"{entry.name}: input {repo} maps to {name}, which is {dep.repo}")
        version = prepare_dataset(name, data_root)  # prepares it, or verifies the existing version
        config["datasets"][repo] = {"revision": dep.revision, "path": str(version.raw_dir)}
        record["datasets"][repo] = {
            "dataset": name,
            "version": version.version,
            "revision": dep.revision,
        }
    for repo, rel in inputs.get("tokenizers", {}).items():
        if models_root is None:
            raise BuildError(f"{entry.name}: tokenizer inputs need paths.models_root")
        root = Path(models_root) / rel
        source = json.loads((root / "SOURCE.json").read_text(encoding="utf-8"))
        if source.get("repo") != repo:
            raise BuildError(f"{entry.name}: {rel} holds {source.get('repo')}, not {repo}")
        dest = work / "tokenizers" / repo.replace("/", "__")
        _copy_tokenizer(root, dest)
        config["tokenizers"][repo] = {"revision": source["revision"], "path": str(dest)}
        record["tokenizers"][repo] = {"weights": rel, "revision": source["revision"]}
    return config, record


def run_steps(steps: list[list[str]], tree: Path, config_path: Path, log_path: Path) -> None:
    env = {**os.environ, **OFFLINE}
    with open(log_path, "a", encoding="utf-8") as log:
        for step in steps:
            log.write(f"$ {' '.join(step)}\n")
            log.flush()
            command = [sys.executable, "-I", "-m", __name__, str(config_path), *step]
            done = subprocess.run(command, cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT)
            if done.returncode:
                tail = log_path.read_text(encoding="utf-8").splitlines()[-20:]
                raise BuildError(
                    f"step `{' '.join(step)}` exited with {done.returncode}:\n" + "\n".join(tail)
                )


def verify_outputs(source: Path, tree: Path, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare built files with the hashes the project published at the pinned commit.

    Each spec names a JSON ``manifest`` in the checkout and the ``key`` (dotted) of a table
    of hashes; the table maps paths under ``root``, or names listed in ``files``. A spec with
    ``required: false`` holds hashes the project records but does not claim to reproduce
    (say, intermediate files of its original build): they are compared and reported, and a
    difference does not fail the build.
    """
    rows = []
    for spec in specs:
        required = spec.get("required", True)
        table: Any = json.loads((source / spec["manifest"]).read_text(encoding="utf-8"))
        for key in spec["key"].split("."):
            table = table[key]
        if "root" in spec:
            expected = {f"{spec['root']}/{rel}": digest for rel, digest in table.items()}
        else:
            expected = {path: table[name] for name, path in spec["files"].items()}
        for path, digest in expected.items():
            built = sha256_file(tree / path) if (tree / path).is_file() else None
            rows.append(
                {
                    "file": path,
                    "published": digest,
                    "built": built,
                    "manifest": spec["manifest"],
                    "required": required,
                }
            )
    checked = [r for r in rows if r["required"]]
    if not checked:
        raise BuildError("params.build.verify lists no required published hashes")
    wrong = [r["file"] for r in checked if r["built"] != r["published"]]
    if wrong:
        raise BuildError(
            f"{len(wrong)} of {len(checked)} built files differ from the published hashes: "
            + ", ".join(wrong)
        )
    return rows


def keep_outputs(tree: Path, raw_dir: Path, patterns: list[str]) -> list[str]:
    """Copy the built files matching ``patterns`` into ``raw_dir`` under their repository paths."""
    kept = set()
    for pattern in patterns:
        matched = sorted(p for p in tree.glob(pattern) if p.is_file())
        if not matched:
            raise BuildError(f"the build produced nothing matching {pattern}")
        for path in matched:
            rel = path.relative_to(tree)
            (raw_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, raw_dir / rel)
            kept.add(rel.as_posix())
    return sorted(kept)


def environment() -> dict[str, str]:
    found = {"python": platform.python_version()}
    for name in RECORDED_PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return found


def build(
    entry: DatasetEntry, data_root: Path, models_root: Path | None, work: Path
) -> dict[str, Any]:
    """Build in ``work`` (outputs stay in ``work/tree``) and verify; returns what to record."""
    from metiscraft.data.fetch import download_root

    spec = entry.params["build"]
    source = checkout(entry, download_root(data_root) / "git")
    tree = Path(work) / "tree"
    shutil.copytree(source, tree, ignore=shutil.ignore_patterns(".git"))
    config, inputs = serve_inputs(entry, data_root, models_root, Path(work))
    config_path = Path(work) / "inputs.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    log_path = Path(work) / "build.log"
    steps = [[str(part) for part in step] for step in spec["steps"]]
    run_steps(steps, tree, config_path, log_path)
    return {
        "steps": [" ".join(step) for step in steps],
        "inputs": inputs,
        "environment": environment(),
        "verified": verify_outputs(source, tree, spec["verify"]),
        "log": log_path.read_text(encoding="utf-8"),
    }


def fetch_build(entry: DatasetEntry, data_root: Path, raw_dir: Path):
    """The fetcher for built entries: build beside the staging version, keep ``files``."""
    from metiscraft.config import resolve_paths
    from metiscraft.data.fetch import FetchResult

    models_root = resolve_paths().models_root
    with tempfile.TemporaryDirectory(prefix=".build-", dir=Path(raw_dir).parent.parent) as tmp:
        result = build(entry, data_root, models_root, Path(tmp))
        files = keep_outputs(Path(tmp) / "tree", raw_dir, entry.files)
    verified = result.pop("verified")
    log = result.pop("log")
    required = [row for row in verified if row["required"]]
    reference = [row for row in verified if not row["required"]]
    result["verified"] = {
        "files": len(required),
        "manifests": sorted({row["manifest"] for row in required}),
    }
    if reference:
        result["verified"]["reference"] = {
            "files": len(reference),
            "differ": [row["file"] for row in reference if row["built"] != row["published"]],
        }
    return FetchResult(
        files=files,
        source={"build": result},
        manifests={"build_verification.json": {"files": verified}, "build.log": log},
    )


def _serve(config: dict[str, Any]) -> None:
    """Answer the builders' Hub loads from the declared local copies, and refuse the rest."""

    def local(kind: str, name: Any, revision: str | None) -> str:
        spec = config[kind].get(str(name))
        if spec is None:
            raise RuntimeError(f"{name} is not declared in params.build.inputs.{kind}")
        if revision not in (None, spec["revision"]):
            raise RuntimeError(f"{name}: asked for {revision}, registry pins {spec['revision']}")
        return spec["path"]

    try:
        import datasets
    except ImportError:
        pass
    else:
        load_dataset = datasets.load_dataset

        def served_load_dataset(path, name=None, *args, revision=None, **kwargs):
            kwargs.setdefault("cache_dir", config["cache_dir"])
            return load_dataset(local("datasets", path, revision), name, *args, **kwargs)

        datasets.load_dataset = served_load_dataset
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pass
    else:
        from_pretrained = AutoTokenizer.from_pretrained

        def served_from_pretrained(name, *args, revision=None, local_files_only=None, **kwargs):
            return from_pretrained(local("tokenizers", name, revision), *args, **kwargs)

        AutoTokenizer.from_pretrained = served_from_pretrained


def _main(argv: list[str]) -> None:
    config = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    script = Path(argv[1]).resolve()
    _serve(config)
    sys.argv = [str(script), *argv[2:]]
    sys.path.insert(0, str(script.parent))  # as `python <script>` does, for sibling imports
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    _main(sys.argv[1:])
