"""Run records take MetisCraft's commit only from its own source checkout."""

import subprocess
from pathlib import Path

import metiscraft.trace as trace
from metiscraft.trace import checkout_of


def git_repo(root: Path, name: str) -> Path:
    root.mkdir(parents=True)
    (root / "pyproject.toml").write_text(f'[project]\nname = "{name}"\n', encoding="utf-8")
    commit = ["-c", "user.name=t", "-c", "user.email=t@example.org", "commit", "-q"]
    for args in (["init", "-q"], [*commit, "--allow-empty", "-m", "init"]):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    return root


def module(root: Path, *parents: str) -> Path:
    path = root.joinpath(*parents, "metiscraft", "trace.py")
    path.parent.mkdir(parents=True)
    path.write_text("", encoding="utf-8")
    return path


def test_the_source_checkout_is_recognised(tmp_path):
    repo = git_repo(tmp_path / "MetisCraft", "metiscraft")
    assert checkout_of(module(repo, "src")) == repo.resolve()


def test_a_package_installed_inside_another_repository_is_not(tmp_path):
    project = git_repo(tmp_path / "project", "their-app")
    venv = (".venv", "lib", "python3.11", "site-packages")
    assert checkout_of(module(project, *venv)) is None
    assert checkout_of(module(project, "src")) is None  # their src layout, their commit


def test_a_copy_outside_git_is_not(tmp_path):
    root = tmp_path / "copy"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname = "metiscraft"\n', encoding="utf-8")
    assert checkout_of(module(root, "src")) is None


def test_an_installed_package_records_its_version(monkeypatch):
    monkeypatch.setattr(trace, "checkout_of", lambda module: None)
    info, diff = trace.git_info()
    assert info == {"commit": None, "source": "installed package", "version": trace.__version__}
    assert diff == ""
