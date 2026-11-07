"""Run records take MetisCraft's commit only from its own source checkout."""

import subprocess
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("terms", [["internal_only"], ["mixed_per_row"], ["custom_contract"]])
def test_custom_terms_are_unverified_in_metadata_and_notice(terms):
    assert trace.commercial_use(terms) is None
    # Re-rendering a historical positive flag must not approve unfamiliar terms.
    notice = trace.notice_text({"dataset": {"terms": terms}, "commercial_use": True})
    assert "eligibility is unverified" in notice
    assert "terms do not restrict" not in notice
    assert "do not grant permission" in notice


def test_restrictions_take_precedence_over_unknown_terms():
    terms = ["research_only", "custom_contract"]
    assert trace.commercial_use(terms) is False
    notice = trace.notice_text({"dataset": {"terms": terms}, "commercial_use": False})
    assert "NOT permitted by the recorded restriction flags" in notice


def test_no_restriction_flags_do_not_claim_licence_permission():
    assert trace.commercial_use([]) is True
    notice = trace.notice_text({"dataset": {"terms": []}, "commercial_use": True})
    assert "No recognized non-commercial restriction flags" in notice
    assert "terms do not restrict" not in notice
    assert "do not grant permission" in notice
