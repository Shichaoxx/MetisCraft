import pytest

from metiscraft import __version__
from metiscraft.cli import main
from metiscraft.registry import Registry


def test_registry_register_and_get():
    reg: Registry[int] = Registry("thing")
    reg.register("b")(2)
    reg.register("a")(1)
    assert reg.get("a") == 1
    assert reg.names() == ["a", "b"]
    assert "b" in reg


def test_registry_errors():
    reg: Registry[int] = Registry("thing")
    reg.register("a")(1)
    with pytest.raises(ValueError):
        reg.register("a")(2)
    with pytest.raises(KeyError, match="known: a"):
        reg.get("missing")


def test_cli_version(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_cli_config_show(tmp_path, capsys):
    recipe = tmp_path / "r.yaml"
    recipe.write_text(
        "task: t\ndata: {dataset: d}\nmodel: {base: m}\nreadout: {kind: answer_token}\n",
        encoding="utf-8",
    )
    assert main(["config", "show", str(recipe), "train.seed=7"]) == 0
    out = capsys.readouterr().out
    assert '"seed": 7' in out
