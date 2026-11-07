from pathlib import Path

import pytest

from metiscraft.config import load_recipe
from metiscraft.data.registry import get_entry

RECIPES = sorted((Path(__file__).resolve().parents[1] / "recipes").rglob("*.yaml"))


def test_recipes_exist():
    assert len(RECIPES) >= 6


@pytest.mark.parametrize("path", RECIPES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_recipe_is_valid(path):
    recipe = load_recipe(path)
    entry = get_entry(recipe.data.dataset)
    assert entry.converter is not None
    if recipe.readout.kind == "answer_token":
        assert "multi_label" not in entry.judgments
