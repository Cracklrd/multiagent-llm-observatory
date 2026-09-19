import csv

import pytest

from tools.data_search import get_model, load_models, recommend_model, search_models


def test_load_and_find_models() -> None:
    models = load_models()

    assert len(models) >= 3
    assert get_model("MODEL_BALANCED").quality_score == 85


def test_search_models_combines_filters() -> None:
    matches = search_models(max_latency=2.5, min_quality=75)

    assert [model.model_name for model in matches] == ["model_balanced", "model_economy"]


@pytest.mark.parametrize(
    ("priority", "expected"),
    [
        ("cost", "model_sprint"),
        ("speed", "model_sprint"),
        ("quality", "model_quality"),
        ("balanced", "model_balanced"),
    ],
)
def test_recommend_model_uses_visible_rules(priority: str, expected: str) -> None:
    assert recommend_model(priority).model_name == expected


def test_invalid_catalog_has_clear_error(tmp_path) -> None:
    catalog = tmp_path / "invalid.csv"
    with catalog.open("w", newline="", encoding="utf-8") as catalog_file:
        writer = csv.writer(catalog_file)
        writer.writerow(["model_name", "input_price"])
        writer.writerow(["broken", "0.1"])

    with pytest.raises(ValueError, match="columns"):
        load_models(catalog)
