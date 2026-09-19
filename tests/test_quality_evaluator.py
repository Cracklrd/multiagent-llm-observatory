import pytest

from tools.quality_evaluator import evaluate_quality, evaluate_quality_details


def test_quality_score_is_reproducible_keyword_coverage() -> None:
    score = evaluate_quality(
        ["cout", "qualite", "latence", "jetons"],
        "La qualite et le cout sont compares avec la latence.",
    )

    assert score == 75.0


def test_quality_matching_ignores_case_accents_and_punctuation() -> None:
    result = evaluate_quality_details(
        ["Coût", "modèle rapide"],
        "Le MODELE rapide réduit le cout!",
    )

    assert result.score == 100.0
    assert result.missing_keywords == ()
    assert result.total_keywords == 2


def test_quality_uses_whole_words_not_substrings() -> None:
    assert evaluate_quality(["cout"], "Un raccourci utile") == 0.0


def test_empty_keywords_have_zero_score() -> None:
    assert evaluate_quality([], "Any generated text") == 0.0


def test_invalid_generated_text_is_rejected() -> None:
    with pytest.raises(TypeError, match="string"):
        evaluate_quality(["cost"], None)  # type: ignore[arg-type]
