import pytest

from tools.cost_calculator import calculate_cost, calculate_cost_breakdown


def test_calculate_cost_uses_separate_input_and_output_prices() -> None:
    cost = calculate_cost("model_balanced", 2_000_000, 1_000_000)

    assert cost == 2.5


def test_cost_breakdown_is_explainable() -> None:
    result = calculate_cost_breakdown("model_fast", 500_000, 250_000)

    assert result.input_cost == 0.075
    assert result.output_cost == 0.15
    assert result.total_cost == 0.225


@pytest.mark.parametrize("input_tokens,output_tokens", [(-1, 0), (0, -1)])
def test_calculate_cost_rejects_negative_tokens(input_tokens: int, output_tokens: int) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        calculate_cost("model_fast", input_tokens, output_tokens)


def test_calculate_cost_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        calculate_cost("model_missing", 100, 100)
