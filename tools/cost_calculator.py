"""Reproducible cost calculations using the local simulated price catalog."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from tools.data_search import DEFAULT_CATALOG_PATH, get_model


TOKENS_PER_PRICE_UNIT = Decimal("1000000")
MONEY_PRECISION = Decimal("0.000001")


@dataclass(frozen=True)
class CostBreakdown:
    model_name: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


def calculate_cost_breakdown(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
) -> CostBreakdown:
    """Calculate input, output, and total costs from prices per million tokens."""

    if isinstance(input_tokens, bool) or isinstance(output_tokens, bool):
        raise TypeError("Token counts must be integers")
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        raise TypeError("Token counts must be integers")
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts must be non-negative")

    model = get_model(model_name, catalog_path)
    input_cost = Decimal(input_tokens) * Decimal(str(model.input_price)) / TOKENS_PER_PRICE_UNIT
    output_cost = Decimal(output_tokens) * Decimal(str(model.output_price)) / TOKENS_PER_PRICE_UNIT

    def money(value: Decimal) -> float:
        return float(value.quantize(MONEY_PRECISION, rounding=ROUND_HALF_UP))

    return CostBreakdown(
        model_name=model.model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost=money(input_cost),
        output_cost=money(output_cost),
        total_cost=money(input_cost + output_cost),
    )


def calculate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
) -> float:
    """Return only the estimated total cost for the required public API."""

    return calculate_cost_breakdown(model_name, input_tokens, output_tokens, catalog_path).total_cost
