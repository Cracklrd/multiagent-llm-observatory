"""Search and compare the simulated LLM model catalog."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "models.csv"


@dataclass(frozen=True)
class ModelProfile:
    """One locally configured model and its observable characteristics."""

    model_name: str
    input_price: float
    output_price: float
    avg_latency: float
    quality_score: float

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


def load_models(catalog_path: str | Path = DEFAULT_CATALOG_PATH) -> list[ModelProfile]:
    """Load and validate model profiles from a CSV catalog."""

    path = Path(catalog_path)
    if not path.exists():
        raise FileNotFoundError(f"Model catalog not found: {path}")

    with path.open(encoding="utf-8", newline="") as catalog_file:
        reader = csv.DictReader(catalog_file)
        required = {"model_name", "input_price", "output_price", "avg_latency", "quality_score"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Catalog must contain these columns: {sorted(required)}")

        models = []
        for line_number, row in enumerate(reader, start=2):
            try:
                model = ModelProfile(
                    model_name=row["model_name"].strip(),
                    input_price=float(row["input_price"]),
                    output_price=float(row["output_price"]),
                    avg_latency=float(row["avg_latency"]),
                    quality_score=float(row["quality_score"]),
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid model data on line {line_number}") from error

            if not model.model_name:
                raise ValueError(f"Missing model name on line {line_number}")
            if min(model.input_price, model.output_price, model.avg_latency) < 0:
                raise ValueError(f"Negative value on line {line_number}")
            if not 0 <= model.quality_score <= 100:
                raise ValueError(f"Quality score outside 0-100 on line {line_number}")
            models.append(model)

    if not models:
        raise ValueError("Model catalog is empty")
    if len({model.model_name.casefold() for model in models}) != len(models):
        raise ValueError("Model names must be unique")
    return models


def get_model(model_name: str, catalog_path: str | Path = DEFAULT_CATALOG_PATH) -> ModelProfile:
    """Find one model by its case-insensitive exact name."""

    normalized_name = model_name.strip().casefold()
    for model in load_models(catalog_path):
        if model.model_name.casefold() == normalized_name:
            return model
    raise ValueError(f"Unknown model: {model_name}")


def search_models(
    query: str = "",
    *,
    max_input_price: float | None = None,
    max_output_price: float | None = None,
    max_latency: float | None = None,
    min_quality: float | None = None,
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
) -> list[ModelProfile]:
    """Search models with optional, combinable constraints."""

    limits = (max_input_price, max_output_price, max_latency, min_quality)
    if any(value is not None and value < 0 for value in limits):
        raise ValueError("Search limits must be non-negative")

    normalized_query = query.strip().casefold()
    matches = []
    for model in load_models(catalog_path):
        if normalized_query and normalized_query not in model.model_name.casefold():
            continue
        if max_input_price is not None and model.input_price > max_input_price:
            continue
        if max_output_price is not None and model.output_price > max_output_price:
            continue
        if max_latency is not None and model.avg_latency > max_latency:
            continue
        if min_quality is not None and model.quality_score < min_quality:
            continue
        matches.append(model)
    return sorted(matches, key=lambda model: (-model.quality_score, model.avg_latency, model.model_name))


def recommend_model(priority: str = "balanced", catalog_path: str | Path = DEFAULT_CATALOG_PATH) -> ModelProfile:
    """Select a model using an explicit cost, speed, quality, or balanced rule."""

    models = load_models(catalog_path)
    normalized_priority = priority.strip().casefold()
    if normalized_priority == "cost":
        return min(models, key=lambda model: (model.input_price + model.output_price, -model.quality_score))
    if normalized_priority == "speed":
        return min(models, key=lambda model: (model.avg_latency, -model.quality_score))
    if normalized_priority == "quality":
        return max(models, key=lambda model: (model.quality_score, -model.avg_latency))
    if normalized_priority != "balanced":
        raise ValueError("Priority must be: balanced, cost, speed, or quality")

    def normalize(value: float, values: list[float]) -> float:
        value_range = max(values) - min(values)
        return 1.0 if value_range == 0 else (value - min(values)) / value_range

    prices = [model.input_price + model.output_price for model in models]
    latencies = [model.avg_latency for model in models]
    qualities = [model.quality_score for model in models]

    def balanced_score(model: ModelProfile) -> float:
        quality = normalize(model.quality_score, qualities)
        affordability = 1 - normalize(model.input_price + model.output_price, prices)
        speed = 1 - normalize(model.avg_latency, latencies)
        return 0.5 * quality + 0.3 * affordability + 0.2 * speed

    return max(models, key=lambda model: (balanced_score(model), model.quality_score))
