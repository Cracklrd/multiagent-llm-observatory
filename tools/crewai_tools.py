"""CrewAI adapters for the deterministic Phase 2 tools."""

import json

from crewai.tools import tool

from tools.cost_calculator import calculate_cost_breakdown
from tools.data_search import load_models, recommend_model
from tools.quality_evaluator import evaluate_quality_details


@tool("search_local_model_catalog")
def search_local_model_catalog(priority: str = "balanced") -> str:
    """Search local simulated LLM data and recommend a model for cost, speed, balance, or quality."""

    recommended = recommend_model(priority)
    payload = {
        "source": "data/models.csv",
        "priority": priority,
        "recommended_model": recommended.to_dict(),
        "all_models": [model.to_dict() for model in load_models()],
    }
    return json.dumps(payload, indent=2)


@tool("calculate_simulated_model_cost")
def calculate_simulated_model_cost(model_name: str, input_tokens: int, output_tokens: int) -> str:
    """Calculate a model's input, output, and total cost from simulated prices per million tokens."""

    result = calculate_cost_breakdown(model_name, input_tokens, output_tokens)
    return json.dumps(result.to_dict(), indent=2)


@tool("evaluate_report_quality")
def evaluate_report_quality(expected_keywords: str, generated_text: str) -> str:
    """Score report coverage from 0 to 100; expected_keywords must be comma-separated."""

    keywords = [keyword.strip() for keyword in expected_keywords.split(",") if keyword.strip()]
    result = evaluate_quality_details(keywords, generated_text)
    return json.dumps(result.to_dict(), indent=2)
