"""CrewAI orchestration for one observable LLM optimization mission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from crewai import Crew, LLM, Process, Task

from agents import (
    create_analyst,
    create_coordinator,
    create_recommender,
    create_researcher,
    create_verifier,
)
from core.config import load_settings
from tools.cost_calculator import calculate_cost_breakdown
from tools.crewai_tools import (
    calculate_simulated_model_cost,
    evaluate_report_quality,
    search_local_model_catalog,
)
from tools.data_search import load_models, recommend_model
from tools.quality_evaluator import evaluate_quality_details


DEFAULT_MISSION = (
    "Recommend the best simulated LLM for a workload with 120,000 input tokens and "
    "30,000 output tokens. Balance cost, latency, and quality."
)


@dataclass(frozen=True)
class CrewExecutionResult:
    mode: str
    model_name: str
    final_output: str
    task_outputs: tuple[str, ...]
    usage: dict[str, Any]


def create_llm(model_name: str | None = None) -> LLM:
    settings = load_settings()
    selected_model = model_name or str(settings["openai_model"])
    return LLM(
        model=selected_model,
        provider="openai",
        base_url=str(settings["openai_base_url"]),
    )


def build_crew(mission: str = DEFAULT_MISSION, model_name: str | None = None) -> Crew:
    """Create five specialized agents and their sequential task handoffs."""

    if not mission.strip():
        raise ValueError("Mission cannot be empty")

    llm = create_llm(model_name)
    coordinator = create_coordinator(llm)
    researcher = create_researcher(llm, [search_local_model_catalog])
    analyst = create_analyst(llm, [calculate_simulated_model_cost])
    verifier = create_verifier(llm, [evaluate_report_quality])
    recommender = create_recommender(llm)

    plan_task = Task(
        name="plan_mission",
        description=(
            "Mission: {mission}\nCreate a four-step execution plan for research, analysis, "
            "verification, and recommendation. State measurable success criteria and assumptions."
        ),
        expected_output="A numbered four-step plan with scope, assumptions, and measurable criteria.",
        agent=coordinator,
    )
    research_task = Task(
        name="research_models",
        description=(
            "Follow the coordinator's plan. You must call search_local_model_catalog at least once. "
            "Select the priority most consistent with the mission and report the complete candidate data, "
            "units, recommendation, and source."
        ),
        expected_output="A factual model comparison citing data/models.csv and the tool call used.",
        agent=researcher,
        context=[plan_task],
    )
    analysis_task = Task(
        name="analyze_tradeoffs",
        description=(
            "Analyze the researched candidates for the mission. You must call "
            "calculate_simulated_model_cost for at least two candidate models using the mission's token "
            "counts. Compare cost, latency, and quality with explicit units and arithmetic results."
        ),
        expected_output="A ranked quantitative comparison of at least two models with cost calculations.",
        agent=analyst,
        context=[plan_task, research_task],
    )
    verification_task = Task(
        name="verify_analysis",
        description=(
            "Audit the analysis for catalog consistency, calculation support, and missing evidence. "
            "You must call evaluate_report_quality with the keywords cost, latency, quality, tokens, model. "
            "Return validation status, score from 0 to 100, errors, and whether a retry is required."
        ),
        expected_output="A verification report with status, numeric score, issues, and retry yes/no.",
        agent=verifier,
        context=[research_task, analysis_task],
    )
    recommendation_task = Task(
        name="recommend_optimization",
        description=(
            "Produce the final answer to the mission using only validated evidence. Name one recommended "
            "model, explain the cost/latency/quality trade-off, give two actionable optimizations, and "
            "briefly summarize execution confidence. Do not introduce new numbers."
        ),
        expected_output="A concise evidence-based recommendation with metrics and two actions.",
        agent=recommender,
        context=[research_task, analysis_task, verification_task],
    )

    return Crew(
        name="LLM Cost Observatory Crew",
        agents=[coordinator, researcher, analyst, verifier, recommender],
        tasks=[plan_task, research_task, analysis_task, verification_task, recommendation_task],
        process=Process.sequential,
        verbose=False,
        memory=False,
        cache=False,
    )


def run_live_crew(mission: str = DEFAULT_MISSION, model_name: str | None = None) -> CrewExecutionResult:
    """Run the five-agent crew through the OpenAI API."""

    settings = load_settings()
    if not settings["openai_api_key_available"]:
        raise RuntimeError("OPENAI_API_KEY is required for live mode")

    selected_model = model_name or str(settings["openai_model"])
    result = build_crew(mission, selected_model).kickoff(inputs={"mission": mission})
    task_outputs = tuple(str(output.raw) for output in result.tasks_output)
    usage_metrics = result.token_usage.model_dump() if result.token_usage else {}
    return CrewExecutionResult(
        mode="live",
        model_name=selected_model,
        final_output=str(result.raw),
        task_outputs=task_outputs,
        usage=usage_metrics,
    )


def run_demo_crew(mission: str = DEFAULT_MISSION) -> CrewExecutionResult:
    """Return a deterministic offline walkthrough with the same five observable stages."""

    if not mission.strip():
        raise ValueError("Mission cannot be empty")
    recommended = recommend_model("balanced")
    cost = calculate_cost_breakdown(recommended.model_name, 120_000, 30_000)
    analysis = (
        f"{recommended.model_name}: ${cost.total_cost:.6f}, "
        f"{recommended.avg_latency:.1f}s latency, {recommended.quality_score:.0f}/100 quality."
    )
    quality = evaluate_quality_details(
        ["cost", "latency", "quality", "tokens", "model"],
        f"The model comparison covers cost, latency, quality, and tokens. {analysis}",
    )
    outputs = (
        "Coordinator: research, calculate, verify, then recommend.",
        f"Researcher: loaded {len(load_models())} candidates from data/models.csv.",
        f"Analyst: {analysis}",
        f"Verifier: success, quality score {quality.score:.2f}, retry no.",
        f"Recommender: use {recommended.model_name} for the balanced workload.",
    )
    return CrewExecutionResult(
        mode="demo",
        model_name="deterministic-local-demo",
        final_output=outputs[-1],
        task_outputs=outputs,
        usage={},
    )
