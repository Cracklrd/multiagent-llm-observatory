import pytest

from core.crew import build_crew, create_llm, run_demo_crew


EXPECTED_ROLES = [
    "Mission Coordinator",
    "Model Researcher",
    "Cost and Performance Analyst",
    "Evidence Verifier",
    "Optimization Recommender",
]


def test_build_crew_has_five_distinct_agents_and_tasks() -> None:
    crew = build_crew("Compare models for a summarization workload.")

    assert [agent.role for agent in crew.agents] == EXPECTED_ROLES
    assert len(crew.tasks) == 5
    assert [task.name for task in crew.tasks] == [
        "plan_mission",
        "research_models",
        "analyze_tradeoffs",
        "verify_analysis",
        "recommend_optimization",
    ]


def test_agent_tools_are_assigned_to_relevant_roles() -> None:
    crew = build_crew("Compare models.")

    assert not crew.agents[0].tools
    assert [tool.name for tool in crew.agents[1].tools] == ["search_local_model_catalog"]
    assert [tool.name for tool in crew.agents[2].tools] == ["calculate_simulated_model_cost"]
    assert [tool.name for tool in crew.agents[3].tools] == ["evaluate_report_quality"]
    assert not crew.agents[4].tools


def test_demo_crew_runs_all_five_stages_without_api() -> None:
    result = run_demo_crew("Compare models without an external API.")

    assert result.mode == "demo"
    assert len(result.task_outputs) == 5
    assert "model_balanced" in result.final_output


def test_empty_mission_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        run_demo_crew("  ")


def test_llm_uses_official_openai_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("OBSERVATORY_OPENAI_BASE_URL", raising=False)
    llm = create_llm("gpt-4.1-mini")

    assert llm.model == "gpt-4.1-mini"
    assert str(llm.base_url).rstrip("/") == "https://api.openai.com/v1"
