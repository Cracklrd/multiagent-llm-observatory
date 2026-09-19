import pytest
from pydantic import ValidationError

from core.models import AgentTraceRecord, RunRecord


def test_run_total_tokens_are_calculated() -> None:
    run = RunRecord(
        run_id="run-1",
        scenario_id="scenario-1",
        scenario_name="Resume simple",
        total_input_tokens=120,
        total_output_tokens=80,
    )

    assert run.total_tokens == 200


def test_trace_total_tokens_are_calculated() -> None:
    trace = AgentTraceRecord(
        trace_id="trace-1",
        run_id="run-1",
        agent_name="Analyste",
        agent_role="Analyse les metriques",
        task_name="Comparer les couts",
        task_order=2,
        model_name="model_balanced",
        input_tokens=30,
        output_tokens=45,
    )

    assert trace.total_tokens == 75


def test_quality_score_must_be_valid() -> None:
    with pytest.raises(ValidationError):
        RunRecord(
            run_id="run-1",
            scenario_id="scenario-1",
            scenario_name="Resume simple",
            quality_score=120,
        )
