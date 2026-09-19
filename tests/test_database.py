from core.database import (
    connect,
    create_agent_trace,
    create_communication,
    create_run,
    create_tool_call,
    get_kpis,
    initialize_database,
)
from core.models import AgentTraceRecord, CommunicationRecord, RunRecord, Status, ToolCallRecord


def test_database_schema_can_be_created(tmp_path) -> None:
    database_path = tmp_path / "observability.db"

    initialize_database(database_path)

    with connect(database_path) as connection:
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    assert {"runs", "agent_traces", "tool_calls", "communications"}.issubset(tables)


def test_can_create_run_trace_tool_call_and_communication(tmp_path) -> None:
    database_path = tmp_path / "observability.db"
    initialize_database(database_path)

    with connect(database_path) as connection:
        create_run(
            connection,
            RunRecord(
                run_id="run-1",
                scenario_id="scenario-1",
                scenario_name="Resume simple",
                status=Status.SUCCESS,
                total_duration_ms=500,
                total_input_tokens=100,
                total_output_tokens=50,
                estimated_cost=0.012,
                quality_score=88,
                final_result="Resultat de demonstration.",
            ),
        )
        create_agent_trace(
            connection,
            AgentTraceRecord(
                trace_id="trace-1",
                run_id="run-1",
                agent_name="Chercheur",
                agent_role="Recupere les donnees locales",
                task_name="Lire les donnees modeles",
                task_order=1,
                model_name="model_fast",
                duration_ms=120,
                input_tokens=40,
                output_tokens=20,
                tool_calls=1,
                quality_score=90,
            ),
        )
        create_tool_call(
            connection,
            ToolCallRecord(
                tool_call_id="tool-call-1",
                run_id="run-1",
                trace_id="trace-1",
                agent_name="Chercheur",
                tool_name="data_search",
                duration_ms=25,
                input_summary="Recherche model_fast",
                output_summary="Modele trouve dans les donnees locales",
            ),
        )
        create_communication(
            connection,
            CommunicationRecord(
                communication_id="communication-1",
                run_id="run-1",
                source_agent="Coordinateur",
                target_agent="Chercheur",
                sequence_number=1,
                message_type="task_assignment",
                message_summary="Demande de lecture des donnees locales",
                token_count=18,
            ),
        )
        connection.commit()

        counts = {
            table: connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()[
                "count"
            ]
            for table in ["runs", "agent_traces", "tool_calls", "communications"]
        }

    assert counts == {
        "runs": 1,
        "agent_traces": 1,
        "tool_calls": 1,
        "communications": 1,
    }


def test_kpis_are_calculated_from_sqlite_data(tmp_path) -> None:
    database_path = tmp_path / "observability.db"
    initialize_database(database_path)

    with connect(database_path) as connection:
        create_run(
            connection,
            RunRecord(
                run_id="run-1",
                scenario_id="scenario-1",
                scenario_name="Resume simple",
                status=Status.SUCCESS,
                total_duration_ms=100,
                total_input_tokens=10,
                total_output_tokens=20,
                estimated_cost=0.03,
            ),
        )
        create_run(
            connection,
            RunRecord(
                run_id="run-2",
                scenario_id="scenario-2",
                scenario_name="Comparaison",
                status=Status.FAILED,
                total_duration_ms=300,
                total_input_tokens=50,
                total_output_tokens=50,
                estimated_cost=0.10,
                error_count=2,
            ),
        )
        create_agent_trace(
            connection,
            AgentTraceRecord(
                trace_id="trace-1",
                run_id="run-1",
                agent_name="Analyste",
                agent_role="Analyse les metriques",
                task_name="Calculer les tendances",
                task_order=1,
                model_name="model_balanced",
                duration_ms=250,
            ),
        )
        create_agent_trace(
            connection,
            AgentTraceRecord(
                trace_id="trace-2",
                run_id="run-2",
                agent_name="Chercheur",
                agent_role="Recupere les donnees locales",
                task_name="Lire les donnees",
                task_order=1,
                model_name="model_fast",
                duration_ms=50,
            ),
        )
        create_agent_trace(
            connection,
            AgentTraceRecord(
                trace_id="trace-3",
                run_id="run-2",
                agent_name="Chercheur",
                agent_role="Recupere les donnees locales",
                task_name="Lire les donnees",
                task_order=2,
                model_name="model_fast",
                duration_ms=60,
            ),
        )
        connection.commit()

        kpis = get_kpis(connection)

    assert kpis["total_runs"] == 2
    assert kpis["success_rate"] == 0.5
    assert kpis["average_duration_ms"] == 200
    assert kpis["total_tokens"] == 130
    assert kpis["total_estimated_cost"] == 0.13
    assert kpis["total_errors"] == 2
    assert kpis["most_used_agent"] == "Chercheur"
    assert kpis["slowest_agent"] == "Analyste"
