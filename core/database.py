import sqlite3
from pathlib import Path
from typing import Any

from core.models import AgentTraceRecord, CommunicationRecord, RunRecord, ToolCallRecord


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    scenario_id TEXT NOT NULL,
    scenario_name TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    total_duration_ms INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    total_input_tokens INTEGER NOT NULL DEFAULT 0,
    total_output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost REAL NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    retry_count INTEGER NOT NULL DEFAULT 0,
    quality_score REAL,
    final_result TEXT NOT NULL DEFAULT '',
    CHECK (total_duration_ms >= 0),
    CHECK (total_input_tokens >= 0),
    CHECK (total_output_tokens >= 0),
    CHECK (total_tokens >= 0),
    CHECK (estimated_cost >= 0),
    CHECK (error_count >= 0),
    CHECK (retry_count >= 0),
    CHECK (quality_score IS NULL OR quality_score BETWEEN 0 AND 100)
);

CREATE TABLE IF NOT EXISTS agent_traces (
    trace_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    agent_role TEXT NOT NULL,
    task_name TEXT NOT NULL,
    task_order INTEGER NOT NULL,
    status TEXT NOT NULL,
    model_name TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost REAL NOT NULL DEFAULT 0,
    tool_calls INTEGER NOT NULL DEFAULT 0,
    retry_count INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    quality_score REAL,
    result_summary TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    CHECK (task_order >= 0),
    CHECK (duration_ms >= 0),
    CHECK (input_tokens >= 0),
    CHECK (output_tokens >= 0),
    CHECK (total_tokens >= 0),
    CHECK (estimated_cost >= 0),
    CHECK (tool_calls >= 0),
    CHECK (retry_count >= 0),
    CHECK (error_count >= 0),
    CHECK (quality_score IS NULL OR quality_score BETWEEN 0 AND 100)
);

CREATE TABLE IF NOT EXISTS tool_calls (
    tool_call_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    input_summary TEXT NOT NULL DEFAULT '',
    output_summary TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (trace_id) REFERENCES agent_traces(trace_id) ON DELETE CASCADE,
    CHECK (duration_ms >= 0)
);

CREATE TABLE IF NOT EXISTS communications (
    communication_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    source_agent TEXT NOT NULL,
    target_agent TEXT NOT NULL,
    sequence_number INTEGER NOT NULL,
    message_type TEXT NOT NULL,
    message_summary TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    CHECK (sequence_number >= 0),
    CHECK (token_count >= 0)
);

CREATE INDEX IF NOT EXISTS idx_agent_traces_run_id ON agent_traces(run_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_run_id ON tool_calls(run_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_trace_id ON tool_calls(trace_id);
CREATE INDEX IF NOT EXISTS idx_communications_run_id ON communications(run_id);
"""


def connect(database_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def initialize_database(database_path: str | Path) -> None:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as connection:
        connection.executescript(SCHEMA_SQL)


def insert_record(connection: sqlite3.Connection, table: str, values: dict[str, Any]) -> None:
    columns = ", ".join(values.keys())
    placeholders = ", ".join(f":{key}" for key in values)
    connection.execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", values)


def create_run(connection: sqlite3.Connection, run: RunRecord) -> None:
    insert_record(connection, "runs", run.to_db_dict())


def create_agent_trace(connection: sqlite3.Connection, trace: AgentTraceRecord) -> None:
    insert_record(connection, "agent_traces", trace.to_db_dict())


def create_tool_call(connection: sqlite3.Connection, tool_call: ToolCallRecord) -> None:
    insert_record(connection, "tool_calls", tool_call.to_db_dict())


def create_communication(connection: sqlite3.Connection, communication: CommunicationRecord) -> None:
    insert_record(connection, "communications", communication.to_db_dict())


def get_kpis(connection: sqlite3.Connection) -> dict[str, float | int | None]:
    row = connection.execute(
        """
        SELECT
            COUNT(*) AS total_runs,
            COALESCE(AVG(CASE WHEN status = 'success' THEN 1.0 ELSE 0.0 END), 0) AS success_rate,
            COALESCE(AVG(total_duration_ms), 0) AS average_duration_ms,
            COALESCE(SUM(total_tokens), 0) AS total_tokens,
            COALESCE(SUM(estimated_cost), 0) AS total_estimated_cost,
            COALESCE(SUM(error_count), 0) AS total_errors
        FROM runs
        """
    ).fetchone()

    agent_row = connection.execute(
        """
        SELECT agent_name
        FROM agent_traces
        GROUP BY agent_name
        ORDER BY COUNT(*) DESC, agent_name ASC
        LIMIT 1
        """
    ).fetchone()

    slowest_row = connection.execute(
        """
        SELECT agent_name
        FROM agent_traces
        GROUP BY agent_name
        ORDER BY AVG(duration_ms) DESC, agent_name ASC
        LIMIT 1
        """
    ).fetchone()

    return {
        "total_runs": row["total_runs"],
        "success_rate": row["success_rate"],
        "average_duration_ms": row["average_duration_ms"],
        "total_tokens": row["total_tokens"],
        "total_estimated_cost": row["total_estimated_cost"],
        "total_errors": row["total_errors"],
        "most_used_agent": agent_row["agent_name"] if agent_row else None,
        "slowest_agent": slowest_row["agent_name"] if slowest_row else None,
    }
