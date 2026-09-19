from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Status(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    RUNNING = "running"


class TraceStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class BaseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def to_db_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class RunRecord(BaseRecord):
    run_id: str
    scenario_id: str
    scenario_name: str
    started_at: str = Field(default_factory=utc_now)
    ended_at: str | None = None
    total_duration_ms: int = 0
    status: Status = Status.RUNNING
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    error_count: int = 0
    retry_count: int = 0
    quality_score: float | None = None
    final_result: str = ""

    @field_validator(
        "total_duration_ms",
        "total_input_tokens",
        "total_output_tokens",
        "total_tokens",
        "error_count",
        "retry_count",
    )
    @classmethod
    def non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("value must be non-negative")
        return value

    @field_validator("quality_score")
    @classmethod
    def quality_between_zero_and_one_hundred(cls, value: float | None) -> float | None:
        if value is not None and not 0 <= value <= 100:
            raise ValueError("quality_score must be between 0 and 100")
        return value

    @model_validator(mode="after")
    def calculate_total_tokens(self) -> "RunRecord":
        self.total_tokens = self.total_input_tokens + self.total_output_tokens
        return self


class AgentTraceRecord(BaseRecord):
    trace_id: str
    run_id: str
    agent_name: str
    agent_role: str
    task_name: str
    task_order: int
    status: TraceStatus = TraceStatus.SUCCESS
    model_name: str
    started_at: str = Field(default_factory=utc_now)
    ended_at: str | None = None
    duration_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    tool_calls: int = 0
    retry_count: int = 0
    error_count: int = 0
    quality_score: float | None = None
    result_summary: str = ""

    @field_validator(
        "task_order",
        "duration_ms",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "tool_calls",
        "retry_count",
        "error_count",
    )
    @classmethod
    def non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("value must be non-negative")
        return value

    @field_validator("quality_score")
    @classmethod
    def quality_between_zero_and_one_hundred(cls, value: float | None) -> float | None:
        if value is not None and not 0 <= value <= 100:
            raise ValueError("quality_score must be between 0 and 100")
        return value

    @model_validator(mode="after")
    def calculate_total_tokens(self) -> "AgentTraceRecord":
        self.total_tokens = self.input_tokens + self.output_tokens
        return self


class ToolCallRecord(BaseRecord):
    tool_call_id: str
    run_id: str
    trace_id: str
    agent_name: str
    tool_name: str
    started_at: str = Field(default_factory=utc_now)
    ended_at: str | None = None
    duration_ms: int = 0
    status: TraceStatus = TraceStatus.SUCCESS
    input_summary: str = ""
    output_summary: str = ""
    error_message: str = ""

    @field_validator("duration_ms")
    @classmethod
    def non_negative_duration(cls, value: int) -> int:
        if value < 0:
            raise ValueError("duration_ms must be non-negative")
        return value


class CommunicationRecord(BaseRecord):
    communication_id: str
    run_id: str
    source_agent: str
    target_agent: str
    sequence_number: int
    message_type: str
    message_summary: str
    timestamp: str = Field(default_factory=utc_now)
    token_count: int = 0

    @field_validator("sequence_number", "token_count")
    @classmethod
    def non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("value must be non-negative")
        return value
