"""Verifier agent definition."""

from crewai import Agent
from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool


def create_verifier(llm: BaseLLM, tools: list[BaseTool]) -> Agent:
    return Agent(
        role="Evidence Verifier",
        goal="Check completeness and consistency, then assign an explainable quality score.",
        backstory=(
            "You audit observable claims and calculations. You identify missing evidence, "
            "use the deterministic quality tool, and request a retry only when justified."
        ),
        llm=llm,
        tools=tools,
        allow_delegation=False,
        verbose=False,
        max_iter=6,
    )
