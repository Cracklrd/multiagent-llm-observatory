"""Analyst agent definition."""

from crewai import Agent
from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool


def create_analyst(llm: BaseLLM, tools: list[BaseTool]) -> Agent:
    return Agent(
        role="Cost and Performance Analyst",
        goal="Compare candidate models using transparent cost, latency, and quality metrics.",
        backstory=(
            "You turn catalog evidence into measurable comparisons. You show calculations, "
            "retain units, and avoid claiming that cost alone proves quality."
        ),
        llm=llm,
        tools=tools,
        allow_delegation=False,
        verbose=False,
        max_iter=8,
    )
