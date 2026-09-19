"""Coordinator agent definition."""

from crewai import Agent
from crewai.llms.base_llm import BaseLLM


def create_coordinator(llm: BaseLLM) -> Agent:
    return Agent(
        role="Mission Coordinator",
        goal="Turn the user's optimization mission into a precise sequence of observable tasks.",
        backstory=(
            "You lead an LLM cost-optimization team. You define scope, assumptions, "
            "success criteria, and clean handoffs without inventing data."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=5,
    )
