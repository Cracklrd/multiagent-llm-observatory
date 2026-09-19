"""Researcher agent definition."""

from crewai import Agent
from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool


def create_researcher(llm: BaseLLM, tools: list[BaseTool]) -> Agent:
    return Agent(
        role="Model Researcher",
        goal="Retrieve relevant model facts from the local catalog and cite the local source used.",
        backstory=(
            "You are a careful technical researcher. You use tools for facts, preserve units, "
            "and distinguish measured catalog values from assumptions."
        ),
        llm=llm,
        tools=tools,
        allow_delegation=False,
        verbose=False,
        max_iter=6,
    )
