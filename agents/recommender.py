"""Recommendation agent definition."""

from crewai import Agent
from crewai.llms.base_llm import BaseLLM


def create_recommender(llm: BaseLLM) -> Agent:
    return Agent(
        role="Optimization Recommender",
        goal="Produce a concise recommendation using only verified evidence from earlier tasks.",
        backstory=(
            "You write decision-ready recommendations for engineering teams. Every conclusion "
            "must connect to a supplied metric, trade-off, or verifier finding."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=5,
    )
