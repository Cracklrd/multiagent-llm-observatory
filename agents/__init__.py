"""Factory functions for the five observatory agents."""

from agents.analyst import create_analyst
from agents.coordinator import create_coordinator
from agents.recommender import create_recommender
from agents.researcher import create_researcher
from agents.verifier import create_verifier

__all__ = [
    "create_analyst",
    "create_coordinator",
    "create_recommender",
    "create_researcher",
    "create_verifier",
]
