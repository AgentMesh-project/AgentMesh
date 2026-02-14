"""
Deep Research Demo for AgentMesh

A multi-agent research workflow demonstrating SRR, TSD, and ESF optimizations.
"""

from agentmesh.examples.deep_research.demo import (
    DeepResearchWorkflow,
    SupervisorAgent,
    ResearcherAgent,
    register_research_tools,
)

__all__ = [
    "DeepResearchWorkflow",
    "SupervisorAgent",
    "ResearcherAgent",
    "register_research_tools",
]
