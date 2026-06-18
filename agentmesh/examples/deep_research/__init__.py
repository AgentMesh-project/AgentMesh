"""
Deep Research Demo for AgentMesh

A multi-agent research workflow demonstrating DTR, BPP, and DES optimizations
over the produce–transfer–consume dataflow.
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
