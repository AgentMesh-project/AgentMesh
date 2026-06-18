"""
AgentMesh Example Applications

Includes:
- deep_research: Supervisor-worker research workflow demonstrating
  DTR delta retrieval, BPP branch-parallel prefill, and DES adaptive
  chunk streaming over the produce–transfer–consume dataflow.
"""

from agentmesh.examples.deep_research import (
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
