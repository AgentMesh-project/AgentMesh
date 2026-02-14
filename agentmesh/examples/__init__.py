"""
AgentMesh Example Applications

Includes:
- deep_research: Supervisor-worker research workflow demonstrating
  SRR semantic caching, TSD parallel prefill, and ESF adaptive streaming.
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
