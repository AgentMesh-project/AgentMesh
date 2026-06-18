"""
Coding Agent Demo for AgentMesh.

A supervisor-worker coding-agent workflow demonstrating DTR, BPP, and DES
optimizations over the produce--transfer--consume dataflow:

  * The supervisor decomposes a repository task into N *independent* file/section
    fixes that are dispatched to parallel worker branches (BPP / branch-parallel
    prefill).
  * Each worker uses coding tools (read file / search / build) whose results are
    served from the DTR cache base + semantic delta (the produce step).
  * Tool and LLM outputs are streamed under DES adaptive chunk control (stage
    overlap).

The structure mirrors ``agentmesh.examples.deep_research`` but for code tasks.
References to OpenCode and SWE-bench Lite are illustrative only; this example
does not depend on either.
"""

from agentmesh.examples.coding_agent.demo import (
    CodingAgentWorkflow,
    CodingSupervisorAgent,
    CodingWorkerAgent,
    register_coding_tools,
    run_demo,
)

__all__ = [
    "CodingAgentWorkflow",
    "CodingSupervisorAgent",
    "CodingWorkerAgent",
    "register_coding_tools",
    "run_demo",
]
