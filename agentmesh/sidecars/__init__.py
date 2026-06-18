"""
AgentMesh Sidecars

Provides the three-layer sidecar architecture for the produce–transfer–consume
dataflow between agents, LLMs, and tools.
"""

from agentmesh.sidecars.sidecar import (
    Message,
    AgentSidecar,
    LLMSidecar,
    ToolSidecar,
    SidecarMesh,
)

__all__ = [
    "Message",
    "AgentSidecar",
    "LLMSidecar",
    "ToolSidecar",
    "SidecarMesh",
]
