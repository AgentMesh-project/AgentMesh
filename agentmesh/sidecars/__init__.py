"""
AgentMesh Sidecars

Provides the three-layer sidecar architecture for
agent-LLM-tool communication optimization.
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
