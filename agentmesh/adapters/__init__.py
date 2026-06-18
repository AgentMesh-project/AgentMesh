"""
AgentMesh Framework Adapters

Provides integration with popular multi-agent frameworks.
"""

from agentmesh.adapters.autogen import (
    AutoGenAdapter,
    WrappedAgent,
    create_autogen_adapter,
)

__all__ = [
    "AutoGenAdapter",
    "WrappedAgent",
    "create_autogen_adapter",
]
