"""
AgentMesh: Semantic Communication for Multi-Agent Systems

A semantic communication layer for multi-agent LLM systems that addresses
the semantic-transmission mismatch through three mechanisms:
- SRR (Semantic Residual Retrieval): Projection-based semantic caching
- TSD (Topology-aware Semantic Decoupling): Parallel KV prefill via TDA
- ESF (Elastic Semantic Flow): Adaptive microbatch streaming
"""

__version__ = "0.1.0"
__author__ = "Anonymous"

from agentmesh.core.runtime import AgentMeshRuntime
from agentmesh.core.config import AgentMeshConfig

__all__ = [
    "AgentMeshRuntime",
    "AgentMeshConfig",
    "__version__",
]
