"""
SRR (Semantic Residual Retrieval)

Projection-based semantic caching with confidence gating,
hotspot token extraction, and SLM query reformulation.
"""

from agentmesh.mechanisms.srr.cache import (
    SRRCache,
    CacheEntry,
    CacheResult,
    SemanticEmbedder,
    HotspotTokenExtractor,
    QueryReformulator,
    create_srr_cache,
)

__all__ = [
    "SRRCache",
    "CacheEntry",
    "CacheResult",
    "SemanticEmbedder",
    "HotspotTokenExtractor",
    "QueryReformulator",
    "create_srr_cache",
]
