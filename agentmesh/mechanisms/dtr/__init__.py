"""
DTR (Delta Tool Retrieval) — the "produce" step.

Projection-based tool-result caching with confidence gating, hotspot token
extraction, and SLM query reformulation. DTR serves the cached *base* of a
tool result and fetches only the semantic *delta*, then async-refreshes the
exact entry — removing REDUNDANCY in the produce stage of a tool→LLM dataflow.
"""

from agentmesh.mechanisms.dtr.cache import (
    DTRCache,
    CacheEntry,
    CacheResult,
    SemanticEmbedder,
    HotspotTokenExtractor,
    QueryReformulator,
    create_dtr_cache,
)

__all__ = [
    "DTRCache",
    "CacheEntry",
    "CacheResult",
    "SemanticEmbedder",
    "HotspotTokenExtractor",
    "QueryReformulator",
    "create_dtr_cache",
]

# Sibling-contributed modules (embeddings.py, reformulator.py, baselines.py)
# are added by another agent. Guard each import so the package keeps importing
# before those files exist.
try:  # pragma: no cover - optional sibling module
    from agentmesh.mechanisms.dtr.embeddings import *  # noqa: F401,F403
except Exception:
    pass

try:  # pragma: no cover - optional sibling module
    from agentmesh.mechanisms.dtr.reformulator import *  # noqa: F401,F403
except Exception:
    pass

try:  # pragma: no cover - optional sibling module
    from agentmesh.mechanisms.dtr.baselines import *  # noqa: F401,F403
except Exception:
    pass
