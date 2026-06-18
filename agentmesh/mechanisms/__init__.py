"""
AgentMesh Core Mechanisms

Provides the three structure-aware mechanisms that make each
produce–transfer–consume dataflow redundancy- and barrier-free:
- DTR: Delta Tool Retrieval          (the "produce" step)
- BPP: Branch-Parallel Prefill       (the "consume" step)
- DES: Dynamic-Equilibrium Streaming (overlaps the stages)

The BPP imports are guarded so this package still imports when ``torch`` is
unavailable (the BPP attention classes only require torch at instantiation).
"""

from agentmesh.mechanisms.dtr import (
    DTRCache,
    CacheEntry,
    CacheResult,
    SemanticEmbedder,
    HotspotTokenExtractor,
    QueryReformulator,
    create_dtr_cache,
)

from agentmesh.mechanisms.des import (
    Observation,
    Sensitivity,
    OnlineSensitivityEstimator,
    DESController,
    create_des_controller,
)

__all__ = [
    # DTR
    "DTRCache",
    "CacheEntry",
    "CacheResult",
    "SemanticEmbedder",
    "HotspotTokenExtractor",
    "QueryReformulator",
    "create_dtr_cache",
    # DES
    "Observation",
    "Sensitivity",
    "OnlineSensitivityEstimator",
    "DESController",
    "create_des_controller",
]

# BPP is imported defensively: its attention classes depend on torch, which is
# an optional extra. If torch is present the names are re-exported here.
try:
    from agentmesh.mechanisms.bpp import (
        WorkerOutput,
        SupervisorContext,
        BPAResult,
        InterBranchMaskGenerator,
        RoPEAligner,
        BranchParallelManager,
        BPAManager,
        KVCacheManager,
        create_bpp_manager,
    )

    __all__ += [
        "WorkerOutput",
        "SupervisorContext",
        "BPAResult",
        "InterBranchMaskGenerator",
        "RoPEAligner",
        "BranchParallelManager",
        "BPAManager",
        "KVCacheManager",
        "create_bpp_manager",
    ]
except Exception:  # pragma: no cover - torch not installed
    pass
