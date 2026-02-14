"""
TSD (Topology-aware Semantic Decoupling)

Topologically Decoupled Attention (TDA) for the supervisor-worker
topology: inter-worker masking, timely parallel prefilling,
KV stitching, and RoPE positional realignment.
"""

from agentmesh.mechanisms.tsd.attention import (
    WorkerOutput,
    SupervisorContext,
    TDAResult,
    InterWorkerMaskGenerator,
    RoPEAligner,
    TDAManager,
    KVCacheManager,
    create_tda_manager,
    create_tsd_manager,
)

__all__ = [
    "WorkerOutput",
    "SupervisorContext",
    "TDAResult",
    "InterWorkerMaskGenerator",
    "RoPEAligner",
    "TDAManager",
    "KVCacheManager",
    "create_tda_manager",
    "create_tsd_manager",
]
