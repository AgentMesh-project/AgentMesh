"""
BPP (Branch-Parallel Prefill) — the "consume" step.

Branch-Parallel Attention (BPA) for the supervisor-worker topology:
inter-branch masking, branch-parallel prefilling, KV stitching, and RoPE
positional realignment. BPP prefills independent worker branches in parallel
against the shared supervisor prefix and removes the BARRIER in the consume
stage of an LLM→LLM dataflow.

A production vLLM integration (a GPUKVConnector) lives in
``agentmesh.mechanisms.bpp.vllm`` and requires vLLM to be installed.
"""

from agentmesh.mechanisms.bpp.attention import (
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

__all__ = [
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

# Production vLLM GPU KV connector (created by another agent; needs vLLM).
# Must not break import when vLLM is unavailable.
try:  # pragma: no cover - vLLM is an optional extra
    from .vllm import GPUKVConnector  # noqa: F401

    __all__ += ["GPUKVConnector"]
except Exception:
    pass
