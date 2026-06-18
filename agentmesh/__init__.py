"""
AgentMesh: Toward Redundancy- and Barrier-Free Dataflow in Multi-Agent Systems

AgentMesh is a structure-aware runtime for multi-agent LLM systems (MAS). It
abstracts every MAS interaction as a produce–transfer–consume dataflow: a
*producer* (a tool, or an upstream agent's LLM decoding) creates an artifact,
it is *transferred*, and a *consumer* (always an LLM prefill) ingests it.

A structure-blind runtime executes each dataflow as one opaque, coarse-grained
block, wasting time on two pathologies:
- REDUNDANCY: re-producing content already held.
- BARRIER:    a consumer idling until its producer fully finishes.

AgentMesh makes each dataflow redundancy- and barrier-free via three mechanisms:
- DTR (Delta Tool Retrieval):       serve the cached base, fetch only the
                                    semantic delta, async-refresh the entry
                                    (the "produce" step).
- BPP (Branch-Parallel Prefill):    prefill independent worker branches in
                                    parallel against the shared supervisor
                                    prefix, then stitch branch-local KVs with
                                    late RoPE realignment (the "consume" step).
- DES (Dynamic-Equilibrium Streaming): drive the pipeline to a dynamic
                                    equilibrium (slack δ→0) via OSE
                                    sensitivities (overlaps produce/transfer/
                                    consume).

The public surface is intentionally small so that ``import agentmesh`` succeeds
with only ``numpy`` + ``pyyaml`` installed; heavy optional dependencies
(torch, sentence-transformers, openai, vllm, autogen) are imported lazily.
"""

__version__ = "2.0.0"
__author__ = "AgentMesh Authors"

from agentmesh.core.runtime import AgentMeshRuntime, create_runtime
from agentmesh.core.config import AgentMeshConfig

__all__ = [
    "AgentMeshRuntime",
    "AgentMeshConfig",
    "create_runtime",
]
