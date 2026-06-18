"""
AgentMesh benchmark suite.

Reviewer-runnable benchmarks that reproduce the per-mechanism and end-to-end
claims of AgentMesh, a structure-aware runtime for multi-agent LLM systems.

Every interaction in AgentMesh is a produce-transfer-consume dataflow. A
structure-blind runtime wastes time on REDUNDANCY (re-producing held content)
and BARRIER (a consumer idling until its producer finishes). AgentMesh removes
both with three mechanisms, each measured by a benchmark module here:

- DTR (Delta Tool Retrieval)        -> shrinks PRODUCE   -> benchmarks.dtr
- BPP (Branch-Parallel Prefill)     -> shrinks CONSUME   -> benchmarks.bpp
- DES (Dynamic-Equilibrium Streaming) -> OVERLAPS stages -> benchmarks.des

plus an end-to-end harness (benchmarks.end_to_end) that toggles the three
mechanisms cumulatively to reproduce the ablation bands.

Each script has a fast synthetic default that runs in seconds with no GPU,
no cluster, and no LLM. Heavy dependencies (torch, transformers, matplotlib)
are optional and guarded; scripts fall back to numeric/CSV output when a
plotting backend is absent. See benchmarks/README.md for the full map from
each script to the paper figure / number it supports.
"""

__all__ = ["dtr", "bpp", "des", "end_to_end"]
