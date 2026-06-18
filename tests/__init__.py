"""AgentMesh test suite.

Tests for the structure-aware multi-agent runtime and its three mechanisms:
DTR (Delta Tool Retrieval), BPP (Branch-Parallel Prefill), and DES
(Dynamic-Equilibrium Streaming).

Heavy-dependency tests (torch / sentence-transformers / vLLM) are guarded with
``pytest.importorskip`` so the suite runs green on a minimal install.
"""
