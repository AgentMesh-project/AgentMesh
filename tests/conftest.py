"""
Pytest configuration and shared fixtures for the AgentMesh test suite.

Design goals:
  - Run green on a minimal install (numpy + pyyaml + pytest + pytest-asyncio).
  - Never import torch / sentence-transformers / vLLM at collection time; tests
    that need them call ``pytest.importorskip`` locally.
  - DTR tests rely on the hash-based embedder fallback (no torch / sbert), which
    yields deterministic 32-dim unit vectors.
"""

import asyncio
from typing import List

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# pytest-asyncio configuration
# ---------------------------------------------------------------------------
# We use the explicit ``@pytest.mark.asyncio`` marker on async tests rather than
# auto mode, so plain async helpers are not accidentally collected.
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "asyncio: mark a coroutine test to be run by pytest-asyncio"
    )


# ---------------------------------------------------------------------------
# Configuration fixtures (no heavy deps)
# ---------------------------------------------------------------------------
@pytest.fixture
def dtr_config():
    """Default DTR configuration for testing (τ=0.8, k=3)."""
    from agentmesh.core.config import DTRConfig

    return DTRConfig(
        enabled=True,
        confidence_threshold=0.8,
        retrieval_depth=10,
        hotspot_k=3,
        cache_size=100,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        slm_model="Qwen/Qwen2.5-0.5B-Instruct",
    )


@pytest.fixture
def bpp_config():
    """Default BPP configuration for testing (max_parallel_branches=3)."""
    from agentmesh.core.config import BPPConfig

    return BPPConfig(
        enabled=True,
        max_parallel_branches=3,
        num_layers=2,
        num_heads=4,
        head_dim=8,
        rope_base=10000.0,
    )


@pytest.fixture
def des_config():
    """Default DES configuration for testing (initial_theta=512)."""
    from agentmesh.core.config import DESConfig

    return DESConfig(
        enabled=True,
        initial_theta=512,
        theta_min=64,
        theta_max=4096,
        damping_factor=1.0,
        window_size=5,
    )


@pytest.fixture
def agentmesh_config(dtr_config, bpp_config, des_config):
    """A complete AgentMesh configuration assembled from the per-mechanism ones."""
    from agentmesh.core.config import AgentMeshConfig, LLMConfig

    return AgentMeshConfig(
        dtr=dtr_config,
        bpp=bpp_config,
        des=des_config,
        llm=LLMConfig(endpoint="http://localhost:8000/v1"),
        device="cpu",
    )


# ---------------------------------------------------------------------------
# DTR fixtures (hash-embedder fallback, no torch / sbert)
# ---------------------------------------------------------------------------
@pytest.fixture
def dtr_cache():
    """A small DTR cache using the hash-based embedder fallback."""
    from agentmesh.mechanisms.dtr import create_dtr_cache

    return create_dtr_cache(
        confidence_threshold=0.8,
        retrieval_depth=10,
        cache_size=100,
        hotspot_k=3,
    )


@pytest.fixture
def sample_queries() -> List[str]:
    """Sample tool queries for testing."""
    return [
        "What is machine learning?",
        "Explain artificial intelligence",
        "How does deep learning work?",
        "What are neural networks?",
        "Describe natural language processing",
    ]


@pytest.fixture
def sample_items() -> List[str]:
    """Sample retrieved items (semantic snippets) for a cache entry."""
    return [f"item-{i}: retrieved snippet" for i in range(10)]


# ---------------------------------------------------------------------------
# DES fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def des_controller():
    """A DES controller with standard paper defaults."""
    from agentmesh.mechanisms.des import create_des_controller

    return create_des_controller(
        initial_theta=512,
        min_theta=64,
        max_theta=4096,
        damping_factor=1.0,
        window_size=5,
    )


# ---------------------------------------------------------------------------
# Fake LLM client (no network) for runtime tests
# ---------------------------------------------------------------------------
class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)
        self.delta = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, reply: str, stream_pieces: List[str]):
        self._reply = reply
        self._stream_pieces = stream_pieces
        self.calls = []

    async def create(self, *, model, messages, max_tokens, temperature, stream=False, **kw):
        self.calls.append({"model": model, "messages": messages, "stream": stream})
        if stream:
            pieces = self._stream_pieces

            async def _gen():
                for piece in pieces:
                    yield _FakeCompletion(piece)

            return _gen()
        return _FakeCompletion(self._reply)


class _FakeChat:
    def __init__(self, reply: str, stream_pieces: List[str]):
        self.completions = _FakeCompletions(reply, stream_pieces)


class FakeLLMClient:
    """A drop-in async stand-in for the OpenAI AsyncClient (no network).

    Mirrors the ``client.chat.completions.create(...)`` surface used by the
    runtime, supporting both single-shot and streaming calls.
    """

    def __init__(self, reply: str = "fake-llm-reply", stream_pieces=None):
        if stream_pieces is None:
            stream_pieces = ["chunk-" + str(i) + " " for i in range(8)]
        self.chat = _FakeChat(reply, stream_pieces)


@pytest.fixture
def fake_llm_client():
    """A fake async LLM client that never touches the network."""
    return FakeLLMClient(reply="machine learning is a subset of AI")


# ---------------------------------------------------------------------------
# Event-loop helper for environments without pytest-asyncio auto mode
# ---------------------------------------------------------------------------
@pytest.fixture
def run_async():
    """Run a coroutine to completion on a fresh event loop."""

    def _runner(coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    return _runner
