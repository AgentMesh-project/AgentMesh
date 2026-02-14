"""
Pytest configuration and fixtures for AgentMesh tests.
"""

import pytest
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def srr_config():
    """Default SRR configuration for testing (defaults: τ=0.7, k=3)."""
    from agentmesh.core.config import SRRConfig
    return SRRConfig(
        confidence_threshold=0.7,
        cache_size=100,
        hotspot_k=3,
        slm_model="Qwen/Qwen2.5-0.5B-Instruct",
        embedding_model="all-MiniLM-L6-v2",
    )


@pytest.fixture
def tsd_config():
    """Default TSD configuration for testing."""
    from agentmesh.core.config import TSDConfig
    return TSDConfig(
        max_parallel_workers=3,
        rope_base=10000.0,
    )


@pytest.fixture
def esf_config():
    """Default ESF configuration for testing."""
    from agentmesh.core.config import ESFConfig
    return ESFConfig(
        initial_microbatch_size=256,
        theta_min=64,
        theta_max=1024,
        damping_factor=0.1,
        window_size=10,
    )


@pytest.fixture
def agentmesh_config(srr_config, tsd_config, esf_config):
    """Complete AgentMesh configuration."""
    from agentmesh.core.config import AgentMeshConfig
    return AgentMeshConfig(
        srr=srr_config,
        tsd=tsd_config,
        esf=esf_config,
    )


# ============================================================================
# Mock Data Fixtures
# ============================================================================

@pytest.fixture
def sample_queries() -> List[str]:
    """Sample queries for testing."""
    return [
        "What is machine learning?",
        "Explain artificial intelligence",
        "How does deep learning work?",
        "What are neural networks?",
        "Describe natural language processing",
    ]


@pytest.fixture
def sample_responses() -> List[str]:
    """Sample responses for testing."""
    return [
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Artificial intelligence is the simulation of human intelligence by machines.",
        "Deep learning uses neural networks with many layers to learn representations.",
        "Neural networks are computing systems inspired by biological neural networks.",
        "NLP is a field focused on interaction between computers and human language.",
    ]


@pytest.fixture
def sample_embeddings() -> torch.Tensor:
    """Sample embeddings for testing."""
    # Create deterministic embeddings
    torch.manual_seed(42)
    return torch.randn(5, 384)


@pytest.fixture
def sample_topology() -> Dict[str, List[str]]:
    """Sample supervisor-worker topology for TSD testing."""
    return {
        "supervisor": [],
        "worker_1": ["supervisor"],
        "worker_2": ["supervisor"],
        "worker_3": ["supervisor"],
    }


# ============================================================================
# Mock Components
# ============================================================================

class MockEmbedder:
    """Mock embedder for testing without loading models."""
    
    def __init__(self, dim: int = 384):
        self.dim = dim
        self._cache = {}
    
    def embed(self, text: str) -> torch.Tensor:
        if text not in self._cache:
            # Deterministic embedding based on text hash
            seed = hash(text) % (2**32)
            torch.manual_seed(seed)
            self._cache[text] = torch.randn(self.dim)
        return self._cache[text]
    
    def embed_batch(self, texts: List[str]) -> torch.Tensor:
        return torch.stack([self.embed(t) for t in texts])


@pytest.fixture
def mock_embedder():
    """Mock embedder fixture."""
    return MockEmbedder()


class MockLLM:
    """Mock LLM for testing."""
    
    def __init__(self, latency_ms: float = 10.0):
        self.latency_ms = latency_ms
        self.call_count = 0
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        self.call_count += 1
        # Simulate latency
        import time
        time.sleep(self.latency_ms / 1000)
        return f"Response to: {prompt[:50]}..."
    
    async def agenerate(self, prompt: str, max_tokens: int = 100) -> str:
        self.call_count += 1
        import asyncio
        await asyncio.sleep(self.latency_ms / 1000)
        return f"Response to: {prompt[:50]}..."


@pytest.fixture
def mock_llm():
    """Mock LLM fixture."""
    return MockLLM()


class MockNetwork:
    """Mock network for ESF testing."""
    
    def __init__(self, base_latency_ms: float = 50.0, jitter_ms: float = 10.0):
        self.base_latency_ms = base_latency_ms
        self.jitter_ms = jitter_ms
        self._rng = np.random.RandomState(42)
    
    def send(self, data: bytes) -> float:
        """Simulate sending data, return latency."""
        size_factor = len(data) / 1000  # 1ms per KB
        latency = self.base_latency_ms + size_factor + self._rng.normal(0, self.jitter_ms)
        return max(0, latency)
    
    def set_conditions(self, base_latency_ms: float, jitter_ms: float):
        """Update network conditions."""
        self.base_latency_ms = base_latency_ms
        self.jitter_ms = jitter_ms


@pytest.fixture
def mock_network():
    """Mock network fixture."""
    return MockNetwork()


# ============================================================================
# Utility Functions
# ============================================================================

@pytest.fixture
def tensor_equal():
    """Utility for comparing tensors."""
    def _equal(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-5) -> bool:
        return torch.allclose(a, b, rtol=rtol)
    return _equal


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Temporary directory for cache testing."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


# ============================================================================
# Markers
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "integration: mark as integration test")


# ============================================================================
# Hooks
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if CUDA not available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
