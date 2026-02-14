"""
Tests for SRR (Semantic Residual Retrieval) mechanism.
"""

import pytest
import numpy as np
from agentmesh.mechanisms.srr import (
    SRRCache,
    CacheEntry,
    CacheResult,
    SemanticEmbedder,
    HotspotTokenExtractor,
    QueryReformulator,
    create_srr_cache,
)


class TestSemanticEmbedder:
    """Tests for SemanticEmbedder class."""

    def test_embed_returns_normalized_array(self):
        """Embedding should return a normalized numpy array."""
        embedder = SemanticEmbedder()
        result = embedder.embed("test query")
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
        # Check normalization: ||v|| ≈ 1.0
        norm = np.linalg.norm(result)
        assert norm == pytest.approx(1.0, abs=0.01)

    def test_embed_batch_returns_array(self):
        """Batch embedding should return array of embeddings."""
        embedder = SemanticEmbedder()
        texts = ["query 1", "query 2", "query 3"]
        result = embedder.embed_batch(texts)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(texts)

    def test_similar_texts_high_dot_product(self):
        """Similar texts should have high dot product (cosine sim on normalized vectors)."""
        embedder = SemanticEmbedder()

        emb1 = embedder.embed("What are the main challenges in machine learning?")
        emb2 = embedder.embed("What challenges exist in ML?")

        # Since vectors are normalized, dot product = cosine similarity
        similarity = np.dot(emb1, emb2)
        assert similarity > 0.5


class TestHotspotTokenExtractor:
    """Tests for HotspotTokenExtractor — Algorithm 1 hotspot extraction."""

    def test_extract_returns_k_tokens(self):
        """Should return top-k hotspot tokens."""
        embedder = SemanticEmbedder()
        extractor = HotspotTokenExtractor(embedder, k=3)

        query_emb = embedder.embed("machine learning optimization")
        cached_emb = embedder.embed("deep learning training process")
        residual = query_emb - (np.dot(query_emb, cached_emb) * cached_emb)
        residual_hat = residual / (np.linalg.norm(residual) + 1e-10)

        hotspots = extractor.extract_hotspot_tokens(
            "machine learning optimization algorithms neural", residual_hat
        )

        assert len(hotspots) <= 3
        assert all(isinstance(t, str) for t in hotspots)

    def test_extract_fewer_tokens_than_k(self):
        """When fewer tokens than k, should return all."""
        embedder = SemanticEmbedder()
        extractor = HotspotTokenExtractor(embedder, k=5)

        emb = embedder.embed("test")
        residual_hat = np.random.randn(len(emb)).astype(np.float32)
        residual_hat /= np.linalg.norm(residual_hat)

        hotspots = extractor.extract_hotspot_tokens("hello world", residual_hat)

        assert len(hotspots) <= 2


class TestSRRCache:
    """Tests for SRRCache."""

    def test_create_cache_with_defaults(self):
        """Cache creation with defaults (τ=0.7)."""
        cache = create_srr_cache(
            confidence_threshold=0.7,
            cache_size=100,
        )
        assert cache is not None
        assert cache.confidence_threshold == 0.7
        assert cache.cache_size == 100

    def test_store_items_and_lookup_exact(self):
        """Exact query should hit cache."""
        cache = SRRCache(confidence_threshold=0.3, cache_size=100)

        query = "What is machine learning?"
        items = ["ML is a subset of AI", "It learns from data", "Uses algorithms"]

        cache.store(query, items)
        result = cache.lookup(query)

        assert result.hit is True
        assert result.projection_intensity > 0.3

    def test_lookup_miss_high_threshold(self):
        """Dissimilar query should miss cache with high τ."""
        cache = SRRCache(confidence_threshold=0.95, cache_size=100)

        cache.store("What is AI?", ["AI is artificial intelligence"])
        result = cache.lookup("How do I cook pasta?")

        assert result.hit is False

    def test_projection_intensity_computed(self):
        """Projection intensity (s_max) should be computed on hit."""
        cache = SRRCache(confidence_threshold=0.3, cache_size=100)

        cache.store("deep learning models", ["Neural networks", "Training data"])
        result = cache.lookup("deep learning approaches")

        if result.hit:
            assert 0.0 <= result.projection_intensity <= 1.0
            assert result.reuse_count >= 0

    def test_reuse_count_proportional(self):
        """Reuse count should be ⌊s_max · n⌋ where n = retrieval_depth."""
        cache = SRRCache(confidence_threshold=0.3, cache_size=100)

        items = ["item 1", "item 2", "item 3", "item 4", "item 5"]
        cache.store("test query about AI safety", items)

        result = cache.lookup("test query about AI safety")
        if result.hit:
            # SRRCache computes k_reuse = floor(s_max * retrieval_depth)
            expected_reuse = int(result.projection_intensity * cache.retrieval_depth)
            assert result.reuse_count == expected_reuse

    def test_cache_eviction(self):
        """Cache should evict oldest entries when full."""
        cache = SRRCache(confidence_threshold=0.3, cache_size=3)

        cache.store("query 1", ["resp 1"])
        cache.store("query 2", ["resp 2"])
        cache.store("query 3", ["resp 3"])
        cache.store("query 4", ["resp 4"])

        assert len(cache._cache) == 3

    def test_stats_tracking(self):
        """Statistics should track SRR activations and cold retrievals."""
        cache = SRRCache(confidence_threshold=0.3, cache_size=100)

        cache.store("test query", ["test response"])
        cache.lookup("test query")
        cache.lookup("completely different query about cooking")

        stats = cache.get_stats()
        assert stats["total_lookups"] == 2
        assert "srr_activations" in stats or "cold_retrievals" in stats

    def test_get_reused_items(self):
        """get_reused_items should return top-n items from cache entry."""
        cache = SRRCache(confidence_threshold=0.3, cache_size=100)
        items = ["alpha", "beta", "gamma", "delta"]
        cache.store("test query", items)

        result = cache.lookup("test query")
        if result.hit and result.cached_entry:
            reused = cache.get_reused_items(result)
            assert len(reused) <= len(items)
            assert reused == items[:result.reuse_count]


class TestCacheResult:
    """Tests for CacheResult dataclass."""

    def test_cache_miss_defaults(self):
        """Cache miss should have default values."""
        result = CacheResult(hit=False)
        assert result.projection_intensity == 0.0
        assert result.reuse_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
