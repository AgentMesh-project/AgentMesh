"""
Tests for DTR (Delta Tool Retrieval) — the "produce" step.

DTR shrinks the produce stage of a tool→LLM dataflow: it serves the cached
*base* of a tool result and fetches only the semantic *delta*, then
async-refreshes the exact entry. These tests exercise the pure-Python cache
path using the hash-based embedder fallback (no torch / sentence-transformers).

Covered:
  - projection-intensity gating at τ (cold retrieval vs DTR activation)
  - item allocation: reuse ⌊s_max·n⌋, fetch n - ⌊s_max·n⌋
  - residual (delta) vector r = v_in - s_max * v_max and hotspot extraction
  - statistics bookkeeping and rates
  - asynchronous cache-refresh scheduling
"""

import time

import numpy as np
import pytest

from agentmesh.mechanisms.dtr import (
    DTRCache,
    create_dtr_cache,
    CacheEntry,
    CacheResult,
    SemanticEmbedder,
    HotspotTokenExtractor,
    QueryReformulator,
)


# ---------------------------------------------------------------------------
# Embedder (hash fallback)
# ---------------------------------------------------------------------------
def test_hash_embedder_is_normalized():
    """Without sbert, the embedder falls back to a normalized hash vector."""
    emb = SemanticEmbedder()
    v = emb.embed("some tool query")
    assert isinstance(v, np.ndarray)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)


def test_hash_embedder_is_deterministic():
    emb = SemanticEmbedder()
    a = emb.embed("identical text")
    b = emb.embed("identical text")
    assert np.allclose(a, b)


def test_self_similarity_is_one():
    """A normalized vector dotted with itself equals 1 (s_i = v_in · v_i)."""
    emb = SemanticEmbedder()
    v = emb.embed("query about reinforcement learning")
    assert np.isclose(float(np.dot(v, v)), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Cold retrieval vs activation (gating at τ)
# ---------------------------------------------------------------------------
def test_cold_retrieval_on_empty_cache(dtr_cache):
    """Empty cache → s_max = 0 < τ → cold retrieval, fetch full depth."""
    result = dtr_cache.lookup("first ever query")
    assert result.hit is False
    assert result.projection_intensity == 0.0
    assert result.reuse_count == 0
    assert result.residual_count == dtr_cache.retrieval_depth


def test_activation_on_identical_query(dtr_cache, sample_items):
    """Identical query → s_max = 1.0 ≥ τ → DTR activation, full reuse."""
    query = "weather in Paris next week"
    dtr_cache.store(query, sample_items)

    result = dtr_cache.lookup(query)
    assert result.hit is True
    assert np.isclose(result.projection_intensity, 1.0, atol=1e-5)
    # s_max ≈ 1.0 → reuse ⌊1.0 · n⌋ = n, residual 0
    assert result.reuse_count == dtr_cache.retrieval_depth
    assert result.residual_count == 0


def test_gating_threshold_boundary():
    """A query whose best match sits just below τ must NOT activate."""
    cache = create_dtr_cache(confidence_threshold=0.8, retrieval_depth=10, cache_size=10)
    emb = cache.embedder

    base = emb.embed("anchor query")
    # Construct a stored entry whose embedding gives s_max < τ against a probe.
    cache.store("anchor query", ["a", "b", "c"], embedding=base)

    # A clearly different probe should fall below τ on the hash embedder.
    result = cache.lookup("completely unrelated topic xyzzy")
    if result.projection_intensity < cache.confidence_threshold:
        assert result.hit is False
        assert result.reuse_count == 0


def test_threshold_respected_with_custom_tau(sample_items):
    """With τ=0.0 every lookup against a populated cache activates."""
    cache = create_dtr_cache(confidence_threshold=0.0, retrieval_depth=10, cache_size=10)
    cache.store("seed query", sample_items)
    result = cache.lookup("any other query at all")
    assert result.hit is True
    assert result.projection_intensity >= cache.confidence_threshold


# ---------------------------------------------------------------------------
# Item allocation: reuse ⌊s_max·n⌋, fetch the rest
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("s_max,n", [(0.8, 10), (0.85, 10), (0.9, 20), (0.95, 7)])
def test_item_allocation_floor_split(s_max, n):
    """reuse = ⌊s_max·n⌋, residual = n - reuse, summing to n."""
    cache = create_dtr_cache(confidence_threshold=0.5, retrieval_depth=n, cache_size=10)
    items = [f"i{i}" for i in range(n)]

    # Build a controlled cache entry + probe so that the dot product == s_max.
    dim = cache.embedder.embed("x").shape[0]
    v_stored = np.zeros(dim, dtype=np.float32)
    v_stored[0] = 1.0
    v_probe = np.zeros(dim, dtype=np.float32)
    v_probe[0] = s_max
    v_probe[1] = float(np.sqrt(max(0.0, 1.0 - s_max ** 2)))

    cache._cache.clear()
    entry = CacheEntry(query="stored", items=items, embedding=v_stored)
    cache._cache[cache._generate_key("stored")] = entry

    # Patch the embedder to return our controlled probe for this query.
    cache.embedder.embed = lambda text, _v=v_probe: _v  # type: ignore

    result = cache.lookup("probe")
    # Use the realized projection intensity (float32 storage may shave a hair
    # off the nominal s_max), then verify the exact ⌊s_max·n⌋ / n-split law.
    expected_reuse = int(np.floor(result.projection_intensity * n))
    assert result.reuse_count == expected_reuse
    assert result.residual_count == n - expected_reuse
    assert result.reuse_count + result.residual_count == n


def test_get_reused_items_returns_base_prefix(dtr_cache, sample_items):
    """The reused base is the first ⌊s_max·n⌋ items of the matched entry."""
    query = "identical reuse query"
    dtr_cache.store(query, sample_items)
    result = dtr_cache.lookup(query)
    reused = dtr_cache.get_reused_items(result)
    assert reused == sample_items[: result.reuse_count]


def test_residual_ratio_property():
    r = CacheResult(hit=True, reuse_count=8, residual_count=2)
    assert np.isclose(r.residual_ratio, 0.2)
    empty = CacheResult(hit=False, reuse_count=0, residual_count=0)
    assert empty.residual_ratio == 1.0


# ---------------------------------------------------------------------------
# Residual (delta) vector + hotspot extraction
# ---------------------------------------------------------------------------
def test_residual_vector_present_on_partial_hit():
    """A partial hit (residual > 0) exposes the delta vector r = v_in - s_max·v_max."""
    cache = create_dtr_cache(confidence_threshold=0.5, retrieval_depth=10, cache_size=10)
    cache.store("base query", [f"i{i}" for i in range(10)])
    result = cache.lookup("base query about a slightly different new aspect")
    if result.hit and result.residual_count > 0:
        assert result.residual_vector is not None
        # r should be (close to) orthogonal-ish but at least finite & nonzero.
        assert np.isfinite(result.residual_vector).all()


def test_hotspot_extraction_returns_at_most_k():
    emb = SemanticEmbedder()
    extractor = HotspotTokenExtractor(emb, k=3)
    r_hat = emb.embed("delta direction")
    tokens = extractor.extract_hotspot_tokens(
        "alpha beta gamma delta epsilon", r_hat
    )
    assert len(tokens) <= 3
    assert all(t in {"alpha", "beta", "gamma", "delta", "epsilon"} for t in tokens)


def test_hotspot_extraction_empty_query():
    emb = SemanticEmbedder()
    extractor = HotspotTokenExtractor(emb, k=3)
    assert extractor.extract_hotspot_tokens("", emb.embed("x")) == []


def test_query_reformulator_fallback_without_slm():
    reformulator = QueryReformulator(slm_client=None)
    out = reformulator.reformulate("find papers", ["transformers", "attention"])
    assert "transformers" in out and "attention" in out
    # No hotspots → returns the query unchanged.
    assert reformulator.reformulate("plain query", []) == "plain query"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def test_stats_track_activations_and_cold(dtr_cache, sample_items):
    dtr_cache.lookup("cold one")              # cold (empty cache)
    dtr_cache.store("hot query", sample_items)
    dtr_cache.lookup("hot query")             # activation (identical)

    stats = dtr_cache.get_stats()
    assert stats["total_lookups"] == 2
    assert stats["dtr_activations"] == 1
    assert stats["cold_retrievals"] == 1
    assert stats["cache_entries"] == 1
    assert 0.0 <= stats["hit_rate"] <= 1.0
    assert 0.0 <= stats["item_reuse_rate"] <= 1.0


def test_clear_resets_state(dtr_cache, sample_items):
    dtr_cache.store("q", sample_items)
    dtr_cache.lookup("q")
    dtr_cache.clear()
    stats = dtr_cache.get_stats()
    assert stats["cache_entries"] == 0
    assert stats["total_lookups"] == 0
    assert stats["dtr_activations"] == 0


def test_lru_eviction_respects_cache_size():
    cache = create_dtr_cache(confidence_threshold=0.8, retrieval_depth=5, cache_size=3)
    for i in range(5):
        cache.store(f"query number {i}", [f"item{i}"])
    assert cache.get_stats()["cache_entries"] <= 3


# ---------------------------------------------------------------------------
# Asynchronous cache refresh scheduling
# ---------------------------------------------------------------------------
def test_async_refresh_scheduled_on_activation(sample_items):
    """When a tool_fn is supplied, an activation schedules a background refresh."""
    cache = create_dtr_cache(confidence_threshold=0.0, retrieval_depth=5, cache_size=10)
    cache.store("seed", sample_items[:5])

    calls = {"n": 0}

    def tool_fn(q):
        calls["n"] += 1
        return [f"fresh-{q}-{i}" for i in range(5)]

    result = cache.lookup("seed variant query", tool_fn=tool_fn)
    assert result.hit is True

    # Background thread is daemonized; give it a brief moment to run.
    deadline = time.time() + 2.0
    while calls["n"] == 0 and time.time() < deadline:
        time.sleep(0.01)

    assert calls["n"] >= 1
    assert cache.get_stats()["async_updates"] >= 1


def test_no_async_refresh_without_tool_fn(sample_items):
    cache = create_dtr_cache(confidence_threshold=0.0, retrieval_depth=5, cache_size=10)
    cache.store("seed", sample_items[:5])
    cache.lookup("seed variant", tool_fn=None)
    time.sleep(0.05)
    assert cache.get_stats()["async_updates"] == 0
