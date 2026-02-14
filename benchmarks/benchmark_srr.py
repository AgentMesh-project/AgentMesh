"""
Benchmark for SRR (Semantic Residual Retrieval).

Measures:
  - Cache lookup latency (embedding + projection + confidence gating)
  - Hotspot extraction time
  - SLM reformulation time
  - End-to-end SRR activation latency
  - Precision at various τ thresholds

Usage:
    python benchmarks/benchmark_srr.py [--num-queries 1000] [--cache-size 512]
"""

import argparse
import time
import statistics
from typing import Dict, List
import numpy as np

from agentmesh.mechanisms.srr import (
    SRRCache,
    SemanticEmbedder,
    HotspotTokenExtractor,
)


def benchmark_embedding(embedder: SemanticEmbedder, texts: List[str]) -> Dict:
    """Benchmark embedding latency."""
    latencies = []
    for text in texts:
        t0 = time.perf_counter()
        embedder.embed(text)
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": statistics.mean(latencies),
        "p50_ms": statistics.median(latencies),
        "p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
        "total_queries": len(texts),
    }


def benchmark_lookup(
    cache: SRRCache,
    queries: List[str],
) -> Dict:
    """Benchmark cache lookup latency and hit rate."""
    latencies = []
    hits = 0
    for q in queries:
        t0 = time.perf_counter()
        result = cache.lookup(q)
        latencies.append((time.perf_counter() - t0) * 1000)
        if result.hit:
            hits += 1

    return {
        "mean_ms": statistics.mean(latencies),
        "p50_ms": statistics.median(latencies),
        "p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
        "hit_rate": hits / len(queries),
        "total_queries": len(queries),
    }


def benchmark_hotspot(
    extractor: HotspotTokenExtractor,
    embedder: SemanticEmbedder,
    num_trials: int = 500,
) -> Dict:
    """Benchmark hotspot token extraction."""
    latencies = []
    # Determine embedding dimension from a sample embedding
    dim = len(embedder.embed("test"))
    for _ in range(num_trials):
        residual_hat = np.random.randn(dim).astype(np.float32)
        residual_hat /= np.linalg.norm(residual_hat)
        query = " ".join([f"token_{i}" for i in range(20)])

        t0 = time.perf_counter()
        extractor.extract_hotspot_tokens(query, residual_hat)
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": statistics.mean(latencies),
        "p50_ms": statistics.median(latencies),
        "p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
    }


def main():
    parser = argparse.ArgumentParser(description="SRR Benchmark")
    parser.add_argument("--num-queries", type=int, default=500)
    parser.add_argument("--cache-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args()

    print("=" * 60)
    print("SRR Benchmark")
    print("=" * 60)

    embedder = SemanticEmbedder()
    cache = SRRCache(
        confidence_threshold=args.threshold,
        cache_size=args.cache_size,
    )

    # Seed cache with some entries
    seed_queries = [f"What is the role of component {i} in distributed systems?" for i in range(50)]
    for q in seed_queries:
        cache.store(q, [f"Response item for {q}"])

    # Generate test queries (mix of similar and dissimilar)
    test_queries = (
        [f"What is the function of component {i} in distributed systems?" for i in range(args.num_queries // 2)]
        + [f"Unrelated topic number {i} about cooking" for i in range(args.num_queries // 2)]
    )

    print(f"\nConfig: τ={args.threshold}, cache_size={args.cache_size}, queries={len(test_queries)}")

    # Embedding benchmark
    print("\n--- Embedding Latency ---")
    emb_stats = benchmark_embedding(embedder, test_queries[:100])
    for k, v in emb_stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # Lookup benchmark
    print("\n--- Lookup Latency ---")
    lookup_stats = benchmark_lookup(cache, test_queries)
    for k, v in lookup_stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # Hotspot benchmark
    print("\n--- Hotspot Extraction ---")
    extractor = HotspotTokenExtractor(embedder, k=3)
    hotspot_stats = benchmark_hotspot(extractor, embedder)
    for k, v in hotspot_stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
