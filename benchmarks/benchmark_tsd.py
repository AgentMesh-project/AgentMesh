"""
Benchmark for TSD (Topology-aware Semantic Decoupling).

Measures:
  - Inter-worker mask generation time
  - RoPE realignment latency
  - Parallel prefill throughput vs. sequential
  - KV stitch + realign latency

Usage:
    python benchmarks/benchmark_tsd.py [--num-workers 8] [--seq-len 512]
"""

import argparse
import time
import statistics
import torch

from agentmesh.mechanisms.tsd import (
    TDAManager,
    SupervisorContext,
    WorkerOutput,
    InterWorkerMaskGenerator,
    RoPEAligner,
)


def benchmark_mask_generation(num_workers: int, seq_len: int, trials: int = 100) -> dict:
    """Benchmark inter-worker mask generation."""
    gen = InterWorkerMaskGenerator()
    prefix_len = 64  # typical supervisor prefix length

    latencies = []
    for _ in range(trials):
        t0 = time.perf_counter()
        # Generate mask for each worker individually (as per actual API)
        for i in range(num_workers):
            gen.generate_worker_mask(prefix_len=prefix_len, worker_len=seq_len)
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": statistics.mean(latencies),
        "p50_ms": statistics.median(latencies),
        "p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
        "total_tokens": num_workers * seq_len,
    }


def benchmark_rope_realignment(
    head_dim: int, num_heads: int, seq_len: int, trials: int = 100
) -> dict:
    """Benchmark RoPE positional realignment."""
    aligner = RoPEAligner(head_dim=head_dim, rope_base=10000.0)
    key = torch.randn(num_heads, seq_len, head_dim)

    latencies = []
    for offset in range(trials):
        t0 = time.perf_counter()
        aligner.realign(key, from_pos=0, to_pos=offset)
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": statistics.mean(latencies),
        "p50_ms": statistics.median(latencies),
        "p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
    }


def benchmark_full_pipeline(
    num_workers: int, num_layers: int, num_heads: int, head_dim: int, seq_len: int
) -> dict:
    """Benchmark TDAManager full pipeline: prefill workers + stitch."""
    manager = TDAManager(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        max_parallel_workers=num_workers,
        rope_base=10000.0,
        device="cpu",
    )

    prefix_len = 64
    prefix_kv = {
        l: (
            torch.randn(num_heads, prefix_len, head_dim),
            torch.randn(num_heads, prefix_len, head_dim),
        )
        for l in range(num_layers)
    }
    manager.set_supervisor_context(SupervisorContext(
        tokens=list(range(prefix_len)),
        kv_states=prefix_kv,
    ))

    # Prefill workers
    t0 = time.perf_counter()
    for i in range(num_workers):
        wo = WorkerOutput(
            worker_id=f"w{i}",
            tokens=list(range(seq_len)),
        )
        manager.prefill_worker(wo)
    prefill_ms = (time.perf_counter() - t0) * 1000

    # Stitch + realign
    t1 = time.perf_counter()
    result = manager.stitch_and_realign()
    stitch_ms = (time.perf_counter() - t1) * 1000

    return {
        "prefill_ms": prefill_ms,
        "stitch_ms": stitch_ms,
        "total_ms": prefill_ms + stitch_ms,
        "num_workers": num_workers,
        "total_tokens": prefix_len + num_workers * seq_len,
    }


def main():
    parser = argparse.ArgumentParser(description="TSD Benchmark")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    args = parser.parse_args()

    print("=" * 60)
    print("TSD Benchmark")
    print("=" * 60)

    print(f"\nConfig: workers={args.num_workers}, seq_len={args.seq_len}")
    print(f"  layers={args.num_layers}, heads={args.num_heads}, head_dim={args.head_dim}")

    # Mask generation
    print("\n--- Mask Generation ---")
    mask_stats = benchmark_mask_generation(args.num_workers, args.seq_len)
    for k, v in mask_stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # RoPE realignment
    print("\n--- RoPE Realignment ---")
    rope_stats = benchmark_rope_realignment(args.head_dim, args.num_heads, args.seq_len)
    for k, v in rope_stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # Full pipeline
    print("\n--- Full Pipeline (prefill + stitch) ---")
    pipe_stats = benchmark_full_pipeline(
        args.num_workers, args.num_layers, args.num_heads, args.head_dim, args.seq_len
    )
    for k, v in pipe_stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # Scaling test
    print("\n--- Worker Scaling ---")
    for n in [1, 2, 4, 8]:
        if n > args.num_workers * 2:
            break
        stats = benchmark_full_pipeline(
            n, args.num_layers, args.num_heads, args.head_dim, args.seq_len
        )
        print(f"  workers={n}: total={stats['total_ms']:.2f}ms "
              f"(prefill={stats['prefill_ms']:.2f}, stitch={stats['stitch_ms']:.2f})")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
