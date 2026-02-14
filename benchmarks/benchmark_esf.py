"""
Benchmark for ESF (Elastic Semantic Flow).

Measures:
  - OSE convergence speed (steps to stable θ)
  - Controller overhead per step
  - θ trajectory under synthetic workloads

Usage:
    python benchmarks/benchmark_esf.py [--steps 100] [--initial-theta 512]
"""

import argparse
import time
import statistics
import math
from typing import Dict, List

from agentmesh.mechanisms.esf import (
    ESFController,
    Observation,
    StateObserver,
    create_esf_controller,
)


def benchmark_controller_overhead(controller: ESFController, steps: int = 1000) -> Dict:
    """Measure per-step compute_next_theta latency."""
    latencies = []
    for i in range(steps):
        t_p = 50.0 + 5.0 * math.sin(i / 10)
        t_c = 45.0 + 3.0 * math.cos(i / 8)
        delta = t_p - t_c

        t0 = time.perf_counter()
        controller.compute_next_theta(t_p, t_c, delta, i)
        latencies.append((time.perf_counter() - t0) * 1e6)  # microseconds

    return {
        "mean_us": statistics.mean(latencies),
        "p50_us": statistics.median(latencies),
        "p99_us": sorted(latencies)[int(len(latencies) * 0.99)],
        "total_steps": steps,
    }


def benchmark_convergence(
    initial_theta: int,
    target_ratio: float = 1.0,
    max_steps: int = 50,
) -> Dict:
    """Simulate workload and measure steps to convergence.

    target_ratio: T_p / T_c ratio at equilibrium.
    Convergence = θ changes by < 1% for 5 consecutive steps.
    """
    controller = create_esf_controller(
        initial_theta=initial_theta,
        min_theta=64,
        max_theta=4096,
    )

    trajectory = []
    converged_at = None
    stable_count = 0

    for step in range(max_steps):
        theta = controller.current_theta
        # Synthetic: producer time scales linearly with θ
        t_p = theta * 0.1 * target_ratio
        t_c = theta * 0.1
        delta = t_p - t_c

        new_theta = controller.compute_next_theta(t_p, t_c, delta, step)
        trajectory.append(new_theta)

        if step > 0 and abs(new_theta - trajectory[-2]) / max(trajectory[-2], 1) < 0.01:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= 5 and converged_at is None:
            converged_at = step - 4

    return {
        "converged_at_step": converged_at,
        "final_theta": trajectory[-1] if trajectory else initial_theta,
        "initial_theta": initial_theta,
        "target_ratio": target_ratio,
        "trajectory_len": len(trajectory),
    }


def benchmark_workload_profiles(initial_theta: int = 512, steps: int = 100) -> List[Dict]:
    """Run ESF under different workload profiles."""
    profiles = [
        ("balanced", lambda s: (50.0, 50.0)),
        ("production_bound", lambda s: (80.0, 40.0)),
        ("consumption_bound", lambda s: (30.0, 70.0)),
        ("oscillating", lambda s: (50.0 + 20 * math.sin(s / 5), 50.0)),
        ("ramp_up", lambda s: (30.0 + s * 0.5, 50.0)),
    ]

    results = []
    for name, workload_fn in profiles:
        controller = create_esf_controller(
            initial_theta=initial_theta, min_theta=64, max_theta=4096
        )
        trajectory = []
        for step in range(steps):
            t_p, t_c = workload_fn(step)
            delta = t_p - t_c
            theta = controller.compute_next_theta(t_p, t_c, delta, step)
            trajectory.append(theta)

        results.append({
            "profile": name,
            "final_theta": trajectory[-1],
            "min_theta": min(trajectory),
            "max_theta": max(trajectory),
            "mean_theta": statistics.mean(trajectory),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="ESF Benchmark")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--initial-theta", type=int, default=512)
    args = parser.parse_args()

    print("=" * 60)
    print("ESF Benchmark")
    print("=" * 60)

    # Controller overhead
    print("\n--- Controller Overhead ---")
    controller = create_esf_controller(
        initial_theta=args.initial_theta, min_theta=64, max_theta=4096
    )
    overhead = benchmark_controller_overhead(controller, steps=args.steps)
    for k, v in overhead.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # Convergence
    print("\n--- Convergence Speed ---")
    for ratio in [0.8, 1.0, 1.2, 1.5, 2.0]:
        conv = benchmark_convergence(args.initial_theta, target_ratio=ratio)
        step_str = str(conv["converged_at_step"]) if conv["converged_at_step"] is not None else "N/A"
        print(f"  ratio={ratio:.1f}: converge@step={step_str}, "
              f"final_θ={conv['final_theta']:.0f}")

    # Workload profiles
    print("\n--- Workload Profiles ---")
    profiles = benchmark_workload_profiles(args.initial_theta, args.steps)
    for p in profiles:
        print(f"  {p['profile']:20s}: θ_final={p['final_theta']:.0f}, "
              f"range=[{p['min_theta']:.0f}, {p['max_theta']:.0f}], "
              f"mean={p['mean_theta']:.0f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
