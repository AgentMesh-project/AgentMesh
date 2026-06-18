#!/usr/bin/env python3
"""
BPP attention-complexity microbench -- the CONSUME (gather) speedup.

In the supervisor-worker gather, full prefill attends all-to-all over the
prefix + every worker branch, so its attention cost grows QUADRATICALLY with
the number of branches N. BPA prefills each branch only against the shared
prefix (cross-branch attention masked), so its cost grows LINEARLY in N. This
microbench shows the O(N^2) -> O(N) reduction as N grows.

Complexity formulas (from agentmesh.mechanisms.bpp.attention)
------------------------------------------------------------
  Full prefill : O( L_prefix * sum_i L_i  +  sum_i sum_{j>=i} L_i * L_j )
  BPA          : O( max_i ( L_prefix * L_i  +  L_i^2 ) )

These are exactly BranchParallelManager.get_complexity_info(); we replicate the
arithmetic here so the microbench runs with NO torch (the manager class needs
torch only to instantiate). With --use-manager and torch installed, the script
cross-checks against the real BranchParallelManager.

Paper mapping
-------------
  "BPP: attention complexity O(N^2)->O(N) in branch count N; 8-61% time
   reduction." Ablation: BPP contributes 19-61% of the CONSUME reduction.

Output: a table of full vs BPA cost and theoretical speedup vs branch count N,
plus an optional PNG plot. A small wall-clock NumPy attention-mass microbench
is also run to show the FLOP trend empirically (still CPU-only).

Fast default: synthetic, no GPU / torch, < 5s.

Usage:
    python -m benchmarks.bpp.bpp_speedup
    python -m benchmarks.bpp.bpp_speedup --prefix 900 --branch-len 620 --max-branches 12
    python -m benchmarks.bpp.bpp_speedup --use-manager   # cross-check w/ torch
    python -m benchmarks.bpp.bpp_speedup --plot out.png
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np


def complexity_full(prefix_len: int, branch_lens: List[int]) -> int:
    """O( L_prefix * sum L_i + sum_i sum_{j>=i} L_i * L_j )."""
    sum_l = sum(branch_lens)
    cross = prefix_len * sum_l
    inter = sum(
        branch_lens[i] * branch_lens[j]
        for i in range(len(branch_lens))
        for j in range(i, len(branch_lens))
    )
    return cross + inter


def complexity_bpa(prefix_len: int, branch_lens: List[int]) -> int:
    """O( max_i ( L_prefix * L_i + L_i^2 ) ) -- the critical-path branch."""
    if not branch_lens:
        return 0
    return max(prefix_len * l + l * l for l in branch_lens)


def cross_check_manager(prefix_len: int, branch_lens: List[int]) -> Dict:
    """Cross-check the formulas against the real BranchParallelManager (needs torch)."""
    from agentmesh.mechanisms.bpp import BranchParallelManager, SupervisorContext, WorkerOutput
    mgr = BranchParallelManager(num_layers=1, num_heads=1, head_dim=8, device="cpu")
    mgr.set_supervisor_context(SupervisorContext(tokens=list(range(prefix_len))))
    for i, l in enumerate(branch_lens):
        mgr.prefill_worker(WorkerOutput(worker_id=f"w{i}", tokens=list(range(l))))
    return mgr.get_complexity_info()


def empirical_mass_flops(prefix_len: int, branch_lens: List[int]) -> Dict[str, float]:
    """
    Tiny CPU wall-clock proxy: compute attention-mass dot products for the
    full vs BPA pattern using NumPy, timing the score computation. This is a
    FLOP-trend illustration only (1-dim features), not a kernel benchmark.
    """
    rng = np.random.default_rng(0)
    d = 8
    prefix = rng.standard_normal((prefix_len, d))
    branches = [rng.standard_normal((l, d)) for l in branch_lens]

    # Full: each branch queries prefix + all (earlier+own) branch keys.
    t0 = time.perf_counter()
    seen_keys = [prefix]
    for b in branches:
        keys = np.concatenate(seen_keys + [b], axis=0)
        _ = b @ keys.T  # (L_i, |keys|)
        seen_keys.append(b)
    full_ms = (time.perf_counter() - t0) * 1000

    # BPA: each branch queries only prefix + its own keys.
    t1 = time.perf_counter()
    for b in branches:
        keys = np.concatenate([prefix, b], axis=0)
        _ = b @ keys.T
    bpa_ms = (time.perf_counter() - t1) * 1000

    return {"full_ms": full_ms, "bpa_ms": bpa_ms,
            "empirical_speedup": full_ms / bpa_ms if bpa_ms > 0 else float("inf")}


def maybe_plot(rows: List[Dict], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    N = [r["N"] for r in rows]
    full = [r["full_cost"] for r in rows]
    bpa = [r["bpa_cost"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))
    ax1.plot(N, full, "o-", label="full prefill  O(N^2)")
    ax1.plot(N, bpa, "s--", label="BPA  O(N)")
    ax1.set_xlabel("branch count N"); ax1.set_ylabel("attention cost (units)")
    ax1.set_title("Attention cost vs N"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.plot(N, [r["speedup"] for r in rows], "d-", color="green")
    ax2.set_xlabel("branch count N"); ax2.set_ylabel("speedup (full / BPA)")
    ax2.set_title("BPA speedup vs N"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="BPP O(N^2)->O(N) complexity microbench.")
    parser.add_argument("--prefix", type=int, default=900, help="Supervisor prefix length.")
    parser.add_argument("--branch-len", type=int, default=620, help="Tokens per worker branch.")
    parser.add_argument("--max-branches", type=int, default=10)
    parser.add_argument("--use-manager", action="store_true",
                        help="Cross-check formulas against BranchParallelManager (needs torch).")
    parser.add_argument("--empirical", action="store_true",
                        help="Also run the tiny NumPy wall-clock FLOP-trend proxy.")
    parser.add_argument("--plot", type=str, default=None, help="PNG path (needs matplotlib).")
    args = parser.parse_args()

    print("=" * 64)
    print("BPP attention-complexity microbench  [O(N^2) -> O(N)]")
    print("Paper: 8-61% CONSUME-stage time reduction in branch count N.")
    print("=" * 64)
    print(f"prefix={args.prefix}, branch_len={args.branch_len}, "
          f"max_branches={args.max_branches}\n")

    rows: List[Dict] = []
    print(f"{'N':>3}  {'full_cost':>14}  {'bpa_cost':>12}  {'speedup':>8}  "
          f"{'time_reduction':>14}")
    print("-" * 60)
    for n in range(1, args.max_branches + 1):
        branch_lens = [args.branch_len] * n
        full = complexity_full(args.prefix, branch_lens)
        bpa = complexity_bpa(args.prefix, branch_lens)
        speedup = full / bpa if bpa else float("inf")
        reduction = 1.0 - bpa / full if full else 0.0
        rows.append({"N": n, "full_cost": full, "bpa_cost": bpa,
                     "speedup": speedup, "time_reduction": reduction})
        print(f"{n:>3}  {full:>14,}  {bpa:>12,}  {speedup:>8.2f}  {reduction*100:>13.1f}%")

    print(f"\nObserved scaling: full grows ~quadratically, BPA stays flat per "
          f"branch -> speedup grows ~linearly with N (O(N^2)->O(N)).")

    if args.use_manager:
        try:
            info = cross_check_manager(args.prefix, [args.branch_len] * min(3, args.max_branches))
            print(f"\nCross-check (BranchParallelManager, N={info.get('num_branches')}): "
                  f"full={info['full_complexity']:,}, bpa={info['bpa_complexity']:,}, "
                  f"speedup={info['speedup']:.2f}")
        except Exception as e:
            print(f"\n[--use-manager] torch unavailable, skipped cross-check: "
                  f"{type(e).__name__}: {e}")

    if args.empirical:
        emp = empirical_mass_flops(args.prefix, [args.branch_len] * args.max_branches)
        print(f"\nEmpirical NumPy proxy (N={args.max_branches}): "
              f"full={emp['full_ms']:.2f}ms, bpa={emp['bpa_ms']:.2f}ms, "
              f"speedup={emp['empirical_speedup']:.2f}x")

    if args.plot:
        if maybe_plot(rows, Path(args.plot)):
            print(f"\nWrote plot -> {args.plot}")
        else:
            print("\nmatplotlib not installed -- skipped PNG plot (table above holds the data).")


if __name__ == "__main__":
    main()
