#!/usr/bin/env python3
"""
DTR confidence-threshold (tau) sweep.

Reproduces the paper's tau sensitivity study: exact-match score (and the
latency / backend-load reductions) as the DTR confidence gate tau sweeps over
[0.75, 0.95]. Lower tau activates DTR more aggressively (more reuse, lower
load, but more risk of serving a stale base); higher tau is conservative.

Paper mapping
-------------
  "DTR ... default tau=0.8, sweep 0.75-0.95; metric = exact-match score."

Output: a numeric table and CSV always; a PNG line plot (exact-match vs tau)
only when matplotlib is installed (otherwise an ASCII sparkline + a note).

Fast default: synthetic, no GPU / LLM, < 10s.

Usage:
    python -m benchmarks.dtr.threshold_sweep
    python -m benchmarks.dtr.threshold_sweep --overlap 0.7 --plot out.png
    python -m benchmarks.dtr.threshold_sweep --trace your_tool_log.jsonl   # optional
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

# Allow direct execution (``python benchmarks/dtr/threshold_sweep.py``) by putting
# the repository root on sys.path, alongside ``python -m benchmarks.dtr.threshold_sweep``.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.dtr.benchmark_dtr import (
    BagOfWordsEmbedder,
    DTRCache,
    load_trace,
    run_dtr,
    synthesize_stream,
)


def sweep(
    stream,
    thresholds: List[float],
    real_embedder: bool,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    depth = stream[0].depth if stream else 10
    for tau in thresholds:
        cache = DTRCache(confidence_threshold=tau, retrieval_depth=depth)
        if not real_embedder:
            bow = BagOfWordsEmbedder()
            cache.embedder = bow
            cache.hotspot_extractor.embedder = bow
        m = run_dtr(stream, cache).finalize(len(stream))
        m["tau"] = round(tau, 3)
        rows.append(m)
    return rows


def ascii_sparkline(values: List[float]) -> str:
    blocks = "_.-=+*#@"  # low -> high
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return "".join(blocks[min(len(blocks) - 1, int((v - lo) / span * (len(blocks) - 1)))]
                   for v in values)


def maybe_plot(rows: List[Dict[str, float]], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    taus = [r["tau"] for r in rows]
    em = [r["exact_match_score"] for r in rows]
    load = [r["backend_load_reduction"] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(taus, em, "o-", label="exact-match score")
    ax.plot(taus, load, "s--", label="backend load reduction")
    ax.set_xlabel(r"confidence threshold $\tau$")
    ax.set_ylabel("score / reduction")
    ax.set_title("DTR threshold sweep")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="DTR tau sweep (0.75-0.95).")
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--overlap", type=float, default=0.65)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--tau-min", type=float, default=0.75)
    parser.add_argument("--tau-max", type=float, default=0.95)
    parser.add_argument("--tau-step", type=float, default=0.05)
    parser.add_argument("--trace", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--real-embedder", action="store_true")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--plot", type=str, default=None,
                        help="PNG path for the exact-match-vs-tau plot (needs matplotlib).")
    args = parser.parse_args()

    print("=" * 64)
    print("DTR confidence-threshold (tau) sweep  [paper: 0.75-0.95]")
    print("=" * 64)

    if args.trace:
        stream = load_trace(Path(args.trace), args.depth)
        print(f"Replaying trace: {args.trace} ({len(stream)} queries)")
    else:
        stream = synthesize_stream(args.num_queries, args.overlap, args.depth, args.seed)
        print(f"Synthetic stream: {len(stream)} queries, overlap={args.overlap}")
    if not stream:
        raise SystemExit("Empty query stream.")

    thresholds = []
    t = args.tau_min
    while t <= args.tau_max + 1e-9:
        thresholds.append(round(t, 3))
        t += args.tau_step

    rows = sweep(stream, thresholds, args.real_embedder)

    print(f"\n{'tau':>6}  {'EM-score':>9}  {'hit':>6}  {'load-red':>9}  "
          f"{'lat-red':>8}  {'speedup':>8}")
    print("-" * 56)
    for r in rows:
        print(f"{r['tau']:>6.2f}  {r['exact_match_score']:>9.3f}  {r['hit_rate']:>6.3f}  "
              f"{r['backend_load_reduction']:>9.3f}  {r['latency_reduction_vs_nc']:>8.3f}  "
              f"{r['speedup_vs_nc']:>8.3f}")

    print(f"\nexact-match vs tau : {ascii_sparkline([r['exact_match_score'] for r in rows])}"
          f"  ({thresholds[0]:.2f} -> {thresholds[-1]:.2f})")
    print(f"load-reduction     : {ascii_sparkline([r['backend_load_reduction'] for r in rows])}")

    if args.csv:
        keys = ["tau"] + [k for k in rows[0] if k != "tau"]
        with Path(args.csv).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote CSV -> {args.csv}")

    if args.plot:
        if maybe_plot(rows, Path(args.plot)):
            print(f"Wrote plot -> {args.plot}")
        else:
            print("matplotlib not installed -- skipped PNG plot (CSV/table above hold the data).")


if __name__ == "__main__":
    main()
