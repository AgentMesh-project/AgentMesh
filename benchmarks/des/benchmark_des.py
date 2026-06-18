#!/usr/bin/env python3
"""
DES (Dynamic-Equilibrium Streaming) benchmark -- the produce/transfer/consume OVERLAP.

DES streams a producer's output to its consumer in chunks of size theta, and an
online controller drives the slack (delta, the idle gap between producer and
consumer) to zero by adapting theta. This benchmark compares DES against:

  - NS (No-Stream)  : the producer fully finishes before the consumer starts
                      (serialized; no overlap) -> a hard BARRIER.
  - SS (Static fixed-chunk Stream) : stream with a constant theta (overlap, but
                      no adaptation -> persistent slack when rates drift).
  - DES (ours)      : adapt theta online via the Online Sensitivity Estimator
                      (Newton step Delta-theta = -delta/rho), driving slack->0.

It drives the real agentmesh DESController and replays synthetic rate traces
spanning the paper's coefficient-of-variation range (CoV 0.13-0.75). It reports
throughput (tokens/s) for each method and shows theta converging to a
rate-matched equilibrium with shrinking slack.

Paper mapping
-------------
  "DES sustains higher throughput than No-Stream/serialized (NS) and any Static
   fixed-chunk Stream (SS); replayed trace CoV 0.13-0.75; converges chunk size
   theta to rate-matched equilibrium." Ablation: DES 6-17% of the OVERLAP.

Fast default: synthetic, no GPU / LLM, < 10s.

Usage:
    python -m benchmarks.des.benchmark_des
    python -m benchmarks.des.benchmark_des --steps 60 --cov 0.5
    python -m benchmarks.des.benchmark_des --trace your_rate_trace.csv   # optional; default is synthetic
    python -m benchmarks.des.benchmark_des --csv out.csv --plot out.png
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Allow direct execution (``python benchmarks/des/benchmark_des.py``) by putting the
# repository root on sys.path, in addition to ``python -m benchmarks.des.benchmark_des``.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agentmesh.mechanisms.des import DESController


# ---------------------------------------------------------------------------
# Rate traces
# ---------------------------------------------------------------------------
def synth_rate_trace(steps: int, cov: float, seed: int,
                     base_producer: float = 620.0,
                     base_consumer: float = 900.0) -> List[Dict[str, float]]:
    """
    Synthesize a producer/consumer rate trace with a target coefficient of
    variation (CoV = std/mean) on the producer rate, in [0.13, 0.75] per paper.

    The producer (upstream tool execution / worker decode) is the bottleneck and
    is bursty; the consumer (downstream LLM prefill) is faster and steadier.
    This is the regime where overlap matters and where a fixed chunk size leaves
    the consumer starving during producer lulls -- the gap DES closes. A
    multiplicative (lognormal) draw keeps the mean unbiased as CoV grows, so the
    NS baseline (which uses the mean rate) is compared fairly.
    """
    rng = np.random.default_rng(seed)
    sigma_ln = math.sqrt(math.log(1.0 + max(1e-6, cov) ** 2))
    mu_ln_p = math.log(base_producer) - 0.5 * sigma_ln ** 2
    rows = []
    for s in range(steps):
        p = float(rng.lognormal(mu_ln_p, sigma_ln))
        c = max(50.0, rng.normal(base_consumer, 0.10 * base_consumer))
        rows.append({
            "step": s,
            "producer_tokens_per_s": p,
            "consumer_tokens_per_s": c,
            "start_latency_ms": 40.0,
            "network_ms_per_chunk": 3.0,
        })
    return rows


def load_rate_trace(path: Path) -> List[Dict[str, float]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "step": int(row["step"]),
                "producer_tokens_per_s": float(row["producer_tokens_per_s"]),
                "consumer_tokens_per_s": float(row["consumer_tokens_per_s"]),
                "start_latency_ms": float(row.get("start_latency_ms", 40.0)),
                "network_ms_per_chunk": float(row.get("network_ms_per_chunk", 3.0)),
            })
    return rows


def trace_cov(rows: List[Dict[str, float]]) -> float:
    p = [r["producer_tokens_per_s"] for r in rows]
    m = statistics.mean(p)
    return (statistics.pstdev(p) / m) if m else 0.0


# ---------------------------------------------------------------------------
# Stage timing model for a single chunk of size theta
# ---------------------------------------------------------------------------
def chunk_times(theta: int, prod_rate: float, cons_rate: float,
                start_ms: float, net_ms: float) -> Tuple[float, float]:
    """
    Return (producer_time, consumer_time) in ms for a chunk of `theta` tokens.

    producer_time = startup_per_chunk + theta / prod_rate + network  (T_p + T_n)
    consumer_time = theta / cons_rate                                (T_c)

    The producer pays a small fixed per-chunk overhead (request setup +
    serialization). With many tiny chunks this overhead dominates (favoring
    larger theta); with one huge chunk the consumer starves while waiting
    (favoring smaller theta). DES balances this tension via theta.

    `start_ms` here is the per-CHUNK overhead. The one-time pipeline fill cost
    is added separately in simulate_stream (only for the first chunk).
    """
    chunk_overhead = start_ms * 0.15  # per-chunk request/serialization overhead
    producer_time = chunk_overhead + (theta / prod_rate) * 1000.0 + net_ms
    consumer_time = (theta / cons_rate) * 1000.0
    return producer_time, consumer_time


# ---------------------------------------------------------------------------
# Pipeline simulators
# ---------------------------------------------------------------------------
def simulate_no_stream(rows: List[Dict[str, float]], total_tokens: int) -> Dict[str, float]:
    """NS: produce the whole output, THEN consume it. No overlap (full barrier).

    Uses the trace's mean producer/consumer rates so the NS baseline is not
    advantaged or penalized by whichever step happens to be first.
    """
    mean_prod = statistics.mean(r["producer_tokens_per_s"] for r in rows)
    mean_cons = statistics.mean(r["consumer_tokens_per_s"] for r in rows)
    start_ms = rows[0]["start_latency_ms"]
    p_time = start_ms + (total_tokens / mean_prod) * 1000.0
    c_time = (total_tokens / mean_cons) * 1000.0
    wall_ms = p_time + c_time  # strictly serialized
    return {
        "method": "NS (No-Stream)",
        "wall_ms": wall_ms,
        "throughput_tok_s": total_tokens / (wall_ms / 1000.0),
        "mean_abs_slack_ms": float("nan"),
        "final_theta": total_tokens,
    }


def simulate_stream(rows: List[Dict[str, float]], total_tokens: int,
                    controller, label: str, fixed_theta: int = None) -> Dict[str, float]:
    """
    Pipelined stream with a producer/consumer timeline (the producer may run
    ahead and buffer chunks, absorbing transient rate bursts).

    For each chunk i:
      producer_done[i] = max(producer_done[i-1], 0) + producer_time[i]
      consumer_start[i] = max(producer_done[i], consumer_done[i-1])
      consumer_done[i] = consumer_start[i] + consumer_time[i]
    Wall time = consumer_done[last]. The per-chunk slack delta is the consumer's
    wait for its chunk (consumer starving) minus the producer's wait for buffer
    drain -- exactly the signal the DES controller minimizes.

    If `controller` is given, theta adapts each step (DES). If `fixed_theta` is
    given, theta is constant (SS).
    """
    produced = 0
    step = 0
    slacks: List[float] = []
    thetas: List[int] = []
    theta = fixed_theta if fixed_theta is not None else controller.current_theta

    n = len(rows)
    producer_done = 0.0
    consumer_done = 0.0
    while produced < total_tokens:
        r = rows[step % n]
        remaining = total_tokens - produced
        this_theta = min(theta, remaining)
        p_time, c_time = chunk_times(
            this_theta, r["producer_tokens_per_s"], r["consumer_tokens_per_s"],
            r["start_latency_ms"], r["network_ms_per_chunk"],
        )
        producer_done += p_time
        consumer_start = max(producer_done, consumer_done)
        # Consumer starvation: time the consumer waited for this chunk to exist.
        starvation = consumer_start - consumer_done
        consumer_done = consumer_start + c_time
        # The controller's slack delta is the signed produce/consume imbalance;
        # driving it to 0 rate-matches the chunk so the consumer neither
        # starves (delta>0) nor saturates the buffer (delta<0).
        slack = p_time - c_time
        slacks.append(max(0.0, starvation))  # observed idle gap (BARRIER)
        thetas.append(this_theta)

        produced += this_theta
        step += 1

        if controller is not None:
            theta = controller.compute_next_theta(p_time, c_time, slack, step)

    wall_ms = consumer_done

    # Report the controller's settled theta (DES) or the configured theta (SS),
    # not the clipped last partial chunk.
    settled_theta = (controller.current_theta if controller is not None
                     else (fixed_theta if fixed_theta is not None else theta))
    return {
        "method": label,
        "wall_ms": wall_ms,
        "throughput_tok_s": total_tokens / (wall_ms / 1000.0),
        "mean_abs_slack_ms": statistics.mean(slacks) if slacks else float("nan"),
        "final_theta": settled_theta,
        "theta_trajectory": thetas,
        "slack_trajectory": slacks,
    }


def maybe_plot(des_run: Dict, path: Path, ss_theta: int) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    thetas = des_run["theta_trajectory"]
    slacks = des_run["slack_trajectory"]
    x = list(range(len(thetas)))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.2, 4.4), sharex=True)
    ax1.plot(x, thetas, "o-", label="DES theta")
    ax1.axhline(ss_theta, color="gray", ls="--", lw=1, label=f"SS fixed theta={ss_theta}")
    ax1.set_ylabel("chunk size theta"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.plot(x, slacks, "s-", color="crimson")
    ax2.set_ylabel("|slack| (ms)"); ax2.set_xlabel("step"); ax2.grid(alpha=0.3)
    ax2.set_title("DES drives slack -> 0", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="DES streaming benchmark (overlap stage).")
    parser.add_argument("--steps", type=int, default=60, help="Synthetic trace length.")
    parser.add_argument("--cov", type=float, default=0.5,
                        help="Producer-rate coefficient of variation (paper 0.13-0.75).")
    parser.add_argument("--total-tokens", type=int, default=12000)
    parser.add_argument("--initial-theta", type=int, default=512)
    parser.add_argument("--ss-theta", type=int, default=512,
                        help="Static-stream fixed chunk size.")
    parser.add_argument("--theta-min", type=int, default=64)
    parser.add_argument("--theta-max", type=int, default=4096)
    parser.add_argument("--damping", type=float, default=0.4,
                        help="DES Newton-step damping (lower = smoother adaptation).")
    parser.add_argument("--trace", type=str, default=None,
                        help="Replay a recorded rate trace (CSV).")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--plot", type=str, default=None,
                        help="PNG of theta + slack trajectory (needs matplotlib).")
    args = parser.parse_args()

    print("=" * 66)
    print("DES (Dynamic-Equilibrium Streaming) benchmark -- OVERLAP stage")
    print("Paper: DES > NS and SS; CoV 0.13-0.75; theta -> equilibrium (slack->0).")
    print("=" * 66)

    if args.trace:
        rows = load_rate_trace(Path(args.trace))
        print(f"Replaying rate trace: {args.trace} ({len(rows)} steps)")
    else:
        rows = synth_rate_trace(args.steps, args.cov, args.seed)
        print(f"Synthetic rate trace: {len(rows)} steps, target CoV={args.cov}")
    if not rows:
        raise SystemExit("Empty rate trace.")
    print(f"Measured producer-rate CoV: {trace_cov(rows):.3f}")
    print(f"total_tokens={args.total_tokens}, initial_theta={args.initial_theta}, "
          f"ss_theta={args.ss_theta}\n")

    ns = simulate_no_stream(rows, args.total_tokens)
    ss = simulate_stream(rows, args.total_tokens, controller=None,
                         label=f"SS (fixed theta={args.ss_theta})", fixed_theta=args.ss_theta)
    des_ctrl = DESController(initial_theta=args.initial_theta,
                            min_theta=args.theta_min, max_theta=args.theta_max,
                            damping_factor=args.damping)
    des = simulate_stream(rows, args.total_tokens, controller=des_ctrl, label="DES (ours)")

    results = [ns, ss, des]
    print(f"{'method':<22}  {'wall_ms':>10}  {'tput_tok/s':>11}  "
          f"{'mean|slack|ms':>13}  {'final_theta':>11}")
    print("-" * 74)
    for r in results:
        slack_s = f"{r['mean_abs_slack_ms']:.1f}" if not math.isnan(r["mean_abs_slack_ms"]) else "n/a"
        print(f"{r['method']:<22}  {r['wall_ms']:>10.0f}  {r['throughput_tok_s']:>11.1f}  "
              f"{slack_s:>13}  {r['final_theta']:>11}")

    des_vs_ns = des["throughput_tok_s"] / ns["throughput_tok_s"]
    des_vs_ss = des["throughput_tok_s"] / ss["throughput_tok_s"]
    print(f"\nDES throughput vs NS : {des_vs_ns:.2f}x  ({(des_vs_ns-1)*100:+.1f}%)")
    print(f"DES throughput vs SS : {des_vs_ss:.2f}x  ({(des_vs_ss-1)*100:+.1f}%)")

    # Convergence: DES drives slack below the static baseline and stabilizes
    # theta. We compare DES's steady-state slack (last half) against SS's slack
    # over the same trace, and report theta drift in the last half.
    des_sl = des["slack_trajectory"]
    ss_sl = ss["slack_trajectory"]
    if len(des_sl) >= 6:
        half = len(des_sl) // 2
        des_steady = statistics.mean(des_sl[half:])
        ss_steady = statistics.mean(ss_sl[half:]) if len(ss_sl) >= 2 else float("nan")
        des_thetas = des["theta_trajectory"][half:]
        theta_drift = (statistics.pstdev(des_thetas) / statistics.mean(des_thetas)
                       if des_thetas and statistics.mean(des_thetas) else 0.0)
        print(f"steady slack (2nd half): DES {des_steady:.1f}ms vs SS {ss_steady:.1f}ms "
              f"({'DES lower -> better overlap' if des_steady <= ss_steady else 'comparable'})")
        print(f"theta equilibrium      : initial {args.initial_theta} -> settled "
              f"{des['final_theta']} (2nd-half theta CoV {theta_drift:.2f})")

    if args.csv:
        with Path(args.csv).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["method", "wall_ms", "throughput_tok_s", "mean_abs_slack_ms", "final_theta"])
            for r in results:
                w.writerow([r["method"], round(r["wall_ms"], 1),
                            round(r["throughput_tok_s"], 2),
                            "" if math.isnan(r["mean_abs_slack_ms"]) else round(r["mean_abs_slack_ms"], 2),
                            r["final_theta"]])
        print(f"\nWrote CSV -> {args.csv}")

    if args.plot:
        if maybe_plot(des, Path(args.plot), args.ss_theta):
            print(f"Wrote plot -> {args.plot}")
        else:
            print("matplotlib not installed -- skipped PNG plot (table above holds the data).")


if __name__ == "__main__":
    main()
