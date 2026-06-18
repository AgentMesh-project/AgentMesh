"""
DES Discrete-Event Streaming Simulator.

A self-contained, timestamped simulator of one produce--transfer--consume
streaming pipeline, used to study and validate the Dynamic-Equilibrium
Streaming (DES) controller offline without a GPU or an LLM backend.

The pipeline mirrors a single AgentMesh dataflow:

    Producer  -->  (network transfer)  -->  Consumer
    (a tool fetch OR an upstream         (ALWAYS an LLM prefill)
     agent's LLM decode)

For each emitted chunk of size ``theta`` (tokens) the simulator advances a
single virtual clock and records:

  * ``produce``  = theta * producer_rate (generation / tool execution time)
  * ``transfer`` = theta * transfer_rate (network time, folded into producer)
  * ``consume``  = theta * consumer_rate (downstream prefill / parsing time)

Following the controller's convention, the *producer* time fed to DES is the
sum ``T_p + T_n`` (produce + transfer), and the *slack* is

    delta = producer_time - consumer_time

The simulator drives the locked :class:`DESController` API
(``compute_next_theta(producer_time, consumer_time, slack_time, time_step)``)
in the loop, choosing the chunk size for the next step.

It also provides synthetic rate generators with a controllable
coefficient-of-variation (CV in the 0.13--0.75 range reported in the paper),
and helpers comparing DES against two baselines:

  * NS  -- No-Stream / serialized: produce the whole response, transfer it,
           then consume it (no overlap; the BARRIER is paid in full).
  * SS  -- Static fixed-chunk Stream: a fixed chunk size theta is used for
           every chunk (overlap, but no equilibrium control).

Pure Python + numpy. No heavy dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "RateFunction",
    "ChunkRecord",
    "SimulationTrace",
    "StreamingPipelineSimulator",
    "make_synthetic_rate_fn",
    "constant_rate_fn",
    "step_rate_fn",
    "coefficient_of_variation",
    "simulate_ns_baseline",
    "simulate_ss_baseline",
    "compare_strategies",
]


# A rate function maps a (time_step, theta) pair to a per-token latency
# (in milliseconds per token).  Both arguments are provided so a generator
# can model time-varying *and* size-dependent behaviour, but most generators
# only use the time step.
RateFunction = Callable[[int, int], float]


# =============================================================================
# Rate generators (synthetic producers / consumers)
# =============================================================================

def coefficient_of_variation(values: Sequence[float]) -> float:
    """
    Compute the coefficient of variation (CV = std / mean) of a series.

    The paper reports replayed-trace CV in the 0.13--0.75 range; this helper
    lets callers verify that a synthetic generator lands in that band.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    mean = float(arr.mean())
    if abs(mean) < 1e-12:
        return 0.0
    return float(arr.std()) / mean


def constant_rate_fn(rate: float) -> RateFunction:
    """Return a rate function with a constant per-token latency (CV = 0)."""

    def _fn(time_step: int, theta: int) -> float:  # noqa: ARG001
        return float(rate)

    return _fn


def make_synthetic_rate_fn(
    base_rate: float,
    cv: float = 0.3,
    seed: Optional[int] = None,
    autocorr: float = 0.0,
    num_steps: int = 4096,
) -> RateFunction:
    """
    Build a synthetic time-varying per-token rate function.

    The returned function yields a pre-sampled lognormal series whose
    coefficient of variation approximates ``cv`` (paper range 0.13--0.75).
    An optional first-order autocorrelation models bursty network/producer
    behaviour (a smoothed random walk over the lognormal innovations).

    Args:
        base_rate: Mean per-token latency in ms/token.
        cv: Target coefficient of variation (std / mean).
        seed: RNG seed for reproducibility.
        autocorr: AR(1) coefficient in [0, 1); 0 = i.i.d. (no memory).
        num_steps: Number of pre-sampled steps (the series wraps if exceeded).

    Returns:
        A RateFunction; calling it with (time_step, theta) returns ms/token.
    """
    rng = np.random.default_rng(seed)

    # For a lognormal with CV = c, the underlying normal has
    # sigma^2 = ln(1 + c^2) and mu chosen so the lognormal mean is base_rate.
    cv = max(0.0, float(cv))
    sigma = math.sqrt(math.log(1.0 + cv * cv)) if cv > 0 else 0.0
    mu = math.log(max(base_rate, 1e-9)) - 0.5 * sigma * sigma

    innovations = rng.normal(loc=0.0, scale=1.0, size=num_steps)

    # Optional AR(1) smoothing of the standardized innovations.
    autocorr = min(max(float(autocorr), 0.0), 0.999)
    if autocorr > 0.0:
        smoothed = np.empty_like(innovations)
        smoothed[0] = innovations[0]
        norm = math.sqrt(1.0 - autocorr * autocorr)
        for i in range(1, num_steps):
            smoothed[i] = autocorr * smoothed[i - 1] + norm * innovations[i]
        innovations = smoothed

    series = np.exp(mu + sigma * innovations)
    # Guard against pathological zeros.
    series = np.clip(series, 1e-6, None)

    def _fn(time_step: int, theta: int) -> float:  # noqa: ARG001
        return float(series[time_step % num_steps])

    return _fn


def step_rate_fn(
    rate_before: float,
    rate_after: float,
    switch_step: int,
) -> RateFunction:
    """
    A piecewise-constant rate that switches at ``switch_step``.

    Useful for studying how fast DES re-converges after an abrupt regime
    change (e.g. a producer slowdown or a faster consumer coming online).
    """

    def _fn(time_step: int, theta: int) -> float:  # noqa: ARG001
        return float(rate_before if time_step < switch_step else rate_after)

    return _fn


# =============================================================================
# Trace records
# =============================================================================

@dataclass
class ChunkRecord:
    """Per-chunk timing record produced by the simulator."""

    step: int
    theta: int                  # chunk size used for this step (tokens)
    produce_time: float         # T_p (ms): producer generation/execution
    transfer_time: float        # T_n (ms): network transfer
    consume_time: float         # T_c (ms): downstream prefill/parsing
    producer_time: float        # T_p + T_n (ms): fed to DES as "producer"
    slack: float                # delta = producer_time - consume_time (ms)
    wall_time: float            # cumulative pipeline wall-clock time (ms)
    tokens_done: int            # cumulative tokens emitted


@dataclass
class SimulationTrace:
    """
    Full trace of a streaming simulation run.

    Attributes:
        records: Per-chunk timing records.
        total_tokens: Total tokens streamed.
        makespan_ms: End-to-end pipeline wall-clock time (ms).
        strategy: Label of the strategy that produced the trace.
    """

    records: List[ChunkRecord] = field(default_factory=list)
    total_tokens: int = 0
    makespan_ms: float = 0.0
    strategy: str = "DES"

    # ---- Convenience views (numpy arrays) ----
    @property
    def thetas(self) -> np.ndarray:
        return np.asarray([r.theta for r in self.records], dtype=float)

    @property
    def slacks(self) -> np.ndarray:
        return np.asarray([r.slack for r in self.records], dtype=float)

    @property
    def producer_times(self) -> np.ndarray:
        return np.asarray([r.producer_time for r in self.records], dtype=float)

    @property
    def consumer_times(self) -> np.ndarray:
        return np.asarray([r.consume_time for r in self.records], dtype=float)

    @property
    def throughput_tok_per_s(self) -> float:
        """Overall throughput in tokens/second."""
        if self.makespan_ms <= 0:
            return 0.0
        return self.total_tokens / (self.makespan_ms / 1000.0)

    def summary(self) -> Dict[str, float]:
        """Return a compact dict of headline metrics."""
        slacks = self.slacks
        thetas = self.thetas
        return {
            "strategy_label": self.strategy,
            "total_tokens": float(self.total_tokens),
            "num_chunks": float(len(self.records)),
            "makespan_ms": self.makespan_ms,
            "throughput_tok_per_s": self.throughput_tok_per_s,
            "mean_abs_slack_ms": float(np.mean(np.abs(slacks))) if slacks.size else 0.0,
            "final_theta": float(thetas[-1]) if thetas.size else 0.0,
            "mean_theta": float(np.mean(thetas)) if thetas.size else 0.0,
            "theta_cv": coefficient_of_variation(thetas) if thetas.size else 0.0,
        }


# =============================================================================
# Core simulator
# =============================================================================

class StreamingPipelineSimulator:
    """
    Discrete-event simulator of a single produce--transfer--consume pipeline.

    The producer emits chunks of ``theta`` tokens; each chunk is transferred
    over the (virtual) network, then consumed by a downstream LLM prefill.
    Because produce, transfer, and consume overlap across consecutive chunks,
    the pipeline behaves like a two-stage producer/consumer system: the wall
    clock advances by ``max(producer_time_i, consumer_time_{i-1})`` plus the
    pipeline fill/drain at the ends.

    The DES controller observes ``(producer_time, consumer_time, slack)`` after
    each chunk and returns the chunk size for the next step, driving slack
    ``delta -> 0`` (the dynamic equilibrium).

    Args:
        producer_rate_fn: ms/token for produce, as f(time_step, theta).
        consumer_rate_fn: ms/token for consume, as f(time_step, theta).
        controller: A DESController-like object exposing
            ``current_theta`` and
            ``compute_next_theta(producer_time, consumer_time, slack_time,
            time_step)``. If None, the simulator runs at a fixed ``theta0``
            (equivalent to the SS baseline).
        theta0: Initial chunk size (tokens).
        transfer_rate_fn: Optional ms/token for the network transfer; folded
            into the producer time (T_p + T_n). Defaults to zero transfer.
        min_theta / max_theta: Hard clamp applied to whatever the controller
            returns (defends against a misconfigured controller).
    """

    def __init__(
        self,
        producer_rate_fn: RateFunction,
        consumer_rate_fn: RateFunction,
        controller: Optional[object] = None,
        theta0: int = 512,
        transfer_rate_fn: Optional[RateFunction] = None,
        min_theta: int = 1,
        max_theta: int = 1 << 20,
    ):
        self.producer_rate_fn = producer_rate_fn
        self.consumer_rate_fn = consumer_rate_fn
        self.transfer_rate_fn = transfer_rate_fn or constant_rate_fn(0.0)
        self.controller = controller
        self.theta0 = int(theta0)
        self.min_theta = int(min_theta)
        self.max_theta = int(max_theta)

    # ------------------------------------------------------------------ #
    def _current_theta(self) -> int:
        """Read the controller's current theta, falling back to theta0."""
        theta = self.theta0
        if self.controller is not None and hasattr(self.controller, "current_theta"):
            theta = int(self.controller.current_theta)
        return int(np.clip(theta, self.min_theta, self.max_theta))

    def run(self, num_chunks: int = 32) -> SimulationTrace:
        """
        Run the streaming simulation for ``num_chunks`` chunks.

        Returns:
            A SimulationTrace with per-chunk records and headline metrics.
        """
        trace = SimulationTrace(strategy="DES" if self.controller is not None else "SS")

        theta = self._current_theta()
        wall = 0.0
        tokens_done = 0
        prev_consume = 0.0  # T_c of the previous chunk (consumer busy time)

        for step in range(num_chunks):
            # Producer + network for this chunk (the "produce" stage).
            p_rate = self.producer_rate_fn(step, theta)
            n_rate = self.transfer_rate_fn(step, theta)
            produce_time = theta * p_rate
            transfer_time = theta * n_rate
            producer_time = produce_time + transfer_time

            # Consumer for this chunk (the "consume" stage, an LLM prefill).
            c_rate = self.consumer_rate_fn(step, theta)
            consume_time = theta * c_rate

            # Pipeline overlap: while the producer makes chunk i, the consumer
            # is still draining chunk i-1. The stage that finishes last gates
            # the clock. delta is the producer/consumer imbalance DES drives
            # toward zero.
            slack = producer_time - consume_time
            wall += max(producer_time, prev_consume)

            tokens_done += theta
            trace.records.append(
                ChunkRecord(
                    step=step,
                    theta=theta,
                    produce_time=produce_time,
                    transfer_time=transfer_time,
                    consume_time=consume_time,
                    producer_time=producer_time,
                    slack=slack,
                    wall_time=wall,
                    tokens_done=tokens_done,
                )
            )

            prev_consume = consume_time

            # Ask DES for the next chunk size (closing the control loop).
            if self.controller is not None and hasattr(self.controller, "compute_next_theta"):
                next_theta = self.controller.compute_next_theta(
                    producer_time=producer_time,
                    consumer_time=consume_time,
                    slack_time=slack,
                    time_step=step,
                )
                theta = int(np.clip(int(next_theta), self.min_theta, self.max_theta))
            # else: fixed theta (SS behaviour).

        # Drain the final chunk through the consumer (pipeline drain).
        wall += prev_consume

        trace.total_tokens = tokens_done
        trace.makespan_ms = wall
        return trace


# =============================================================================
# Baselines
# =============================================================================

def simulate_ns_baseline(
    producer_rate_fn: RateFunction,
    consumer_rate_fn: RateFunction,
    total_tokens: int,
    transfer_rate_fn: Optional[RateFunction] = None,
) -> SimulationTrace:
    """
    No-Stream / serialized (NS) baseline.

    The producer generates the entire response, transfers it, and only then
    does the consumer begin prefill. There is no overlap, so the consumer
    idles for the full produce+transfer duration (the BARRIER is paid in
    full). This is the structure-blind worst case DES improves upon.
    """
    transfer_rate_fn = transfer_rate_fn or constant_rate_fn(0.0)

    produce_time = total_tokens * producer_rate_fn(0, total_tokens)
    transfer_time = total_tokens * transfer_rate_fn(0, total_tokens)
    consume_time = total_tokens * consumer_rate_fn(0, total_tokens)
    producer_time = produce_time + transfer_time

    makespan = producer_time + consume_time  # fully serialized

    trace = SimulationTrace(strategy="NS")
    trace.records.append(
        ChunkRecord(
            step=0,
            theta=total_tokens,
            produce_time=produce_time,
            transfer_time=transfer_time,
            consume_time=consume_time,
            producer_time=producer_time,
            slack=producer_time - consume_time,
            wall_time=makespan,
            tokens_done=total_tokens,
        )
    )
    trace.total_tokens = total_tokens
    trace.makespan_ms = makespan
    return trace


def simulate_ss_baseline(
    producer_rate_fn: RateFunction,
    consumer_rate_fn: RateFunction,
    total_tokens: int,
    theta: int,
    transfer_rate_fn: Optional[RateFunction] = None,
) -> SimulationTrace:
    """
    Static fixed-chunk Stream (SS) baseline.

    A fixed chunk size ``theta`` is streamed for every chunk. The stages
    overlap (unlike NS), but there is no equilibrium control, so a mismatched
    static theta leaves persistent slack.
    """
    num_chunks = max(1, math.ceil(total_tokens / max(1, theta)))
    sim = StreamingPipelineSimulator(
        producer_rate_fn=producer_rate_fn,
        consumer_rate_fn=consumer_rate_fn,
        controller=None,  # no controller -> fixed theta
        theta0=theta,
        transfer_rate_fn=transfer_rate_fn,
    )
    trace = sim.run(num_chunks=num_chunks)
    trace.strategy = "SS"
    return trace


def compare_strategies(
    producer_rate_fn: RateFunction,
    consumer_rate_fn: RateFunction,
    controller: object,
    total_tokens: int = 8192,
    des_num_chunks: int = 32,
    ss_theta: Optional[int] = None,
    transfer_rate_fn: Optional[RateFunction] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Run DES vs NS vs SS over the same synthetic pipeline and return summaries.

    Args:
        producer_rate_fn / consumer_rate_fn: pipeline rate generators.
        controller: a DESController-like instance for the DES run.
        total_tokens: total tokens to stream (NS/SS sized to match).
        des_num_chunks: number of chunks for the DES run.
        ss_theta: fixed chunk size for SS (defaults to the controller's
            initial theta if available, else total_tokens // des_num_chunks).
        transfer_rate_fn: optional network rate generator.

    Returns:
        ``{"DES": summary, "NS": summary, "SS": summary}`` plus a
        ``"speedup_vs_NS"`` / ``"speedup_vs_SS"`` field on the DES summary.
    """
    if ss_theta is None:
        ss_theta = getattr(controller, "initial_theta", None)
        if ss_theta is None:
            ss_theta = max(1, total_tokens // max(1, des_num_chunks))

    # Reset the controller so DES starts fresh.
    if hasattr(controller, "reset"):
        controller.reset()

    des_sim = StreamingPipelineSimulator(
        producer_rate_fn=producer_rate_fn,
        consumer_rate_fn=consumer_rate_fn,
        controller=controller,
        theta0=getattr(controller, "current_theta", ss_theta),
        transfer_rate_fn=transfer_rate_fn,
    )
    des_trace = des_sim.run(num_chunks=des_num_chunks)

    ns_trace = simulate_ns_baseline(
        producer_rate_fn, consumer_rate_fn, des_trace.total_tokens, transfer_rate_fn
    )
    ss_trace = simulate_ss_baseline(
        producer_rate_fn,
        consumer_rate_fn,
        des_trace.total_tokens,
        theta=int(ss_theta),
        transfer_rate_fn=transfer_rate_fn,
    )

    des_summary = des_trace.summary()
    ns_summary = ns_trace.summary()
    ss_summary = ss_trace.summary()

    des_summary["speedup_vs_NS"] = (
        ns_summary["makespan_ms"] / des_summary["makespan_ms"]
        if des_summary["makespan_ms"] > 0
        else 0.0
    )
    des_summary["speedup_vs_SS"] = (
        ss_summary["makespan_ms"] / des_summary["makespan_ms"]
        if des_summary["makespan_ms"] > 0
        else 0.0
    )

    return {"DES": des_summary, "NS": ns_summary, "SS": ss_summary}
