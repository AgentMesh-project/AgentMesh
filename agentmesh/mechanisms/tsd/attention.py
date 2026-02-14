"""
TSD (Topology-aware Semantic Decoupling)

Implements Topologically Decoupled Attention (TDA) for the supervisor-worker
scatter-gather topology in multi-agent systems.

Core idea: exploit the semantic orthogonality of parallel worker outputs to
transform monolithic context gathering into a decoupled stream, enabling
communication-computation overlap.

TDA components:
1. Inter-Worker Masking  — each worker only attends to supervisor prefix
2. Timely Parallel Prefill — prefill workers as they arrive (no straggler wait)
3. Positional Realignment — RoPE rotation after KV stitching

Complexity reduction:
    Full:  O(L_prefix · Σ L_i + Σ_i Σ_j L_i · L_j)
    TDA:   O(max(L_prefix · L_i + L_i²))
    → Linear in number of workers (vs. quadratic)

Note: TSD is orthogonal to KV blending algorithms (CacheBlend, EPIC). They
focus on token-level KV selection; TSD operates at the orchestration layer.
"""

import torch
import numpy as np
import math
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class WorkerOutput:
    """
    Output from a single worker agent in the scatter-gather topology.

    Attributes:
        worker_id: Unique identifier for the worker agent.
        tokens: Token IDs produced by this worker's decoding.
        arrival_time: Timestamp when output reached the supervisor transport layer.
        kv_states: Optional pre-computed KV tensors (key, value) per layer.
                   Shape per layer: (num_heads, seq_len, head_dim).
    """
    worker_id: str
    tokens: List[int]
    arrival_time: float = field(default_factory=time.time)
    kv_states: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass
class SupervisorContext:
    """
    Fixed supervisor prefix context (system prompt + user task + role setting).

    In TDA, each worker's KV computation only attends to this prefix,
    NOT to other workers' outputs.

    Attributes:
        tokens: Token IDs of the supervisor prefix.
        kv_states: Pre-computed KV states for the prefix (per layer).
        length: Number of tokens in the prefix.
    """
    tokens: List[int]
    kv_states: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass
class TDAResult:
    """
    Result of TDA parallel prefill + KV stitching.

    Attributes:
        combined_kv: Stitched KV states per layer after positional realignment.
        total_tokens: Total tokens across all workers (excluding prefix).
        worker_order: Order of workers in the stitched sequence.
        prefill_times: Per-worker prefill timing.
        realignment_time: Time spent on RoPE positional realignment.
    """
    combined_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    total_tokens: int
    worker_order: List[str]
    prefill_times: Dict[str, float] = field(default_factory=dict)
    realignment_time: float = 0.0


# ---------------------------------------------------------------------------
# Inter-Worker Attention Mask Generator
# ---------------------------------------------------------------------------
class InterWorkerMaskGenerator:
    """
    Generates attention masks for Inter-Worker Masking.

    During prefill of worker i's tokens (L_i), the supervisor only computes
    cross-attention between those tokens and the fixed supervisor prefix
    (L_prefix). Attention between concurrent worker results is bypassed.

    This reduces attention complexity from:
        O(L_prefix · Σ L_i + Σ_i Σ_{j≥i} L_i · L_j)
    to:
        O(max_i(L_prefix · L_i + L_i²))
    """

    @staticmethod
    def generate_worker_mask(
        prefix_len: int,
        worker_len: int,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Generate attention mask for a single worker's prefill.

        The mask allows:
        - Worker tokens attend to all prefix tokens (cross-attention)
        - Worker tokens attend to preceding worker tokens (self-attention, causal)
        - Prefix tokens attend to other prefix tokens (already computed)

        Args:
            prefix_len: Length of supervisor prefix L_prefix.
            worker_len: Length of this worker's output L_i.
            device: Device for the mask tensor.

        Returns:
            Boolean mask of shape (worker_len, prefix_len + worker_len).
            True = attention allowed.
        """
        total_len = prefix_len + worker_len

        # Worker tokens can attend to:
        #   1. All prefix tokens (columns 0..prefix_len-1): True
        #   2. Worker tokens causally (columns prefix_len..prefix_len+j for j≤current)
        mask = torch.zeros(worker_len, total_len, dtype=torch.bool, device=device)

        # Attend to all prefix tokens
        mask[:, :prefix_len] = True

        # Causal self-attention within worker tokens
        for i in range(worker_len):
            mask[i, prefix_len : prefix_len + i + 1] = True

        return mask

    @staticmethod
    def generate_full_decoding_mask(
        prefix_len: int,
        worker_lengths: List[int],
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Generate the full causal mask for decoding after KV stitching.

        After TDA parallel prefill and positional realignment, the supervisor
        decodes using the concatenated KV states. The decoding mask is a
        standard causal mask over the full sequence.

        Args:
            prefix_len: Length of supervisor prefix.
            worker_lengths: Lengths of each worker's output.
            device: Device for the mask tensor.

        Returns:
            Full causal mask of shape (total_len, total_len).
        """
        total_len = prefix_len + sum(worker_lengths)
        # Standard causal mask (lower triangular)
        return torch.tril(
            torch.ones(total_len, total_len, dtype=torch.bool, device=device)
        )


# ---------------------------------------------------------------------------
# RoPE Positional Realignment
# ---------------------------------------------------------------------------
class RoPEAligner:
    """
    Positional Realignment via Rotary Position Embeddings (RoPE).

    After parallel prefill, each worker's KV states were computed assuming
    the same starting position (directly after the supervisor prefix). TDA
    performs late-stage positional realignment to assign distinct global
    positions after stitching.

    To adjust a key vector k from position i to position j:

        k_j = R_{j-i} · k_i

    where R_{j-i} is the RoPE rotation matrix:

        R_{j-i} = [ cos((j-i)θ)  -sin((j-i)θ) ]
                  [ sin((j-i)θ)   cos((j-i)θ) ]

    (2D illustration; extends to higher dimensions pairwise)

    Paper note: "This vectorized operation is performed in-place on GPUs,
    with negligible (< 1 ms) overhead."
    """

    def __init__(self, head_dim: int = 128, rope_base: float = 10000.0):
        """
        Args:
            head_dim: Dimension per attention head (must be even).
            rope_base: Base frequency for RoPE (default 10000).
        """
        self.head_dim = head_dim
        self.rope_base = rope_base

        # Pre-compute inverse frequencies: θ_d = base^{-2d/D}
        self.inv_freq = 1.0 / (
            rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )

    def realign(
        self,
        key: torch.Tensor,
        from_pos: int,
        to_pos: int,
    ) -> torch.Tensor:
        """
        Rotate key vectors from position `from_pos` to `to_pos` using RoPE.

        Implements: k_j = R_{j-i} · k_i

        Args:
            key: Key tensor of shape (..., seq_len, head_dim).
            from_pos: Original starting position i.
            to_pos: Target starting position j.

        Returns:
            Rotated key tensor with corrected positions.
        """
        if from_pos == to_pos:
            return key

        offset = to_pos - from_pos
        seq_len = key.shape[-2]
        device = key.device

        # Compute rotation angles for each position in the sequence
        positions = torch.arange(seq_len, device=device).float()
        angles = positions.unsqueeze(-1) * 0 + offset  # offset is constant
        angles = angles * self.inv_freq.to(device)  # (seq_len, head_dim//2)

        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        # Apply rotation pairwise: (x, y) -> (x cos - y sin, x sin + y cos)
        key_even = key[..., 0::2]
        key_odd = key[..., 1::2]

        rotated_even = key_even * cos_angles - key_odd * sin_angles
        rotated_odd = key_even * sin_angles + key_odd * cos_angles

        # Interleave back
        rotated = torch.stack([rotated_even, rotated_odd], dim=-1)
        return rotated.flatten(-2)

    def batch_realign(
        self,
        keys: List[torch.Tensor],
        from_positions: List[int],
        to_positions: List[int],
    ) -> List[torch.Tensor]:
        """
        Realign multiple key tensors in batch.

        Args:
            keys: List of key tensors.
            from_positions: Original positions for each tensor.
            to_positions: Target positions for each tensor.

        Returns:
            List of realigned key tensors.
        """
        return [
            self.realign(k, f, t)
            for k, f, t in zip(keys, from_positions, to_positions)
        ]


# ---------------------------------------------------------------------------
# TDA Manager (core TSD component)
# ---------------------------------------------------------------------------
class TDAManager:
    """
    Topologically Decoupled Attention (TDA) Manager.

    Orchestrates the complete TSD workflow for supervisor-worker topology:

    1. Worker outputs arrive asynchronously (scatter-execute phase)
    2. For each arrival: parallel prefill with inter-worker masking
       (only attend to supervisor prefix, not other workers)
    3. After all workers complete: KV stitching + RoPE positional realignment
    4. Supervisor proceeds with standard decoding on the stitched KV states

    Design tradeoff:
    - Direct KV reuse from workers is efficient but severely degrades accuracy
      (workers' KVs lack semantic alignment with supervisor's instructions)
    - Full all-to-all attention is precise but incurs quadratic latency
    - TDA: lightweight re-computation with supervisor prefix achieves near-full
      precision at linear-scale complexity
    """

    def __init__(
        self,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        max_parallel_workers: int = 3,
        rope_base: float = 10000.0,
        device: str = "cuda",
    ):
        """
        Initialize TDA Manager.

        Args:
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            head_dim: Dimension per attention head.
            max_parallel_workers: Maximum concurrent worker prefills.
            rope_base: Base frequency for RoPE.
            device: Default device for tensors.
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_parallel_workers = max_parallel_workers
        self.device = device

        # Components
        self.mask_generator = InterWorkerMaskGenerator()
        self.rope_aligner = RoPEAligner(head_dim=head_dim, rope_base=rope_base)

        # State tracking
        self.supervisor_ctx: Optional[SupervisorContext] = None
        self.worker_kv_states: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}
        self.worker_lengths: Dict[str, int] = {}
        self.arrival_order: List[str] = []

        # Statistics
        self.stats = {
            "total_prefills": 0,
            "total_workers_processed": 0,
            "total_stitches": 0,
            "total_realignments": 0,
            "avg_prefill_time_ms": 0.0,
            "avg_realignment_time_ms": 0.0,
        }

    def set_supervisor_context(self, ctx: SupervisorContext):
        """
        Set the supervisor prefix context.

        This must be called before any worker prefill. The prefix includes
        the supervisor's system prompt, user task, role setting, etc.
        """
        self.supervisor_ctx = ctx
        logger.debug(f"Supervisor context set: {ctx.length} tokens")

    def prefill_worker(
        self,
        worker_output: WorkerOutput,
        prefill_fn: Optional[callable] = None,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Prefill a single worker's output with inter-worker masking.

        Implements "Timely Parallel Prefill": the supervisor initiates
        prefilling each worker's output as soon as its segments reach the
        transport layer — no need to wait for all workers.

        During this prefill, TDA computes KV states by only attending to:
        - The supervisor prefix (L_prefix tokens)
        - The worker's own content (L_i tokens, causal)
        All other workers' cross-attention is masked out.

        Args:
            worker_output: Output from a worker agent.
            prefill_fn: Optional custom prefill function. If None, uses
                        worker's pre-computed KV states or initializes
                        zero-valued placeholders.

        Returns:
            Per-layer KV states: {layer_idx: (key, value)}.
        """
        assert self.supervisor_ctx is not None, "Supervisor context must be set first"
        start_time = time.time()

        prefix_len = self.supervisor_ctx.length
        worker_len = worker_output.length
        worker_id = worker_output.worker_id

        # Generate inter-worker mask for this worker
        mask = self.mask_generator.generate_worker_mask(
            prefix_len=prefix_len,
            worker_len=worker_len,
            device=self.device,
        )

        if prefill_fn is not None:
            # Use custom prefill function (e.g., vLLM KV connector)
            kv_states = prefill_fn(
                supervisor_prefix=self.supervisor_ctx,
                worker_tokens=worker_output.tokens,
                attention_mask=mask,
            )
        elif worker_output.kv_states is not None:
            # Use pre-computed KV states from the worker
            kv_states = worker_output.kv_states
        else:
            # Compute placeholder KV states via masked attention simulation.
            # In production, this path is replaced by the vLLM KV connector
            # which performs actual transformer forward passes.
            kv_states = {}
            for layer in range(self.num_layers):
                # Initialize KV states with zeros (neutral values)
                # to represent positions awaiting real computation.
                key = torch.zeros(
                    self.num_heads, worker_len, self.head_dim, device=self.device
                )
                value = torch.zeros(
                    self.num_heads, worker_len, self.head_dim, device=self.device
                )
                kv_states[layer] = (key, value)

        # Store results
        self.worker_kv_states[worker_id] = kv_states
        self.worker_lengths[worker_id] = worker_len
        self.arrival_order.append(worker_id)

        # Update statistics
        prefill_time_ms = (time.time() - start_time) * 1000
        self.stats["total_prefills"] += 1
        self.stats["total_workers_processed"] += 1
        n = self.stats["total_prefills"]
        self.stats["avg_prefill_time_ms"] += (
            prefill_time_ms - self.stats["avg_prefill_time_ms"]
        ) / n

        logger.debug(
            f"Worker '{worker_id}' prefilled: {worker_len} tokens, "
            f"{prefill_time_ms:.2f}ms (mask: {prefix_len}+{worker_len})"
        )

        return kv_states

    def stitch_and_realign(self) -> TDAResult:
        """
        KV Stitching + Positional Realignment — final TDA step.

        After all workers complete parallel prefill:
        1. Concatenate each worker's KV states in arrival order
        2. Apply RoPE rotation to correct positions:
           Each worker was prefilled assuming position starts at prefix_len.
           After stitching, worker i should start at:
             prefix_len + sum(L_j for j < i)

        Returns:
            TDAResult with combined KV states and timing information.
        """
        assert len(self.worker_kv_states) > 0, "No worker KV states to stitch"
        start_time = time.time()

        worker_order = list(self.arrival_order)
        combined_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        prefill_times: Dict[str, float] = {}
        prefix_len = self.supervisor_ctx.length if self.supervisor_ctx else 0

        for layer in range(self.num_layers):
            all_keys = []
            all_values = []

            # Compute cumulative positions for each worker
            cumulative_pos = prefix_len
            for worker_id in worker_order:
                kv = self.worker_kv_states[worker_id][layer]
                key, value = kv

                # Each worker was prefilled at position prefix_len
                # Target position for this worker: cumulative_pos
                from_pos = prefix_len
                to_pos = cumulative_pos

                # Apply RoPE realignment if position differs
                if from_pos != to_pos:
                    key = self.rope_aligner.realign(key, from_pos, to_pos)

                all_keys.append(key)
                all_values.append(value)

                cumulative_pos += self.worker_lengths[worker_id]

            # Concatenate along sequence dimension
            combined_kv[layer] = (
                torch.cat(all_keys, dim=-2),
                torch.cat(all_values, dim=-2),
            )

        realignment_time = (time.time() - start_time) * 1000
        self.stats["total_stitches"] += 1
        self.stats["total_realignments"] += len(worker_order)
        n = self.stats["total_stitches"]
        self.stats["avg_realignment_time_ms"] += (
            realignment_time - self.stats["avg_realignment_time_ms"]
        ) / n

        total_tokens = sum(self.worker_lengths[w] for w in worker_order)

        logger.debug(
            f"KV stitching complete: {len(worker_order)} workers, "
            f"{total_tokens} tokens, realignment={realignment_time:.2f}ms"
        )

        return TDAResult(
            combined_kv=combined_kv,
            total_tokens=total_tokens,
            worker_order=worker_order,
            prefill_times=prefill_times,
            realignment_time=realignment_time,
        )

    def get_parallel_groups(self) -> List[Set[str]]:
        """
        Get groups of workers that can be prefilled in parallel.

        In the supervisor-worker topology, ALL workers are independent
        (they don't depend on each other's outputs), so they can all
        be prefilled concurrently.

        Returns:
            Single group containing all pending worker IDs.
        """
        pending = set(self.worker_lengths.keys()) - set(
            w for w in self.worker_kv_states.keys()
        )
        if not pending:
            return [set(self.arrival_order)]
        return [pending]

    def get_complexity_info(self) -> Dict:
        """
        Get complexity analysis info for current configuration.

        Full attention:  O(L_prefix · Σ L_i + Σ_i Σ_{j≥i} L_i · L_j)
        TDA attention:   O(max_i(L_prefix · L_i + L_i²))
        """
        prefix_len = self.supervisor_ctx.length if self.supervisor_ctx else 0
        worker_lens = list(self.worker_lengths.values())

        if not worker_lens:
            return {"full_complexity": 0, "tda_complexity": 0, "speedup": 1.0}

        # Full attention complexity
        sum_l = sum(worker_lens)
        full_cross = prefix_len * sum_l
        full_inter = sum(
            worker_lens[i] * worker_lens[j]
            for i in range(len(worker_lens))
            for j in range(i, len(worker_lens))
        )
        full_complexity = full_cross + full_inter

        # TDA complexity
        tda_complexity = max(prefix_len * l + l * l for l in worker_lens)

        speedup = full_complexity / tda_complexity if tda_complexity > 0 else float("inf")

        return {
            "full_complexity": full_complexity,
            "tda_complexity": tda_complexity,
            "speedup": speedup,
            "num_workers": len(worker_lens),
            "prefix_len": prefix_len,
            "worker_lengths": worker_lens,
        }

    def get_stats(self) -> Dict:
        """Get TDA statistics."""
        return {
            **self.stats,
            "num_workers_stored": len(self.worker_kv_states),
            "complexity": self.get_complexity_info(),
        }

    def clear(self):
        """Reset all state for a new workflow round."""
        self.supervisor_ctx = None
        self.worker_kv_states.clear()
        self.worker_lengths.clear()
        self.arrival_order.clear()
        self.stats = {
            "total_prefills": 0,
            "total_workers_processed": 0,
            "total_stitches": 0,
            "total_realignments": 0,
            "avg_prefill_time_ms": 0.0,
            "avg_realignment_time_ms": 0.0,
        }


# Convenience function
def create_tda_manager(
    num_layers: int = 32,
    num_heads: int = 32,
    head_dim: int = 128,
    max_parallel_workers: int = 3,
    rope_base: float = 10000.0,
    device: str = "cuda",
) -> TDAManager:
    """
    Create a TDA manager instance for the supervisor-worker topology.

    Args:
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        max_parallel_workers: Maximum concurrent worker prefills.
        rope_base: RoPE base frequency.
        device: Device for tensors.

    Returns:
        Configured TDAManager instance.
    """
    return TDAManager(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        max_parallel_workers=max_parallel_workers,
        rope_base=rope_base,
        device=device,
    )


KVCacheManager = TDAManager
create_tsd_manager = create_tda_manager
