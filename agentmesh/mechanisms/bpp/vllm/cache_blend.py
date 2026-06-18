# SPDX-License-Identifier: Apache-2.0
"""
Selective KV recomputation (CacheBlend / EPIC) for BPP.

BPP is orthogonal to, and composable with, selective KV-recompute schemes. When
independent worker branches are prefilled in parallel and their KV is stitched
into the supervisor sequence (Branch-Parallel Prefill), the cross-branch
attention that a serial prefill would have produced is missing. Two well-known
recompute schemes recover it cheaply by recomputing a small fraction of tokens:

  * CacheBlend (S7/S8): dynamically selects High-KV-Deviation (HKVD) tokens by
    comparing freshly computed K against cached K, then blends new K/V at those
    positions and keeps the cached values elsewhere.
  * EPIC / LegoLink (S9/S10): statically selects the first ``k`` tokens of each
    branch (the "attention-sink" tokens), recomputing only those.

This is the BPP-side integration glue. The actual branch-parallel attention
masking and KV stitching live in ``agentmesh.mechanisms.bpp``; here we provide
the selective-recompute primitives used by ``GPUKVConnector`` for the S7–S10
ablation scenarios. Heavy deps are imported lazily so the module compiles on
hosts without a GPU stack.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext

logger = init_logger(__name__)

# Check if LMCache ops are available for fused position adjustment
HAS_LMCACHE_OPS = False
try:
    import lmcache.c_ops as lmc_ops
    HAS_LMCACHE_OPS = True
except ImportError:
    pass


@dataclass
class CacheBlendConfig:
    """Configuration for CacheBlend selective KV recomputation."""
    enabled: bool = False
    recompute_ratio: float = 0.15  # Default: recompute 15% of tokens
    check_layer: int = 0  # Layer where HKVD token selection happens


@dataclass
class EPICConfig:
    """Configuration for EPIC (LegoLink) static token selection.

    EPIC differs from CacheBlend:
    - CacheBlend: Dynamic selection based on KV deviation (O(15%N²))
    - EPIC: Static selection of first k tokens per chunk (O(kN), k≪N)

    Key insight: "attention sink" phenomenon - initial tokens of each chunk
    disproportionately absorb attention when compiled independently.
    Recomputing first k tokens allows them to recognize their non-initial
    position, releasing attention to more relevant positions.
    """
    enabled: bool = False
    k: int = 16  # Number of initial tokens to recompute per chunk


@dataclass
class BlendSegment:
    """Information about a branch segment that needs selective recompute.

    A *segment* is one worker branch whose KV was produced independently and is
    being stitched into the stitched supervisor sequence.
    """
    cache_key: str  # Key to retrieve cached KV (e.g., "worker1" / branch 1)
    num_tokens: int  # Number of tokens in this segment
    old_start_pos: int  # Original position (from branch prefill)
    new_start_pos: int  # Target position (in stitched sequence)
    slot_start: int  # Start index in slot_mapping
    slot_end: int  # End index in slot_mapping


@dataclass
class CacheBlendMetadata:
    """Metadata for selective-recompute processing during forward pass."""
    enabled: bool = False
    blend_segments: List[BlendSegment] = field(default_factory=list)
    hkvd_indices: Optional[Dict[str, torch.Tensor]] = None  # Per-segment HKVD indices
    recompute_ratio: float = 0.15
    check_layer: int = 0


class CacheBlendProcessor:
    """
    Processor for CacheBlend selective KV recomputation.

    This implements the core CacheBlend algorithm:
    1. Load pre-computed KV cache from branches
    2. Compute new KV from current context (with correct cross-branch attention)
    3. Compute KV deviation between new and cached KV
    4. Select top-k% tokens with highest deviation (HKVD tokens)
    5. Update cached KV at HKVD positions with new values
    """

    def __init__(
        self,
        config: CacheBlendConfig,
        rope_adjuster: Optional[Any] = None,
    ):
        self.config = config
        self.rope_adjuster = rope_adjuster

        # State during processing
        self._metadata: Optional[CacheBlendMetadata] = None
        self._cached_kv: Dict[str, Dict[str, torch.Tensor]] = {}  # {key: {layer: kv_tensor}}
        self._hkvd_indices: Dict[str, torch.Tensor] = {}  # Per-segment HKVD indices

    def set_metadata(self, metadata: CacheBlendMetadata) -> None:
        """Set metadata for current forward pass."""
        self._metadata = metadata
        self._hkvd_indices.clear()

    def set_cached_kv(self, cached_kv: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """Set the cached KV from the global branch KV store."""
        self._cached_kv = cached_kv

    def is_enabled(self) -> bool:
        """Check if CacheBlend is enabled for current request."""
        return self._metadata is not None and self._metadata.enabled

    @staticmethod
    def compute_kv_deviation(
        new_k: torch.Tensor,
        cached_k: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KV deviation between new and cached K values.

        Args:
            new_k: New K values computed with current context [num_tokens, hidden_dim]
            cached_k: Cached K values from branch prefill [num_tokens, hidden_dim]

        Returns:
            deviation: Per-token deviation scores [num_tokens]
        """
        # Ensure same shape
        assert new_k.shape == cached_k.shape, f"Shape mismatch: {new_k.shape} vs {cached_k.shape}"

        # Compute L2 deviation per token
        diff = new_k - cached_k
        deviation = torch.sum(diff ** 2, dim=-1)  # [num_tokens]

        return deviation

    @staticmethod
    def select_hkvd_tokens(
        deviation: torch.Tensor,
        recompute_ratio: float,
    ) -> torch.Tensor:
        """
        Select High KV Deviation (HKVD) tokens.

        Args:
            deviation: Per-token deviation scores [num_tokens]
            recompute_ratio: Fraction of tokens to select (e.g., 0.15)

        Returns:
            hkvd_indices: Indices of selected HKVD tokens, sorted [num_hkvd]
        """
        num_tokens = deviation.shape[0]
        num_hkvd = max(1, int(num_tokens * recompute_ratio))

        # Select top-k tokens with highest deviation
        _, topk_indices = torch.topk(deviation, k=num_hkvd)

        # Sort indices for efficient memory access
        hkvd_indices = torch.sort(topk_indices).values

        return hkvd_indices

    @staticmethod
    def blend_kv(
        new_kv: torch.Tensor,
        cached_kv: torch.Tensor,
        hkvd_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Blend new KV values into cached KV at HKVD positions.

        Args:
            new_kv: New KV values [2, num_tokens, hidden_dim] or [num_tokens, hidden_dim]
            cached_kv: Cached KV values (same shape as new_kv)
            hkvd_indices: Indices of HKVD tokens to update [num_hkvd]

        Returns:
            blended_kv: Updated KV with new values at HKVD positions
        """
        blended_kv = cached_kv.clone()

        if cached_kv.dim() == 3 and cached_kv.shape[0] == 2:
            # KV combined format: [2, num_tokens, hidden_dim]
            blended_kv[:, hkvd_indices, :] = new_kv[:, hkvd_indices, :]
        else:
            # Single K or V: [num_tokens, hidden_dim]
            blended_kv[hkvd_indices, :] = new_kv[hkvd_indices, :]

        return blended_kv

    def process_layer(
        self,
        layer_id: int,
        layer_name: str,
        new_kv: torch.Tensor,
        segment: BlendSegment,
        segment_slice: slice,
    ) -> torch.Tensor:
        """
        Process one layer for a blend segment.

        Args:
            layer_id: Current layer index
            layer_name: Layer name for KV cache lookup
            new_kv: Newly computed KV for this layer [2, num_tokens, hidden_dim]
            segment: BlendSegment info
            segment_slice: Slice into new_kv for this segment

        Returns:
            blended_kv: KV values after blending for this segment
        """
        cache_key = segment.cache_key

        # Get cached KV for this layer
        if cache_key not in self._cached_kv:
            logger.warning(f"[CACHEBLEND] No cached KV for key={cache_key}")
            return new_kv[..., segment_slice, :]

        if layer_name not in self._cached_kv[cache_key]:
            logger.warning(f"[CACHEBLEND] No cached KV for key={cache_key}, layer={layer_name}")
            return new_kv[..., segment_slice, :]

        cached_kv = self._cached_kv[cache_key][layer_name]
        new_segment_kv = new_kv[..., segment_slice, :]

        # Realign position encoding of cached K if needed
        if segment.old_start_pos != segment.new_start_pos and self.rope_adjuster is not None:
            cached_kv = self._adjust_k_position(
                cached_kv,
                segment.old_start_pos,
                segment.new_start_pos,
                segment.num_tokens,
            )

        # On check_layer, compute deviation and select HKVD tokens
        if layer_id == self._metadata.check_layer:
            # Extract K for deviation computation
            if cached_kv.dim() == 3 and cached_kv.shape[0] == 2:
                cached_k = cached_kv[0]
                new_k = new_segment_kv[0] if new_segment_kv.dim() == 3 else new_segment_kv
            else:
                cached_k = cached_kv
                new_k = new_segment_kv

            # Ensure same number of tokens
            num_tokens = min(cached_k.shape[0], new_k.shape[0])
            cached_k = cached_k[:num_tokens]
            new_k = new_k[:num_tokens]

            # Compute deviation and select HKVD
            deviation = self.compute_kv_deviation(new_k, cached_k)
            hkvd_indices = self.select_hkvd_tokens(deviation, self._metadata.recompute_ratio)
            self._hkvd_indices[cache_key] = hkvd_indices

            logger.info(
                f"[CACHEBLEND] Layer {layer_id}: selected {len(hkvd_indices)}/{num_tokens} "
                f"HKVD tokens for {cache_key}"
            )

        # Blend KV using HKVD indices
        if cache_key in self._hkvd_indices:
            hkvd_indices = self._hkvd_indices[cache_key]
            blended_kv = self.blend_kv(new_segment_kv, cached_kv, hkvd_indices)
            return blended_kv
        else:
            # No HKVD selected yet, return cached KV
            return cached_kv

    def _adjust_k_position(
        self,
        kv_data: torch.Tensor,
        old_start: int,
        new_start: int,
        num_tokens: int,
    ) -> torch.Tensor:
        """Realign K position encoding using RoPE."""
        if self.rope_adjuster is None:
            return kv_data

        try:
            from agentmesh.mechanisms.bpp.vllm.position_encoding import (
                adjust_kv_positions,
            )
            return adjust_kv_positions(kv_data, old_start, new_start, self.rope_adjuster)
        except Exception as e:
            logger.error(f"Position adjustment failed: {e}")
            return kv_data

    def get_hkvd_indices(self, cache_key: str) -> Optional[torch.Tensor]:
        """Get HKVD indices for a segment."""
        return self._hkvd_indices.get(cache_key)

    def clear(self) -> None:
        """Clear processor state."""
        self._metadata = None
        self._cached_kv.clear()
        self._hkvd_indices.clear()


def create_blend_metadata_s7(
    master_tokens: int,
    worker1_tokens: int,
    worker2_tokens: int,
    worker3_tokens: int,
    worker1_prompt_len: int,
    worker2_prompt_len: int,
    worker3_prompt_len: int,
    recompute_ratio: float = 0.15,
) -> CacheBlendMetadata:
    """
    Create CacheBlend metadata for S7 scenario (sequential blending).

    In S7, each branch is blended with the accumulated prefix:
    - branch1: prefix = supervisor
    - branch2: prefix = supervisor + branch1
    - branch3: prefix = supervisor + branch1 + branch2
    """
    segments = [
        BlendSegment(
            cache_key="worker1",
            num_tokens=worker1_tokens,
            old_start_pos=worker1_prompt_len,
            new_start_pos=master_tokens,
            slot_start=master_tokens,
            slot_end=master_tokens + worker1_tokens,
        ),
        BlendSegment(
            cache_key="worker2",
            num_tokens=worker2_tokens,
            old_start_pos=worker2_prompt_len,
            new_start_pos=master_tokens + worker1_tokens,
            slot_start=master_tokens + worker1_tokens,
            slot_end=master_tokens + worker1_tokens + worker2_tokens,
        ),
        BlendSegment(
            cache_key="worker3",
            num_tokens=worker3_tokens,
            old_start_pos=worker3_prompt_len,
            new_start_pos=master_tokens + worker1_tokens + worker2_tokens,
            slot_start=master_tokens + worker1_tokens + worker2_tokens,
            slot_end=master_tokens + worker1_tokens + worker2_tokens + worker3_tokens,
        ),
    ]

    return CacheBlendMetadata(
        enabled=True,
        blend_segments=segments,
        recompute_ratio=recompute_ratio,
        check_layer=0,
    )


def create_blend_metadata_s9(
    master_tokens: int,
    worker1_tokens: int,
    worker2_tokens: int,
    worker3_tokens: int,
    worker1_prompt_len: int,
    worker2_prompt_len: int,
    worker3_prompt_len: int,
    epic_k: int = 16,
) -> CacheBlendMetadata:
    """
    Create EPIC metadata for S9 scenario (sequential EPIC/LegoLink).

    S9 uses EPIC's static token selection (first k tokens) instead of
    CacheBlend's dynamic HKVD selection:
    - branch1: prefix = supervisor, recompute first k tokens
    - branch2: prefix = supervisor + branch1, recompute first k tokens
    - branch3: prefix = supervisor + branch1 + branch2, recompute first k tokens

    This is more efficient than CacheBlend but may be slightly less accurate.
    """
    segments = [
        BlendSegment(
            cache_key="worker1",
            num_tokens=worker1_tokens,
            old_start_pos=worker1_prompt_len,
            new_start_pos=master_tokens,
            slot_start=master_tokens,
            slot_end=master_tokens + worker1_tokens,
        ),
        BlendSegment(
            cache_key="worker2",
            num_tokens=worker2_tokens,
            old_start_pos=worker2_prompt_len,
            new_start_pos=master_tokens + worker1_tokens,
            slot_start=master_tokens + worker1_tokens,
            slot_end=master_tokens + worker1_tokens + worker2_tokens,
        ),
        BlendSegment(
            cache_key="worker3",
            num_tokens=worker3_tokens,
            old_start_pos=worker3_prompt_len,
            new_start_pos=master_tokens + worker1_tokens + worker2_tokens,
            slot_start=master_tokens + worker1_tokens + worker2_tokens,
            slot_end=master_tokens + worker1_tokens + worker2_tokens + worker3_tokens,
        ),
    ]

    # Note: recompute_ratio is not used in EPIC mode; epic_k is used instead
    return CacheBlendMetadata(
        enabled=True,
        blend_segments=segments,
        recompute_ratio=0.0,  # Not used for EPIC
        check_layer=0,
    )


def create_blend_metadata_s10(
    master_tokens: int,
    worker1_tokens: int,
    worker2_tokens: int,
    worker3_tokens: int,
    worker1_prompt_len: int,
    worker2_prompt_len: int,
    worker3_prompt_len: int,
    epic_k: int = 16,
) -> CacheBlendMetadata:
    """
    Create EPIC metadata for S10 scenario (parallel EPIC/LegoLink + position adjust).

    S10 is the parallel variant of S9:
    - All branches are blended as if appended to the supervisor (parallel)
    - First k tokens of each branch segment are recomputed
    - Position encoding is realigned after blending

    This combines EPIC's efficiency with BPP's parallel KV stitching.
    """
    # For S10, initial positions are all at master_tokens (parallel blend)
    # Final position realignment happens after blending
    segments = [
        BlendSegment(
            cache_key="worker1",
            num_tokens=worker1_tokens,
            old_start_pos=worker1_prompt_len,
            new_start_pos=master_tokens,  # Blend position
            slot_start=master_tokens,
            slot_end=master_tokens + worker1_tokens,
        ),
        BlendSegment(
            cache_key="worker2",
            num_tokens=worker2_tokens,
            old_start_pos=worker2_prompt_len,
            new_start_pos=master_tokens,  # Blend position (parallel)
            slot_start=master_tokens + worker1_tokens,
            slot_end=master_tokens + worker1_tokens + worker2_tokens,
        ),
        BlendSegment(
            cache_key="worker3",
            num_tokens=worker3_tokens,
            old_start_pos=worker3_prompt_len,
            new_start_pos=master_tokens,  # Blend position (parallel)
            slot_start=master_tokens + worker1_tokens + worker2_tokens,
            slot_end=master_tokens + worker1_tokens + worker2_tokens + worker3_tokens,
        ),
    ]

    return CacheBlendMetadata(
        enabled=True,
        blend_segments=segments,
        recompute_ratio=0.0,  # Not used for EPIC
        check_layer=0,
    )


def select_legolink_tokens(num_tokens: int, k: int) -> torch.Tensor:
    """
    EPIC (LegoLink): Select first k tokens for recomputation.

    Unlike CacheBlend's dynamic HKVD selection, EPIC statically selects
    the first k tokens of each chunk based on the "attention sink" phenomenon:
    - Initial tokens of independently-compiled chunks absorb disproportionate attention
    - Recomputing first k tokens allows them to recognize non-initial position
    - This releases attention to more semantically relevant positions

    Complexity: O(kN) vs CacheBlend's O(15%N²), where k << N

    Args:
        num_tokens: Total number of tokens in the segment
        k: Number of initial tokens to select for recomputation

    Returns:
        legolink_indices: Indices of selected tokens [min(k, num_tokens)]
    """
    actual_k = min(k, num_tokens)
    return torch.arange(actual_k, dtype=torch.long)


def create_blend_metadata_s8(
    master_tokens: int,
    worker1_tokens: int,
    worker2_tokens: int,
    worker3_tokens: int,
    worker1_prompt_len: int,
    worker2_prompt_len: int,
    worker3_prompt_len: int,
    recompute_ratio: float = 0.15,
) -> CacheBlendMetadata:
    """
    Create CacheBlend metadata for S8 scenario (parallel blending + position adjust).

    In S8, all branches are blended as if appended to the supervisor (parallel):
    - branch1: prefix = supervisor, position realigned from branch1 prompt_len to supervisor_len
    - branch2: prefix = supervisor, position realigned from branch2 prompt_len to supervisor_len
    - branch3: prefix = supervisor, position realigned from branch3 prompt_len to supervisor_len

    Then position encoding is further realigned:
    - branch2: supervisor_len → supervisor_len + branch1_len
    - branch3: supervisor_len → supervisor_len + branch1_len + branch2_len
    """
    # For S8, initial positions are all at master_tokens (parallel blend)
    # Final position realignment happens after blending
    segments = [
        BlendSegment(
            cache_key="worker1",
            num_tokens=worker1_tokens,
            old_start_pos=worker1_prompt_len,
            new_start_pos=master_tokens,  # Blend position
            slot_start=master_tokens,
            slot_end=master_tokens + worker1_tokens,
        ),
        BlendSegment(
            cache_key="worker2",
            num_tokens=worker2_tokens,
            old_start_pos=worker2_prompt_len,
            new_start_pos=master_tokens,  # Blend position (parallel)
            slot_start=master_tokens + worker1_tokens,
            slot_end=master_tokens + worker1_tokens + worker2_tokens,
        ),
        BlendSegment(
            cache_key="worker3",
            num_tokens=worker3_tokens,
            old_start_pos=worker3_prompt_len,
            new_start_pos=master_tokens,  # Blend position (parallel)
            slot_start=master_tokens + worker1_tokens + worker2_tokens,
            slot_end=master_tokens + worker1_tokens + worker2_tokens + worker3_tokens,
        ),
    ]

    return CacheBlendMetadata(
        enabled=True,
        blend_segments=segments,
        recompute_ratio=recompute_ratio,
        check_layer=0,
    )
