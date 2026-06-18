# SPDX-License-Identifier: Apache-2.0
"""
BPP late RoPE realignment for stitched KV.

When AgentMesh prefills independent worker *branches* in parallel against a
shared supervisor prefix (Branch-Parallel Prefill), each branch's KV is produced
at branch-local positions. To stitch the branches into one contiguous sequence
for the consuming LLM prefill, the cached K of each branch must be re-rotated
from its produce-time position to its final position in the merged sequence.
This module performs that *late RoPE realignment*: it undoes the original RoPE
phase and re-applies RoPE at the new offset, on K only (V is position-free).

Two backends are provided:
  * a fused LMCache CUDA kernel (``rotary_embedding_k_fused``) when available,
  * a pure-torch fallback that reverses the old rotation via a shuffle trick and
    re-rotates at the new positions.

Heavy dependencies (``torch``, vLLM's ``get_rope``, optional ``lmcache``) are
imported lazily so that this module can be byte-compiled / imported on hosts
without a GPU stack.
"""

from typing import Any, Dict, Optional

import torch

from vllm.model_executor.layers.rotary_embedding import get_rope as vllm_get_rope
from vllm.logger import init_logger

logger = init_logger(__name__)

HAS_CUDA = torch.cuda.is_available()

HAS_LMCACHE_OPS = False
if HAS_CUDA:
    try:
        import lmcache.c_ops as lmc_ops
        HAS_LMCACHE_OPS = True
    except ImportError:
        pass


class FusedRopeAdjuster:
    """Fused RoPE realigner for BPP-stitched KV.

    Re-rotates the K of a branch segment from its produce-time positions to the
    target positions it must occupy in the stitched sequence consumed by the
    downstream LLM prefill.
    """

    def __init__(self, rope, is_neox_style: bool = True):
        self.rope = rope
        self.is_neox_style = is_neox_style
        self.head_size = rope.head_size
        self.cos_sin_cache = rope.cos_sin_cache
        self.rotary_dim = rope.rotary_dim

    def adjust_k_positions(
        self,
        k: torch.Tensor,
        old_positions: torch.Tensor,
        new_positions: torch.Tensor
    ) -> torch.Tensor:
        """Realign K cache position encoding."""
        if HAS_LMCACHE_OPS:
            return self._fused_adjust(k, old_positions, new_positions)
        else:
            return self._fallback_adjust(k, old_positions, new_positions)

    def _fused_adjust(
        self,
        k: torch.Tensor,
        old_positions: torch.Tensor,
        new_positions: torch.Tensor
    ) -> torch.Tensor:
        """Use LMCache fused kernel for realignment."""
        num_tokens = k.shape[0]
        original_shape = k.shape

        if k.dim() == 2:
            hidden_dim = k.shape[1]
        else:
            hidden_dim = k.numel() // num_tokens

        if hidden_dim % self.head_size != 0:
            return self._fallback_adjust(k, old_positions, new_positions)

        k = k.view(num_tokens, -1, self.head_size)

        lmc_ops.rotary_embedding_k_fused(
            old_positions,
            new_positions,
            k,
            self.head_size,
            self.cos_sin_cache.to(k.device),
            self.is_neox_style,
        )

        return k.view(original_shape)

    def _fallback_adjust(
        self,
        k: torch.Tensor,
        old_positions: torch.Tensor,
        new_positions: torch.Tensor
    ) -> torch.Tensor:
        """Fallback: reverse old encoding, apply new encoding."""
        num_tokens = k.shape[0]
        original_shape = k.shape

        k = k.view(num_tokens, -1)
        hidden_dim = k.shape[1]

        if hidden_dim % self.rotary_dim != 0:
            return k.view(original_shape)

        try:
            k_reversed = self._reverse_rope(k, old_positions)
            dummy_q = torch.zeros_like(k_reversed)
            _, k_new = self.rope(new_positions, dummy_q, k_reversed)
            return k_new.view(original_shape)
        except Exception:
            return k.view(original_shape)

    def _reverse_rope(self, k: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Reverse RoPE encoding via shuffle trick."""
        num_tokens = k.shape[0]
        hidden_dim = k.shape[1]

        if hidden_dim % self.rotary_dim != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) not divisible by rotary_dim ({self.rotary_dim})")

        k = k.view(num_tokens, -1, self.rotary_dim)

        if self.is_neox_style:
            o1, o2 = torch.chunk(k, 2, dim=-1)
            k_shuffled = torch.cat((o2, o1), dim=-1)
        else:
            o1 = k[..., ::2]
            o2 = k[..., 1::2]
            k_shuffled = torch.stack((o2, o1), dim=-1).reshape(k.shape)

        k_shuffled = k_shuffled.view(num_tokens, hidden_dim)
        dummy_q = torch.zeros_like(k_shuffled)
        _, k_roped = self.rope(positions, dummy_q, k_shuffled)

        k_roped = k_roped.view(num_tokens, -1, self.rotary_dim)
        if self.is_neox_style:
            o1, o2 = torch.chunk(k_roped, 2, dim=-1)
            k_reversed = torch.cat((o2, o1), dim=-1)
        else:
            o1 = k_roped[..., ::2]
            o2 = k_roped[..., 1::2]
            k_reversed = torch.stack((o2, o1), dim=-1).reshape(k_roped.shape)

        return k_reversed.view(num_tokens, hidden_dim)

    def __call__(
        self,
        k: torch.Tensor,
        old_positions: torch.Tensor,
        new_positions: torch.Tensor
    ) -> torch.Tensor:
        return self.adjust_k_positions(k, old_positions, new_positions)


def get_rope_adjuster_v2(
    head_size: int,
    max_position: int,
    is_neox_style: bool = True,
    rope_parameters: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Optional[FusedRopeAdjuster]:
    """Get RoPE realigner (new vLLM API)."""
    try:
        rope = vllm_get_rope(
            head_size=head_size,
            max_position=max_position,
            is_neox_style=is_neox_style,
            rope_parameters=rope_parameters,
            dtype=dtype,
        )
        return FusedRopeAdjuster(rope, is_neox_style)
    except Exception as e:
        logger.error("Failed to create RoPE: %s", e)
        return None


def get_rope_adjuster(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    partial_rotary_factor: float = 1.0,
) -> Optional[FusedRopeAdjuster]:
    """Get RoPE realigner (legacy API)."""
    rope_parameters: Dict[str, Any] = {
        "rope_theta": base,
        "partial_rotary_factor": partial_rotary_factor,
    }

    if rope_scaling is not None:
        rope_parameters["rope_type"] = rope_scaling.get("type", "default")
        rope_parameters.update(rope_scaling)
    else:
        rope_parameters["rope_type"] = "default"

    return get_rope_adjuster_v2(
        head_size=head_size,
        max_position=max_position,
        is_neox_style=is_neox_style,
        rope_parameters=rope_parameters,
        dtype=dtype,
    )


def adjust_kv_positions(
    k_cache: torch.Tensor,
    old_start_pos: int,
    new_start_pos: int,
    rope_adjuster: FusedRopeAdjuster,
) -> torch.Tensor:
    """Realign a branch's K cache from ``old_start_pos`` to ``new_start_pos``.

    The contiguous block of tokens is re-rotated so the stitched sequence is
    positionally consistent for the consuming LLM prefill. Only K is touched.
    """
    original_dtype = k_cache.dtype
    device = k_cache.device

    # Check if KV combined format [2, num_tokens, ...]
    is_kv_combined = len(k_cache.shape) >= 2 and k_cache.shape[0] == 2

    if is_kv_combined:
        k = k_cache[0]  # Only realign K, V unchanged
    else:
        k = k_cache

    num_tokens = k.shape[0]

    old_positions = torch.arange(
        old_start_pos, old_start_pos + num_tokens,
        dtype=torch.long, device=device
    )
    new_positions = torch.arange(
        new_start_pos, new_start_pos + num_tokens,
        dtype=torch.long, device=device
    )

    rope_dtype = rope_adjuster.cos_sin_cache.dtype
    if k.dtype != rope_dtype:
        k = k.to(rope_dtype)

    k_adjusted = rope_adjuster(k, old_positions, new_positions)

    if k_adjusted.dtype != original_dtype:
        k_adjusted = k_adjusted.to(original_dtype)

    if is_kv_combined:
        result = k_cache.clone()
        result[0] = k_adjusted
        return result
    else:
        return k_adjusted
