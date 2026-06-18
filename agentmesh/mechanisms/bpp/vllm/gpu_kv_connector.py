# SPDX-License-Identifier: Apache-2.0
"""
GPUKVConnector — the production vLLM realization of BPP (Branch-Parallel Prefill).

This connector is how AgentMesh removes the BARRIER in the *consume* stage of an
LLM→LLM dataflow. A structure-blind runtime prefills the supervisor sequence
serially: each worker branch waits for the previous one to finish. BPP instead
prefills every independent worker branch in parallel against the shared
supervisor prefix, then *stitches* the per-branch KV into one contiguous
sequence for the final consuming LLM prefill. Two ingredients make the stitch
correct:

  1. Slot-based block-diagonal isolation — Branch-Parallel Attention (BPA).
     Each branch occupies a disjoint range of KV slots, so during the parallel
     prefill a branch attends only to the shared prefix plus its own tokens and
     never to a sibling branch. The disjoint slot layout *is* the cross-branch
     attention mask.
  2. Late RoPE realignment (see ``position_encoding.py``). A branch's K is
     produced at branch-local positions; before the consuming prefill reads the
     stitched sequence, each branch's K is re-rotated to its final offset.

Optional selective KV recompute (CacheBlend / EPIC, see ``cache_blend.py``) is
*orthogonal and composable*: it recovers the small amount of cross-branch
attention that parallel prefill omits by recomputing a few tokens per branch.

Protocol summary (request types, kept intact for scheduler compatibility):
1. Store KV from the supervisor prompt          (key="master")
2. Store incremental KV from branch 1           (key="worker1")
3. Store incremental KV from branch 2/3         (key="worker2"/"worker3")
4. For the FINAL request, load and stitch all KV segments with RoPE realignment.

The scenario matrix (S1–S10) selected by the ``KV_SCENARIO`` env var maps onto
the BPP ablations; see ``README.md``. vLLM-internal imports are kept verbatim so
the connector drops into a vLLM tree unchanged; they are unused on hosts without
vLLM but do not affect byte-compilation.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from enum import Enum
import os

import torch

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput

from agentmesh.mechanisms.bpp.vllm.position_encoding import (
    adjust_kv_positions,
    FusedRopeAdjuster,
    get_rope_adjuster_v2,
)
from agentmesh.mechanisms.bpp.vllm.cache_blend import (
    CacheBlendConfig,
    CacheBlendMetadata,
    CacheBlendProcessor,
    BlendSegment,
    create_blend_metadata_s7,
    create_blend_metadata_s8,
    # EPIC/LegoLink support
    EPICConfig,
    create_blend_metadata_s9,
    create_blend_metadata_s10,
    select_legolink_tokens,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class RequestType(Enum):
    """Request type enumeration.

    ``MASTER`` carries the supervisor prefix; ``WORKER1..3`` carry the
    independent worker branches; ``FINAL`` triggers the stitched consuming
    prefill. The string values are part of the on-wire protocol and must not
    change.
    """
    MASTER = "master"
    WORKER1 = "worker1"
    WORKER2 = "worker2"
    WORKER3 = "worker3"
    FINAL = "final"
    UNKNOWN = "unknown"


# Global GPU KV Cache storage
# Structure: {key: {layer_name: kv_tensor}}
GPU_KV_CACHE: Dict[str, Dict[str, torch.Tensor]] = {}

# Scheduler state for cross-request tracking
SCHEDULER_STATE = {
    "master_stored": False,
    "worker1_stored": False,
    "worker2_stored": False,
    "worker3_stored": False,
    "master_tokens": 0,
    "worker1_tokens": 0,
    "worker2_tokens": 0,
    "worker3_tokens": 0,
    "master_prompt": None,
    "worker1_prompt": None,
    "worker2_prompt": None,
    "worker3_prompt": None,
    "scenario_config": None,  # ScenarioConfig from driver script
}


def clear_gpu_kv_cache():
    """Clear GPU KV Cache and scheduler state."""
    global GPU_KV_CACHE
    GPU_KV_CACHE.clear()
    SCHEDULER_STATE.clear()
    SCHEDULER_STATE.update({
        "master_stored": False,
        "worker1_stored": False,
        "worker2_stored": False,
        "worker3_stored": False,
        "master_tokens": 0,
        "worker1_tokens": 0,
        "worker2_tokens": 0,
        "worker3_tokens": 0,
        "master_prompt": None,
        "worker1_prompt": None,
        "worker2_prompt": None,
        "worker3_prompt": None,
        "scenario_config": None,
    })
    logger.info("GPU KV Cache cleared")


def get_gpu_kv_cache_info() -> Dict[str, Any]:
    """Get cache information."""
    info = {
        "cached_keys": list(GPU_KV_CACHE.keys()),
        "scheduler_state": SCHEDULER_STATE.copy(),
    }
    for key, layers in GPU_KV_CACHE.items():
        if layers:
            first_layer = next(iter(layers.values()))
            info[f"{key}_shape"] = list(first_layer.shape)
            info[f"{key}_num_layers"] = len(layers)
    return info


@dataclass
class PosAdjustment:
    """Late-RoPE realignment config for one branch KV segment."""
    old_pos: int = 0
    new_pos: int = 0


@dataclass
class ReqMeta:
    """Metadata for a single request."""
    request_type: RequestType
    token_ids: List[int]
    slot_mapping: torch.Tensor
    # Save config
    save_start: int = 0
    save_end: int = 0
    save_key: str = ""
    # Load config
    load_keys: List[str] = field(default_factory=list)
    # Late-RoPE realignment config: {key: PosAdjustment}
    adjust_configs: Dict[str, PosAdjustment] = field(default_factory=dict)


@dataclass
class GPUKVConnectorMetadata(KVConnectorMetadata):
    """GPU KV Connector metadata."""
    requests: List[ReqMeta] = field(default_factory=list)


class GPUKVConnector(KVConnectorBase_V1):
    """vLLM KV connector implementing BPP branch-parallel prefill and KV stitching."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: Dict[str, "Request"] = {}
        self._vllm_config = vllm_config

        # RoPE realigner for late position realignment (lazy init)
        self._rope_adjuster: Optional[FusedRopeAdjuster] = None
        self._rope_adjuster_init_attempted = False

        # Track which layers have been saved
        self._save_logged: Dict[str, bool] = {}

        # CacheBlend processor for S7/S8 scenarios
        self._cacheblend_processor: Optional[CacheBlendProcessor] = None
        self._cacheblend_config = CacheBlendConfig(
            enabled=False,
            recompute_ratio=float(os.environ.get("CACHEBLEND_RECOMPUTE_RATIO", "0.15")),
            check_layer=int(os.environ.get("CACHEBLEND_CHECK_LAYER", "0")),
        )

        # Track CacheBlend state for FINAL request
        self._cacheblend_active = False
        self._cacheblend_hkvd_indices: Dict[str, torch.Tensor] = {}  # Per-segment HKVD indices
        self._cacheblend_segment_info: Dict[str, BlendSegment] = {}  # Segment info for save_kv_layer
        self._cacheblend_first_layer_done = False

        # CacheBlend: Store cached KV copies for comparison in save_kv_layer
        self._cacheblend_cached_kv_copy: Dict[str, Dict[str, torch.Tensor]] = {}  # {key: {layer: kv}}
        self._cacheblend_slot_mapping: Optional[torch.Tensor] = None
        self._cacheblend_master_len: int = 0
        self._cacheblend_blend_segments: List[BlendSegment] = []
        self._cacheblend_parallel_mode: bool = False

        logger.info("GPUKVConnector initialized, block_size=%d", self._block_size)

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Load (and stitch) branch KV from the GPU cache into vLLM's paged buffer.

        For S7/S8 scenarios, this also handles CacheBlend selective KV recompute.
        """
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return

        metadata = self._get_connector_metadata()
        if not isinstance(metadata, GPUKVConnectorMetadata):
            return

        if self._rope_adjuster is None:
            self._init_rope_adjuster(forward_context)

        # Check if CacheBlend is enabled (S7/S8 scenarios)
        _, _, _, _, _, use_cacheblend, parallel_blend = self._get_scenario_from_env()

        for req_meta in metadata.requests:
            if not req_meta.load_keys:
                continue

            # For FINAL request in S7/S8, use CacheBlend processing
            if (use_cacheblend and
                req_meta.request_type == RequestType.FINAL and
                "worker1" in req_meta.load_keys):
                self._start_load_kv_with_cacheblend(
                    forward_context, req_meta, attn_metadata, parallel_blend
                )
                continue

            total_loaded = 0
            load_details = []

            slot_offset = 0
            for key in req_meta.load_keys:
                if key not in GPU_KV_CACHE:
                    logger.warning("[LOAD KV] missing cache key=%s", key)
                    continue

                cached_layers = GPU_KV_CACHE[key]
                if not cached_layers:
                    logger.warning("[LOAD KV] missing cache key=%s", key)
                    continue

                first_kv = next(iter(cached_layers.values()))
                if first_kv.dim() == 3:
                    # [2, num_tokens, hidden_dim]
                    num_tokens = first_kv.shape[1]
                else:
                    # [num_tokens, hidden_dim] (MLA)
                    num_tokens = first_kv.shape[0]

                if num_tokens == 0:
                    continue

                slot_start = slot_offset
                slot_end = slot_offset + num_tokens

                if slot_end > len(req_meta.slot_mapping):
                    logger.warning(
                        f"Not enough slots: need {slot_end} but only have {len(req_meta.slot_mapping)}"
                    )
                    slot_end = len(req_meta.slot_mapping)
                    num_tokens = slot_end - slot_start
                    if num_tokens <= 0:
                        continue

                target_slots = req_meta.slot_mapping[slot_start:slot_end].cuda()

                # Check if this key needs late-RoPE realignment
                adjust_config = req_meta.adjust_configs.get(key)
                need_adjust = adjust_config is not None

                if need_adjust:
                    logger.info(
                        "[ADJUST POS] key=%s: pos %d -> %d for %d tokens",
                        key, adjust_config.old_pos, adjust_config.new_pos, num_tokens
                    )

                for layer_name in forward_context.no_compile_layers:
                    layer = forward_context.no_compile_layers[layer_name]
                    kv_cache_attr = getattr(layer, "kv_cache", None)
                    if kv_cache_attr is None or layer_name not in cached_layers:
                        continue

                    kv_cache_layer = kv_cache_attr[forward_context.virtual_engine]
                    kv_data = cached_layers[layer_name]

                    if kv_data.dim() == 3:
                        kv_data = kv_data[:, :num_tokens, :]
                    else:
                        kv_data = kv_data[:num_tokens, :]

                    if need_adjust:
                        kv_data = self._adjust_k_position(
                            kv_data, adjust_config.old_pos,
                            adjust_config.new_pos, num_tokens)

                    self._inject_kv(kv_cache_layer, kv_data, target_slots, attn_metadata)

                slot_offset += num_tokens
                total_loaded += num_tokens
                load_details.append(f"{key}:{num_tokens}")

            if total_loaded > 0:
                logger.info(
                    "[LOAD KV] %s request: loaded %d tokens (%s)",
                    req_meta.request_type.value, total_loaded, ", ".join(load_details)
                )

    def _inject_kv(
        self,
        dst_layer: torch.Tensor,
        src_kv: torch.Tensor,
        slot_mapping: torch.Tensor,
        attn_metadata: "AttentionMetadata",
    ) -> None:
        """Inject KV into destination layer at the branch's disjoint slot range."""
        if isinstance(attn_metadata, MLACommonMetadata):
            num_pages, page_size = dst_layer.shape[0], dst_layer.shape[1]
            dst_flat = dst_layer.reshape(num_pages * page_size, -1)
            dst_flat[slot_mapping, ...] = src_kv
        else:
            num_pages, page_size = dst_layer.shape[1], dst_layer.shape[2]
            dst_flat = dst_layer.reshape(2, num_pages * page_size, -1)
            dst_flat[:, slot_mapping, ...] = src_kv

    def _get_rotary_emb_from_model(self, forward_context: "ForwardContext") -> Optional[Any]:
        """Get rotary_emb instance from model layers.

        This extracts the actual RoPE configuration used by the model,
        ensuring correct late RoPE realignment for all model types
        (Qwen3, gpt-oss, Llama3, etc.).
        """
        for layer_name, layer in forward_context.no_compile_layers.items():
            # Check self_attn (common in most models)
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                if hasattr(attn, 'rotary_emb') and attn.rotary_emb is not None:
                    return attn.rotary_emb
            # Check attn (used in some models like gpt-oss)
            if hasattr(layer, 'attn'):
                attn = layer.attn
                if hasattr(attn, 'rotary_emb') and attn.rotary_emb is not None:
                    return attn.rotary_emb
            # Check layer itself (some architectures)
            if hasattr(layer, 'rotary_emb') and layer.rotary_emb is not None:
                return layer.rotary_emb
        return None

    def _init_rope_adjuster(self, forward_context: "ForwardContext") -> None:
        """Initialize the late-RoPE realigner.

        Priority:
        1. Get rotary_emb directly from model (most accurate, works for all models)
        2. Fallback to hf_config.rope_parameters (gpt-oss, qwen3, etc.)
        3. Fallback to hf_config.rope_scaling (legacy models)
        """
        if self._rope_adjuster_init_attempted:
            return
        self._rope_adjuster_init_attempted = True

        try:
            # Priority 1: Get rotary_emb directly from model layers
            # This is the most accurate method as it uses the exact RoPE config
            # that the model is using (including YARN, Llama3, etc.)
            rotary_emb = self._get_rotary_emb_from_model(forward_context)

            if rotary_emb is not None:
                # Extract config from the rotary_emb instance
                head_size = getattr(rotary_emb, 'head_size', None)
                is_neox_style = getattr(rotary_emb, 'is_neox_style', True)
                cos_sin_cache = getattr(rotary_emb, 'cos_sin_cache', None)
                base = getattr(rotary_emb, 'base', 10000)
                rotary_dim = getattr(rotary_emb, 'rotary_dim', head_size)

                if head_size is not None and cos_sin_cache is not None:
                    # Create FusedRopeAdjuster directly with the model's rotary_emb
                    self._rope_adjuster = FusedRopeAdjuster(rotary_emb, is_neox_style)
                    self._head_dim = head_size

                    # Get rope_type from class name if possible
                    rope_type = type(rotary_emb).__name__

                    logger.info(
                        "RoPE realigner from model: head_size=%d, base=%.0f, "
                        "rotary_dim=%d, is_neox=%s, type=%s",
                        head_size, base, rotary_dim, is_neox_style, rope_type
                    )
                    return

            # Priority 2 & 3: Fallback to config-based initialization
            logger.info("Falling back to config-based RoPE realigner initialization")
            hf_config = self._vllm_config.model_config.hf_config

            hidden_size = getattr(hf_config, "hidden_size", None)
            num_heads = getattr(hf_config, "num_attention_heads", None)
            num_kv_heads = (
                getattr(hf_config, "num_key_value_heads", None)
                or getattr(hf_config, "num_kv_heads", None)
                or num_heads
            )
            head_dim = getattr(hf_config, "head_dim", None)
            if head_dim is None and hidden_size is not None and num_heads:
                head_dim = hidden_size // num_heads
            max_position = (
                getattr(hf_config, "max_position_embeddings", None)
                or getattr(hf_config, "max_sequence_length", None)
                or getattr(hf_config, "seq_length", None)
            )
            rope_style = getattr(hf_config, "rope_style", "neox")
            is_neox_style = rope_style.lower() != "gptj"

            if None in (hidden_size, num_heads, num_kv_heads, head_dim, max_position):
                raise ValueError(
                    "Missing rope parameters: hidden_size=%s num_heads=%s num_kv_heads=%s "
                    "head_dim=%s max_position=%s"
                    % (hidden_size, num_heads, num_kv_heads, head_dim, max_position)
                )

            self._head_dim = head_dim
            self._num_kv_heads = num_kv_heads
            self._kv_hidden_dim = num_kv_heads * head_dim

            # Priority 2: Check rope_parameters (used by gpt-oss, qwen3, etc.)
            rope_parameters_from_config = getattr(hf_config, "rope_parameters", None)
            if rope_parameters_from_config is not None:
                rope_parameters = dict(rope_parameters_from_config)
                if "rope_type" not in rope_parameters:
                    rope_parameters["rope_type"] = "default"
                rope_theta = rope_parameters.get("rope_theta", 10000)
                logger.info(
                    "Using rope_parameters from config: rope_type=%s, rope_theta=%.0f",
                    rope_parameters.get("rope_type", "default"), rope_theta
                )
            else:
                # Priority 3: Fallback to rope_scaling (legacy models)
                rope_theta = getattr(hf_config, "rope_theta", None) or getattr(
                    hf_config, "rotary_emb_base", None
                ) or 10000
                rope_scaling = getattr(hf_config, "rope_scaling", None) or getattr(
                    hf_config, "rope_scaling_factor", None
                )
                rope_parameters = {"rope_theta": rope_theta}
                if rope_scaling is not None:
                    rope_parameters["rope_type"] = rope_scaling.get("type", "default")
                    rope_parameters.update(rope_scaling)
                else:
                    rope_parameters["rope_type"] = "default"

            self._rope_adjuster = get_rope_adjuster_v2(
                head_size=head_dim,
                max_position=max_position,
                is_neox_style=is_neox_style,
                rope_parameters=rope_parameters,
                dtype=getattr(self._vllm_config, "dtype", None),
            )

            logger.info(
                "RoPE realigner initialized from config: head_dim=%d, num_kv_heads=%d, "
                "theta=%.0f, rope_type=%s, style=%s",
                head_dim, num_kv_heads, rope_theta,
                rope_parameters.get("rope_type", "default"),
                "neox" if is_neox_style else "gptj"
            )
        except Exception as e:
            logger.warning("Failed to initialize rope_adjuster: %s", e)

    def _adjust_k_position(
        self,
        kv_data: torch.Tensor,
        old_start: int,
        new_start: int,
        num_tokens: int,
    ) -> torch.Tensor:
        """Late-RoPE realign K of a branch segment from old to new start position."""
        if self._rope_adjuster is None:
            return kv_data

        try:
            return adjust_kv_positions(kv_data, old_start, new_start, self._rope_adjuster)
        except Exception as e:
            logger.error("Position realignment failed: %s", e)
            return kv_data

    def _start_load_kv_with_cacheblend(
        self,
        forward_context: "ForwardContext",
        req_meta: ReqMeta,
        attn_metadata: "AttentionMetadata",
        parallel_blend: bool,
    ) -> None:
        """
        CacheBlend Plan A: Load KV with selective recompute for S7/S8 scenarios.

        CacheBlend workflow:
        1. Load supervisor KV directly (no blending needed for the shared prefix)
        2. For each branch segment:
           a. Load cached KV from the branch
           b. Late-RoPE realign if needed
           c. Select HKVD tokens (first layer only)
           d. Blend new KV with cached KV at HKVD positions
           e. Inject blended KV into paged buffer

        CacheBlend Plan B: Load cached KV and prepare for selective fusion in save_kv_layer.

        Workflow:
        1. Load supervisor KV directly (no blending needed for the shared prefix)
        2. For each branch segment:
           a. Load cached KV (with realignment) into paged buffer
           b. Save a copy of the cached KV for later comparison
        3. Model forward will compute new KV with correct cross-branch attention
        4. In save_kv_layer: compare new KV with cached KV, select HKVD, blend results
        """
        master_len = SCHEDULER_STATE["master_tokens"]
        worker1_len = SCHEDULER_STATE["worker1_tokens"]
        worker2_len = SCHEDULER_STATE["worker2_tokens"]
        worker3_len = SCHEDULER_STATE["worker3_tokens"]

        prompt_worker1_len = int(os.environ.get("KV_PROMPT_WORKER1_LEN", "0"))
        prompt_worker2_len = int(os.environ.get("KV_PROMPT_WORKER2_LEN", "0"))
        prompt_worker3_len = int(os.environ.get("KV_PROMPT_WORKER3_LEN", "0"))
        recompute_ratio = self._cacheblend_config.recompute_ratio

        # Create CacheBlend metadata
        if parallel_blend:
            blend_meta = create_blend_metadata_s8(
                master_len, worker1_len, worker2_len, worker3_len,
                prompt_worker1_len, prompt_worker2_len, prompt_worker3_len,
                recompute_ratio
            )
            logger.info("[CACHEBLEND S8] Parallel blend mode, ratio=%.2f", recompute_ratio)
        else:
            blend_meta = create_blend_metadata_s7(
                master_len, worker1_len, worker2_len, worker3_len,
                prompt_worker1_len, prompt_worker2_len, prompt_worker3_len,
                recompute_ratio
            )
            logger.info("[CACHEBLEND S7] Sequential blend mode, ratio=%.2f", recompute_ratio)

        # Activate CacheBlend mode and store metadata for save_kv_layer
        self._cacheblend_active = True
        self._cacheblend_parallel_mode = parallel_blend
        self._cacheblend_slot_mapping = req_meta.slot_mapping.clone()
        self._cacheblend_master_len = master_len
        self._cacheblend_blend_segments = list(blend_meta.blend_segments)
        self._cacheblend_hkvd_indices.clear()
        self._cacheblend_cached_kv_copy.clear()
        self._cacheblend_first_layer_done = False

        # Step 1: Load supervisor KV directly (no blending needed for the shared prefix)
        if "master" in GPU_KV_CACHE:
            master_slots = req_meta.slot_mapping[:master_len].cuda()
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]
                kv_cache_attr = getattr(layer, "kv_cache", None)
                if kv_cache_attr is None or layer_name not in GPU_KV_CACHE["master"]:
                    continue
                kv_cache_layer = kv_cache_attr[forward_context.virtual_engine]
                master_kv_full = GPU_KV_CACHE["master"][layer_name]
                if master_kv_full.dim() == 3:
                    master_kv = master_kv_full[:, :master_len, :]
                else:
                    master_kv = master_kv_full[:master_len, :]
                self._inject_kv(kv_cache_layer, master_kv, master_slots, attn_metadata)
            logger.info("[CACHEBLEND] Loaded supervisor KV: %d tokens", master_len)

        # Step 2: Load each branch segment and save cached KV copy for later comparison
        total_loaded = 0
        for segment in blend_meta.blend_segments:
            cache_key = segment.cache_key
            if cache_key not in GPU_KV_CACHE:
                logger.warning("[CACHEBLEND] Missing cache key: %s", cache_key)
                continue

            cached_layers = GPU_KV_CACHE[cache_key]
            if not cached_layers:
                continue

            num_tokens = segment.num_tokens
            target_slots = req_meta.slot_mapping[segment.slot_start:segment.slot_end].cuda()

            # Save segment info for use in save_kv_layer
            self._cacheblend_segment_info[cache_key] = segment

            # Initialize storage for cached KV copy for this segment
            self._cacheblend_cached_kv_copy[cache_key] = {}

            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]
                kv_cache_attr = getattr(layer, "kv_cache", None)
                if kv_cache_attr is None or layer_name not in cached_layers:
                    continue

                kv_cache_layer = kv_cache_attr[forward_context.virtual_engine]
                cached_kv = cached_layers[layer_name]

                # Trim to actual tokens
                if cached_kv.dim() == 3:
                    cached_kv = cached_kv[:, :num_tokens, :].clone()
                else:
                    cached_kv = cached_kv[:num_tokens, :].clone()

                # Late-RoPE realign position encoding
                # For S8 (parallel blend): first realign to blend position (new_start_pos),
                # then after blending, realign again to final position (slot_start)
                # For S7 (sequential blend): new_start_pos == slot_start, single realignment
                if parallel_blend:
                    # S8: Two-phase position realignment
                    # Phase 1: Realign to parallel blend position (all at master_len)
                    blend_pos = segment.new_start_pos
                    if segment.old_start_pos != blend_pos and self._rope_adjuster is not None:
                        cached_kv = self._adjust_k_position(
                            cached_kv, segment.old_start_pos, blend_pos, num_tokens
                        )
                    # Store blend_pos for later use (phase 2 realignment after blending)
                    segment.blend_pos = blend_pos
                else:
                    # S7: Single realignment to final position
                    final_pos = segment.slot_start
                    if segment.old_start_pos != final_pos and self._rope_adjuster is not None:
                        cached_kv = self._adjust_k_position(
                            cached_kv, segment.old_start_pos, final_pos, num_tokens
                        )

                # Save the realigned cached KV for comparison in save_kv_layer
                self._cacheblend_cached_kv_copy[cache_key][layer_name] = cached_kv.clone()

                # Inject cached KV into paged buffer
                # Model forward will compute new KV on top of this
                self._inject_kv(kv_cache_layer, cached_kv, target_slots, attn_metadata)

            total_loaded += num_tokens
            logger.info(
                "[CACHEBLEND] Loaded %s: %d tokens at pos [%d:%d], old_pos=%d",
                cache_key, num_tokens, segment.slot_start, segment.slot_end, segment.old_start_pos
            )

        logger.info(
            "[CACHEBLEND] Prepared: supervisor=%d + branches=%d tokens, CacheBlend active",
            master_len, total_loaded
        )

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Save a KV cache layer (a produced branch/supervisor segment) to GPU storage.

        For CacheBlend (S7/S8): Also performs selective KV fusion by comparing
        newly computed KV with cached KV, selecting HKVD tokens, and blending.
        """
        connector_metadata = self._get_connector_metadata()
        if not isinstance(connector_metadata, GPUKVConnectorMetadata):
            return

        # Check if CacheBlend mode is active for FINAL request
        if self._cacheblend_active:
            self._save_kv_layer_with_cacheblend(layer_name, kv_layer, attn_metadata)
            return

        for req_meta in connector_metadata.requests:
            if not req_meta.save_key:
                continue

            start, end = req_meta.save_start, req_meta.save_end
            if end <= start:
                continue

            slots = req_meta.slot_mapping[start:end].cuda()
            kv_data = self._extract_kv(kv_layer, slots, attn_metadata)

            key = req_meta.save_key
            if key not in GPU_KV_CACHE:
                GPU_KV_CACHE[key] = {}

            GPU_KV_CACHE[key][layer_name] = kv_data.clone()

            # Update scheduler state
            num_tokens = end - start
            if key == "master":
                SCHEDULER_STATE["master_stored"] = True
                SCHEDULER_STATE["master_tokens"] = num_tokens
                if SCHEDULER_STATE["master_prompt"] is None:
                    SCHEDULER_STATE["master_prompt"] = list(req_meta.token_ids[:num_tokens])
            elif key == "worker1":
                SCHEDULER_STATE["worker1_stored"] = True
                SCHEDULER_STATE["worker1_tokens"] = num_tokens
            elif key == "worker2":
                SCHEDULER_STATE["worker2_stored"] = True
                SCHEDULER_STATE["worker2_tokens"] = num_tokens
            elif key == "worker3":
                SCHEDULER_STATE["worker3_stored"] = True
                SCHEDULER_STATE["worker3_tokens"] = num_tokens

            log_key = f"{key}_{id(req_meta)}"
            if log_key not in self._save_logged:
                self._save_logged[log_key] = True
                logger.info(
                    "[SAVE KV] %s request: saved %d tokens (key=%s)",
                    req_meta.request_type.value, num_tokens, key
                )

    def _save_kv_layer_with_cacheblend(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
    ) -> None:
        """
        CacheBlend: Perform selective KV fusion for branch segments.

        For each branch segment:
        1. Extract newly computed KV from model forward (has correct cross-branch attention)
        2. Compare with cached KV to compute KV deviation
        3. Select top-k% HKVD tokens (first layer only)
        4. Blend: keep new KV at HKVD positions, use cached KV elsewhere
        5. Inject blended KV back into paged buffer
        """
        recompute_ratio = self._cacheblend_config.recompute_ratio
        is_first_layer = not self._cacheblend_first_layer_done

        for segment in self._cacheblend_blend_segments:
            cache_key = segment.cache_key

            # Check if we have cached KV for this segment
            if cache_key not in self._cacheblend_cached_kv_copy:
                continue
            if layer_name not in self._cacheblend_cached_kv_copy[cache_key]:
                continue

            # Get slot range for this segment
            slot_start = segment.slot_start
            slot_end = segment.slot_end
            num_tokens = segment.num_tokens

            if self._cacheblend_slot_mapping is None:
                continue

            # Extract newly computed KV for this segment's positions
            segment_slots = self._cacheblend_slot_mapping[slot_start:slot_end].cuda()
            new_kv = self._extract_kv(kv_layer, segment_slots, attn_metadata)

            # Get cached KV (already realigned in start_load_kv)
            cached_kv = self._cacheblend_cached_kv_copy[cache_key][layer_name]

            # Ensure shape compatibility
            if new_kv.dim() == 3 and new_kv.shape[0] == 2:
                # KV combined: [2, num_tokens, hidden_dim]
                actual_new_tokens = new_kv.shape[1]
                actual_cached_tokens = cached_kv.shape[1] if cached_kv.dim() == 3 else cached_kv.shape[0]
            else:
                # MLA or single: [num_tokens, hidden_dim]
                actual_new_tokens = new_kv.shape[0]
                actual_cached_tokens = cached_kv.shape[0]

            min_tokens = min(actual_new_tokens, actual_cached_tokens, num_tokens)

            # Trim to matching size
            if new_kv.dim() == 3:
                new_kv = new_kv[:, :min_tokens, :]
                cached_kv = cached_kv[:, :min_tokens, :] if cached_kv.dim() == 3 else cached_kv[:min_tokens, :]
            else:
                new_kv = new_kv[:min_tokens, :]
                cached_kv = cached_kv[:min_tokens, :]

            # On first layer, compute HKVD indices
            if is_first_layer:
                # Check if EPIC mode (S9/S10) - use static token selection
                scenario_name = os.environ.get("KV_SCENARIO", "")
                epic_k = int(os.environ.get("EPIC_K", "0"))
                is_epic_mode = scenario_name in ("S9", "S10") and epic_k > 0

                if is_epic_mode:
                    # EPIC/LegoLink: select first k tokens (static selection)
                    hkvd_indices = select_legolink_tokens(min_tokens, epic_k)
                    self._cacheblend_hkvd_indices[cache_key] = hkvd_indices

                    log_key = f"epic_{cache_key}"
                    if log_key not in self._save_logged:
                        self._save_logged[log_key] = True
                        logger.info(
                            "[EPIC %s] %s: selected first %d/%d tokens (k=%d)",
                            scenario_name, cache_key, len(hkvd_indices), min_tokens, epic_k
                        )
                else:
                    # CacheBlend: dynamic HKVD selection based on KV deviation
                    # Extract K for deviation computation
                    if new_kv.dim() == 3 and new_kv.shape[0] == 2:
                        new_k = new_kv[0]
                        cached_k = cached_kv[0] if cached_kv.dim() == 3 else cached_kv
                    else:
                        new_k = new_kv
                        cached_k = cached_kv

                    # Compute KV deviation (L2 norm per token)
                    deviation = CacheBlendProcessor.compute_kv_deviation(new_k, cached_k)

                    # Select HKVD tokens
                    hkvd_indices = CacheBlendProcessor.select_hkvd_tokens(deviation, recompute_ratio)
                    self._cacheblend_hkvd_indices[cache_key] = hkvd_indices

                    log_key = f"hkvd_{cache_key}"
                    if log_key not in self._save_logged:
                        self._save_logged[log_key] = True
                        logger.info(
                            "[CACHEBLEND] %s: selected %d/%d HKVD tokens (ratio=%.2f)",
                            cache_key, len(hkvd_indices), min_tokens, recompute_ratio
                        )

            # Blend KV: new KV at HKVD positions, cached KV elsewhere
            if cache_key in self._cacheblend_hkvd_indices:
                hkvd_indices = self._cacheblend_hkvd_indices[cache_key]
                blended_kv = CacheBlendProcessor.blend_kv(new_kv, cached_kv, hkvd_indices)
            else:
                # No HKVD selected, use cached KV
                blended_kv = cached_kv

            # S8: Phase 2 position realignment after blending
            # Realign from blend_pos (master_len) to final_pos (slot_start)
            if self._cacheblend_parallel_mode and cache_key != "worker1":
                blend_pos = getattr(segment, 'blend_pos', segment.new_start_pos)
                final_pos = segment.slot_start
                if blend_pos != final_pos and self._rope_adjuster is not None:
                    blended_kv = self._adjust_k_position(
                        blended_kv, blend_pos, final_pos, min_tokens
                    )
                    if is_first_layer:
                        log_key = f"phase2_adj_{cache_key}"
                        if log_key not in self._save_logged:
                            self._save_logged[log_key] = True
                            logger.info(
                                "[CACHEBLEND S8] %s: phase2 pos realign %d -> %d",
                                cache_key, blend_pos, final_pos
                            )

            # Inject blended KV back into paged buffer
            target_slots = segment_slots[:min_tokens]
            self._inject_kv(kv_layer, blended_kv, target_slots, attn_metadata)

        # Mark first layer as done
        if is_first_layer:
            self._cacheblend_first_layer_done = True

    def _extract_kv(
        self,
        layer: torch.Tensor,
        slot_mapping: torch.Tensor,
        attn_metadata: "AttentionMetadata",
    ) -> torch.Tensor:
        """Extract KV from layer."""
        if isinstance(attn_metadata, MLACommonMetadata):
            num_pages, page_size = layer.shape[0], layer.shape[1]
            return layer.reshape(num_pages * page_size, -1)[slot_mapping, ...]
        num_pages, page_size = layer.shape[1], layer.shape[2]
        return layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping, ...]

    def wait_for_save(self):
        """Clean up after save operations complete."""
        self._save_logged.clear()

        # Clear CacheBlend state after FINAL request completes
        if self._cacheblend_active:
            logger.info("[CACHEBLEND] Request completed, clearing CacheBlend state")
            self._cacheblend_active = False
            self._cacheblend_hkvd_indices.clear()
            self._cacheblend_segment_info.clear()
            self._cacheblend_cached_kv_copy.clear()
            self._cacheblend_slot_mapping = None
            self._cacheblend_blend_segments.clear()
            self._cacheblend_first_layer_done = False

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Check how many KV tokens can be loaded/stitched from cache."""
        req_type = self._detect_request_type(request)
        is_independent, _, _, _, is_sequential, _, _ = self._get_scenario_from_env()
        num_tokens = len(list(request.prompt_token_ids or []))

        # S3/S4/S5: Branch requests can reuse the supervisor KV
        if req_type == RequestType.WORKER1:
            if not is_independent and SCHEDULER_STATE["master_stored"]:
                master_tokens = SCHEDULER_STATE["master_tokens"]
                max_external = num_tokens - num_computed_tokens - 1
                loadable = min(master_tokens, max_external)

                if loadable > 0:
                    logger.info(
                        "[GET MATCHED] worker1: master=%d, loadable=%d",
                        master_tokens, loadable
                    )
                    return loadable, False

        if req_type == RequestType.WORKER2:
            if not is_independent and SCHEDULER_STATE["master_stored"]:
                if is_sequential and SCHEDULER_STATE["worker1_stored"]:
                    # S5: branch2 reuses supervisor + branch1
                    total = (SCHEDULER_STATE["master_tokens"] +
                            SCHEDULER_STATE["worker1_tokens"])
                    max_external = num_tokens - num_computed_tokens - 1
                    loadable = min(total, max_external)

                    if loadable > 0:
                        logger.info(
                            "[GET MATCHED] worker2 (S5): master+w1=%d, loadable=%d",
                            total, loadable
                        )
                        return loadable, False
                else:
                    # S3/S4: branch2 reuses only the supervisor
                    master_tokens = SCHEDULER_STATE["master_tokens"]
                    max_external = num_tokens - num_computed_tokens - 1
                    loadable = min(master_tokens, max_external)

                    if loadable > 0:
                        logger.info(
                            "[GET MATCHED] worker2: master=%d, loadable=%d",
                            master_tokens, loadable
                        )
                        return loadable, False

        # S5: branch3 reuses supervisor + branch1 + branch2
        if req_type == RequestType.WORKER3:
            if not is_independent and SCHEDULER_STATE["master_stored"]:
                if is_sequential and SCHEDULER_STATE["worker1_stored"] and SCHEDULER_STATE["worker2_stored"]:
                    total = (SCHEDULER_STATE["master_tokens"] +
                            SCHEDULER_STATE["worker1_tokens"] +
                            SCHEDULER_STATE["worker2_tokens"])
                    max_external = num_tokens - num_computed_tokens - 1
                    loadable = min(total, max_external)

                    if loadable > 0:
                        logger.info(
                            "[GET MATCHED] worker3 (S5): master+w1+w2=%d, loadable=%d",
                            total, loadable
                        )
                        return loadable, False
                else:
                    # Non-sequential: branch3 reuses only the supervisor
                    master_tokens = SCHEDULER_STATE["master_tokens"]
                    max_external = num_tokens - num_computed_tokens - 1
                    loadable = min(master_tokens, max_external)

                    if loadable > 0:
                        logger.info(
                            "[GET MATCHED] worker3: master=%d, loadable=%d",
                            master_tokens, loadable
                        )
                        return loadable, False

        # Final request: load and stitch all four KV segments
        if req_type == RequestType.FINAL:
            if (SCHEDULER_STATE["master_stored"] and
                SCHEDULER_STATE["worker1_stored"] and
                SCHEDULER_STATE["worker2_stored"] and
                SCHEDULER_STATE["worker3_stored"]):
                # Total cached KV tokens
                cached_kv_total = (SCHEDULER_STATE["master_tokens"] +
                                  SCHEDULER_STATE["worker1_tokens"] +
                                  SCHEDULER_STATE["worker2_tokens"] +
                                  SCHEDULER_STATE["worker3_tokens"])

                # Load all cached KV, then incremental prefill the suffix
                # loadable = cached_kv_total (no -1, we want to prefill suffix after)
                loadable = cached_kv_total

                # Calculate suffix length for incremental prefill
                suffix_tokens = num_tokens - cached_kv_total

                if suffix_tokens == 0:
                    loadable -= 1  # Leave one token for decoding

                logger.info(
                    "[GET MATCHED] FINAL: cached_kv=%d, num_tokens=%d, "
                    "loadable=%d, suffix_prefill=%d",
                    cached_kv_total, num_tokens, loadable, suffix_tokens
                )

                if loadable > 0:
                    return loadable, False

        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int
    ):
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build connector metadata."""
        meta = GPUKVConnectorMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = list(new_req.prompt_token_ids or [])
            block_ids = new_req.block_ids[0]

            block_ids_tensor = torch.tensor(block_ids)
            num_blocks = block_ids_tensor.shape[0]
            block_offsets = torch.arange(0, self._block_size)
            full_slot_mapping = (
                block_offsets.reshape((1, self._block_size))
                + block_ids_tensor.reshape((num_blocks, 1)) * self._block_size
            ).flatten()

            req_type = self._detect_request_type_by_tokens(token_ids)

            req_meta = ReqMeta(
                request_type=req_type,
                token_ids=token_ids,
                slot_mapping=full_slot_mapping,
            )

            if new_req.req_id in self._requests_need_load:
                _, _, _, _, is_sequential, _, _ = self._get_scenario_from_env()
                # Determine which keys to load based on request type
                if req_type == RequestType.FINAL:
                    req_meta.load_keys = ["master", "worker1", "worker2", "worker3"]
                    self._configure_load_adjustments(req_meta)
                elif req_type == RequestType.WORKER1:
                    req_meta.load_keys = ["master"]
                elif req_type == RequestType.WORKER2:
                    if is_sequential:
                        req_meta.load_keys = ["master", "worker1"]
                    else:
                        req_meta.load_keys = ["master"]
                elif req_type == RequestType.WORKER3:
                    if is_sequential:
                        req_meta.load_keys = ["master", "worker1", "worker2"]
                    else:
                        req_meta.load_keys = ["master"]
                # Also configure save for branch requests (after load)
                if req_type in (RequestType.WORKER1, RequestType.WORKER2, RequestType.WORKER3):
                    self._configure_save(req_meta, token_ids)
            else:
                self._configure_save(req_meta, token_ids)

            meta.requests.append(req_meta)

        self._requests_need_load.clear()
        return meta

    def _detect_request_type(self, request: "Request") -> RequestType:
        """Detect request type."""
        return self._detect_request_type_by_tokens(list(request.prompt_token_ids or []))

    def _detect_request_type_by_tokens(self, token_ids: List[int]) -> RequestType:
        """Detect request type by token IDs.

        Uses stored state progression instead of prefix matching to avoid
        tokenization inconsistency issues (e.g., tokenize("A"+"B") != tokenize("A") + tokenize("B"))
        """
        num_tokens = len(token_ids)
        is_independent, _, _, _, is_sequential, _, _ = self._get_scenario_from_env()

        # First request (no supervisor stored yet) -> MASTER
        if not SCHEDULER_STATE.get("master_stored", False):
            return RequestType.MASTER

        # Check if all branches are stored -> FINAL
        all_workers_stored = (
            SCHEDULER_STATE["worker1_stored"] and
            SCHEDULER_STATE["worker2_stored"] and
            SCHEDULER_STATE["worker3_stored"]
        )

        if all_workers_stored:
            return RequestType.FINAL

        # Determine branch type based on what's stored (in order)
        if not SCHEDULER_STATE["worker1_stored"]:
            return RequestType.WORKER1
        elif not SCHEDULER_STATE["worker2_stored"]:
            return RequestType.WORKER2
        elif not SCHEDULER_STATE["worker3_stored"]:
            return RequestType.WORKER3

        logger.info("[DETECT] Fallback to FINAL")
        return RequestType.FINAL

    def _get_scenario_from_env(self):
        """Get scenario config from the KV_SCENARIO environment variable.

        Returns: (independent, adjust_w1, adjust_w2, adjust_w3, sequential, cacheblend, parallel_blend)
        - independent: branch prompts are just token_1/token_2/token_3, not prefixed
        - adjust_w1: late-RoPE realign branch1 during final stitch
        - adjust_w2: late-RoPE realign branch2 during final stitch
        - adjust_w3: late-RoPE realign branch3 during final stitch
        - sequential: S5 mode where branches accumulate KV sequentially
        - cacheblend: S7/S8/S9/S10 mode - use selective KV recompute
        - parallel_blend: S8/S10 mode - parallel blend (all appended to supervisor)

        Note: S9/S10 use EPIC's static token selection (first k tokens) instead of
        CacheBlend's dynamic HKVD selection. The difference is handled in
        _save_kv_layer_with_cacheblend by checking the EPIC_K environment variable.
        """
        scenario_name = os.environ.get("KV_SCENARIO", "")
        # (independent, adjust_w1, adjust_w2, adjust_w3, sequential, cacheblend, parallel_blend)
        scenarios = {
            "S1": (True, False, False, False, False, False, False),
            "S2": (True, True, True, True, False, False, False),
            "S3": (False, False, False, False, False, False, False),
            "S4": (False, False, True, True, False, False, False),
            "S5": (False, False, False, False, True, False, False),
            "S7": (True, True, True, True, False, True, False),   # Sequential CacheBlend
            "S8": (True, True, True, True, False, True, True),    # Parallel CacheBlend
            "S9": (True, True, True, True, False, True, False),   # Sequential EPIC/LegoLink
            "S10": (True, True, True, True, False, True, True),   # Parallel EPIC/LegoLink
        }
        return scenarios.get(scenario_name, (False, False, True, True, False, False, False))

    def _configure_load_adjustments(self, req_meta: ReqMeta) -> None:
        """Configure late-RoPE realignments for the final stitch based on scenario."""
        master_len = SCHEDULER_STATE["master_tokens"]
        worker1_len = SCHEDULER_STATE["worker1_tokens"]
        worker2_len = SCHEDULER_STATE["worker2_tokens"]

        is_independent, adjust_w1, adjust_w2, adjust_w3, _, _, _ = self._get_scenario_from_env()

        logger.info(
            "[CONFIG] scenario: independent=%s, adjust_w1=%s, adjust_w2=%s, adjust_w3=%s",
            is_independent, adjust_w1, adjust_w2, adjust_w3
        )

        if is_independent:
            # S1/S2: Independent branches - their KV starts at pos prompt_worker_i_len
            # (because branch request is prompt_worker_i + token_i, we save token_i part)
            prompt_worker1_len = int(os.environ.get("KV_PROMPT_WORKER1_LEN", "0"))
            prompt_worker2_len = int(os.environ.get("KV_PROMPT_WORKER2_LEN", "0"))
            prompt_worker3_len = int(os.environ.get("KV_PROMPT_WORKER3_LEN", "0"))

            if adjust_w1:
                req_meta.adjust_configs["worker1"] = PosAdjustment(
                    old_pos=prompt_worker1_len,
                    new_pos=master_len
                )
                logger.info("[CONFIG] realign worker1 pos %d -> %d", prompt_worker1_len, master_len)
            if adjust_w2:
                req_meta.adjust_configs["worker2"] = PosAdjustment(
                    old_pos=prompt_worker2_len,
                    new_pos=master_len + worker1_len
                )
                logger.info(
                    "[CONFIG] realign worker2 pos %d -> %d",
                    prompt_worker2_len, master_len + worker1_len
                )
            if adjust_w3:
                req_meta.adjust_configs["worker3"] = PosAdjustment(
                    old_pos=prompt_worker3_len,
                    new_pos=master_len + worker1_len + worker2_len
                )
                logger.info(
                    "[CONFIG] realign worker3 pos %d -> %d",
                    prompt_worker3_len, master_len + worker1_len + worker2_len
                )
        else:
            # S3/S4: branches with prefix - their KV starts at pos master_len
            if adjust_w2:
                req_meta.adjust_configs["worker2"] = PosAdjustment(
                    old_pos=master_len,
                    new_pos=master_len + worker1_len
                )
                logger.info(
                    "[CONFIG] realign worker2 pos %d -> %d",
                    master_len, master_len + worker1_len
                )
            if adjust_w3:
                req_meta.adjust_configs["worker3"] = PosAdjustment(
                    old_pos=master_len,
                    new_pos=master_len + worker1_len + worker2_len
                )
                logger.info(
                    "[CONFIG] realign worker3 pos %d -> %d",
                    master_len, master_len + worker1_len + worker2_len
                )

    def _configure_save(self, req_meta: ReqMeta, token_ids: List[int]) -> None:
        """Configure save parameters based on scenario."""
        is_independent, _, _, _, is_sequential, _, _ = self._get_scenario_from_env()

        if req_meta.request_type == RequestType.MASTER:
            req_meta.save_key = "master"
            req_meta.save_start = 0
            req_meta.save_end = len(token_ids)
            SCHEDULER_STATE["master_stored"] = True
            SCHEDULER_STATE["master_tokens"] = len(token_ids)
            SCHEDULER_STATE["master_prompt"] = list(token_ids)
            logger.info(
                "[CONFIG SAVE] MASTER: tokens=%d, save_range=[%d:%d]",
                len(token_ids), 0, len(token_ids)
            )

        elif req_meta.request_type == RequestType.WORKER1:
            req_meta.save_key = "worker1"
            if is_independent:
                # S1/S2: branch prefills prompt_worker1 + token1, saves only token1
                # Get prompt_worker1 length from environment variable
                prompt_worker1_len = int(os.environ.get("KV_PROMPT_WORKER1_LEN", "0"))
                req_meta.save_start = prompt_worker1_len
                req_meta.save_end = len(token_ids)
                SCHEDULER_STATE["worker1_tokens"] = len(token_ids) - prompt_worker1_len
                logger.info(
                    "[CONFIG SAVE] WORKER1 (S1/S2): total_tokens=%d, prompt_worker1_len=%d, "
                    "save_range=[%d:%d], save_tokens=%d",
                    len(token_ids), prompt_worker1_len, prompt_worker1_len, len(token_ids),
                    len(token_ids) - prompt_worker1_len
                )
            else:
                master_len = SCHEDULER_STATE["master_tokens"]
                req_meta.save_start = master_len
                req_meta.save_end = len(token_ids)
                SCHEDULER_STATE["worker1_tokens"] = len(token_ids) - master_len
                # Debug: verify prefix match
                master_prompt = SCHEDULER_STATE.get("master_prompt", [])
                prefix_match = (len(token_ids) >= master_len and
                               list(token_ids[:master_len]) == master_prompt)
                logger.info(
                    "[CONFIG SAVE] WORKER1: total_tokens=%d, master_len=%d, "
                    "save_range=[%d:%d], prefix_match=%s",
                    len(token_ids), master_len, master_len, len(token_ids), prefix_match
                )
            SCHEDULER_STATE["worker1_stored"] = True
            SCHEDULER_STATE["worker1_prompt"] = list(token_ids)

        elif req_meta.request_type == RequestType.WORKER2:
            req_meta.save_key = "worker2"
            if is_independent:
                # S1/S2: branch prefills prompt_worker2 + token2, saves only token2
                # Get prompt_worker2 length from environment variable
                prompt_worker2_len = int(os.environ.get("KV_PROMPT_WORKER2_LEN", "0"))
                req_meta.save_start = prompt_worker2_len
                req_meta.save_end = len(token_ids)
                SCHEDULER_STATE["worker2_tokens"] = len(token_ids) - prompt_worker2_len
                logger.info(
                    "[CONFIG SAVE] WORKER2 (S1/S2): total_tokens=%d, prompt_worker2_len=%d, "
                    "save_range=[%d:%d], save_tokens=%d",
                    len(token_ids), prompt_worker2_len, prompt_worker2_len, len(token_ids),
                    len(token_ids) - prompt_worker2_len
                )
            elif is_sequential:
                prefix_len = (SCHEDULER_STATE["master_tokens"] +
                             SCHEDULER_STATE["worker1_tokens"])
                req_meta.save_start = prefix_len
                req_meta.save_end = len(token_ids)
                SCHEDULER_STATE["worker2_tokens"] = len(token_ids) - prefix_len
                # Debug: verify prefix match
                worker1_prompt = SCHEDULER_STATE.get("worker1_prompt", [])
                prefix_match = (len(token_ids) >= prefix_len and
                               list(token_ids[:prefix_len]) == worker1_prompt[:prefix_len])
                logger.info(
                    "[CONFIG SAVE] WORKER2 (S5): total_tokens=%d, prefix_len=%d, "
                    "save_range=[%d:%d], prefix_match=%s",
                    len(token_ids), prefix_len, prefix_len, len(token_ids), prefix_match
                )
            else:
                master_len = SCHEDULER_STATE["master_tokens"]
                req_meta.save_start = master_len
                req_meta.save_end = len(token_ids)
                SCHEDULER_STATE["worker2_tokens"] = len(token_ids) - master_len
            SCHEDULER_STATE["worker2_stored"] = True
            SCHEDULER_STATE["worker2_prompt"] = list(token_ids)

        elif req_meta.request_type == RequestType.WORKER3:
            req_meta.save_key = "worker3"
            if is_independent:
                # S1/S2: branch prefills prompt_worker3 + token3, saves only token3
                # Get prompt_worker3 length from environment variable
                prompt_worker3_len = int(os.environ.get("KV_PROMPT_WORKER3_LEN", "0"))
                req_meta.save_start = prompt_worker3_len
                req_meta.save_end = len(token_ids)
                SCHEDULER_STATE["worker3_tokens"] = len(token_ids) - prompt_worker3_len
                logger.info(
                    "[CONFIG SAVE] WORKER3 (S1/S2): total_tokens=%d, prompt_worker3_len=%d, "
                    "save_range=[%d:%d], save_tokens=%d",
                    len(token_ids), prompt_worker3_len, prompt_worker3_len, len(token_ids),
                    len(token_ids) - prompt_worker3_len
                )
            elif is_sequential:
                # S5: branch3 = final prompt, save from master+w1+w2 length
                prefix_len = (SCHEDULER_STATE["master_tokens"] +
                             SCHEDULER_STATE["worker1_tokens"] +
                             SCHEDULER_STATE["worker2_tokens"])
                req_meta.save_start = prefix_len
                req_meta.save_end = len(token_ids)
                SCHEDULER_STATE["worker3_tokens"] = len(token_ids) - prefix_len
            else:
                # S3/S4: branch with prefix, save from master length
                master_len = SCHEDULER_STATE["master_tokens"]
                req_meta.save_start = master_len
                req_meta.save_end = len(token_ids)
                SCHEDULER_STATE["worker3_tokens"] = len(token_ids) - master_len
            SCHEDULER_STATE["worker3_stored"] = True
            SCHEDULER_STATE["worker3_prompt"] = list(token_ids)
