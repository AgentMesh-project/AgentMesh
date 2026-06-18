# SPDX-License-Identifier: Apache-2.0
"""
Production vLLM integration for BPP (Branch-Parallel Prefill).

This subpackage is the real, drop-in vLLM ``KVConnector`` that realizes BPP
inside the vLLM serving engine. It prefills independent worker *branches* in
parallel against a shared supervisor prefix, isolates them via slot-based
block-diagonal Branch-Parallel Attention (BPA), and stitches the per-branch KV
into one contiguous sequence with late RoPE realignment. It is optionally
composable with selective KV recompute (CacheBlend / EPIC).

Public names:
    * ``GPUKVConnector``        — the vLLM KV connector (register as "GPUKVConnector")
    * ``FusedRopeAdjuster``     — late RoPE realignment for stitched KV
    * ``CacheBlendProcessor``   — selective KV recompute (CacheBlend/EPIC)

Importing this subpackage requires vLLM (and a torch/CUDA stack) to be
installed. When those are unavailable the imports below fail with a clear
message, but the module still byte-compiles and the parent package degrades
gracefully (it guards this import with try/except).
"""

_IMPORT_ERROR = None

try:
    from agentmesh.mechanisms.bpp.vllm.gpu_kv_connector import (
        GPUKVConnector,
        GPUKVConnectorMetadata,
        RequestType,
        ReqMeta,
        clear_gpu_kv_cache,
        get_gpu_kv_cache_info,
        GPU_KV_CACHE,
        SCHEDULER_STATE,
    )
    from agentmesh.mechanisms.bpp.vllm.position_encoding import (
        FusedRopeAdjuster,
        get_rope_adjuster,
        get_rope_adjuster_v2,
        adjust_kv_positions,
    )
    from agentmesh.mechanisms.bpp.vllm.cache_blend import (
        CacheBlendConfig,
        CacheBlendMetadata,
        CacheBlendProcessor,
        BlendSegment,
        create_blend_metadata_s7,
        create_blend_metadata_s8,
        EPICConfig,
        create_blend_metadata_s9,
        create_blend_metadata_s10,
        select_legolink_tokens,
    )

    __all__ = [
        # KV Connector
        "GPUKVConnector",
        "GPUKVConnectorMetadata",
        "RequestType",
        "ReqMeta",
        "clear_gpu_kv_cache",
        "get_gpu_kv_cache_info",
        "GPU_KV_CACHE",
        "SCHEDULER_STATE",
        # Late RoPE realignment
        "FusedRopeAdjuster",
        "get_rope_adjuster",
        "get_rope_adjuster_v2",
        "adjust_kv_positions",
        # Selective KV recompute (CacheBlend)
        "CacheBlendConfig",
        "CacheBlendMetadata",
        "CacheBlendProcessor",
        "BlendSegment",
        "create_blend_metadata_s7",
        "create_blend_metadata_s8",
        # EPIC/LegoLink
        "EPICConfig",
        "create_blend_metadata_s9",
        "create_blend_metadata_s10",
        "select_legolink_tokens",
    ]

except ImportError as exc:  # pragma: no cover - vLLM is an optional extra
    _IMPORT_ERROR = exc

    def __getattr__(name: str):
        raise ImportError(
            "agentmesh.mechanisms.bpp.vllm requires vLLM (and a torch/CUDA "
            "stack) to be installed. Install vLLM and register the connector "
            "as 'GPUKVConnector'. Original import error: %s" % (_IMPORT_ERROR,)
        )

    __all__ = []
