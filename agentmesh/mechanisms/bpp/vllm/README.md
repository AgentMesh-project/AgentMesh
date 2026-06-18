# BPP vLLM Integration — `GPUKVConnector`

This subpackage is the **production vLLM integration** for AgentMesh's
**Branch-Parallel Prefill (BPP)** mechanism. It is a drop-in vLLM
`KVConnector` that prefills independent worker *branches* in parallel against a
shared *supervisor* prefix, isolates them with slot-based block-diagonal
**Branch-Parallel Attention (BPA)**, and stitches the per-branch KV into one
contiguous sequence for the consuming LLM prefill using **late RoPE
realignment**. It is optionally composable with selective KV recompute
(CacheBlend / EPIC).

In AgentMesh terms, every interaction is a produce-transfer-consume dataflow
whose consumer is always an LLM prefill. BPP attacks the **consume** stage: a
structure-blind runtime prefills the merged sequence serially, so each branch
*idles* (a BARRIER) until the previous one finishes. BPP turns
serial-prefill-then-decode into parallel-prefill + KV-stitch, reducing
consume-stage time by **8%-61%** with precision matching or exceeding full
prefill for 8B/20B models.

## Files

| File | Role |
| --- | --- |
| `gpu_kv_connector.py` | The `GPUKVConnector` — scheduler + worker side; slot-based block-diagonal isolation (the cross-branch attention mask); request-type protocol; scenario matrix S1-S10. |
| `position_encoding.py` | `FusedRopeAdjuster` — late RoPE realignment for stitched KV (LMCache fused kernel + pure-torch fallback). |
| `cache_blend.py` | `CacheBlendProcessor` + EPIC/LegoLink selective KV recompute for the S7-S10 scenarios. |
| `__init__.py` | Public exports, guarded so importing without vLLM fails with a clear message. |

## Installation

The connector keeps vLLM-internal imports verbatim, so it drops into a vLLM
checkout unchanged. There are two ways to make vLLM see it.

### Option A — copy into vLLM's connector tree (simplest)

Copy this directory into vLLM and register it in
`vllm/distributed/kv_transfer/kv_connector/factory.py`:

```
cp -r agentmesh/mechanisms/bpp/vllm \
   <vllm>/vllm/distributed/kv_transfer/kv_connector/v1/gpu_kv_connector
```

```python
# in factory.py
KVConnectorFactory.register_connector(
    "GPUKVConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.gpu_kv_connector.gpu_kv_connector",
    "GPUKVConnector",
)
```

> When copied into the vLLM tree, change the in-module imports from
> `agentmesh.mechanisms.bpp.vllm.*` back to the local
> `vllm.distributed.kv_transfer.kv_connector.v1.gpu_kv_connector.*` paths
> (or leave the package importable on `PYTHONPATH`).

### Option B — load as an external module

Recent vLLM can load a connector class from an arbitrary import path via
`KVTransferConfig.kv_connector_module_path`, avoiding any edit to the vLLM
source tree:

```python
from vllm import LLM
from vllm.config import KVTransferConfig

llm = LLM(
    model="Qwen/Qwen3-1.7B",
    enforce_eager=True,
    disable_hybrid_kv_cache_manager=True,   # BPP needs a homogeneous KV manager
    enable_prefix_caching=False,            # exact, controlled KV reuse
    kv_transfer_config=KVTransferConfig(
        kv_connector="GPUKVConnector",
        kv_role="kv_both",
        kv_connector_module_path="agentmesh.mechanisms.bpp.vllm.gpu_kv_connector",
    ),
)
```

## Configuration (environment variables)

The scheduler relies on these env vars; their names are part of the protocol
contract and are kept intact.

| Variable | Meaning | Default |
| --- | --- | --- |
| `KV_SCENARIO` | Selects the scenario `S1`-`S10` (see the matrix below). | `""` |
| `KV_PROMPT_WORKER1_LEN` | Token length of branch-1's local prefix (independent branches, S1/S2/S7-S10). | `0` |
| `KV_PROMPT_WORKER2_LEN` | Token length of branch-2's local prefix. | `0` |
| `KV_PROMPT_WORKER3_LEN` | Token length of branch-3's local prefix. | `0` |
| `CACHEBLEND_RECOMPUTE_RATIO` | Fraction of tokens recomputed per branch in CacheBlend (S7/S8). | `0.15` |
| `CACHEBLEND_CHECK_LAYER` | Layer at which HKVD token selection happens. | `0` |
| `EPIC_K` | Number of leading ("attention-sink") tokens recomputed per branch in EPIC/LegoLink (S9/S10). | `0` |

## Scenario matrix (S1-S10) → paper ablations

`KV_SCENARIO` selects how branch KV is produced, position-realigned, and
optionally selectively-recomputed. The tuple returned internally is
`(independent, adjust_w1, adjust_w2, adjust_w3, sequential, cacheblend, parallel_blend)`.

| Scenario | Mode | What it exercises | Maps to |
| --- | --- | --- | --- |
| `S1` | Independent branches, no realign | Parallel produce, branch-local KV stitched as-is | Direct BPP w/o realignment (**DBPP** baseline) |
| `S2` | Independent branches + RoPE realign | Full BPP: parallel produce + late RoPE realignment | **BPP** (KV-Reuse with realignment, **KVR**) |
| `S3` | Prefixed branches, single realign | Branches share the supervisor prefix in-prompt | BPP with shared-prefix reuse |
| `S4` | Prefixed branches, multi-branch realign | Realign branch 2/3 to their stitched offsets | BPP, 3-branch realignment |
| `S5` | Sequential KV accumulation | Branches accumulate KV serially | Serial-prefill reference (**FP**-like ordering) |
| `S7` | Sequential + CacheBlend | BPP + dynamic HKVD selective recompute | BPP ∘ CacheBlend (composability) |
| `S8` | Parallel + CacheBlend + realign | Parallel blend at prefix end, two-phase RoPE realign | BPP ∘ CacheBlend, parallel |
| `S9` | Sequential + EPIC/LegoLink | BPP + static first-`k` recompute | BPP ∘ EPIC |
| `S10` | Parallel + EPIC/LegoLink + realign | Parallel blend, first-`k` recompute, two-phase realign | BPP ∘ EPIC, parallel |

Paper reference points: BPP reduces consume-stage time **8-61%** and turns
attention complexity from O(N²) to O(N) in the branch count N, with precision
matching/exceeding **Full Prefill (FP)** for 8B/20B models. BPP is orthogonal
to and composable with **CacheBlend** and **EPIC** (the S7-S10 rows above).

## Request-type protocol

The connector tracks cross-request state and classifies each incoming request
by progression (not prefix matching, to avoid tokenization inconsistencies):

1. `MASTER`  — supervisor prefix; stored under key `master`.
2. `WORKER1/2/3` — branch produce; incremental KV stored under `worker1/2/3`.
3. `FINAL` — consuming prefill; loads `master + worker1 + worker2 + worker3`,
   realigns each branch's K via `FusedRopeAdjuster`, and decodes.

## Notes

- **Homogeneous KV manager**: set `disable_hybrid_kv_cache_manager=True`.
- **Exact reuse**: set `enable_prefix_caching=False` so reuse is fully
  controlled by the connector.
- **Memory**: branch KV lives in GPU memory; lower `gpu_memory_utilization`
  if needed.
- **Debugging**: use `enforce_eager=True`.

See [`../../../examples/bpp_vllm/`](../../../examples/bpp_vllm/) for a runnable
example and [`docs/vllm_integration.md`](../../../../docs/vllm_integration.md)
for the architecture write-up.
