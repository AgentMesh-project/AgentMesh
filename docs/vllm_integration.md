# vLLM Integration — How BPP Runs in a Real Serving Engine

This document describes how AgentMesh's **Branch-Parallel Prefill (BPP)**
mechanism is realized inside the [vLLM](https://github.com/vllm-project/vllm)
serving engine as a production `KVConnector` named `GPUKVConnector`. It
complements the mechanism write-up in [`bpp.md`](bpp.md): that page explains the
*algorithm* (Branch-Parallel Attention + late RoPE realignment); this page
explains the *engineering* — how those ideas map onto vLLM's paged KV cache,
scheduler, and connector hooks.

The implementation lives in
[`agentmesh/mechanisms/bpp/vllm/`](../agentmesh/mechanisms/bpp/vllm/) and a
runnable example in
[`agentmesh/examples/bpp_vllm/`](../agentmesh/examples/bpp_vllm/).

## 1. Where BPP fits in the dataflow

Every AgentMesh interaction is a produce-transfer-consume dataflow whose
consumer is **always** an LLM prefill. In the dominant supervisor-worker
(scatter-gather) pattern, the supervisor concatenates `N` worker outputs and
prefills the merged context. A structure-blind engine does this **serially**:
the consume stage is a BARRIER — the supervisor idles until the last branch
arrives, and full attention re-derives cross-branch terms that the semantically
independent branches barely use.

BPP makes the consume stage barrier- and redundancy-free by prefilling each
branch **in parallel** against only the shared supervisor prefix and itself, then
**stitching** the branch-local KV into one coherent context for decode. This is
the §4.3 mechanism evaluated in §5 (BPP eval) of the paper.

## 2. Why a KV connector

vLLM exposes a `KVConnector` extension point that brackets the model forward
pass with hooks on both the **scheduler** side and the **worker** side. This is
exactly the seam BPP needs: it lets us *inject* externally produced KV into
vLLM's paged buffer before a forward pass and *extract* freshly produced KV
afterward — without forking the attention kernels. `GPUKVConnector` therefore
implements BPP as a connector with `kv_role="kv_both"` (it both stores and
loads KV).

### Scheduler-side hooks

| Hook | BPP role |
| --- | --- |
| `get_num_new_matched_tokens()` | Decide how many cached branch/prefix tokens the incoming request can reuse (per scenario). |
| `update_state_after_alloc()` | Mark requests that will load external KV. |
| `build_connector_meta()` | Compute the slot mapping for the request's blocks and attach the save/load plan + per-branch position-realignment configs. |

### Worker-side hooks

| Hook | BPP role |
| --- | --- |
| `start_load_kv()` | Inject stored prefix + branch KV into the paged buffer at the request's slots, applying late RoPE realignment per branch; optionally run CacheBlend/EPIC. |
| `save_kv_layer()` | Extract newly produced KV per layer into GPU storage (and, in S7-S10, blend selectively recomputed tokens). |
| `wait_for_save()` | Finalize and clear per-request blend state. |

## 3. Slot-based block-diagonal isolation = the cross-branch attention mask

vLLM's KV cache is *paged*: each request's tokens map to a list of physical KV
slots via a slot mapping derived from its block ids and the block size. BPP
exploits this directly.

`build_connector_meta()` materializes each request's full slot mapping:

```
block_offsets = arange(block_size)
slot_mapping  = (block_offsets[None, :] + block_ids[:, None] * block_size).flatten()
```

Each branch then occupies a **disjoint, contiguous range of KV slots**. During
the parallel prefill, a branch's queries can only reach the shared-prefix slots
plus its own slots; sibling branches live in different slot ranges and are never
attended to. **The disjoint slot layout *is* Branch-Parallel Attention (BPA)** —
a block-diagonal cross-branch mask realized through memory layout rather than an
explicit attention bias. This is what drops attention complexity from O(N²) to
O(N) in the branch count N.

KV is injected/extracted with a single gather/scatter over the slot mapping,
specialized for the two paged layouts vLLM uses:

- standard attention: KV stored combined as `[2, num_tokens, hidden]`;
- MLA attention: a single latent tensor `[num_tokens, hidden]`.

## 4. KV stitching with late RoPE realignment

Because every branch is prefilled at the *same* starting position (immediately
after the shared prefix), the branch-local keys carry colliding RoPE phases.
Before the consuming prefill reads the stitched sequence, each branch's **K**
(only K; V is position-free) is re-rotated from its produce-time offset to its
final offset in the merged layout.

This is handled by `FusedRopeAdjuster` in
[`position_encoding.py`](../agentmesh/mechanisms/bpp/vllm/position_encoding.py),
which obtains the model's exact rotary configuration directly from the live
`rotary_emb` module (correct for Qwen3, GPT-OSS, Llama-3, including YARN / Llama3
scaling). Moving a key from position `i` to `j` is a rotation by `(j - i)`; the
adjuster either:

- calls the fused LMCache CUDA kernel `rotary_embedding_k_fused` when available, or
- falls back to pure torch: reverse the old rotation via a shuffle trick, then
  re-apply RoPE at the new positions.

`get_num_new_matched_tokens()` and `_configure_load_adjustments()` compute each
branch's `old_pos -> new_pos` so the FINAL request stitches
`master + worker1 + worker2 + worker3` into a positionally consistent sequence.

## 5. Composability with CacheBlend and EPIC

BPP is **orthogonal to, and composable with**, selective KV-recompute schemes.
Parallel prefill omits the (small) genuine cross-branch attention; CacheBlend and
EPIC recover it by recomputing a few tokens per branch
([`cache_blend.py`](../agentmesh/mechanisms/bpp/vllm/cache_blend.py)):

- **CacheBlend** (S7/S8): compute fresh K under the stitched context, measure
  per-token deviation from cached K, select the top High-KV-Deviation (HKVD)
  fraction (`CACHEBLEND_RECOMPUTE_RATIO`, default 0.15), and blend new K/V at
  those positions only.
- **EPIC / LegoLink** (S9/S10): statically recompute the first `EPIC_K`
  "attention-sink" tokens of each branch — O(kN) instead of CacheBlend's
  O(15%·N²).

For the parallel variants (S8/S10) the connector applies a two-phase RoPE
realignment: first to the common blend position at the prefix end, then to each
branch's final stitched offset after blending.

## 6. Scenario matrix → paper ablations

The `KV_SCENARIO` env var (S1-S10) selects the produce/realign/recompute
combination. It maps onto the §5 BPP baselines and ablations:

| Scenario | Behavior | Paper mapping |
| --- | --- | --- |
| `S1` | Parallel branches, **no** realignment | Direct BPP w/o realignment (**DBPP**) |
| `S2` | Parallel branches **+ RoPE realignment** | BPP / KV-Reuse with realignment (**KVR**) |
| `S3`/`S4` | Shared-prefix branches, single / multi-branch realign | BPP shared-prefix reuse |
| `S5` | Serial KV accumulation | Full-Prefill ordering reference (**FP**) |
| `S7`/`S8` | BPP ∘ CacheBlend (sequential / parallel) | Composability with CacheBlend |
| `S9`/`S10` | BPP ∘ EPIC/LegoLink (sequential / parallel) | Composability with EPIC |

Reported results (paper §5, BPP eval): **8-61%** consume-stage time reduction,
attention complexity O(N²)→O(N) in branch count N, and precision matching or
exceeding **Full Prefill** for 8B/20B models. The ablation contribution of BPP
to the cumulative speedup is **19-61%** of the consume stage. BPP composes with
CacheBlend and EPIC without precision regressions.

## 7. Engine configuration

```python
from vllm import LLM
from vllm.config import KVTransferConfig

llm = LLM(
    model="Qwen/Qwen3-1.7B",
    enforce_eager=True,                    # easier to debug; optional
    disable_hybrid_kv_cache_manager=True,  # BPP needs a homogeneous KV manager
    enable_prefix_caching=False,           # reuse is controlled by the connector
    kv_transfer_config=KVTransferConfig(
        kv_connector="GPUKVConnector",
        kv_role="kv_both",
    ),
)
```

Register `GPUKVConnector` either by copying this subpackage into vLLM's
connector tree (and adding a `KVConnectorFactory.register_connector(...)` line)
or by passing `kv_connector_module_path` in `KVTransferConfig`. Full
installation steps, env vars, and the request-type protocol are in
[`agentmesh/mechanisms/bpp/vllm/README.md`](../agentmesh/mechanisms/bpp/vllm/README.md).
