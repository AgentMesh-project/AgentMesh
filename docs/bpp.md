# BPP — Branch-Parallel Prefill

**BPP shrinks the *consume* step of an `LLM_TO_LLM` dataflow, making it both redundancy- and
barrier-free.**

In the dominant supervisor–worker (scatter–gather) pattern, the supervisor concatenates all `N`
worker outputs and starts prefilling only after the *last* worker arrives. This consume step is
wasteful in two ways:

- **BARRIER** — the supervisor idles until the straggler branch delivers its last byte.
- **REDUNDANCY** — with branch lengths `L_i` and a shared prefix `L_prefix`, full attention costs
  `O(L_prefix · Σ L_i + Σ_i Σ_j L_i L_j)`, where the cross-branch terms (`i ≠ j`) grow
  *quadratically* with the worker count `N` — yet the branches are semantically independent, so
  cross-branch attention yields marginal value.

BPP prefills each branch *as it arrives*, *in parallel*, against only the shared prefix and itself,
then stitches the branch-local KVs into one coherent context for decoding.

## (1) Parallel prefill via Branch-Parallel Attention (BPA)

**BPA** masks cross-branch attention: branch `i`'s tokens attend to the supervisor prefix (user
prompt, system instructions, role setting) and to branch `i` itself, but *not* to any sibling branch
`j`. Each branch's KV computation thus costs `O(L_prefix · L_i + L_i²)` and is independent of its
siblings, so:

- the supervisor can prefill a branch **the moment it arrives** — no longer blocked by the straggler;
- multi-branch prefill runs **concurrently**, leaving a critical path of
  `O(max_{i≤N}(L_prefix · L_i + L_i²))`;
- total attention work falls from **quadratic to linear in the branch count `N`** in the worst case:
  `O(Σ_i (L_prefix · L_i + L_i²))`.

Because no branch's computation depends on another, the parallel prefills can further **overlap with
worker decodings** — the seam where [DES](des.md) takes effect.

## (2) KV stitching with late positional realignment

BPA computes every branch in parallel at the *same* starting position (right after the shared
prefix), so the branch-local KVs carry colliding positions. Before decoding, BPP realigns each key to
its global position in the stitched layout. Under rotary position embeddings (RoPE), moving a key `k`
from position `i` to `j` is a rotation:

```
k_j = R_{j-i} · k_i = [ cos((j−i)θ)  −sin((j−i)θ) ] k_i
                      [ sin((j−i)θ)   cos((j−i)θ) ]
```

(shown two-dimensional for simplicity; the higher-dimensional form follows the standard RoPE
construction). The rotation is performed in-place on GPU with negligible (< 1 ms) overhead. The
mechanism is exposed as `RoPEAligner` in `agentmesh.mechanisms.bpp`.

## Why cross-attention masking is safe

Consider the two extremes:

- **Full attention** is exact but pays the full barrier and quadratic cost.
- **Directly reusing a worker's own decoding-time KVs** is fastest but breaks correctness — those
  KVs were computed under each worker's local context and never attended to the supervisor's
  instructions.

BPP takes the balance point: it **recomputes** each branch's KVs *under the supervisor prefix*
(absorbing the instruction context that direct reuse loses) and masks only the cross-branch attention
of marginal value. Crucially, **masking happens only during prefill**: after stitching, the decoding
pass attends over the *complete* KV set of all branches with full attention, so cross-branch
synthesis — comparing and integrating worker findings — still occurs as before. What BPP skips is
merely conditioning branch `i`'s *semantic encoding* on branch `j`'s content.

## Composability with KV-blending algorithms

KV-blending methods for RAG (e.g. **CacheBlend**, **EPIC**) face the same RoPE/semantic obstacle —
identical token segments recurring under different prefixes — and reuse most KVs while recomputing a
small, significant fraction. These are **orthogonal** to BPP: they prune *token-level* recomputation
*within* one context, whereas BPP decouples multiple *branches* at the dataflow layer. They compose
naturally: after BPP splits the gather into per-branch dataflows, a blending algorithm can replace
the incremental prefill *within* each parallel branch, and BPP still performs the final stitching and
realignment — further reducing recomputation overhead.

## Configuration

`BPPConfig` ([`agentmesh/core/config.py`](../agentmesh/core/config.py)):

```python
from agentmesh.core.config import BPPConfig

bpp = BPPConfig(
    enabled=True,
    max_parallel_branches=3,   # max concurrent branch prefills
    num_layers=32,             # transformer layers
    num_heads=32,              # attention heads
    head_dim=128,              # dimension per head
    rope_base=10000.0,         # RoPE base frequency
)
```

Public BPP surface (`agentmesh.mechanisms.bpp`): `BranchParallelManager` (aliases `BPAManager`,
`KVCacheManager`), `create_bpp_manager`, `WorkerOutput`, `SupervisorContext`, `BPAResult`,
`InterBranchMaskGenerator`, `RoPEAligner`. The pure-Python classes import without torch; torch is
required at instantiation. See [api_reference.md](api_reference.md).

## vLLM integration

The production realization of BPA is a custom **GPU KV connector** inside vLLM
([`agentmesh/mechanisms/bpp/vllm/`](../agentmesh/mechanisms/bpp/vllm/)), exposed as
`GPUKVConnector`. It performs branch-parallel prefill, KV stitching, and late RoPE realignment
directly on the GPU KV cache. See [vllm_integration.md](vllm_integration.md) for setup and the
`agentmesh/examples/bpp_vllm/` demo.

## Results

- **Latency:** reduces attention complexity `O(N²) → O(N)` in the worker count `N`; **8%–61%** time
  reduction on the deep-research benchmark, and it *enables* DES for further speedup.
- **Precision:** for medium models (8B/20B), BPP **matches or exceeds** Full Prefill (FP) — under
  full attention, mutually irrelevant cross-branch attention becomes "noise", whereas BPP
  concentrates on the supervisor-to-branch relationship. Baselines: **KV-Reuse with realignment
  (KVR)**, **Direct BPP without realignment (DBPP)**, **Full Prefill (FP)**. Direct KV reuse never
  sees the supervisor's instructions and routinely fails to produce a complete report; BPP without
  realignment shows greater variance from positional collisions.
