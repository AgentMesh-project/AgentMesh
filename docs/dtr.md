# DTR — Delta Tool Retrieval

**DTR shrinks the *produce* step of a `TOOL_TO_LLM` dataflow, making it redundancy-free.**

A tool call follows a *query–search–fetch* pattern and is retrieval-heavy: a lightweight query
returns a list of `n` lengthy, discrete items (web pages, snippets, documents). This produce step
dominates deep-research latency and is highly **redundant** — queries that are lexically different
but *semantically overlapping*, both across requests on similar topics and within a request across
refinement steps. Exact-match caches miss this fuzzy redundancy; hit-or-miss semantic caches reuse a
result *in full or not at all*. DTR instead reuses the overlapping cached **base** and fetches only
the missing **delta**.

## Algorithm

DTR decomposes an incoming query `Q_in` (retrieving `n` items) into a base covered by cache and a
delta of novel intent, in three steps (Algorithm 1).

```
Algorithm 1: Delta Tool Retrieval
Require: query Q_in, cache V_cache, threshold τ, depth n
 1:  v_in   ← Encoder(Q_in)                              # embed input (normalized)
 2:  s_max  ← max_{v_i ∈ V_cache} (v_in · v_i)           # best projection intensity
 3:  v_max  ← argmax_{v_i ∈ V_cache} (v_in · v_i)
 4:  if s_max ≤ τ then                                   # ── COLD RETRIEVAL ──
 5:      fetch all n items via tool execution
 6:      update V_cache with {Q_in, items}
 7:  else                                                # ── DELTA RETRIEVAL ──
 8:      reuse k_reuse = ⌊s_max · n⌋ items from v_max     # the cached BASE
 9:      r = v_in − s_max·v_max ;  r̂ = r / ‖r‖           # the DELTA (residual)
10:      top-k tokens {t_j} ← argmax_j (e_j · r̂)         # hotspot tokens (k=3)
11:      Q_diff ← SLM_Reformulate(Q_in, {t_j})           # differential query
12:      fetch n − k_reuse delta items via Q_diff
13:      execute Q_in in background; refresh V_cache      # ── ASYNC REFRESH ──
14:  end if
```

### (1) Base reuse via confidence-gated projection (lines 1–8)

DTR embeds `Q_in` into a normalized vector `v_in` and projects it onto each cached query `v_i`:

```
p_i = (v_in · v_i / ‖v_i‖²) v_i = (v_in · v_i) v_i = s_i v_i
```

Because vectors are normalized, the **projection intensity** `s_i = v_in · v_i` is the cosine
similarity — it measures how much of `Q_in` the cached result of `Q_i` can fulfill. A **confidence
gate** `τ ∈ [0,1]` decides reuse on the best match `s_max`:

- `s_max ≤ τ` → **cold-fetch** all `n` items (prevents low-similarity cache content from polluting
  the LLM context) and update the cache pool.
- `s_max > τ` → reuse `⌊s_max · n⌋` items from the best match — turning similarity into a *graded
  reuse fraction* rather than a binary hit.

### (2) Delta retrieval via residual query reformulation (lines 9–12)

The unreused `⌈(1 − s_max)·n⌉` items form the **delta**. DTR computes the residual `r = v_in −
s_max·v_max` (the intent uncovered by the base), normalizes it to `r̂`, and scores each query token
`t_j` (embedded as `e_j`) by alignment `c_j = e_j · r̂`. The **top-k** delta-characterizing tokens
are handed to a small language model (SLM) that reformulates a *differential query* `Q_diff`
emphasizing them while preserving the original subject. Fetching incrementally with `Q_diff` returns
items concentrated on the delta intent. Default `k = 3`: fewer hotspot tokens underspecify the
delta, more dilute its focus and reintroduce redundancy.

### (3) Asynchronous refresh (line 13)

Partial reuse risks **drift** — small base–delta misalignments accumulating across successive hits.
DTR serves the reused and delta items immediately at the frontend, while a low-priority background
task executes the exact `Q_in` and overwrites the cache entry with the authentic result. The refresh
only pulls items not already cached, so it bounds drift *without* adding foreground latency and
*reduces* total backend fetch load.

## Overhead

DTR's control overhead is three orders of magnitude below a cold tool execution — negligible against
the WAN fetch it removes:

| Component | Cost |
|-----------|------|
| Query embedding (`all-MiniLM-L6-v2`, SentenceTransformers) | ~4 ms |
| FAISS IVF-Flat nearest-neighbor lookup (`O(log N)`) | sub-millisecond |
| SLM reformulation (`Qwen2.5-0.5B-Instruct`, CPU) | < 15 ms |

## Configuration

`DTRConfig` ([`agentmesh/core/config.py`](../agentmesh/core/config.py)):

```python
from agentmesh.core.config import DTRConfig

dtr = DTRConfig(
    enabled=True,
    confidence_threshold=0.8,                                # τ (paper deployment default)
    retrieval_depth=10,                                      # n — items per query
    hotspot_k=3,                                             # k — top-k delta tokens
    cache_size=1000,                                         # max entries (LRU)
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    slm_model="Qwen/Qwen2.5-0.5B-Instruct",
)
```

Override at runtime with env vars `AGENTMESH_CONFIDENCE_THRESHOLD`, `AGENTMESH_EMBEDDING_MODEL`,
`AGENTMESH_SLM_MODEL`. The public DTR surface is `DTRCache`, `create_dtr_cache`, `CacheEntry`,
`CacheResult`, `SemanticEmbedder`, `HotspotTokenExtractor`, `QueryReformulator`
(`agentmesh.mechanisms.dtr`); see [api_reference.md](api_reference.md).

## Baselines & results

DTR is compared against: **No-Cache (NC)**, **Exact-Match (EM)**, **Fixed-Fraction-Reuse (FFR)**
(reuse a fraction without a new fetch above a threshold), and **Cortex** (a knowledge-caching
architecture for agentic search with an all-or-nothing LLM-judged policy). Metric: exact-match score
(the same metric Cortex uses).

- **Latency & load:** by serving the cached base and fetching only the delta, DTR cuts total
  scraping time and tool latency by **71.5%**, and lowers backend fetch volume by **57%** (the async
  refresh pulls only absent items).
- **Precision & safety:** DTR outperforms FFR and Cortex on exact-match score and stability — delta
  fetch recovers the necessary semantics that partial hits lack.
- **Threshold sweep:** `τ` swept `0.75–0.95` (deployment default `0.8`). Exact-match score degrades
  *shallowly* as `τ` decreases, confirming robustness to this hyperparameter (which trades efficiency
  for accuracy).
