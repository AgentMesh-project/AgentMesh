# SRR — Semantic Residual Retrieval

## Problem

In multi-agent workflows, successive tool calls and research queries often share substantial semantic overlap.  Sending the full prompt each time wastes both network bandwidth and LLM prefill compute.

## Mechanism

SRR maintains a **semantic cache** indexed by normalized embedding vectors (all-MiniLM-L6-v2).

### SRR Lookup & Store

1. **Embed** the incoming query: $\mathbf{q} = \text{embed}(Q)$, $\|\mathbf{q}\|=1$.
2. **Project** onto each cached entry $\mathbf{e}_i$: projection intensity $s_i = \mathbf{q} \cdot \mathbf{e}_i$.
3. **Confidence gating**: if $s_{\max} \geq \tau$ (default $\tau = 0.7$), the cache activates.
4. **Proportional item allocation**: reuse $\lfloor s_{\max} \cdot n \rfloor$ items from the matched entry.
5. **Residual vector**: $\mathbf{r} = \mathbf{q} - s_{\max}\,\mathbf{e}_{\max}$, $\hat{\mathbf{r}} = \mathbf{r}/\|\mathbf{r}\|$.
6. **Hotspot token extraction**: for each token $j$ in $Q$, compute $c_j = \mathbf{e}_j \cdot \hat{\mathbf{r}}$ and take top-$k$ (default $k=3$).
7. **SLM reformulation**: prompt Qwen2.5-0.5B-Instruct with hotspot tokens to generate a compact residual query.
8. **Offset variance correction**: after the full response arrives, update the cached embedding to track semantic drift.

### Key Design Choices

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| τ (confidence threshold) | 0.7 | Balances precision (99.4%) vs. activation rate |
| k (hotspot tokens) | 3 | Captures residual semantics without noise |
| SLM | Qwen2.5-0.5B-Instruct | Lightweight, <5 ms reformulation |

## API

```python
from agentmesh.mechanisms.srr import SRRCache, create_srr_cache

cache = create_srr_cache(confidence_threshold=0.7, cache_size=1024)

# Store after receiving a response
cache.store(query="What is ML?", items=["ML is a subset of AI", ...])

# Lookup on new query
result = cache.lookup("What is machine learning?")
if result.hit:
    reused = cache.get_reused_items(result)
    # result.hotspot_tokens, result.reformulated_query available for residual
```
