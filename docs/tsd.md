# TSD — Topology-aware Semantic Decoupling

## Problem

In a supervisor-worker topology the supervisor dispatches tasks to N workers.  Standard LLM serving processes each worker sequentially, re-computing the shared supervisor prefix N times.  With causal attention, the total cost is:

$$O\bigl(L_{\text{prefix}} \cdot \sum_i L_i + \sum_i \sum_j L_i \cdot L_j\bigr)$$

## Mechanism — Topologically Decoupled Attention (TDA)

TDA exploits two structural invariants:

1. **Workers share the supervisor prefix** — its KV cache can be computed once.
2. **Workers never attend to sibling workers** — cross-worker KV entries are masked out.

### Steps

1. **Inter-worker masking**: generate a block-diagonal attention mask that allows each worker to attend to the prefix and itself, but not to other workers.
2. **Timely parallel prefill**: all workers prefill concurrently in a single batched forward pass.
3. **KV stitching**: merge per-worker KV caches back into the supervisor's decoding context.
4. **RoPE positional realignment**: correct position embeddings via the rotation formula $k_j = R_{j-i} \cdot k_i$ to maintain globally consistent positions.

### Complexity Reduction

$$O\bigl(L_{\text{prefix}} \cdot \sum_i L_i + \sum_i \sum_j L_i \cdot L_j\bigr) \;\longrightarrow\; O\bigl(\max(L_{\text{prefix}} \cdot L_i + L_i^2)\bigr)$$

## API

```python
from agentmesh.mechanisms.tsd import TDAManager, SupervisorContext, WorkerOutput

manager = TDAManager(
    num_layers=32, num_heads=32, head_dim=128,
    max_parallel_workers=8, rope_base=10000.0, device="cuda",
)

# 1. Register supervisor prefix
manager.set_supervisor_context(SupervisorContext(
    tokens=prefix_ids, kv_states=prefix_kv,
))

# 2. Parallel prefill for each worker
for wid, tokens, kv in workers:
    manager.prefill_worker(WorkerOutput(
        worker_id=wid, tokens=tokens, kv_states=kv,
    ))

# 3. Stitch + realign for decoding
result = manager.stitch_and_realign()
# result.combined_kv, result.total_tokens, result.worker_order
```
