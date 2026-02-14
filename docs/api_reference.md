# API Reference

> Auto-generated overview. For full docstrings, see source code.

## Core

### `AgentMeshConfig`
```python
from agentmesh.core.config import AgentMeshConfig

cfg = AgentMeshConfig(
    srr=SRRConfig(confidence_threshold=0.7, hotspot_k=3),
    tsd=TSDConfig(max_parallel_workers=8, rope_base=10000.0),
    esf=ESFConfig(initial_microbatch_size=512, theta_min=64, theta_max=4096),
)
```

### `AgentMeshRuntime`
```python
from agentmesh.core.runtime import AgentMeshRuntime, AgentRequest

runtime = AgentMeshRuntime(config=cfg)
response = await runtime.process(AgentRequest(
    request_id="req-1", agent_id="researcher_1", prompt="..."
))
```

**Key methods:**
| Method | Description |
|--------|-------------|
| `process(request: AgentRequest) → AgentResponse` | Full SRR pipeline: lookup → reuse/reformulate → LLM call → store |
| `stream(request: AgentRequest) → AsyncIterator[StreamChunk]` | Streaming variant with ESF microbatch control |
| `batch_process(requests: List[AgentRequest]) → List[AgentResponse]` | TDA-accelerated parallel prefill |
| `get_stats()` | Runtime statistics (srr_activations, cold_retrievals, etc.) |
| `reset()` | Clear all caches and counters |

---

## SRR

### `SRRCache`
```python
from agentmesh.mechanisms.srr import SRRCache, create_srr_cache
```

| Method | Description |
|--------|-------------|
| `store(query, items, embedding=None)` | Cache a query with its response items (List[str]) |
| `lookup(query, tool_fn=None) → CacheResult` | Project query; return hit/miss with projection_intensity |
| `get_reused_items(result: CacheResult) → List[str]` | Return top-⌊s_max·n⌋ items from the cached entry |
| `get_stats()` | Activation/cold counts |

### `SemanticEmbedder`
Normalized sentence embeddings via all-MiniLM-L6-v2.

### `HotspotTokenExtractor`
Extracts top-k tokens with highest $c_j = \mathbf{e}_j \cdot \hat{\mathbf{r}}$.

### `QueryReformulator`
SLM-based residual query generation using Qwen2.5-0.5B-Instruct.

---

## TSD

### `TDAManager`
```python
from agentmesh.mechanisms.tsd import TDAManager, create_tda_manager
```

| Method | Description |
|--------|-------------|
| `set_supervisor_context(ctx: SupervisorContext)` | Register shared prefix KV |
| `prefill_worker(worker_output: WorkerOutput, prefill_fn=None)` | Prefill a worker with inter-worker masking |
| `stitch_and_realign() → TDAResult` | Merge KVs + RoPE correction |
| `get_parallel_groups()` | List parallelizable worker groups |
| `get_complexity_info()` | Original vs. reduced complexity |
| `clear()` | Reset all state |

### `InterWorkerMaskGenerator`
Block-diagonal attention mask: each worker sees prefix + self only.

### `RoPEAligner`
Applies $k_j = R_{j-i} \cdot k_i$ for positional realignment.

---

## ESF

### `ESFController`
```python
from agentmesh.mechanisms.esf import ESFController, create_esf_controller
```

| Method | Description |
|--------|-------------|
| `compute_next_theta(producer_time, consumer_time, slack_time, time_step)` | Core algorithm |
| `get_state()` | Current θ, step count, history |
| `reset()` | Restore initial state |

### `StateObserver`
Sliding-window observer that records `Observation` and estimates `Sensitivity`.

---

## Adapters

### `agentmesh.adapters.autogen`
Drop-in integration with AutoGen v0.4.2+. See [Getting Started](getting_started.md).

## Sidecars

### `agentmesh.sidecars.sidecar`
gRPC sidecar mesh for distributed deployment. See [Architecture](architecture.md).
