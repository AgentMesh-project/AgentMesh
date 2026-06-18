# API Reference

The AgentMesh public API. All names below are stable and importable as shown.

```python
import agentmesh
agentmesh.__version__          # "2.0.0"
```

## Top level — `agentmesh`

```python
from agentmesh import AgentMeshRuntime, AgentMeshConfig, create_runtime, __version__
```

| Name | Kind | Summary |
|------|------|---------|
| `AgentMeshRuntime` | class | The runtime orchestrating DTR, BPP, DES over the dataflow. |
| `AgentMeshConfig` | dataclass | Top-level config bundling `dtr`, `bpp`, `des`, `llm`. |
| `create_runtime(...)` | function | Convenience constructor with per-mechanism toggles. |
| `__version__` | str | `"2.0.0"`. |

## Configuration — `agentmesh.core.config`

```python
from agentmesh.core.config import (
    DTRConfig, BPPConfig, DESConfig, LLMConfig, AgentMeshConfig, load_config,
)
```

### `DTRConfig` — Delta Tool Retrieval (produce)

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `True` | enable DTR |
| `confidence_threshold` | `0.8` | gate `τ` (sweep `0.75–0.95`) |
| `retrieval_depth` | `10` | items per query `n` |
| `hotspot_k` | `3` | top-k delta tokens |
| `cache_size` | `1000` | max cached entries |
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | query embedder |
| `slm_model` | `Qwen/Qwen2.5-0.5B-Instruct` | reformulation SLM |

### `BPPConfig` — Branch-Parallel Prefill (consume)

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `True` | enable BPP |
| `max_parallel_branches` | `3` | max concurrent branch prefills |
| `num_layers` | `32` | transformer layers |
| `num_heads` | `32` | attention heads |
| `head_dim` | `128` | dimension per head |
| `rope_base` | `10000.0` | RoPE base frequency |

### `DESConfig` — Dynamic-Equilibrium Streaming (overlap)

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `True` | enable DES |
| `initial_theta` | `512` | initial chunk size `θ₀` (tokens) |
| `theta_min` | `64` | `θ_min` clip |
| `theta_max` | `4096` | `θ_max` clip |
| `damping_factor` | `1.0` | step smoothing |
| `window_size` | `5` | OSE sliding window |

### `LLMConfig`

| Field | Default |
|-------|---------|
| `endpoint` | `http://localhost:8000/v1` |
| `model` | `Qwen/Qwen3-32B` |
| `api_key` | `None` |
| `timeout` | `120` |
| `max_tokens` | `4096` |
| `temperature` | `0.7` |

### `AgentMeshConfig`

Fields: `.dtr` (`DTRConfig`), `.bpp` (`BPPConfig`), `.des` (`DESConfig`), `.llm` (`LLMConfig`),
`log_level` (`"INFO"`), `device` (`"cuda"`), `num_workers` (`3`).

Constructors / serializers:
`AgentMeshConfig.from_dict(data)`, `from_yaml(path)`, `from_env()`, `to_dict()`, `to_yaml(path)`.
`load_config(config_path=None, use_env=True)` merges **file > env > defaults**.

## Runtime — `agentmesh.core.runtime`

```python
from agentmesh.core.runtime import (
    AgentMeshRuntime, AgentRequest, AgentResponse, StreamChunk, create_runtime,
)
```

### `AgentMeshRuntime`

```python
AgentMeshRuntime(config=None, config_path=None, llm_client=None)
```

Attributes: `dtr_cache`, `bpp_manager`, `des_controller` (each `None` when its mechanism is
disabled), `config`, `stats`.

| Method | Signature | Description |
|--------|-----------|-------------|
| `process` | `async (request: AgentRequest) -> AgentResponse` | One dataflow with DTR delta retrieval (produce). |
| `stream` | `async (request: AgentRequest) -> AsyncIterator[StreamChunk]` | DES-paced streaming (overlap). |
| `batch_process` | `async (requests: List[AgentRequest]) -> List[AgentResponse]` | Independent worker branches via BPP (consume); sequential fallback without BPP or for ≤ 1 request. |
| `get_stats` | `() -> Dict[str, Any]` | Aggregate runtime + per-mechanism telemetry. |
| `reset` / `reset_stats` | `() -> None` | Clear stats and per-mechanism state. |

### `AgentRequest`

`request_id: str`, `agent_id: str`, `prompt: str`, `metadata: dict = {}`, `timestamp: float`.

### `AgentResponse`

`content: str`, `request_id`, `agent_id`, `tokens_generated`, `latency_ms`,
`dtr_activated: bool`, `projection_intensity: float`, `reuse_count: int`, `residual_count: int`,
`metadata: dict`.

### `StreamChunk`

`content: str`, `chunk_index: int`, `request_id`, `is_final: bool`, `tokens: int`,
`chunk_size: int`.

### `create_runtime`

```python
create_runtime(dtr_enabled=True, bpp_enabled=True, des_enabled=True,
               llm_endpoint="http://localhost:8000/v1", **kwargs) -> AgentMeshRuntime
```

## Dataflow abstraction — `agentmesh.core.dataflow`

```python
from agentmesh.core.dataflow import Dataflow, Producer, Consumer, DataflowEdge, Stage
```

- `DataflowEdge` — enum: `TOOL_TO_LLM`, `LLM_TO_LLM`.
- `Stage` — enum: `PRODUCE`, `TRANSFER`, `CONSUME`.
- `Producer(node_id, is_tool=False, metadata={})` — source side (tool exec or upstream decode).
- `Consumer(node_id, metadata={})` — sink side (always an LLM prefill).
- `Dataflow(producer, consumer, edge, produce_cost=0.0, transfer_cost=0.0, consume_cost=0.0,
  payload_tokens=0, metadata={})` — one edge.
  - `.total_cost` — `produce + transfer + consume`.
  - `.slack` — `δ = (produce_cost + transfer_cost) − consume_cost` (positive = BARRIER).
  - `.primary_mechanism()` — `"DTR"` for `TOOL_TO_LLM`, `"BPP"` for `LLM_TO_LLM`.

## DTR — `agentmesh.mechanisms.dtr`

```python
from agentmesh.mechanisms.dtr import (
    DTRCache, create_dtr_cache, CacheEntry, CacheResult,
    SemanticEmbedder, HotspotTokenExtractor, QueryReformulator,
)
```

| Name | Summary |
|------|---------|
| `DTRCache` | Projection-based tool-result cache: `lookup`, `get_reused_items`, `store`, `clear`, `get_stats`. |
| `create_dtr_cache(confidence_threshold, cache_size, embedding_model, ...)` | Factory used by the runtime. |
| `CacheResult` | Lookup result: `hit`, `projection_intensity`, `reuse_count`, `residual_count`, `reformulated_query`. |
| `CacheEntry` | A stored `{query, items}` entry. |
| `SemanticEmbedder` | Wraps the sentence-transformers embedder (~4 ms/query). |
| `HotspotTokenExtractor` | Top-k delta-characterizing tokens from the residual `r̂`. |
| `QueryReformulator` | SLM-based differential query (`Q_diff`) builder. |

See [dtr.md](dtr.md).

## BPP — `agentmesh.mechanisms.bpp`

```python
from agentmesh.mechanisms.bpp import (
    BranchParallelManager, BPAManager, KVCacheManager, create_bpp_manager,
    WorkerOutput, SupervisorContext, BPAResult, InterBranchMaskGenerator, RoPEAligner,
)
```

| Name | Summary |
|------|---------|
| `BranchParallelManager` | Orchestrates branch-parallel prefill, masking, KV stitch, realignment. Aliases: `BPAManager`, `KVCacheManager`. |
| `create_bpp_manager(num_layers, num_heads, head_dim, max_parallel_branches, rope_base, device)` | Factory used by the runtime. |
| `InterBranchMaskGenerator` | Builds the BPA cross-branch mask (prefill-only). |
| `RoPEAligner` | Late positional realignment `k_j = R_{j−i} k_i` (in-place, < 1 ms). |
| `WorkerOutput`, `SupervisorContext`, `BPAResult` | Branch I/O and result dataclasses. |
| `GPUKVConnector` | (optional, requires vLLM) production KV connector — see [vllm_integration.md](vllm_integration.md). |

The torch-dependent classes import lazily; the names are re-exported from
`agentmesh.mechanisms` when torch is installed. See [bpp.md](bpp.md).

## DES — `agentmesh.mechanisms.des`

```python
from agentmesh.mechanisms.des import (
    DESController, create_des_controller, OnlineSensitivityEstimator, Observation, Sensitivity,
)
```

| Name | Summary |
|------|---------|
| `DESController` | Drives slack `δ→0` via the Newton step `Δθ = −δ/ρ`. Attributes: `current_theta`; methods `compute_next_theta(producer_time, consumer_time, slack_time, time_step)`, `get_state`, `reset`. |
| `create_des_controller(initial_theta, min_theta, max_theta, damping_factor, window_size)` | Factory used by the runtime. |
| `OnlineSensitivityEstimator` | Finite-difference `Ŝ_pro`, `Ŝ_con`. |
| `Observation`, `Sensitivity` | Telemetry / sensitivity dataclasses. |

See [des.md](des.md).

## Sidecars & adapters

```python
from agentmesh.sidecars import Message, AgentSidecar, LLMSidecar, ToolSidecar, SidecarMesh
from agentmesh.adapters.autogen import AutoGenAdapter, WrappedAgent, create_autogen_adapter
```

`SidecarMesh` wires `AgentSidecar`/`LLMSidecar`/`ToolSidecar` to the runtime's `dtr_cache`,
`bpp_manager`, and `des_controller`. `AutoGenAdapter` wraps an AutoGen v0.4.2+ runtime. See
[architecture.md](architecture.md).
