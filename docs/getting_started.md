# Getting Started

AgentMesh is a structure-aware runtime for multi-agent LLM systems. It models every interaction as a
produce–transfer–consume dataflow and makes each dataflow redundancy- and barrier-free via three
mechanisms — **DTR** (produce), **BPP** (consume), and **DES** (overlap). This guide installs
AgentMesh, configures it, and runs a demo.

## Prerequisites

- **Python ≥ 3.10**
- **AutoGen v0.4.2** (deep-research orchestrator) and/or **OpenCode v1.14.51** (coding agent) —
  optional, via extras.
- **vLLM v0.13.0** for LLM inference — optional, via extras. A CUDA-capable GPU is recommended for
  BPP parallel prefill and the vLLM backend.

The core runtime (`import agentmesh`, the config classes, the DTR cache, and the DES controller)
works on CPU with only `numpy` + `pyyaml`. Heavy components (torch, sentence-transformers, FAISS,
AutoGen, vLLM) are optional extras imported lazily.

## Installation

```bash
git clone https://github.com/AgentMesh-project/AgentMesh.git
cd AgentMesh
pip install -e .                 # core runtime

# Optional extras
pip install -e ".[autogen]"      # AutoGen v0.4.2 adapter
pip install -e ".[vllm]"         # vLLM v0.13.0 backend + BPP GPU KV connector
pip install -e ".[dev]"          # tests, linters, type checking
```

## Configure

### Option A — in Python

```python
from agentmesh import AgentMeshConfig
from agentmesh.core.config import DTRConfig, BPPConfig, DESConfig, LLMConfig

config = AgentMeshConfig(
    dtr=DTRConfig(confidence_threshold=0.8, hotspot_k=3),   # produce
    bpp=BPPConfig(max_parallel_branches=3),                 # consume
    des=DESConfig(initial_theta=512),                       # overlap
    llm=LLMConfig(endpoint="http://localhost:8000/v1", model="Qwen/Qwen3-32B"),
)
```

### Option B — environment variables

`AgentMeshConfig.from_env()` (and the runtime's default loader) read `AGENTMESH_*` variables:

```bash
export AGENTMESH_LLM_ENDPOINT="http://localhost:8000/v1"
export AGENTMESH_LLM_MODEL="Qwen/Qwen3-32B"
export AGENTMESH_API_KEY="EMPTY"                  # placeholder for a local vLLM endpoint
export AGENTMESH_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export AGENTMESH_SLM_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
export AGENTMESH_CONFIDENCE_THRESHOLD="0.8"       # DTR τ
export AGENTMESH_LOG_LEVEL="INFO"
export AGENTMESH_DEVICE="cuda"
```

### Option C — YAML

Copy [`config/agentmesh.example.yaml`](../config/agentmesh.example.yaml), edit it, and load it.
Precedence is **file > env > defaults**:

```python
from agentmesh import AgentMeshConfig
config = AgentMeshConfig.from_yaml("agentmesh.yaml")
```

## Initialize the runtime

```python
from agentmesh import AgentMeshRuntime
runtime = AgentMeshRuntime(config=config)          # or AgentMeshRuntime(config_path="agentmesh.yaml")
```

## Run a demo

```bash
# Deep research: supervisor + parallel researchers, exercising DTR + BPP + DES
python -m agentmesh.examples.deep_research.demo \
    --topic "Impact of LLMs on software engineering" \
    --num-workers 3 \
    --llm-backend http://localhost:8000/v1
```

The demo prints a research report and per-mechanism statistics (DTR cache hit rate, item reuse rate,
parallel batches). Additional demos live in `agentmesh/examples/coding_agent/` (OpenCode-style,
tool-light) and `agentmesh/examples/bpp_vllm/` (BPP on a real vLLM backend — see
[vllm_integration.md](vllm_integration.md)).

## Enable / disable mechanisms

Each mechanism has an `enabled` flag; disabling one degrades to a safe sequential fallback (e.g.
`batch_process` without BPP runs branches one at a time). For quick toggling, use `create_runtime`:

```python
from agentmesh import create_runtime

# Ablation: DTR only
runtime = create_runtime(dtr_enabled=True, bpp_enabled=False, des_enabled=False,
                         llm_endpoint="http://localhost:8000/v1")
```

Or set the flags on a config:

```python
config.dtr.enabled = True
config.bpp.enabled = False
config.des.enabled = True
```

## The three entry points

| Method | Dataflow shape | Mechanism |
|--------|----------------|-----------|
| `await runtime.process(req)` | single dataflow | DTR delta retrieval (produce) |
| `await runtime.batch_process(reqs)` | independent worker branches | BPP parallel prefill (consume) |
| `async for chunk in runtime.stream(req)` | streamed chunks | DES-paced overlap |

```python
import asyncio
from agentmesh.core.runtime import AgentRequest

async def main():
    resp = await runtime.process(AgentRequest(
        request_id="r1", agent_id="researcher_0",
        prompt="What are the key challenges in distributed systems?",
    ))
    print("DTR activated:", resp.dtr_activated,
          "| projection intensity:", resp.projection_intensity,
          "| reused items:", resp.reuse_count)

asyncio.run(main())
```

## AutoGen integration

```python
from agentmesh.adapters.autogen import AutoGenAdapter, create_autogen_adapter

adapter = create_autogen_adapter(runtime=runtime)   # wraps an AutoGen v0.4.2+ runtime
# use the adapter in place of the standard AutoGen runtime
```

## Run the tests

```bash
pytest tests/ -v
```

## Next steps

- [architecture.md](architecture.md) — the produce–transfer–consume dataflow model.
- [dtr.md](dtr.md) · [bpp.md](bpp.md) · [des.md](des.md) — each mechanism in depth.
- [api_reference.md](api_reference.md) — the full public API.
