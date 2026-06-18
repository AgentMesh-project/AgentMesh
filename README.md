# AgentMesh: Toward Redundancy- and Barrier-Free Dataflow in Multi-Agent Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**AgentMesh is an efficient runtime for multi-agent LLM systems (MAS).** It models every
interaction in a MAS as a *produce–transfer–consume* dataflow — where the **producer** is a tool or
an upstream agent's LLM decode, the transfer crosses a transport layer, and the **consumer** is
an LLM prefill. A structure-blind runtime executes each dataflow as one opaque,
coarse-grained block, wasting end-to-end time on two pathologies: **REDUNDANCY** (re-producing
content already held) and **BARRIER** (a consumer idling until its producer fully finishes).
AgentMesh makes each dataflow redundancy- and barrier-free with three interlocking mechanisms (DTR,
BPP, DES), reducing end-to-end latency by **up to 62%** on deep-research workloads (and up to 21% on
coding agents) with **no statistically significant quality loss** — generalizing across three GPU
generations, six models, and two agentic application classes.

---

## Architecture

AgentMesh sits *between* the agent orchestrator and the tool/LLM execution engines, intercepting
every produce–transfer–consume dataflow on its two dominant edges — `TOOL_TO_LLM` (a tool result fed
to an agent's prefill) and `LLM_TO_LLM` (an upstream agent's decode fed to a downstream prefill). Its
three mechanisms act directly on those edges.

```
                  ┌────────────────────────────────────────────────┐
                  │            Agent Orchestrator (MAS)            │
                  │   AutoGen (deep research) · OpenCode (coding)  │
                  └───────────────────────────┬────────────────────┘
                                              │  produce─transfer─consume dataflows
   ┌──────────────────────────────────────────▼────────────────────────────────────────────┐
   │                                AgentMesh Runtime                                      │
   │                                                                                       │
   │     tool ──TOOL_TO_LLM──▶ [ DTR ]            agent ──LLM_TO_LLM──▶ [ BPP ]            │
   │     serve cached base,                       parallel-prefill independent             │
   │     fetch only the delta,                    worker branches vs. shared prefix,       │
   │     async-refresh  (PRODUCE)                 stitch KV w/ late RoPE  (CONSUME)        │
   │                                                                                       │
   │   ──────────────────────  [ DES ] overlaps PRODUCE│TRANSFER│CONSUME  ───────────────  │
   │                            online controller drives slack δ→0 (OSE Newton step)       │
   │                                                                                       │
   │            Sidecar mesh (gRPC)  ·  framework adapters  ·  vLLM GPU KV connector       │
   └──────────────────────────────────────────┬────────────────────────────────────────────┘
                                              │  inference / tool API
                  ┌───────────────────────────▼────────────────────┐
                  │      Tool executors       ·       vLLM         │
                  │   (web search, scrape,           (LLM prefill  │
                  │    code/file ops)                 + decode)    │
                  └────────────────────────────────────────────────┘
```

## Mechanisms

Each mechanism targets a different stage of the dataflow, so their gains **compose rather than
overlap**.

| Mechanism | Full name | Stage | Edge | What it does |
|-----------|-----------|-------|------|--------------|
| **DTR** | Delta Tool Retrieval | **produce** (removes REDUNDANCY) | `TOOL_TO_LLM` | Confidence-gated projection of a query onto cached results: serves the cached *base* (`⌊s_max·n⌋` items), fetches only the semantic *delta* via an SLM-reformulated query, and async-refreshes the exact entry. |
| **BPP** | Branch-Parallel Prefill | **consume** (removes BARRIER) | `LLM_TO_LLM` | Prefills independent worker branches in parallel against the shared supervisor prefix, then stitches branch-local KVs with late RoPE realignment. |
| └ **BPA** | Branch-Parallel Attention | (sub-mechanism of BPP) | — | Masks cross-branch attention *during prefill only*; collapses attention cost from `O(N²)` to `O(N)` in the branch count `N`. Decode still attends the full stitched KV. |
| **DES** | Dynamic-Equilibrium Streaming | **overlap** (drives slack δ→0) | both | Online controller paces streamed chunk size θ to the runtime rates of producer and consumer, overlapping produce/transfer/consume. |
| └ **OSE** | Online Sensitivity Estimator | (sub-mechanism of DES) | — | Finite-difference estimates of marginal sensitivities `Ŝ_pro`, `Ŝ_con`; Newton step `Δθ = −δ/ρ`, `ρ = Ŝ_pro − Ŝ_con`, clipped to `[θ_min, θ_max]`. |

See [docs/architecture.md](docs/architecture.md) for the full dataflow model, and
[docs/dtr.md](docs/dtr.md), [docs/bpp.md](docs/bpp.md), [docs/des.md](docs/des.md) for each mechanism.

## Install

AgentMesh has a deliberately small core: `import agentmesh` and the config classes work with only
`numpy` + `pyyaml`. Heavy components (torch, sentence-transformers, FAISS, AutoGen, vLLM) are
optional extras imported lazily.

```bash
git clone https://github.com/AgentMesh-project/AgentMesh.git
cd AgentMesh
pip install -e .                 # core runtime (numpy, pyyaml, openai, aiohttp, ...)

# Optional extras
pip install -e ".[autogen]"      # AutoGen adapter
pip install -e ".[vllm]"         # vLLM backend + BPP GPU KV connector
pip install -e ".[dev]"          # tests, linters, type checking
```

Requires Python ≥ 3.10. A CUDA-capable GPU is recommended for BPP parallel prefill and the vLLM
backend; the pure-Python runtime, DTR cache, and DES controller run on CPU.

## Quick Start

```python
import asyncio
from agentmesh import AgentMeshRuntime, AgentMeshConfig
from agentmesh.core.config import DTRConfig, BPPConfig, DESConfig, LLMConfig
from agentmesh.core.runtime import AgentRequest

config = AgentMeshConfig(
    dtr=DTRConfig(enabled=True, confidence_threshold=0.8, hotspot_k=3),   # produce
    bpp=BPPConfig(enabled=True, max_parallel_branches=3),                 # consume
    des=DESConfig(enabled=True, initial_theta=512),                       # overlap
    llm=LLMConfig(endpoint="http://localhost:8000/v1", model="Qwen/Qwen3-32B"),
)
runtime = AgentMeshRuntime(config=config)

async def main():
    # process(): single dataflow with DTR delta retrieval (the produce step)
    resp = await runtime.process(AgentRequest(
        request_id="r1", agent_id="researcher_0",
        prompt="What are the key challenges in distributed systems?",
    ))
    print(resp.dtr_activated, resp.projection_intensity, resp.reuse_count)

    # batch_process(): independent worker branches prefilled in parallel via BPP
    responses = await runtime.batch_process([
        AgentRequest(request_id="b0", agent_id="worker_0", prompt="subtask 0"),
        AgentRequest(request_id="b1", agent_id="worker_1", prompt="subtask 1"),
        AgentRequest(request_id="b2", agent_id="worker_2", prompt="subtask 2"),
    ])

    # stream(): DES adapts chunk size θ online to overlap producer/consumer
    async for chunk in runtime.stream(AgentRequest(
        request_id="s1", agent_id="writer_0", prompt="Write a detailed analysis...",
    )):
        print(chunk.content, end="", flush=True)

    print(runtime.get_stats())

asyncio.run(main())
```

### Demos

```bash
# Deep research (supervisor + parallel researchers; DTR + BPP + DES)
python -m agentmesh.examples.deep_research.demo \
    --topic "Impact of LLMs on software engineering" --num-workers 3

# Coding agent (OpenCode-style; tool-light, prefill-bound)
# see agentmesh/examples/coding_agent/

# BPP on a real vLLM backend (GPU KV connector)
# see agentmesh/examples/bpp_vllm/ and docs/vllm_integration.md
```

A convenience constructor toggles mechanisms without a full config:

```python
from agentmesh import create_runtime
runtime = create_runtime(dtr_enabled=True, bpp_enabled=False, des_enabled=True,
                         llm_endpoint="http://localhost:8000/v1")
```

## Evaluation

All experiments live under [`benchmarks/`](benchmarks/), one subdirectory per claim:
`benchmarks/dtr/`, `benchmarks/bpp/`, `benchmarks/des/`, `benchmarks/end_to_end/`. Every script
has a fast synthetic default that runs with no GPU, cluster, or LLM — see
[benchmarks/README.md](benchmarks/README.md) for the script→claim map and how to supply your
own traces for full reproduction.

## Repository Structure

```
AgentMesh/
├── agentmesh/
│   ├── core/                     # runtime, config, dataflow abstraction
│   │   ├── config.py             # DTRConfig, BPPConfig, DESConfig, LLMConfig, AgentMeshConfig
│   │   ├── runtime.py            # AgentMeshRuntime, AgentRequest/Response, StreamChunk
│   │   └── dataflow.py           # Dataflow, Producer, Consumer, DataflowEdge, Stage
│   ├── mechanisms/
│   │   ├── dtr/                  # Delta Tool Retrieval (cache, embedder, reformulator)
│   │   ├── bpp/                  # Branch-Parallel Prefill (BPA, RoPE realign, KV stitch)
│   │   │   └── vllm/             # GPU KV connector for the vLLM backend
│   │   └── des/                  # Dynamic-Equilibrium Streaming (controller, OSE)
│   ├── adapters/                 # framework adapters (AutoGen)
│   ├── sidecars/                 # gRPC sidecar mesh
│   ├── proto/                    # protobuf service definitions
│   └── examples/                 # deep_research, coding_agent, bpp_vllm demos
├── benchmarks/                   # dtr, bpp, des, end_to_end (synthetic defaults)
├── config/                       # agentmesh.example.yaml
├── docs/                         # architecture, dtr, bpp, des, getting_started, api_reference, vllm_integration
└── tests/
```

## Citation

```bibtex
@inproceedings{agentmesh2026,
  title     = {AgentMesh: Toward Redundancy- and Barrier-Free Dataflow in Multi-Agent Systems},
  author    = {{AgentMesh Authors}},
  year      = {2026},
  note      = {\url{https://github.com/AgentMesh-project/AgentMesh}},
}
```

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 AgentMesh Authors.
