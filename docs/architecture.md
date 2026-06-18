# Architecture

AgentMesh is a **structure-aware runtime for multi-agent LLM systems (MAS)**. This document explains
the dataflow model it is built on, the two pathologies it eliminates, how the runtime intercepts
each dataflow, the mechanism-to-stage mapping, and the sidecar/proto middleware that carries it all.

## The produce–transfer–consume dataflow

A MAS is a dependency graph of agents, tools, and LLM engines that exchange intermediate artifacts.
AgentMesh abstracts every such exchange as a single **produce–transfer–consume dataflow** — one
directed edge in the interaction graph:

```
   producer ──produce──▶ [ artifact ] ──transfer──▶ consumer
   (tool exec OR          (tool result OR             (ALWAYS an
    upstream agent's       decoded tokens)              LLM prefill)
    LLM decode)
```

- **produce** — a *producer* creates an artifact. The producer is either a **tool execution** (e.g.
  a web search returning a list of pages) or an **upstream agent's LLM decoding** (a sequence of
  decoded tokens).
- **transfer** — the artifact crosses a transport layer (in-process queue, gRPC stream, network hop).
- **consume** — a *consumer* ingests the artifact. The consumer is **always an LLM prefill**: a
  downstream agent reads the artifact into its context window.

A MAS's end-to-end latency is the aggregate cost of its produce–transfer–consume dataflows, chained
or parallel-branched. This abstraction is formalized in
[`agentmesh/core/dataflow.py`](../agentmesh/core/dataflow.py) as `Dataflow`, `Producer`, `Consumer`,
and the `Stage` enum.

## The two dominant edge types

In practice, two edge types dominate (modeled as `DataflowEdge` in `core/dataflow.py`):

| Edge | Producer → Consumer | Example | Primary mechanism |
|------|---------------------|---------|-------------------|
| `TOOL_TO_LLM` | tool result → LLM prefill | web search/scrape result fed to a researcher agent | **DTR** |
| `LLM_TO_LLM`  | upstream decode → LLM prefill | worker outputs gathered by a supervisor | **BPP** |

`Dataflow.primary_mechanism()` returns `"DTR"` for `TOOL_TO_LLM` and `"BPP"` for `LLM_TO_LLM`; DES
overlaps the stages of either edge.

## Why a structure-blind runtime is slow

A runtime that treats each dataflow as one opaque, indivisible block — fully materializing the
artifact before the consumer may touch it — wastes end-to-end time two ways:

- **REDUNDANCY** — the producer re-produces content already held. A tool re-fetches semantically
  overlapping results across similar requests and refinement steps; a shared consumer computes full
  cross-attention over mutually independent parallel branches (the cross-branch terms carry
  marginal value). *Addressed at the produce stage by DTR and at the consume stage by BPP.*
- **BARRIER** — the producer must finish the *whole* message before releasing *any* part, so the
  consumer idles until the last byte arrives. An LLM cannot start prefilling until the tool/agent
  finishes; a supervisor stalls on the straggler branch. *Addressed at the consume stage by BPP and
  smoothed across all three stages by DES.*

`Dataflow.slack` exposes the barrier slack `δ = (produce_cost + transfer_cost) − consume_cost`:
positive slack means the consumer is waiting on the producer.

## How the runtime intercepts dataflows

`AgentMeshRuntime` (in [`agentmesh/core/runtime.py`](../agentmesh/core/runtime.py)) sits across the
orchestrator and the tool/LLM engines and exposes three async entry points, one per dataflow shape:

```
                  ┌──────────────────────────────────────────────┐
                  │            Agent Orchestrator (MAS)           │
                  │     AutoGen (research) · OpenCode (coding)    │
                  └────────────────────────┬─────────────────────┘
                                           │  dataflows
   ┌────────────────────────────────────────▼──────────────────────────────────────┐
   │                            AgentMeshRuntime                                     │
   │   .process(req)      ── single dataflow,  DTR delta retrieval     (produce)     │
   │   .batch_process(rs) ── independent worker branches, BPP parallel (consume)     │
   │   .stream(req)       ── DES-paced chunks, slack δ→0               (overlap)     │
   │                                                                                │
   │   dtr_cache · bpp_manager · des_controller   ·   get_stats()                   │
   └────────────────────────────────────────┬──────────────────────────────────────┘
                                           │  inference / tool API
                  ┌────────────────────────▼─────────────────────┐
                  │     Tool executors      ·       vLLM          │
                  └──────────────────────────────────────────────┘
```

Each mechanism is constructed only when enabled in `AgentMeshConfig`; disabled mechanisms degrade to
sequential fallbacks (e.g. `batch_process` without BPP runs branches one at a time). The runtime
holds the three mechanism objects directly as `runtime.dtr_cache`, `runtime.bpp_manager`, and
`runtime.des_controller`, and aggregates per-mechanism telemetry via `get_stats()`.

## Mechanism → stage mapping

| Mechanism | Full name | Stage | Pathology removed |
|-----------|-----------|-------|-------------------|
| **DTR** | Delta Tool Retrieval | produce | REDUNDANCY |
| **BPP** (with **BPA**) | Branch-Parallel Prefill (Branch-Parallel Attention) | consume | BARRIER + REDUNDANCY |
| **DES** (with **OSE**) | Dynamic-Equilibrium Streaming (Online Sensitivity Estimator) | overlap | BARRIER |

Because the three mechanisms target different stages, their gains **compose rather than overlap**.
Chained together, AgentMesh turns a rigid, stop-and-go dataflow into an organic pipeline:
delta-fetched producers and branch-parallel consumers interact over a dynamic streaming transfer
fabric. Details: [dtr.md](dtr.md), [bpp.md](bpp.md), [des.md](des.md).

## Sidecar and proto middleware

AgentMesh is realized as a distributed middleware stack so it can intercept dataflows without
rewriting the orchestrator:

- **Sidecar mesh** ([`agentmesh/sidecars/`](../agentmesh/sidecars/)) — a three-layer sidecar
  architecture (`AgentSidecar`, `LLMSidecar`, `ToolSidecar`, coordinated by `SidecarMesh`) that
  carries `Message` objects between agents, LLMs, and tools. The mesh is wired to the runtime's
  `dtr_cache`, `bpp_manager`, and `des_controller`, so each hop applies the right mechanism for its
  edge.
- **Proto definitions** ([`agentmesh/proto/`](../agentmesh/proto/)) — gRPC service definitions for
  the produce–transfer–consume dataflow. Generated `*_pb2.py` files are excluded from version
  control; regenerate them with `grpc_tools.protoc` (see the module docstring).
- **Framework adapters** ([`agentmesh/adapters/`](../agentmesh/adapters/)) — `AutoGenAdapter` wraps
  an AutoGen v0.4.2+ runtime so existing agent code runs unchanged on top of AgentMesh.
- **vLLM GPU KV connector** ([`agentmesh/mechanisms/bpp/vllm/`](../agentmesh/mechanisms/bpp/vllm/)) —
  the production realization of BPA inside vLLM; see [vllm_integration.md](vllm_integration.md).

## Further reading

- [getting_started.md](getting_started.md) — install, configure, run a demo.
- [api_reference.md](api_reference.md) — the locked public API.
