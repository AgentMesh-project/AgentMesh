# AgentMesh Architecture

> Reference: *AgentMesh: Semantic Communication for Multi-Agent Systems* (System Design)

## Overview

AgentMesh sits between multi-agent frameworks (e.g. AutoGen) and LLM inference backends (e.g. vLLM), introducing a **semantic communication layer** that eliminates redundant data transmission by exploiting the structure of agent interactions.

```
┌────────────────────────────────┐
│      Agent Framework (AutoGen) │
└──────────┬─────────────────────┘
           │ Agent API
┌──────────▼─────────────────────┐
│        AgentMesh Runtime       │
│  ┌──────┐  ┌──────┐  ┌──────┐  │
│  │ SRR  │  │ TSD  │  │ ESF  │  │
│  └──────┘  └──────┘  └──────┘  │
│       Sidecar Mesh (gRPC)      │
└──────────┬─────────────────────┘
           │ Inference API
┌──────────▼─────────────────────┐
│    LLM Backend (vLLM)          │
└────────────────────────────────┘
```

## Core Insight

> *Semantic duplication is equivalent to communication redundancy.*

When multiple agents in a collaborative workflow (e.g. deep-research) invoke an LLM, their prompts overlap heavily — shared system instructions, tool descriptions, and intermediate results. AgentMesh detects and removes this redundancy at the **semantic level**, yielding up to **62% latency reduction** with **<1% average precision loss**.

## Three Mechanisms

### SRR — Semantic Residual Retrieval

Projects a new query onto cached semantic vectors.  When projection intensity exceeds τ (default 0.7), reuses proportional cached items and reformulates only the residual via a small language model (SLM, Qwen2.5-0.5B-Instruct).

### TSD — Topology-aware Semantic Decoupling

Exploits the supervisor-worker topology: since sibling workers are generated from the same supervisor prefix and never attend to each other, their KV caches can be **prefilled in parallel** and later **stitched** with RoPE positional realignment.

### ESF — Elastic Semantic Flow

Adaptively sizes microbatches via Online Sensitivity Estimation (OSE).  Observes per-step producer time $T_p + T_n$, consumer time $T_c$, and slack $\delta$, then estimates sensitivity $\rho_i = S_{pro,i} - S_{con,i-1}$ to compute $\Delta\theta_i = -\delta_i / \rho_i$.

## Module Layout

```
agentmesh/
├── core/
│   ├── config.py       # SRRConfig, TSDConfig, ESFConfig, AgentMeshConfig
│   └── runtime.py      # AgentMeshRuntime orchestrator
├── mechanisms/
│   ├── srr/cache.py    # SRRCache, SemanticEmbedder, HotspotTokenExtractor, QueryReformulator
│   ├── tsd/attention.py# TDAManager, InterWorkerMaskGenerator, RoPEAligner
│   └── esf/controller.py# ESFController, StateObserver, Observation, Sensitivity
├── sidecars/sidecar.py # gRPC sidecar mesh
├── adapters/autogen.py # AutoGen integration
├── proto/agentmesh.proto
└── examples/deep_research/demo.py
```
