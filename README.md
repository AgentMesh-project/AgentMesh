# AgentMesh: Semantic Communication for Multi-Agent Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **AgentMesh** is a semantic communication framework that bridges the gap between agentic orchestration workflows and tool/LLM execution. By treating **semantic duplication as communication redundancy**, AgentMesh reduces end-to-end latency by up to 62% with negligible precision loss (< 1% on average).

## Key Insights

Modern multi-agent systems (MAS) suffer from a fundamental **semantic-transmission mismatch**: while MAS interactions are inherently semantic (natural language, tool calls, reasoning chains), the underlying transport layer remains byte-oriented and semantically agnostic.

AgentMesh is designed as a **semantic communication layer** that transforms the rigid, sequential agentic lifecycle into an active, semantic-aware pipeline via three integrated primitives:

1. **Fuzzy Redundancy** -- Tool queries exhibit inter- and intra-request semantic overlap, yet byte-level caching cannot capture fuzzy similarities.
2. **Topology-Dependent Serialization** -- In supervisor-worker topologies, the supervisor must wait for all workers and perform exhaustive cross-attention, unaware of workers' semantic independence.
3. **Dynamic Flows** -- Semantic-aware optimizations introduce compound payload and arrival heterogeneity that static pipelines cannot handle.

## Architecture Overview

```
                         AgentMesh Semantic Communication Layer
  ┌──────────────────────────────────────────────────────────────────────┐
  │   ┌──────────────────────────────────────────────────────────────┐   │
  │   │              Elastic Semantic Flow (ESF)                     │   │
  │   │        Adaptive streaming fabric for all data flows          │   │
  │   └──────────────────────────────────────────────────────────────┘   │
  │                                                                      │
  │   ┌────────────────────────┐  ┌──────────────────────────────────┐   │
  │   │  Semantic Residual     │  │  Topology-aware Semantic         │   │
  │   │  Retrieval (SRR)       │  │  Decoupling (TSD)                │   │
  │   │                        │  │                                  │   │
  │   │  Agent ←→ Tool         │  │  Worker Agents → Supervisor      │   │
  │   │  semantic delta        │  │  decoupled parallel prefill      │   │
  │   │  encoding for tools    │  │  with TDA                        │   │
  │   └────────────────────────┘  └──────────────────────────────────┘   │
  └──────────────────────────────────────────────────────────────────────┘
          │                │                │               │
    ┌─────┴─────┐   ┌──────┴──────┐  ┌──────┴──────┐  ┌─────┴─────┐
    │  Agent    │   │   Tool      │  │  Worker LLM │  │ Supervisor│
    │ (AutoGen) │   │  (Search,   │  │  (vLLM)     │  │    LLM    │
    │           │   │   Scrape)   │  │             │  │  (vLLM)   │
    └───────────┘   └─────────────┘  └─────────────┘  └───────────┘
```

## Core Mechanisms

### 1. Semantic Residual Retrieval (SRR)

**Goal**: Minimize redundant tool execution and data transfer by decomposing a tool query into a *semantic base* (reusable from cache) and a *semantic residual* (novel intent requiring incremental retrieval).

**Implementation**: agentmesh/mechanisms/srr/

### 2. Topology-aware Semantic Decoupling (TSD)

**Goal**: Exploit the semantic orthogonality of parallel workers in supervisor-worker topologies to transform monolithic context gathering into a decoupled stream, enabling parallel prefill.

**Core Innovation — Topologically Decoupled Attention (TDA)**: Inter-worker masking, timely parallel prefill, positional realignment.

**Compatibility**: TSD is orthogonal to KV blending algorithms such as CacheBlend and EPIC — they can be optionally integrated to further decrease computation overhead.

**Implementation**: agentmesh/mechanisms/tsd/

### 3. Elastic Semantic Flow (ESF)

**Goal**: Serve as a unified streaming fabric that adaptively calibrates microbatch granularity to maximize communication-computation overlap, mastering both semantic and systemic stochasticity.

**ESF supports two pipelines**: (1) tool-to-agent (SRR items streaming) and (2) agent-to-agent (worker decode → supervisor prefill via TSD).

**Implementation**: agentmesh/mechanisms/esf/

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AgentMesh-project/AgentMesh.git
cd AgentMesh

# Install dependencies
pip install -e .

# Install with agent/LLM-related dependencies
pip install -e ".[autogen,vllm]"
```

### Basic Usage

```python
from agentmesh import AgentMeshRuntime, AgentMeshConfig
from agentmesh.core.config import SRRConfig, TSDConfig, ESFConfig
from agentmesh.adapters import AutoGenAdapter

# Initialize the runtime with all three mechanisms
config = AgentMeshConfig(
    srr=SRRConfig(confidence_threshold=0.7, retrieval_depth=10),
    tsd=TSDConfig(max_parallel_workers=3),
    esf=ESFConfig(initial_microbatch_size=512, theta_min=64, theta_max=4096),
)
runtime = AgentMeshRuntime(config=config)

# Wrap your AutoGen agents
adapter = AutoGenAdapter(runtime=runtime)
agents = adapter.wrap_agents(your_agents)

# Process requests through SRR-enabled pipeline
from agentmesh.core.runtime import AgentRequest
response = await runtime.process(AgentRequest(
    request_id="req-1", agent_id="researcher_1",
    prompt="Your research query",
))
```

### Running the Demo

```bash
# Deep Research Demo (3 researcher workers + 1 supervisor)
python -m agentmesh.examples.deep_research --topic "Networked systems for machine learning (ML)" --num-workers 3

# With custom LLM backend
python -m agentmesh.examples.deep_research \
    --llm-backend http://localhost:8000/v1 \
    --model Qwen/Qwen3-32B \
    --enable-srr \
    --enable-tsd \
    --enable-esf
```

## Evaluation

### Reproducing Paper Results

```bash
# Individual mechanism benchmarks
python benchmarks/benchmark_srr.py          # SRR cache hit, latency, precision
python benchmarks/benchmark_tsd.py          # TSD precision (KVR/DTSD/TSD/FP)
python benchmarks/benchmark_esf.py          # ESF throughput and dynamics
```

## Repository Structure

```
AgentMesh/
├── agentmesh/                # Main package
│   ├── core/                 # Core runtime and utilities
│   │   ├── runtime.py        # AgentMesh runtime orchestrating SRR/TSD/ESF
│   │   └── config.py         # Configuration management
│   ├── mechanisms/           # Three core mechanisms
│   │   ├── srr/              # Semantic Residual Retrieval
│   │   │   └── cache.py      # SRR cache with projection + residual + SLM reformulation
│   │   ├── tsd/              # Topology-aware Semantic Decoupling
│   │   │   └── attention.py  # TDA: inter-worker masking, parallel prefill, RoPE realignment
│   │   └── esf/              # Elastic Semantic Flow
│   │       └── controller.py # ESF controller with OSE + sensitivity-aware adaptation
│   ├── sidecars/             # Communication interceptors
│   │   └── sidecar.py        # Agent/LLM/Tool sidecar mesh
│   ├── proto/                # gRPC protocol definitions
│   │   └── agentmesh.proto   # Service definitions for agent/LLM/tool communication
│   ├── adapters/             # Framework adapters
│   │   └── autogen.py        # AutoGen adapter
│   └── examples/             # Example applications
│       └── deep_research/    # Supervisor-worker Deep Research demo
├── benchmarks/               # Evaluation scripts
│   ├── benchmark_srr.py      # SRR cache hit rate and latency
│   ├── benchmark_tsd.py      # TSD precision and throughput
│   └── benchmark_esf.py      # ESF dynamics and adaptation
├── tests/                    # Unit tests
└── docs/                     # Documentation
```

## Configuration

### Environment Variables

```bash
export AGENTMESH_LLM_ENDPOINT="http://localhost:8000/v1"
export AGENTMESH_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export AGENTMESH_SLM_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
export AGENTMESH_LOG_LEVEL="INFO"
```

### Configuration File

```yaml
# config.yaml
srr:
  enabled: true
  confidence_threshold: 0.7      # τ — semantic confidence gate
  retrieval_depth: 10             # n — number of items per tool query
  hotspot_k: 3                    # top-k residual-characterized tokens
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  slm_model: "Qwen/Qwen2.5-0.5B-Instruct"  # SLM for query reformulation
  cache_size: 1000

tsd:
  enabled: true
  max_parallel_workers: 3
  
esf:
  enabled: true
  initial_microbatch_size: 512    # θ₁ — initial microbatch size (tokens)
  theta_min: 64                   # θ_min — minimum microbatch size
  theta_max: 4096                 # θ_max — maximum microbatch size
```

## API Reference

### SRRCache

```python
class SRRCache:
    def __init__(self, confidence_threshold: float = 0.7, retrieval_depth: int = 10, ...):
        """Initialize semantic cache with confidence gating."""
    
    def lookup(self, query: str, tool_fn: Optional[Callable] = None) -> CacheResult:
        """
        Projection-based semantic cache lookup.
        Returns reuse_count, residual items, and reformulated query.
        """
    
    def store(self, query: str, items: List[str], embedding: Optional[np.ndarray] = None):
        """Store query-items pair in cache (with async full-execution update)."""
    
    def get_reused_items(self, result: CacheResult) -> List[str]:
        """Return top ⌊s_max·n⌋ items from the cached entry."""
```

### TDAManager

```python
class TDAManager:
    def __init__(self, num_layers: int = 32, num_heads: int = 32, head_dim: int = 128,
                 max_parallel_workers: int = 3, rope_base: float = 10000.0, device: str = "cuda"):
        """Initialize TDA manager for supervisor-worker topology."""
    
    def set_supervisor_context(self, ctx: SupervisorContext):
        """Set the supervisor prefix context (shared across all workers)."""
    
    def prefill_worker(self, worker_output: WorkerOutput, prefill_fn: Optional[callable] = None):
        """Timely parallel prefill with inter-worker masking."""
    
    def stitch_and_realign(self) -> TDAResult:
        """Merge per-worker KV caches + RoPE positional realignment."""
```

### ESFController

```python
class ESFController:
    def __init__(self, initial_theta: int = 512, min_theta: int = 64, max_theta: int = 4096,
                 damping_factor: float = 1.0, window_size: int = 5):
        """Initialize ESF controller with OSE."""
    
    def compute_next_theta(self, producer_time: float, consumer_time: float,
                           slack_time: float, time_step: int) -> int:
        """Sensitivity-aware microbatch size adaptation."""
    
    def get_state(self) -> Dict:
        """Get current controller state including sensitivity estimates."""
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific mechanism tests
pytest tests/test_srr.py -v
pytest tests/test_tsd.py -v
pytest tests/test_esf.py -v

# Run with coverage
pytest --cov=agentmesh tests/
```

## Implementation Stack

- **Agent Framework**: [AutoGen](https://github.com/microsoft/autogen)
- **LLM Serving**: [vLLM](https://github.com/vllm-project/vllm) (with custom KV connector for TDA)
- **Embedding**: [sentence-transformers](https://www.sbert.net) (all-MiniLM-L6-v2)
- **Similarity Index**: [FAISS](https://github.com/facebookresearch/faiss) (IVF-Flat)
- **SLM**: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
