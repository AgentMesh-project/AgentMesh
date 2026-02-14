# Getting Started

## Prerequisites

- Python ≥ 3.10
- [AutoGen](https://github.com/microsoft/autogen) ≥ 0.4.2
- [vLLM](https://github.com/vllm-project/vllm) ≥ 0.13.0 (for LLM inference)
- CUDA-capable GPU (recommended for TSD parallel prefill)

## Installation

```bash
# From source
git clone https://github.com/AgentMesh-project/AgentMesh.git
cd AgentMesh
pip install -e ".[dev]"
```

## Quick Start

### 1. Configure

```python
from agentmesh.core.config import AgentMeshConfig, SRRConfig

config = AgentMeshConfig(
    srr=SRRConfig(confidence_threshold=0.7),  # Paper default τ = 0.7
)
```

Or via environment variables:

```bash
export AGENTMESH_CONFIDENCE_THRESHOLD=0.7
export AGENTMESH_SLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
```

### 2. Initialize Runtime

```python
from agentmesh.core.runtime import AgentMeshRuntime

runtime = AgentMeshRuntime(config=config)
```

### 3. Process Queries (SRR)

```python
from agentmesh.core.runtime import AgentRequest

# First call — cold retrieval
response = await runtime.process(AgentRequest(
    request_id="req-1",
    agent_id="researcher_1",
    prompt="What are the key challenges in distributed systems?",
))

# Second call — SRR may activate if semantically similar
response = await runtime.process(AgentRequest(
    request_id="req-2",
    agent_id="researcher_2",
    prompt="What challenges exist in distributed computing?",
))

print(f"SRR activated: {response.srr_activated}")
print(f"Projection intensity: {response.projection_intensity:.2f}")
```

### 4. Parallel Prefill (TSD)

```python
from agentmesh.core.runtime import AgentRequest

# Batch process with TDA parallel prefill
responses = await runtime.batch_process(
    requests=[
        AgentRequest(request_id="r1", agent_id="worker_1", prompt="subtask 1"),
        AgentRequest(request_id="r2", agent_id="worker_2", prompt="subtask 2"),
        AgentRequest(request_id="r3", agent_id="worker_3", prompt="subtask 3"),
    ]
)
```

### 5. Streaming with ESF

```python
async for chunk in runtime.stream(AgentRequest(
    request_id="req-s1",
    agent_id="writer_1",
    prompt="Generate a detailed analysis...",
)):
    print(chunk.content, end="", flush=True)
    # ESF automatically adapts microbatch_size
```

## AutoGen Integration

```python
from agentmesh.adapters.autogen import AutoGenAdapter

# Wraps AutoGen v0.4.2+ runtime with AgentMesh
adapter = AutoGenAdapter(runtime=runtime)
# Use adapter in place of standard AutoGen runtime
```

## Running the Deep Research Demo

```bash
python -m agentmesh.examples.deep_research.demo
```

## Running Tests

```bash
pytest tests/ -v
```

## Next Steps

- [Architecture Overview](architecture.md)
- [SRR Details](srr.md)
- [TSD Details](tsd.md)
- [ESF Details](esf.md)
- [API Reference](api_reference.md)
