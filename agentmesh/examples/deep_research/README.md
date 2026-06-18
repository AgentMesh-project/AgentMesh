# Deep Research Example

The canonical **supervisor–worker (scatter–gather)** workload for AgentMesh, and
the headline application in the paper. A supervisor agent scopes a research
topic, decomposes it into *N independent* sub-topics dispatched to parallel
researcher branches, each researcher gathers evidence through tools, and the
supervisor synthesizes the branch drafts into a final report. The
[`coding_agent`](../coding_agent) example mirrors this structure for code-repair
tasks.

Every interaction is a **produce–transfer–consume** dataflow (producer = a tool
or an upstream agent's LLM decode; consumer = always an LLM prefill). AgentMesh
makes each dataflow redundancy- and barrier-free with three mechanisms:

| Mechanism | Stage it optimizes | Role in this example |
|-----------|--------------------|----------------------|
| **DTR** — Delta Tool Retrieval | shrinks **PRODUCE** (tool → LLM) | Researchers' `web_search` queries on related sub-topics share a cached base; only the semantic delta is fetched, then async-refreshed — the dominant cost in deep research, where WAN fetch of long HTML documents bottlenecks. |
| **BPP** — Branch-Parallel Prefill | shrinks **CONSUME** (gather) | The *N* researcher drafts are prefilled in parallel against the shared supervisor prefix (topic + role brief), with cross-branch attention masked (BPA). Attention cost in branch count goes O(N²) → O(N), and the supervisor no longer waits on the straggler branch. |
| **DES** — Dynamic-Equilibrium Streaming | **OVERLAPS** produce/transfer/consume | Tool and LLM outputs stream under an online controller that drives slack δ → 0 (rate-matched chunk size θ), hiding transfer/prefill behind production. |

## Running

This demo drives a real LLM through an **OpenAI-compatible endpoint** (the
`--llm-backend` flag is that endpoint's URL — e.g. a local vLLM server). The
runtime degrades gracefully if the endpoint is unreachable (LLM calls return
error strings rather than crashing), but a served model is needed for a
meaningful run.

```bash
# Against a local vLLM (or any OpenAI-compatible) server
python -m agentmesh.examples.deep_research \
    --topic "The impact of large language models on software engineering" \
    --num-workers 3 \
    --llm-backend http://localhost:8000/v1 \
    --model Qwen/Qwen3-4B
```

Serve a model first, for example:

```bash
pip install -e ".[vllm]"
vllm serve Qwen/Qwen3-4B --port 8000
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--topic` | *(LLMs on software engineering)* | Research topic to investigate. |
| `--num-workers` | `3` | Number of researcher branches (one BPP branch each). |
| `--llm-backend` | `http://localhost:8000/v1` | OpenAI-compatible endpoint URL. |
| `--model` | `Qwen/Qwen3-4B` | LLM model name. |
| `--enable-dtr` / `--enable-bpp` / `--enable-des` | on | Toggle each mechanism (use for ablations). |
| `--log-level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR`. |

## Programmatic use

```python
import asyncio
from agentmesh.examples.deep_research import run_demo

result = asyncio.run(run_demo(
    topic="Post-quantum cryptography standards",
    num_workers=3,
    llm_endpoint="http://localhost:8000/v1",
    enable_dtr=True, enable_bpp=True, enable_des=True,
))
print(result["report"])
print(result["stats"])   # DTR cache hit rate, per-sidecar counters, total duration
```

The package also exposes the building blocks for custom workflows:
`DeepResearchWorkflow`, `SupervisorAgent`, `ResearcherAgent`, and
`register_research_tools`.

## Workflow phases

Mirrors the supervisor–worker lifecycle described in the paper (Appendix):

1. **Scope & dispatch (scatter)** — the supervisor scopes the topic and
   decomposes it into *N* independent sub-tasks, instantiating one researcher
   branch per sub-task.
2. **Parallel worker execution** — each researcher runs a tool loop
   (`web_search` / `analyze` / `summarize`) to draft its section. Tool results
   flow through DTR (delta retrieval); drafts stream under DES. Without
   AgentMesh this phase is gated by the slowest (straggler) branch.
3. **Synthesis (gather)** — the supervisor prefills the *N* drafts in parallel
   via BPP and synthesizes them into a single report. Cross-branch attention is
   masked only during prefill; the final decode attends the full stitched KV.

Statistics (DTR cache hit rate, per-sidecar counters, total duration) are
printed at the end and returned in `result["stats"]`.
