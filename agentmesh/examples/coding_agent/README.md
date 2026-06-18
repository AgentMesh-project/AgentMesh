# Coding Agent Example

A supervisor–worker **coding agent** built on the AgentMesh structure-aware
runtime. It mirrors the [`deep_research`](../deep_research) example, but for
code-repair tasks: the supervisor decomposes a repository issue into *N
independent* file/section fixes that are dispatched to parallel worker
branches, each worker uses coding tools to gather context, and all dataflows
are streamed under adaptive chunk control.

Every interaction is a **produce–transfer–consume** dataflow (producer = a tool
or an upstream agent's LLM decode; consumer = always an LLM prefill). AgentMesh
makes each dataflow redundancy- and barrier-free with three mechanisms:

| Mechanism | Stage it optimizes | Role in this example |
|-----------|--------------------|----------------------|
| **DTR** — Delta Tool Retrieval | shrinks **PRODUCE** (tool → LLM) | `read_file` / `search_code` / `build` results for related files share a cached base; only the semantic delta per file is fetched, then async-refreshed. |
| **BPP** — Branch-Parallel Prefill | shrinks **CONSUME** (gather) | The *N* independent worker branches are prefilled in parallel against the shared supervisor prefix (task brief + repo summary), with cross-branch attention masked (BPA). Attention cost in branch count goes O(N²) → O(N). |
| **DES** — Dynamic-Equilibrium Streaming | **OVERLAPS** produce/transfer/consume | Tool and LLM outputs stream under an online controller that drives slack δ → 0 (rate-matched chunk size θ). |

## Running

Like the [`deep_research`](../deep_research) demo, this runs against a local
**OpenAI-compatible endpoint** (the `--llm-backend` flag is that endpoint's URL,
e.g. a vLLM server). The runtime degrades gracefully if the endpoint is
unreachable (LLM calls return error strings rather than crashing), but a served
model is needed for a meaningful run.

```bash
# serve a model first, e.g.:  pip install -e ".[vllm]" && vllm serve Qwen/Qwen3-4B --port 8000
python -m agentmesh.examples.coding_agent \
    --issue "fix the off-by-one in the tokenizer" \
    --files src/tokenizer.py src/decoder.py src/utils.py \
    --num-workers 3 \
    --llm-backend http://localhost:8000/v1 \
    --model Qwen/Qwen3-4B
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--issue` | demo issue | Bug/feature description to resolve. |
| `--files` | *(synthetic)* | Independent target files, one per worker branch. |
| `--num-workers` | `3` | Number of code-fixing worker branches (BPP). |
| `--llm-backend` | `http://localhost:8000/v1` | OpenAI-compatible endpoint URL. |
| `--model` | `Qwen/Qwen3-4B` | LLM model name. |
| `--enable-dtr` / `--enable-bpp` / `--enable-des` | on | Toggle each mechanism. |
| `--log-level` | `INFO` | Logging verbosity. |

## Programmatic use

```python
import asyncio
from agentmesh.examples.coding_agent import run_demo

result = asyncio.run(run_demo(
    issue="fix the race in process()",
    target_files=["src/handler.py", "src/pool.py"],
    num_workers=2,
    llm_endpoint="http://localhost:8000/v1",
))
print(result["changeset"])
```

## Workflow phases

1. **Decompose** — the supervisor splits the issue into independent per-file
   fix tasks (one BPP branch each).
2. **Fix in parallel** — each worker reads / searches / builds (DTR produce
   steps) and proposes a scoped patch; the branches prefill in parallel (BPP).
3. **Assemble** — the supervisor reviews the patches and assembles a single
   changeset.

Statistics (DTR cache hit rate, per-sidecar counters, total duration) are
printed at the end and returned in `result["stats"]`.
