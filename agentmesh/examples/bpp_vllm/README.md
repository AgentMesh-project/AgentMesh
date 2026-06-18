# BPP vLLM Example — Supervisor + Parallel Branches

A small, runnable demonstration of **Branch-Parallel Prefill (BPP)** as realized
by AgentMesh's `GPUKVConnector` inside vLLM.

It walks the produce-transfer-consume dataflow BPP makes barrier-free:

1. **Supervisor** prefills a shared prefix prompt and stores its KV.
2. **Branch 1** and **Branch 2** each prefill `prefix + delta_i` and store
   incremental KV. These branches are independent — in a real BPP run they are
   prefilled in parallel against the shared prefix, isolated by slot-based
   block-diagonal Branch-Parallel Attention.
3. The **Final** request loads and *stitches* all KV segments with late RoPE
   realignment, then decodes.

The script then compares the stitched-KV (BPP) output against a plain
full-prefill baseline on the concatenated prompt; they should match for the
prefix-reuse scenarios.

## Requirements

- A GPU host with **vLLM** installed.
- The `GPUKVConnector` registered or loadable (see
  [`../../mechanisms/bpp/vllm/README.md`](../../mechanisms/bpp/vllm/README.md)
  and [`docs/vllm_integration.md`](../../../docs/vllm_integration.md)).

The example imports vLLM lazily. On a host **without** vLLM/GPU it prints a
clear message and exits cleanly (return code 0) instead of crashing.

## Run

```bash
# Default scenario S3 (prefixed branches, shared-prefix reuse).
KV_SCENARIO=S3 python run_supervisor_workers.py --model Qwen/Qwen3-1.7B

# Full BPP with late RoPE realignment of independent branches.
KV_SCENARIO=S2 python run_supervisor_workers.py --model Qwen/Qwen3-1.7B

# BPP composed with CacheBlend selective recompute.
KV_SCENARIO=S7 CACHEBLEND_RECOMPUTE_RATIO=0.15 \
    python run_supervisor_workers.py --model Qwen/Qwen3-1.7B

# BPP composed with EPIC/LegoLink (recompute first k tokens per branch).
KV_SCENARIO=S9 EPIC_K=16 \
    python run_supervisor_workers.py --model Qwen/Qwen3-1.7B
```

### Arguments

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | `Qwen/Qwen3-1.7B` | Any HuggingFace model id vLLM can load. |
| `--max-tokens` | `64` | Decode length for the final request. |
| `--gpu-memory-utilization` | `0.8` | Passed through to vLLM. |

See the scenario matrix (`S1`-`S10`) and the full env-var list in
[`../../mechanisms/bpp/vllm/README.md`](../../mechanisms/bpp/vllm/README.md).

## What to expect

For shared-prefix scenarios the BPP output should match the baseline exactly.
The prompts are intentionally tiny and synthetic (a toy arithmetic prefix plus
two short branch deltas) so the example is fast and ships no data files.
