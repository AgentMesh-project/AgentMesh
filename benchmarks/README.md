# AgentMesh Benchmark Suite

Reviewer-runnable benchmarks that reproduce AgentMesh's per-mechanism and
end-to-end claims. AgentMesh is a **structure-aware runtime for multi-agent LLM
systems**: every interaction is a **produce–transfer–consume** dataflow
(producer = a tool *or* an upstream agent's LLM decode; consumer = always an LLM
prefill). A structure-blind runtime wastes time on **REDUNDANCY** (re-producing
held content) and **BARRIER** (a consumer idling until its producer finishes).
AgentMesh removes both with three mechanisms:

| Mechanism | Full name | Dataflow stage it improves |
|-----------|-----------|----------------------------|
| **DTR** | Delta Tool Retrieval | shrinks **PRODUCE** (tool→LLM): serve cached base, fetch only the semantic delta, async-refresh |
| **BPP** | Branch-Parallel Prefill | shrinks **CONSUME** (gather): parallel-prefill independent worker branches vs. the shared prefix, mask cross-branch attention (BPA), stitch KV with late RoPE realignment |
| **DES** | Dynamic-Equilibrium Streaming | **OVERLAPS** produce/transfer/consume: an online controller drives slack δ→0 via OSE |

Every script has a **fast synthetic default** that runs in seconds with **no
GPU, no cluster, and no LLM**. Heavy dependencies (`torch`, `transformers`,
`matplotlib`, `sentence-transformers`) are optional and guarded — scripts fall
back to numeric / CSV output when a plotting or model backend is absent.

## Quick start

```bash
# from the repository root
pip install -e .                 # installs agentmesh (the core runtime)
export PYTHONPATH=$PWD           # or `pip install -e .`

python -m benchmarks.dtr.benchmark_dtr
python -m benchmarks.dtr.threshold_sweep
python -m benchmarks.bpp.attention_orthogonality
python -m benchmarks.bpp.bpp_speedup
python -m benchmarks.des.benchmark_des
python -m benchmarks.end_to_end.run_e2e
```

Each script accepts `--help`. All are self-contained by default.

## Script → paper claim map

| Script | Reproduces (paper figure / number) | Headline metric(s) | Runtime |
|--------|-----------------------------------|--------------------|---------|
| `dtr/benchmark_dtr.py` | DTR cuts tool/scrape latency **71.5%**, lowers backend fetch volume **57%**; precision matches/exceeds NC / EM / FFR / Cortex | exact-match score, backend-load reduction, foreground-latency reduction | self-contained (<10 s) |
| `dtr/threshold_sweep.py` | DTR τ sensitivity, **sweep 0.75–0.95**, default τ=0.8 | exact-match vs. τ, load reduction vs. τ | self-contained (<10 s) |
| `bpp/attention_orthogonality.py` | Motivation for BPA masking: cross-branch attention is negligible | `last_branch_cross_fraction`, `self_over_cross_ratio`, `prefix_fraction` | self-contained `--synthetic` (<5 s); real capture needs GPU + `transformers` |
| `bpp/bpp_speedup.py` | BPP attention complexity **O(N²)→O(N)**; **8–61%** CONSUME-stage time reduction | full vs. BPA cost, speedup vs. branch count N | self-contained (<5 s); `--use-manager` cross-check needs `torch` |
| `des/benchmark_des.py` | DES > No-Stream (NS) and Static-Stream (SS); replayed-trace **CoV 0.13–0.75**; θ converges to equilibrium (slack→0) | throughput vs. NS/SS, steady-state slack, θ trajectory | self-contained (<10 s) |
| `end_to_end/run_e2e.py` | Deep-research latency reduced **40.6–62.4%**, overall **up to 2.1×**; cumulative ablation DTR **15–78%**, BPP **19–61%**, DES **6–17%** | per-stage + total % reduction, speedup | self-contained (<5 s); `--trace` for recorded timings |

## What is synthetic vs. what you supply for full reproduction

The fast defaults synthesize tiny, plausible inputs so the **smoke path** runs
anywhere. The reduction *bands* are reproduced from synthetic cost models; the
**absolute** paper numbers (e.g. exactly 71.5%) come from the full hardware
testbed and recorded traces. To move toward full reproduction:

- **DTR**: `--trace <tool_log.jsonl>` replays a real tool log (one JSON object
  per line with `query`, `depth`, `items`). `--real-embedder` uses the
  `agentmesh` `SemanticEmbedder` (install `sentence-transformers`); the default
  uses a deterministic bag-of-words embedder so cosine similarity is meaningful
  on CPU.
- **BPP**: `bpp_speedup.py --use-manager` cross-checks the complexity formulas
  against the real `BranchParallelManager` (needs `torch`).
  `attention_orthogonality.py --model Qwen/Qwen3-4B` is the documented GPU
  capture path (needs `torch` + `transformers` + a GPU); the self-contained
  study uses `--synthetic`.
- **DES**: `--trace <rate_trace.csv>` replays recorded producer/consumer rates
  (columns `step, producer_tokens_per_s, consumer_tokens_per_s,
  start_latency_ms, network_ms_per_chunk`).
- **End-to-end**: `--perf-fits <fits.json>` / `--trace <timings.json>` (the JSON
  schema is documented in `end_to_end/run_e2e.py`) plug in profiled per-stage
  timings; the default uses a built-in synthetic cost model.

## Input data

Every benchmark **generates its inputs synthetically** by default, so no data
files are shipped or required — the smoke path runs anywhere. To run against
your own recorded data, pass the optional flags (each script's header documents
the expected format):

- `benchmark_dtr.py --trace <tool_log.jsonl>` — a tool log: one JSON object per
  line with `query`, `depth`, `items`.
- `benchmark_des.py --trace <rate_trace.csv>` — a producer/consumer rate trace
  (columns `step, producer_tokens_per_s, consumer_tokens_per_s,
  start_latency_ms, network_ms_per_chunk`).
- `run_e2e.py --perf-fits <fits.json>` / `--trace <timings.json>` — a per-stage
  cost model (schema in `end_to_end/run_e2e.py`).
- `PerformancePredictor(<fits.json>)` (`agentmesh.mechanisms.des`) — fitted
  latency curves; with no path it uses built-in illustrative fits.

Any files you drop under `benchmarks/data/` are git-tracked (see the
`!benchmarks/data/*` rules in `.gitignore`), but none are required to run.

## Installing optional extras

```bash
pip install matplotlib              # PNG plots / heatmaps (otherwise numeric/CSV)
pip install sentence-transformers   # DTR --real-embedder
pip install torch transformers      # BPP --use-manager / real attention capture
```
