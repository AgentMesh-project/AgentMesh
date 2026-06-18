#!/usr/bin/env python3
"""
DTR (Delta Tool Retrieval) benchmark -- the PRODUCE step of a tool->LLM dataflow.

Reproduces the DTR headline claims:
  - cuts tool/scrape latency by ~71.5%  (foreground critical path)
  - lowers backend fetch volume by ~57% (items fetched vs. a no-cache run)
  - exact-match precision matches/exceeds the baselines

Paper mapping
-------------
  Section: "DTR cuts tool/scrape latency 71.5%; lowers backend fetch volume 57%"
  Ablation: DTR contributes 15-78% of the PRODUCE-stage reduction.
  Baselines: NC (No-Cache), EM (Exact-Match), FFR (Fixed-Fraction-Reuse), Cortex.
  Default confidence threshold tau = 0.8 (sweep 0.75-0.95 in threshold_sweep.py).
  Metric: exact-match score = fraction of the no-cache reference items recovered.

Method (mirrors srr_simulation_package/agentmesh_srr/trace_scale_eval.py)
------------------------------------------------------------------------
Each method replays the same query stream. We compare against a "direct"
(no-cache) reference that always fetches the full item set per query:
  - NC      : always cold-fetch n items (the reference); load = 100%.
  - EM       : reuse items ONLY on a byte-exact repeat of a prior query.
  - FFR      : on any near-repeat, reuse a fixed fraction f of items.
  - Cortex   : similarity judge -- reuse ALL items when sim >= judge_threshold,
               else cold-fetch (all-or-nothing cache).
  - DTR (us) : the agentmesh DTRCache -- reuse floor(s_max * n) base items and
               fetch only ceil((1 - s_max) * n) delta items when s_max >= tau,
               with asynchronous exact refresh.

Latency model: foreground critical path counts only the items actually fetched
on the request path (delta items for DTR; all items for cold fetches); the
async refresh runs off the critical path. Backend load counts every item the
backend must produce on the foreground path.

Fast default: fully synthetic, no GPU / LLM, < 10s. The agentmesh embedder
falls back to a deterministic hash embedding when sentence-transformers is
absent, so results are reproducible without heavy deps.

Usage:
    python -m benchmarks.dtr.benchmark_dtr
    python -m benchmarks.dtr.benchmark_dtr --num-queries 240 --overlap 0.7
    python -m benchmarks.dtr.benchmark_dtr --trace your_tool_log.jsonl   # optional; default is synthetic
    python -m benchmarks.dtr.benchmark_dtr --csv out.csv
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Allow direct execution (``python benchmarks/dtr/benchmark_dtr.py``) by putting the
# repository root on sys.path, in addition to ``python -m benchmarks.dtr.benchmark_dtr``.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agentmesh.mechanisms.dtr import DTRCache


# ---------------------------------------------------------------------------
# Lightweight deterministic semantic embedder
# ---------------------------------------------------------------------------
# The agentmesh SemanticEmbedder falls back to a *hash* embedding when
# sentence-transformers is absent, which destroys semantic similarity
# (unrelated queries get arbitrary cosine). For a credible CPU-only benchmark
# we substitute a deterministic bag-of-words embedding: cosine similarity then
# tracks token overlap, so near-repeat queries score high and unrelated
# queries score low -- exactly the structure DTR exploits. Install
# `sentence-transformers` and pass --real-embedder for the production encoder.
_VOCAB: Dict[str, int] = {}


def _token_index(tok: str) -> int:
    idx = _VOCAB.get(tok)
    if idx is None:
        idx = len(_VOCAB)
        _VOCAB[tok] = idx
    return idx


class BagOfWordsEmbedder:
    """Deterministic, normalized bag-of-words embedder (no heavy deps)."""

    dim = 4096

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in text.lower().split():
            vec[_token_index(tok) % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_batch(self, texts):
        return np.array([self.embed(t) for t in texts])

    def embed_tokens(self, tokens):
        return self.embed_batch(tokens)


# ---------------------------------------------------------------------------
# Synthetic tool workload with controllable semantic overlap
# ---------------------------------------------------------------------------
TOPICS = [
    ("japan elderly population", "aging demographics housing mobility healthcare consumption pension dependency fertility migration"),
    ("ai accelerator power", "datacenter energy gpu tdp cooling efficiency pue grid perf-per-watt liquid"),
    ("gpu cluster networking", "inference serving nvlink rdma collective topology pcie bandwidth kv offload"),
    ("renewable energy storage", "grid battery solar wind demand-response capacity flexibility curtailment policy"),
    ("quantum error correction", "surface-code logical qubit fidelity decoder threshold lattice syndrome stabilizer noise"),
    ("crispr gene therapy", "delivery vector off-target editing trial efficacy immunogenicity dosing regulatory safety"),
]


@dataclass
class ToolQuery:
    """One tool call in the replayed stream."""
    step: int
    query: str
    depth: int
    # The "true" item set the backend would return for this exact query.
    items: List[str]
    direct_latency_ms: float


def _topic_items(topic_key: str, modifiers: List[str], depth: int) -> List[str]:
    """Deterministic item set: a topic core plus query-specific modifier items."""
    core = [f"{topic_key} :: core-doc-{i}" for i in range(max(1, depth - len(modifiers)))]
    extra = [f"{topic_key} :: {m}-doc" for m in modifiers]
    items = (core + extra)[:depth]
    while len(items) < depth:
        items.append(f"{topic_key} :: filler-{len(items)}")
    return items


def synthesize_stream(
    num_queries: int,
    overlap: float,
    depth: int,
    seed: int,
) -> List[ToolQuery]:
    """
    Build a query stream with controllable semantic overlap.

    `overlap` in [0, 1] is the probability that a query is a near-repeat of a
    recent query on the same topic (sharing most modifier tokens) rather than a
    fresh topic+modifier draw. Higher overlap => more DTR-cacheable structure.
    """
    rng = random.Random(seed)
    stream: List[ToolQuery] = []
    recent: List[Tuple[str, List[str]]] = []  # (topic_key, modifiers)

    for step in range(num_queries):
        if recent and rng.random() < overlap:
            topic_key, base_mods = recent[rng.randrange(len(recent))]
            mods = list(base_mods)
            # Perturb a small part of the intent: the "semantic delta".
            if mods and rng.random() < 0.8:
                mods[rng.randrange(len(mods))] = rng.choice(
                    ["trend", "forecast", "economics", "2025", "limits", "outlook"]
                )
        else:
            topic_key, pool = TOPICS[rng.randrange(len(TOPICS))]
            pool_tokens = pool.split()
            mods = rng.sample(pool_tokens, k=min(4, len(pool_tokens)))

        query = f"{topic_key} {' '.join(mods)}"
        items = _topic_items(topic_key, mods, depth)
        base = 3900.0 + 300.0 * rng.random()
        stream.append(
            ToolQuery(
                step=step,
                query=query,
                depth=depth,
                items=items,
                direct_latency_ms=base,
            )
        )
        recent.append((topic_key, mods))
        if len(recent) > 12:
            recent.pop(0)
    return stream


def load_trace(path: Path, depth: int) -> List[ToolQuery]:
    """Replay a recorded tool log (JSONL). Each line: {query, depth, items, ...}."""
    stream: List[ToolQuery] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw_items = row.get("items", [])
            items = [
                it if isinstance(it, str) else (it.get("url") or it.get("title") or json.dumps(it))
                for it in raw_items
            ]
            d = int(row.get("depth", depth) or depth)
            if not items:
                items = [f"{row['query']} :: item-{j}" for j in range(d)]
            stream.append(
                ToolQuery(
                    step=int(row.get("step", i)),
                    query=row["query"],
                    depth=d,
                    items=items,
                    direct_latency_ms=float(row.get("direct_latency_ms", 4000.0)),
                )
            )
    return stream


# ---------------------------------------------------------------------------
# Latency model
# ---------------------------------------------------------------------------
def fetch_latency_ms(query: ToolQuery, fetched_items: int) -> float:
    """
    Foreground latency for fetching `fetched_items` of `query.depth` items.

    A fixed start cost plus a per-item cost; reusing cached base items removes
    their per-item cost from the critical path.
    """
    if query.depth <= 0:
        return 0.0
    start = query.direct_latency_ms * 0.25
    per_item = (query.direct_latency_ms * 0.75) / query.depth
    if fetched_items <= 0:
        # Pure cache hit: only a tiny lookup cost.
        return start * 0.10
    return start + per_item * fetched_items


# ---------------------------------------------------------------------------
# Method result accumulator
# ---------------------------------------------------------------------------
@dataclass
class MethodMetrics:
    name: str
    query_count: int = 0
    foreground_items_fetched: int = 0      # backend load on the critical path
    reference_items: int = 0               # total items a no-cache run produces
    matched_items: int = 0                 # exact-match score numerator
    returned_items: int = 0                # precision denominator
    foreground_latency_ms: float = 0.0
    direct_latency_ms: float = 0.0
    cache_hits: int = 0

    def finalize(self, num_queries: int) -> Dict[str, float]:
        em_score = self.matched_items / self.reference_items if self.reference_items else 0.0
        precision = self.matched_items / self.returned_items if self.returned_items else 0.0
        load_reduction = (
            1.0 - self.foreground_items_fetched / self.reference_items
            if self.reference_items else 0.0
        )
        latency_reduction = (
            1.0 - self.foreground_latency_ms / self.direct_latency_ms
            if self.direct_latency_ms else 0.0
        )
        speedup = (
            self.direct_latency_ms / self.foreground_latency_ms
            if self.foreground_latency_ms else float("inf")
        )
        return {
            "method": self.name,
            "queries": num_queries,
            "exact_match_score": round(em_score, 4),
            "precision": round(precision, 4),
            "hit_rate": round(self.cache_hits / num_queries, 4) if num_queries else 0.0,
            "fg_items_fetched": self.foreground_items_fetched,
            "backend_load_reduction": round(load_reduction, 4),
            "fg_latency_ms": round(self.foreground_latency_ms, 1),
            "latency_reduction_vs_nc": round(latency_reduction, 4),
            "speedup_vs_nc": round(speedup, 3),
        }


# ---------------------------------------------------------------------------
# Baselines + DTR
# ---------------------------------------------------------------------------
def _served_set(reused: List[str], fetched: List[str]) -> List[str]:
    """Items returned to the agent = reused base + freshly fetched delta."""
    return list(reused) + list(fetched)


def run_no_cache(stream: List[ToolQuery]) -> MethodMetrics:
    """NC: always cold-fetch the full item set. This is the reference."""
    m = MethodMetrics("NC (No-Cache)")
    for q in stream:
        m.query_count += 1
        m.reference_items += q.depth
        m.returned_items += len(q.items)
        m.matched_items += len(q.items)          # served == reference
        m.foreground_items_fetched += q.depth
        lat = fetch_latency_ms(q, q.depth)
        m.foreground_latency_ms += lat
        m.direct_latency_ms += lat
    return m


def run_exact_match(stream: List[ToolQuery]) -> MethodMetrics:
    """EM: reuse cached items only on a byte-exact repeat of a prior query."""
    m = MethodMetrics("EM (Exact-Match)")
    seen: Dict[str, List[str]] = {}
    for q in stream:
        m.query_count += 1
        m.reference_items += q.depth
        m.direct_latency_ms += fetch_latency_ms(q, q.depth)
        if q.query in seen:
            m.cache_hits += 1
            served = seen[q.query]
            m.foreground_latency_ms += fetch_latency_ms(q, 0)
        else:
            served = q.items
            seen[q.query] = list(q.items)
            m.foreground_items_fetched += q.depth
            m.foreground_latency_ms += fetch_latency_ms(q, q.depth)
        m.returned_items += len(served)
        m.matched_items += len(set(served) & set(q.items))
    return m


def run_fixed_fraction(stream: List[ToolQuery], embedder, sim_threshold: float,
                       reuse_fraction: float) -> MethodMetrics:
    """FFR: on a near-repeat (sim >= threshold), reuse a fixed fraction f of items."""
    m = MethodMetrics(f"FFR (f={reuse_fraction:.2f})")
    keys: List[str] = []
    vecs: List[np.ndarray] = []
    item_store: List[List[str]] = []
    for q in stream:
        m.query_count += 1
        m.reference_items += q.depth
        m.direct_latency_ms += fetch_latency_ms(q, q.depth)
        v = embedder.embed(q.query)
        best_sim, best_idx = _best_match(v, vecs)
        if best_idx is not None and best_sim >= sim_threshold:
            m.cache_hits += 1
            k_reuse = min(q.depth, int(math.floor(reuse_fraction * q.depth)))
            k_reuse = max(1, k_reuse)
            reused = item_store[best_idx][:k_reuse]
            k_fetch = q.depth - k_reuse
            fetched = q.items[k_reuse:k_reuse + k_fetch]
            served = _served_set(reused, fetched)
            m.foreground_items_fetched += k_fetch
            m.foreground_latency_ms += fetch_latency_ms(q, k_fetch)
        else:
            served = q.items
            m.foreground_items_fetched += q.depth
            m.foreground_latency_ms += fetch_latency_ms(q, q.depth)
        keys.append(q.query)
        vecs.append(v)
        item_store.append(list(q.items))
        m.returned_items += len(served)
        m.matched_items += len(set(served) & set(q.items))
    return m


def run_cortex(stream: List[ToolQuery], embedder, judge_threshold: float) -> MethodMetrics:
    """Cortex: a similarity judge that reuses ALL cached items when approved."""
    m = MethodMetrics(f"Cortex (j={judge_threshold:.2f})")
    vecs: List[np.ndarray] = []
    item_store: List[List[str]] = []
    for q in stream:
        m.query_count += 1
        m.reference_items += q.depth
        m.direct_latency_ms += fetch_latency_ms(q, q.depth)
        v = embedder.embed(q.query)
        best_sim, best_idx = _best_match(v, vecs)
        if best_idx is not None and best_sim >= judge_threshold:
            m.cache_hits += 1
            served = item_store[best_idx][:q.depth]
            m.foreground_latency_ms += fetch_latency_ms(q, 0)
        else:
            served = q.items
            m.foreground_items_fetched += q.depth
            m.foreground_latency_ms += fetch_latency_ms(q, q.depth)
        vecs.append(v)
        item_store.append(list(q.items))
        m.returned_items += len(served)
        m.matched_items += len(set(served) & set(q.items))
    return m


def run_dtr(stream: List[ToolQuery], cache: DTRCache) -> MethodMetrics:
    """
    DTR (ours): reuse floor(s_max * n) base items and fetch only the
    ceil((1 - s_max) * n) delta items when s_max >= tau, plus async exact
    refresh. We model the refresh as off-critical-path (does not add foreground
    latency or foreground load), and let it eventually correct the served set.
    """
    m = MethodMetrics("DTR (ours)")
    for q in stream:
        m.query_count += 1
        m.reference_items += q.depth
        m.direct_latency_ms += fetch_latency_ms(q, q.depth)
        result = cache.lookup(q.query)
        if result.hit and result.cached_entry is not None:
            m.cache_hits += 1
            k_reuse = min(result.reuse_count, len(result.cached_entry.items))
            reused = result.cached_entry.items[:k_reuse]
            k_fetch = q.depth - k_reuse
            # Delta items are fetched fresh via the reformulated query, so they
            # are exactly the novel items for this query.
            fetched = q.items[k_reuse:k_reuse + k_fetch] if k_fetch > 0 else []
            served = _served_set(reused, fetched)
            m.foreground_items_fetched += max(0, k_fetch)
            m.foreground_latency_ms += fetch_latency_ms(q, max(0, k_fetch))
        else:
            served = q.items
            m.foreground_items_fetched += q.depth
            m.foreground_latency_ms += fetch_latency_ms(q, q.depth)
        # Store the authentic full result (models the async exact refresh).
        cache.store(q.query, list(q.items))
        m.returned_items += len(served)
        m.matched_items += len(set(served) & set(q.items))
    return m


def _best_match(v: np.ndarray, vecs: List[np.ndarray]) -> Tuple[float, Optional[int]]:
    best_sim, best_idx = 0.0, None
    for i, u in enumerate(vecs):
        s = float(np.dot(v, u))
        if s > best_sim:
            best_sim, best_idx = s, i
    return best_sim, best_idx


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_all(
    stream: List[ToolQuery],
    tau: float,
    ffr_fraction: float,
    cortex_threshold: float,
    sim_threshold: float,
    real_embedder: bool = False,
) -> List[Dict[str, float]]:
    n = len(stream)
    cache = DTRCache(confidence_threshold=tau, retrieval_depth=stream[0].depth if stream else 10)
    if not real_embedder:
        # Replace the hash-fallback embedder with a deterministic BoW embedder
        # so cosine similarity is semantically meaningful on CPU.
        bow = BagOfWordsEmbedder()
        cache.embedder = bow
        cache.hotspot_extractor.embedder = bow
    embedder = cache.embedder  # share the same embedder for baselines

    results = [
        run_no_cache(stream).finalize(n),
        run_exact_match(stream).finalize(n),
        run_fixed_fraction(stream, embedder, sim_threshold, ffr_fraction).finalize(n),
        run_cortex(stream, embedder, cortex_threshold).finalize(n),
        run_dtr(stream, cache).finalize(n),
    ]
    return results


def print_table(rows: List[Dict[str, float]]) -> None:
    cols = [
        ("method", "method", 18),
        ("exact_match_score", "EM-score", 9),
        ("precision", "prec", 7),
        ("hit_rate", "hit", 7),
        ("backend_load_reduction", "load-red", 9),
        ("latency_reduction_vs_nc", "lat-red", 9),
        ("speedup_vs_nc", "speedup", 8),
    ]
    header = "  ".join(f"{label:>{w}}" if key != "method" else f"{label:<{w}}"
                       for key, label, w in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = []
        for key, _label, w in cols:
            val = r[key]
            if key == "method":
                cells.append(f"{val:<{w}}")
            elif isinstance(val, float):
                cells.append(f"{val:>{w}.3f}")
            else:
                cells.append(f"{val:>{w}}")
        print("  ".join(cells))


def write_csv(rows: List[Dict[str, float]], path: Path) -> None:
    import csv
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote CSV -> {path}")


def main():
    parser = argparse.ArgumentParser(description="DTR benchmark (produce-stage).")
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--overlap", type=float, default=0.65,
                        help="Semantic overlap in [0,1] of the synthetic stream.")
    parser.add_argument("--depth", type=int, default=6, help="Items per tool query (n).")
    parser.add_argument("--tau", type=float, default=0.8,
                        help="DTR confidence threshold (paper default 0.8).")
    parser.add_argument("--ffr-fraction", type=float, default=0.5,
                        help="Fixed-Fraction-Reuse fraction f.")
    parser.add_argument("--cortex-threshold", type=float, default=0.85,
                        help="Cortex similarity-judge approval threshold.")
    parser.add_argument("--sim-threshold", type=float, default=0.8,
                        help="Near-repeat similarity threshold for FFR.")
    parser.add_argument("--trace", type=str, default=None,
                        help="Replay a recorded tool log (JSONL) instead of synthesizing.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--real-embedder", action="store_true",
                        help="Use the agentmesh SemanticEmbedder (needs sentence-transformers).")
    parser.add_argument("--csv", type=str, default=None, help="Write results CSV to this path.")
    args = parser.parse_args()

    print("=" * 72)
    print("DTR (Delta Tool Retrieval) benchmark -- PRODUCE step (tool->LLM)")
    print("Paper targets: ~71.5% foreground latency cut, ~57% backend load cut.")
    print("=" * 72)

    if args.trace:
        stream = load_trace(Path(args.trace), args.depth)
        print(f"Replaying trace: {args.trace} ({len(stream)} queries)")
    else:
        stream = synthesize_stream(args.num_queries, args.overlap, args.depth, args.seed)
        print(f"Synthetic stream: {len(stream)} queries, overlap={args.overlap}, "
              f"depth={args.depth}, seed={args.seed}")

    if not stream:
        raise SystemExit("Empty query stream.")

    print(f"DTR tau={args.tau}  |  baselines: NC, EM, FFR(f={args.ffr_fraction}), "
          f"Cortex(j={args.cortex_threshold})\n")

    t0 = time.perf_counter()
    rows = run_all(stream, args.tau, args.ffr_fraction,
                   args.cortex_threshold, args.sim_threshold,
                   real_embedder=args.real_embedder)
    elapsed = time.perf_counter() - t0

    print_table(rows)

    dtr = next(r for r in rows if r["method"] == "DTR (ours)")
    print()
    print(f"DTR foreground latency reduction vs NC : {dtr['latency_reduction_vs_nc']*100:5.1f}%  "
          f"(paper target ~71.5%)")
    print(f"DTR backend load reduction vs NC       : {dtr['backend_load_reduction']*100:5.1f}%  "
          f"(paper target ~57%)")
    print(f"DTR exact-match score                  : {dtr['exact_match_score']:.3f}")
    print(f"\nWall time: {elapsed:.2f}s")

    if args.csv:
        write_csv(rows, Path(args.csv))


if __name__ == "__main__":
    main()
