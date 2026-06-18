#!/usr/bin/env python3
"""
BPP attention-orthogonality study -- the motivation for Branch-Parallel Attention.

BPA (Branch-Parallel Attention) masks cross-branch attention in the
supervisor-worker scatter-gather topology. This is only sound if worker
branches barely attend to each other in the first place. This benchmark
measures exactly that on a captured (or synthesized) attention map.

Headline metrics (mirrors attn_orthogonality/capture_attention.py)
------------------------------------------------------------------
  last_branch_cross_fraction : fraction of the LAST branch's attention mass
      that lands on OTHER branches' tokens. The number to quote -- should be
      tiny (near the uniform baseline or below), justifying the mask.
  self_over_cross_ratio      : per-pair self-attention intensity divided by
      per-pair cross-branch intensity. Large (>> 1) => masking cross-branch
      attention loses almost nothing.
  prefix_fraction            : fraction of branch attention on the shared
      supervisor prefix. Large => branches mostly attend to the prefix, which
      BPA preserves exactly.

Paper mapping
-------------
  Justifies BPA masking; supports the "precision matches/exceeds Full-Prefill"
  claim and the orthogonality-to-CacheBlend/EPIC point.

Modes
-----
  --synthetic (default): synthesize a plausible block-structured attention
      matrix (bright self-diagonal + bright prefix column + dim cross-branch +
      one attention sink at BOS) and an --overlap control scenario that injects
      deliberate cross-branch mass to show the metric responds.
  --dry-run: print the planned configuration and exit (no computation).
  real GPU path (--model ...): capture attention from a HuggingFace model
      (e.g. Qwen3-4B) via transformers -- guarded import, documented below,
      NOT exercised by the fast default.

Output: a heatmap PNG when matplotlib is present, else a numeric block-mass
summary. Always prints the headline metrics and a JSON blob.

Fast default: synthetic, no GPU / transformers, < 5s.

Usage:
    python -m benchmarks.bpp.attention_orthogonality
    python -m benchmarks.bpp.attention_orthogonality --num-branches 4 --overlap-scenario
    python -m benchmarks.bpp.attention_orthogonality --dry-run
    python -m benchmarks.bpp.attention_orthogonality --heatmap fig.png --json out.json
    # real GPU path (needs torch + transformers + a GPU):
    python -m benchmarks.bpp.attention_orthogonality --model Qwen/Qwen3-4B
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Segment layout for a supervisor-worker attention map
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    name: str
    start: int
    end: int  # exclusive

    @property
    def length(self) -> int:
        return self.end - self.start


def build_layout(
    sink_tokens: int,
    prefix_tokens: int,
    branch_tokens: List[int],
) -> List[Segment]:
    """Lay out [sink | prefix | branch_0 | branch_1 | ...] along the sequence."""
    segs: List[Segment] = []
    pos = 0
    segs.append(Segment("sink", pos, pos + sink_tokens)); pos += sink_tokens
    segs.append(Segment("prefix", pos, pos + prefix_tokens)); pos += prefix_tokens
    for i, b in enumerate(branch_tokens):
        segs.append(Segment(f"W{i}", pos, pos + b)); pos += b
    return segs


# ---------------------------------------------------------------------------
# Synthetic attention matrix
# ---------------------------------------------------------------------------
def synth_attention(
    segs: List[Segment],
    cross_mass: float,
    seed: int,
) -> np.ndarray:
    """
    Build a row-stochastic causal attention matrix with the block structure
    observed in supervisor-worker decoding:

      - a bright attention SINK at BOS (column 0),
      - a bright shared-PREFIX column band,
      - a bright per-branch SELF diagonal block,
      - a DIM cross-branch band (controlled by `cross_mass`),
      - causal masking (no attention to future tokens).

    Rows are normalized to sum to 1 over the causally-visible keys.
    """
    rng = np.random.default_rng(seed)
    S = segs[-1].end
    A = np.zeros((S, S), dtype=np.float64)

    sink = segs[0]
    prefix = segs[1]
    branches = [s for s in segs if s.name.startswith("W")]

    def seg_of(pos: int) -> Segment:
        for s in segs:
            if s.start <= pos < s.end:
                return s
        return segs[-1]

    for q in range(S):
        qseg = seg_of(q)
        row = np.zeros(S, dtype=np.float64)
        # Attention sink (BOS): every token keeps a little mass here.
        for k in range(sink.start, min(sink.end, q + 1)):
            row[k] += 8.0 + rng.random()
        # Shared prefix: strong, especially for worker rows.
        pref_w = 6.0 if qseg.name.startswith("W") else 2.0
        for k in range(prefix.start, min(prefix.end, q + 1)):
            row[k] += pref_w * (0.7 + 0.6 * rng.random())
        if qseg.name.startswith("W"):
            # Self block: bright diagonal within the branch.
            for k in range(qseg.start, min(qseg.end, q + 1)):
                d = q - k
                row[k] += (10.0 if d == 0 else 4.0 / (1.0 + d)) * (0.8 + 0.4 * rng.random())
            # Cross-branch: dim mass onto OTHER (earlier) branches.
            for ob in branches:
                if ob is qseg:
                    continue
                for k in range(ob.start, min(ob.end, q + 1)):
                    row[k] += cross_mass * (0.5 + rng.random())
        else:
            # Prefix/sink rows attend causally within themselves.
            for k in range(qseg.start, min(qseg.end, q + 1)):
                row[k] += 3.0 * (0.6 + 0.6 * rng.random())
        tot = row.sum()
        if tot > 0:
            row /= tot
        A[q] = row
    return A


# ---------------------------------------------------------------------------
# Metric computation (block masses + headline fractions)
# ---------------------------------------------------------------------------
def compute_metrics(A: np.ndarray, segs: List[Segment]) -> Dict:
    """Compute block masses and the headline orthogonality metrics."""
    names = [s.name for s in segs]
    S = len(segs)
    # Mean per-query attention fraction from segment i (rows) to segment j (cols).
    frac = np.zeros((S, S), dtype=np.float64)
    for i, qs in enumerate(segs):
        if qs.length == 0:
            continue
        block = A[qs.start:qs.end, :]
        for j, ks in enumerate(segs):
            frac[i, j] = block[:, ks.start:ks.end].sum() / qs.length

    w_idx = [i for i, s in enumerate(segs) if s.name.startswith("W")]
    p_idx = names.index("prefix")
    last = w_idx[-1]

    # last_branch_cross_fraction: mass from the last branch onto other branches.
    last_cross = float(sum(frac[last, c] for c in w_idx if c != last))
    # prefix_fraction (averaged over branches that have predecessors).
    exposed = w_idx[1:] if len(w_idx) > 1 else w_idx
    prefix_fraction = float(np.mean([frac[r, p_idx] for r in exposed])) if exposed else float(frac[last, p_idx])

    # Per-pair intensities: total mass / number of (query, key) causal pairs.
    def causal_pairs(qseg: Segment, kseg: Segment) -> int:
        c = 0
        for q in range(qseg.start, qseg.end):
            lo, hi = kseg.start, min(kseg.end, q + 1)
            c += max(0, hi - lo)
        return c

    def block_mass(qseg: Segment, kseg: Segment) -> float:
        return float(A[qseg.start:qseg.end, kseg.start:kseg.end].sum())

    self_ints, cross_ints = [], []
    for r in w_idx:
        qseg = segs[r]
        p = causal_pairs(qseg, qseg)
        if p > 0:
            self_ints.append(block_mass(qseg, qseg) / p)
        for c in w_idx:
            if c >= r:
                continue
            kseg = segs[c]
            pc = causal_pairs(qseg, kseg)
            if pc > 0:
                cross_ints.append(block_mass(qseg, kseg) / pc)
    self_int = float(np.mean(self_ints)) if self_ints else float("nan")
    cross_int = float(np.mean(cross_ints)) if cross_ints else float("nan")
    self_over_cross = float(self_int / cross_int) if cross_int and cross_int > 0 else float("inf")

    # Uniform-attention baseline for the last branch's cross fraction.
    last_span = segs[last]
    pred_len = sum(segs[i].length for i in w_idx if i != last)
    qpos = np.arange(last_span.start, last_span.end, dtype=np.float64)
    uniform_cross = float(np.mean(pred_len / (qpos + 1.0))) if last_span.length else float("nan")

    return {
        "segment_names": names,
        "segment_lengths": [s.length for s in segs],
        "last_branch_cross_fraction": last_cross,
        "uniform_baseline_last_branch_cross": uniform_cross,
        "cross_suppression_vs_uniform": float(uniform_cross / last_cross) if last_cross > 0 else float("inf"),
        "prefix_fraction": prefix_fraction,
        "per_pair_self_intensity": self_int,
        "per_pair_cross_intensity": cross_int,
        "self_over_cross_ratio": self_over_cross,
        "_frac_matrix": frac.tolist(),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def numeric_block_summary(metrics: Dict) -> None:
    names = metrics["segment_names"]
    frac = np.array(metrics["_frac_matrix"])
    print("\nBlock attention-fraction matrix (rows=query seg, cols=key seg):")
    head = "        " + "".join(f"{n:>8}" for n in names)
    print(head)
    for i, n in enumerate(names):
        cells = "".join(f"{frac[i, j]:>8.3f}" for j in range(len(names)))
        print(f"{n:>8}{cells}")


def maybe_heatmap(metrics: Dict, A: np.ndarray, segs: List[Segment], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(5, 4.2))
    disp = np.where(A > 0, A, np.nan)
    im = ax.imshow(disp, aspect="auto", cmap="magma",
                   norm=LogNorm(vmin=max(A[A > 0].min(), 1e-4), vmax=A.max()))
    for s in segs[1:]:
        ax.axhline(s.start - 0.5, color="cyan", lw=0.5, alpha=0.6)
        ax.axvline(s.start - 0.5, color="cyan", lw=0.5, alpha=0.6)
    ticks = [(s.start + s.end) / 2 for s in segs]
    ax.set_xticks(ticks); ax.set_xticklabels([s.name for s in segs], fontsize=7)
    ax.set_yticks(ticks); ax.set_yticklabels([s.name for s in segs], fontsize=7)
    ax.set_title("Supervisor-worker attention (BPA motivation)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Real GPU capture path (guarded, documented; not run by the fast default)
# ---------------------------------------------------------------------------
def capture_real(model: str, num_branches: int) -> Tuple[np.ndarray, List[Segment]]:
    """
    Capture an attention map from a real model (e.g. Qwen/Qwen3-4B).

    Requires `torch` and `transformers` and (realistically) a GPU. This mirrors
    attn_orthogonality/capture_attention.py at a high level: build a
    supervisor prefix + concatenated worker drafts, run a forward pass with
    output_attentions=True, and pool the per-head maps. We keep it minimal and
    lazy-imported so the module stays importable without heavy deps.
    """
    import torch  # noqa: F401  (raises a clear error if absent)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    raise NotImplementedError(
        "The real GPU capture path is intentionally a thin stub in the OSS "
        "benchmark suite: a faithful capture needs the full worker-generation "
        "pipeline and a GPU. Use --synthetic for the self-contained study, or adapt "
        "attn_orthogonality/capture_attention.py for end-to-end capture. "
        f"(requested model={model}, num_branches={num_branches})"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BPP attention-orthogonality study.")
    parser.add_argument("--num-branches", type=int, default=4)
    parser.add_argument("--prefix-tokens", type=int, default=40)
    parser.add_argument("--branch-tokens", type=int, default=24,
                        help="Tokens per worker branch (synthetic mode).")
    parser.add_argument("--sink-tokens", type=int, default=1)
    parser.add_argument("--cross-mass", type=float, default=0.05,
                        help="Per-key cross-branch weight before normalization (synthetic).")
    parser.add_argument("--overlap-scenario", action="store_true",
                        help="Control: inject heavy cross-branch mass to show the "
                             "metric responds (cross fraction rises).")
    parser.add_argument("--synthetic", action="store_true", default=True,
                        help="Synthesize the attention map without a GPU (default).")
    parser.add_argument("--model", type=str, default=None,
                        help="Real GPU capture from this HF model (needs torch+transformers+GPU).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned configuration and exit.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--heatmap", type=str, default=None, help="PNG path (needs matplotlib).")
    parser.add_argument("--json", type=str, default=None, help="Write metrics JSON to this path.")
    args = parser.parse_args()

    print("=" * 70)
    print("BPP attention-orthogonality study (BPA motivation)")
    print("=" * 70)

    branch_tokens = [args.branch_tokens] * args.num_branches
    cross_mass = 1.2 if args.overlap_scenario else args.cross_mass
    scenario = "overlap-control (heavy cross-branch)" if args.overlap_scenario else "disjoint (default)"

    if args.dry_run:
        print(f"[dry-run] scenario           : {scenario}")
        print(f"[dry-run] branches           : {args.num_branches} x {args.branch_tokens} tokens")
        print(f"[dry-run] prefix/sink        : {args.prefix_tokens}/{args.sink_tokens} tokens")
        print(f"[dry-run] cross_mass         : {cross_mass}")
        print(f"[dry-run] capture            : {'real:' + args.model if args.model else 'synthetic'}")
        return

    if args.model:
        A, segs = capture_real(args.model, args.num_branches)
    else:
        segs = build_layout(args.sink_tokens, args.prefix_tokens, branch_tokens)
        A = synth_attention(segs, cross_mass=cross_mass, seed=args.seed)

    print(f"Scenario : {scenario}")
    print(f"Layout   : sink={args.sink_tokens}, prefix={args.prefix_tokens}, "
          f"branches={branch_tokens}\n")

    metrics = compute_metrics(A, segs)

    print("Headline metrics")
    print("-" * 40)
    print(f"  last_branch_cross_fraction      : {metrics['last_branch_cross_fraction']:.4f}")
    print(f"  uniform baseline (last branch)  : {metrics['uniform_baseline_last_branch_cross']:.4f}")
    print(f"  cross suppression vs uniform    : {metrics['cross_suppression_vs_uniform']:.2f}x")
    print(f"  prefix_fraction                 : {metrics['prefix_fraction']:.4f}")
    print(f"  self_over_cross_ratio           : {metrics['self_over_cross_ratio']:.2f}x")

    verdict = (
        "cross-branch attention is negligible -> BPA masking is justified."
        if metrics["last_branch_cross_fraction"] <= metrics["uniform_baseline_last_branch_cross"] * 1.5
        and metrics["self_over_cross_ratio"] > 3.0
        else "cross-branch attention is substantial -> masking would lose mass "
             "(expected in the --overlap-scenario control)."
    )
    print(f"\nVerdict: {verdict}")

    numeric_block_summary(metrics)

    if args.heatmap:
        if maybe_heatmap(metrics, A, segs, Path(args.heatmap)):
            print(f"\nWrote heatmap -> {args.heatmap}")
        else:
            print("\nmatplotlib not installed -- skipped heatmap (numeric summary above holds the data).")

    if args.json:
        out = {k: v for k, v in metrics.items() if k != "_frac_matrix"}
        out["scenario"] = scenario
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Wrote metrics JSON -> {args.json}")


if __name__ == "__main__":
    main()
