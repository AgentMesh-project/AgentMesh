#!/usr/bin/env python3
"""
End-to-end latency + cumulative-ablation harness.

Simulates a supervisor-worker deep-research workflow (1 supervisor, N
researchers) and toggles AgentMesh's three mechanisms cumulatively:

    none  ->  +DTR  ->  +DTR+BPP  ->  +DTR+BPP+DES

reporting the per-stage and total % latency reduction. Each mechanism shrinks
or overlaps a different part of the produce-transfer-consume dataflow:

    DTR -> shrinks PRODUCE  (tool->LLM tool/scrape cost)
    BPP -> shrinks CONSUME  (supervisor gather/prefill over worker branches)
    DES -> OVERLAPS produce/transfer/consume (streaming)

Paper mapping
-------------
  End-to-end: deep-research latency reduced 40.6-62.4%; overall up to 2.1x.
  Cumulative ablation bands: DTR 15-78% (produce), BPP 19-61% (consume),
  DES 6-17% (overlap). This harness reproduces those bands from a synthetic
  per-stage cost model (built into the script; override with --perf-fits) and
  labels each stage with the band it targets; pass --trace for recorded timings.

Cost model (per workflow round)
-------------------------------
  produce_ms : N researchers each run a tool call (base + per-item) -> the
               slowest researcher is on the critical path (they run in parallel).
               DTR cuts each researcher's foreground tool cost by ~71.5% (and
               backend load by ~57%) for cache-eligible queries.
  consume_ms : supervisor prefill over the concatenated worker branches.
               Full prefill cost grows with the SQUARE of total gathered length;
               BPP makes it grow with the per-branch length (linear in N).
  transfer_ms: worker->supervisor streaming. DES overlaps it with produce/
               consume, removing ~`des_overlap_fraction` of the serial transfer.
  fixed_ms   : supervisor decode + routing, unaffected by the three mechanisms.

Fast default: synthetic, no GPU / LLM, < 5s.

Usage:
    python -m benchmarks.end_to_end.run_e2e
    python -m benchmarks.end_to_end.run_e2e --num-researchers 4 --rounds 3
    python -m benchmarks.end_to_end.run_e2e --perf-fits my_perf_fits.json   # optional override
    python -m benchmarks.end_to_end.run_e2e --trace recorded_timings.json --csv out.csv
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Optional bundled perf-fit file: auto-loaded if present. The repository ships
# without it by default, so DEFAULT_FITS below keeps run_e2e fully self-contained.
BUNDLED_PERF_FITS = (
    Path(__file__).resolve().parents[1] / "data" / "sample_perf_fits.json"
)

# Built-in illustrative synthetic per-stage cost model (deep-research-like,
# produce/consume balanced). These are NOT measured hardware numbers; they
# reproduce the cumulative-ablation *bands*, not the absolute paper figures.
# Supply profiled timings via --perf-fits or --trace for true reproduction.
DEFAULT_FITS: Dict = {
    "_comment": "Illustrative synthetic cost model; pass --perf-fits/--trace for measured data.",
    "stages": {
        "produce_tool_ms": {"base_ms": 400, "per_item_ms": 400, "items": 6,
                            "dtr_latency_reduction": 0.715},
        "consume_gather_ms": {"prefix_tokens": 700, "branch_tokens": [1100, 1000, 900],
                              "prefill_ms_per_ktok2": 0.205},
        "transfer_overlap": {"serial_transfer_ms": 747, "des_overlap_fraction": 0.75},
        "fixed_overhead_ms": {"ms": 3653},
    },
    "ablation_targets": {"dtr_produce_pct": [15, 78], "bpp_consume_pct": [19, 61],
                         "des_overlap_pct": [6, 17], "total_speedup_max": 2.1},
}


# ---------------------------------------------------------------------------
# Stage cost model
# ---------------------------------------------------------------------------
@dataclass
class Stages:
    produce_ms: float
    transfer_ms: float
    consume_ms: float
    fixed_ms: float

    @property
    def total(self) -> float:
        return self.produce_ms + self.transfer_ms + self.consume_ms + self.fixed_ms


def base_stage_costs(fits: Dict, num_researchers: int) -> Stages:
    """Baseline (no mechanism) per-round costs from the perf-fit model."""
    s = fits["stages"]
    p = s["produce_tool_ms"]
    produce = p["base_ms"] + p["per_item_ms"] * p["items"]  # one researcher (parallel -> critical path)

    c = s["consume_gather_ms"]
    prefix = c["prefix_tokens"]
    branches = c["branch_tokens"][:num_researchers] or [600] * num_researchers
    while len(branches) < num_researchers:
        branches.append(branches[-1])
    total_len = prefix + sum(branches)
    # Full prefill: quadratic in the total gathered length.
    consume = c["prefill_ms_per_ktok2"] * (total_len / 1000.0) ** 2 * 1000.0

    transfer = s["transfer_overlap"]["serial_transfer_ms"]
    fixed = s["fixed_overhead_ms"]["ms"]
    return Stages(produce_ms=produce, transfer_ms=transfer, consume_ms=consume, fixed_ms=fixed)


def apply_mechanisms(base: Stages, fits: Dict, num_researchers: int,
                     dtr: bool, bpp: bool, des: bool) -> Stages:
    """Apply the enabled mechanisms cumulatively to the baseline stages."""
    s = fits["stages"]
    produce = base.produce_ms
    transfer = base.transfer_ms
    consume = base.consume_ms

    if dtr:
        # DTR removes ~71.5% of the foreground PRODUCE cost for cache-eligible
        # tool calls (the rest is the cold-miss tail).
        red = s["produce_tool_ms"]["dtr_latency_reduction"]
        produce = produce * (1.0 - red)

    if bpp:
        # BPP turns the quadratic gather into the per-branch critical path:
        # O(L_prefix*L_i + L_i^2) for the longest branch instead of O(L_total^2).
        c = s["consume_gather_ms"]
        prefix = c["prefix_tokens"]
        branches = (c["branch_tokens"][:num_researchers] or [600] * num_researchers)
        while len(branches) < num_researchers:
            branches.append(branches[-1])
        longest = max(branches)
        consume = c["prefill_ms_per_ktok2"] * (
            (prefix * longest + longest * longest) / 1e6
        ) * 1000.0

    if des:
        # DES overlaps transfer with produce/consume, removing a fraction of the
        # serial transfer cost.
        ov = s["transfer_overlap"]["des_overlap_fraction"]
        transfer = transfer * (1.0 - ov)

    return Stages(produce_ms=produce, transfer_ms=transfer,
                  consume_ms=consume, fixed_ms=base.fixed_ms)


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------
ABLATION_STEPS = [
    ("none", False, False, False),
    ("+DTR", True, False, False),
    ("+BPP", True, True, False),
    ("+DES", True, True, True),
]


def run_ablation(fits: Dict, num_researchers: int, rounds: int) -> List[Dict]:
    base = base_stage_costs(fits, num_researchers)
    rows: List[Dict] = []
    prev_total: Optional[float] = None
    baseline_total: Optional[float] = None

    for label, dtr, bpp, des in ABLATION_STEPS:
        st = apply_mechanisms(base, fits, num_researchers, dtr, bpp, des)
        total = st.total * rounds
        if baseline_total is None:
            baseline_total = total
        cum_red = 1.0 - total / baseline_total if baseline_total else 0.0
        step_red = (1.0 - total / prev_total) if prev_total else 0.0
        rows.append({
            "config": label,
            "produce_ms": round(st.produce_ms * rounds, 1),
            "transfer_ms": round(st.transfer_ms * rounds, 1),
            "consume_ms": round(st.consume_ms * rounds, 1),
            "fixed_ms": round(st.fixed_ms * rounds, 1),
            "total_ms": round(total, 1),
            "step_reduction": round(step_red, 4),
            "cumulative_reduction": round(cum_red, 4),
            "speedup_vs_none": round(baseline_total / total, 3) if total else float("inf"),
        })
        prev_total = total
    return rows


def load_trace_fits(path: Path) -> Dict:
    """Load recorded timings / a perf-fit model (the JSON schema documented above)."""
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_table(rows: List[Dict]) -> None:
    print(f"{'config':<8}  {'produce':>9}  {'transfer':>9}  {'consume':>9}  "
          f"{'fixed':>9}  {'total_ms':>10}  {'step%':>7}  {'cum%':>7}  {'speedup':>7}")
    print("-" * 88)
    for r in rows:
        print(f"{r['config']:<8}  {r['produce_ms']:>9.0f}  {r['transfer_ms']:>9.0f}  "
              f"{r['consume_ms']:>9.0f}  {r['fixed_ms']:>9.0f}  {r['total_ms']:>10.0f}  "
              f"{r['step_reduction']*100:>6.1f}%  {r['cumulative_reduction']*100:>6.1f}%  "
              f"{r['speedup_vs_none']:>7.2f}")


def check_bands(rows: List[Dict], fits: Dict) -> None:
    tg = fits.get("ablation_targets", {})
    by = {r["config"]: r for r in rows}
    print("\nPer-mechanism step contribution vs paper bands:")

    def band(name, step_pct, lo_hi):
        lo, hi = lo_hi
        ok = lo - 1e-6 <= step_pct <= hi + 1e-6
        flag = "in band" if ok else "OUT of band"
        print(f"  {name:<5} step {step_pct:5.1f}%  (paper {lo}-{hi}%)  [{flag}]")

    if "+DTR" in by:
        band("DTR", by["+DTR"]["step_reduction"] * 100, tg.get("dtr_produce_pct", [15, 78]))
    if "+BPP" in by:
        band("BPP", by["+BPP"]["step_reduction"] * 100, tg.get("bpp_consume_pct", [19, 61]))
    if "+DES" in by:
        band("DES", by["+DES"]["step_reduction"] * 100, tg.get("des_overlap_pct", [6, 17]))

    final = rows[-1]
    target_speedup = tg.get("total_speedup_max", 2.1)
    print(f"\nTotal latency reduction : {final['cumulative_reduction']*100:.1f}%  "
          f"(paper deep-research 40.6-62.4%)")
    print(f"Total speedup           : {final['speedup_vs_none']:.2f}x  "
          f"(paper up to {target_speedup}x)")


def main():
    parser = argparse.ArgumentParser(description="End-to-end cumulative-ablation harness.")
    parser.add_argument("--num-researchers", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=1,
                        help="Workflow rounds to aggregate (multiplies all stages).")
    parser.add_argument("--perf-fits", type=str, default=None,
                        help="Per-stage cost model JSON (optional; defaults to a "
                             "built-in synthetic model, or the bundled file if present).")
    parser.add_argument("--trace", type=str, default=None,
                        help="Recorded timings JSON (same schema as perf-fits) for full reproduction.")
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    print("=" * 90)
    print("AgentMesh end-to-end cumulative ablation (supervisor-worker deep research)")
    print("none -> +DTR (produce) -> +BPP (consume) -> +DES (overlap)")
    print("=" * 90)

    if args.trace:
        p = Path(args.trace)
        if not p.exists():
            raise SystemExit(f"Trace not found: {p}")
        fits, src = load_trace_fits(p), f"recorded trace ({p.name})"
    elif args.perf_fits:
        p = Path(args.perf_fits)
        if not p.exists():
            raise SystemExit(f"Cost model not found: {p}")
        fits, src = load_trace_fits(p), f"perf-fit file ({p.name})"
    elif BUNDLED_PERF_FITS.exists():
        fits, src = load_trace_fits(BUNDLED_PERF_FITS), f"bundled perf-fit ({BUNDLED_PERF_FITS.name})"
    else:
        fits, src = DEFAULT_FITS, "built-in synthetic cost model"
    print(f"Cost model: {src}")
    print(f"Workflow  : 1 supervisor + {args.num_researchers} researchers, "
          f"{args.rounds} round(s)\n")

    rows = run_ablation(fits, args.num_researchers, args.rounds)
    print_table(rows)
    check_bands(rows, fits)

    if args.csv:
        import csv
        with Path(args.csv).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote CSV -> {args.csv}")


if __name__ == "__main__":
    main()
