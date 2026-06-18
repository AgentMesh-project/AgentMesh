"""
DTR (Delta Tool Retrieval) benchmarks.

Measure the PRODUCE step of a tool->LLM dataflow: cache hit-rate, item reuse,
foreground latency reduction, backend-load reduction, and exact-match precision
versus the No-Cache / Exact-Match / Fixed-Fraction-Reuse / Cortex baselines.

Scripts:
- benchmark_dtr.py   -- the headline DTR table (latency, load, precision).
- threshold_sweep.py -- exact-match vs confidence threshold tau in [0.75, 0.95].
"""
