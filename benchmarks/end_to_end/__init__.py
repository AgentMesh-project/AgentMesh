"""
End-to-end benchmarks.

A latency + cumulative-ablation harness for a supervisor-worker workflow with
the three mechanisms toggled cumulatively (none -> +DTR -> +BPP -> +DES),
reporting the per-stage and total % reductions consistent with the paper.

Scripts:
- run_e2e.py -- the cumulative-ablation table and total speedup.
"""
