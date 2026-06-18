"""
DES (Dynamic-Equilibrium Streaming) benchmarks.

Measure the OVERLAP of produce/transfer/consume stages: throughput of DES
versus No-Stream (NS) and Static fixed-chunk Stream (SS) across synthetic
rate traces, and convergence of the chunk size theta to a rate-matched
equilibrium (slack delta -> 0).

Scripts:
- benchmark_des.py -- throughput comparison and theta-convergence study.
"""
