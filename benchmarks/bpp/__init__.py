"""
BPP (Branch-Parallel Prefill) benchmarks.

Measure the CONSUME (gather) step of an LLM->LLM dataflow in the
supervisor-worker scatter-gather topology.

Scripts:
- attention_orthogonality.py -- evidence that cross-branch attention is
  negligible (last_branch_cross_fraction, self_over_cross_ratio,
  prefix_fraction), justifying Branch-Parallel Attention (BPA) masking.
- bpp_speedup.py -- attention-complexity microbench showing the
  O(N^2) -> O(N) reduction in branch count N.
"""
