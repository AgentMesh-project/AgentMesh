# SPDX-License-Identifier: Apache-2.0
"""
BPP vLLM example: supervisor + parallel worker branches.

Demonstrates how AgentMesh's ``GPUKVConnector`` (Branch-Parallel Prefill)
prefills independent worker branches in parallel against a shared supervisor
prefix and stitches their KV into one sequence for the final consuming prefill,
verifying the stitched output against a full-prefill baseline.

Requires vLLM + a GPU. See ``run_supervisor_workers.py`` and ``README.md``.
"""
