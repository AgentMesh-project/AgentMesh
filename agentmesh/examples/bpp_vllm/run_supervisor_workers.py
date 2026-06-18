#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
BPP supervisor + parallel-branch example (vLLM).

This is a small, runnable demonstration of Branch-Parallel Prefill (BPP) as
realized by AgentMesh's ``GPUKVConnector`` inside vLLM. It mirrors the
produce-transfer-consume dataflow that BPP makes barrier-free:

  1. Supervisor : prefill the shared prefix prompt -> store KV (key="master")
  2. Branch 1   : prefill (prefix + delta_1)       -> store incremental KV
  3. Branch 2   : prefill (prefix + delta_2)       -> store incremental KV
  4. Final      : load + stitch all KV segments with late RoPE realignment,
                  then decode.

The branches are independent: in a real BPP run they are prefilled in parallel
against the shared prefix, isolated by slot-based block-diagonal Branch-Parallel
Attention. Here they are issued sequentially only so the example runs as a
single-process script; the connector performs the same KV stitch either way.

Verification: the FINAL (stitched-KV) output is compared against a plain
full-prefill baseline on the concatenated prompt. They should match for the
prefix-reuse scenarios (e.g. ``KV_SCENARIO=S3``).

This example REQUIRES vLLM and a GPU. With neither installed it prints a clear
message and exits cleanly. Prompts are tiny and synthetic — no data files.

Usage:
    # Register the connector (either copy this subpackage into vLLM's connector
    # tree, or rely on vLLM's kv_connector_module_path; see ../README.md).
    KV_SCENARIO=S3 python run_supervisor_workers.py --model Qwen/Qwen3-1.7B
"""

import argparse
import os
import sys
from typing import Dict

# Set multiprocessing method before importing vllm.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def _vllm_available() -> bool:
    """Return True iff vLLM (and a usable torch/CUDA stack) is importable."""
    try:
        import torch  # noqa: F401
        import vllm  # noqa: F401
    except Exception:
        return False
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="BPP supervisor + parallel-branch example (vLLM)"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    return parser.parse_args()


def cleanup_distributed():
    """Clean up the distributed process group if one was created."""
    import torch.distributed as dist

    if dist.is_initialized():
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass


def run_baseline(model_name: str, prompts: Dict[str, str], max_tokens: int, gpu_util: float):
    """Baseline: full prefill + decode on the concatenated final prompt."""
    import torch
    from vllm import LLM, SamplingParams

    print("\n" + "=" * 60)
    print("Baseline: full prefill + decode (no BPP)")
    print("=" * 60)

    llm = LLM(model=model_name, enforce_eager=True, gpu_memory_utilization=gpu_util)
    sampling = SamplingParams(temperature=0, max_tokens=max_tokens)

    final_prompt = prompts["final"]
    output = llm.generate([final_prompt], sampling)[0]
    result = output.outputs[0].text

    print(f"Final prompt: {final_prompt}")
    print(f"Output: {result}")

    del llm
    torch.cuda.empty_cache()
    cleanup_distributed()
    return result


def run_bpp(model_name: str, prompts: Dict[str, str], max_tokens: int, gpu_util: float):
    """BPP: produce per-branch KV, then load+stitch for the consuming prefill."""
    import torch
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    # The connector exposes clear_gpu_kv_cache() to reset cross-request state.
    try:
        # When the subpackage has been copied into vLLM's connector tree:
        from vllm.distributed.kv_transfer.kv_connector.v1.gpu_kv_connector import (
            clear_gpu_kv_cache,
        )
    except Exception:
        # When loaded as an external module from this package:
        from agentmesh.mechanisms.bpp.vllm import clear_gpu_kv_cache

    print("\n" + "=" * 60)
    print("BPP: per-branch produce + stitched consume (GPUKVConnector)")
    print("=" * 60)

    clear_gpu_kv_cache()

    llm = LLM(
        model=model_name,
        enforce_eager=True,
        gpu_memory_utilization=gpu_util,
        kv_transfer_config=KVTransferConfig(
            kv_connector="GPUKVConnector",
            kv_role="kv_both",
        ),
        disable_hybrid_kv_cache_manager=True,
        enable_prefix_caching=False,
    )

    prefill_sampling = SamplingParams(temperature=0, max_tokens=1)
    decode_sampling = SamplingParams(temperature=0, max_tokens=max_tokens)

    # Step 1: supervisor prefix produce.
    print("\n[Step 1] Supervisor prefill (shared prefix)")
    _ = llm.generate([prompts["master"]], prefill_sampling)
    print(f"  Prompt: {prompts['master']}")

    # Step 2: branch 1 produce.
    print("\n[Step 2] Branch 1 prefill (prefix + delta_1)")
    _ = llm.generate([prompts["worker1"]], prefill_sampling)
    print(f"  Prompt: {prompts['worker1']}")

    # Step 3: branch 2 produce.
    print("\n[Step 3] Branch 2 prefill (prefix + delta_2)")
    _ = llm.generate([prompts["worker2"]], prefill_sampling)
    print(f"  Prompt: {prompts['worker2']}")

    # Step 4: stitched consume — load all KV segments, realign RoPE, decode.
    print("\n[Step 4] Final request (load + stitch branch KV, then decode)")
    output = llm.generate([prompts["final"]], decode_sampling)[0]
    result = output.outputs[0].text
    print(f"  Final prompt: {prompts['final']}")
    print(f"  Output: {result}")

    del llm
    torch.cuda.empty_cache()
    cleanup_distributed()
    return result


def main():
    if not _vllm_available():
        print(
            "[bpp_vllm] vLLM (and a torch/CUDA GPU stack) is not available in "
            "this environment, so this example cannot run.\n"
            "Install vLLM on a GPU host, register the GPUKVConnector "
            "(see ../README.md and docs/vllm_integration.md), then re-run:\n"
            "    KV_SCENARIO=S3 python run_supervisor_workers.py "
            "--model Qwen/Qwen3-1.7B"
        )
        return 0

    args = parse_args()
    scenario = os.environ.get("KV_SCENARIO", "S3")

    print("=" * 60)
    print("BPP supervisor + parallel-branch example")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"KV_SCENARIO: {scenario}")
    print(f"Max tokens:  {args.max_tokens}")

    # Tiny synthetic prompts: a shared prefix plus two independent branch deltas.
    prefix = "Suppose a+b+c=10. What is a?"
    branch_delta_1 = " Given b=3."
    branch_delta_2 = " Given c=5."

    prompts = {
        "master": prefix,
        "worker1": prefix + branch_delta_1,
        "worker2": prefix + branch_delta_2,
        "final": prefix + branch_delta_1 + branch_delta_2,
    }

    print("\nPrompts:")
    print(f"  Supervisor (prefix): '{prompts['master']}'")
    print(f"  Branch 1           : '{prompts['worker1']}'")
    print(f"  Branch 2           : '{prompts['worker2']}'")
    print(f"  Final (stitched)   : '{prompts['final']}'")

    baseline_result = run_baseline(
        args.model, prompts, args.max_tokens, args.gpu_memory_utilization
    )
    bpp_result = run_bpp(
        args.model, prompts, args.max_tokens, args.gpu_memory_utilization
    )

    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    print(f"Baseline (full prefill): {baseline_result}")
    print(f"BPP (stitched KV)      : {bpp_result}")
    match = baseline_result == bpp_result
    print(f"Match: {match}")

    return 0 if match else 1


if __name__ == "__main__":
    sys.exit(main())
