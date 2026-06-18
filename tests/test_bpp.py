"""
Tests for BPP (Branch-Parallel Prefill) — the "consume" step.

BPP removes the BARRIER in the consume stage of an LLM→LLM dataflow by
prefilling independent worker branches in parallel against the shared
supervisor prefix, masking cross-branch attention (Branch-Parallel Attention,
BPA), and stitching the branch-local KVs with late RoPE realignment.

The attention module imports cleanly without torch (the import is guarded), but
its tensor-producing pieces (mask generation, RoPE realignment, the manager)
require torch and are skipped via ``pytest.importorskip`` when torch is absent.
Pure-Python pieces (dataclasses, import-guard behavior) are always tested.
"""

import pytest

from agentmesh.mechanisms.bpp import (
    WorkerOutput,
    SupervisorContext,
    BPAResult,
    InterBranchMaskGenerator,
    RoPEAligner,
    BranchParallelManager,
    BPAManager,
    KVCacheManager,
    create_bpp_manager,
)
from agentmesh.mechanisms.bpp import attention as bpp_attention


# ---------------------------------------------------------------------------
# Import-guard behavior (no torch required)
# ---------------------------------------------------------------------------
def test_module_imports_without_torch():
    """The BPP module must import even when torch is unavailable."""
    assert WorkerOutput is not None
    assert InterBranchMaskGenerator is not None
    assert create_bpp_manager is not None


def test_aliases_point_to_manager():
    """BPAManager and KVCacheManager are aliases of BranchParallelManager."""
    assert BPAManager is BranchParallelManager
    assert KVCacheManager is BranchParallelManager


def test_torch_gated_construction_raises_when_torch_absent():
    """Without torch, instantiating the manager raises a clear ImportError."""
    if bpp_attention._TORCH_AVAILABLE:
        pytest.skip("torch present; guard not exercised")
    with pytest.raises(ImportError):
        create_bpp_manager(num_layers=1, num_heads=1, head_dim=2, device="cpu")


def test_torch_gated_mask_raises_when_torch_absent():
    if bpp_attention._TORCH_AVAILABLE:
        pytest.skip("torch present; guard not exercised")
    with pytest.raises(ImportError):
        InterBranchMaskGenerator.generate_worker_mask(prefix_len=4, worker_len=2)


# ---------------------------------------------------------------------------
# Pure-Python dataclasses (no torch)
# ---------------------------------------------------------------------------
def test_worker_output_length():
    out = WorkerOutput(worker_id="w1", tokens=[1, 2, 3, 4])
    assert out.length == 4


def test_supervisor_context_length():
    ctx = SupervisorContext(tokens=list(range(7)))
    assert ctx.length == 7


def test_bpa_result_defaults():
    res = BPAResult(combined_kv={}, total_tokens=12, worker_order=["a", "b"])
    assert res.total_tokens == 12
    assert res.worker_order == ["a", "b"]
    assert res.realignment_time == 0.0
    assert res.prefill_times == {}


# ---------------------------------------------------------------------------
# torch-dependent pieces — skipped gracefully without torch
# ---------------------------------------------------------------------------
def test_worker_mask_shape_and_contents():
    torch = pytest.importorskip("torch")

    prefix_len, worker_len = 5, 3
    mask = InterBranchMaskGenerator.generate_worker_mask(
        prefix_len=prefix_len, worker_len=worker_len, device="cpu"
    )
    # Shape: (worker_len, prefix_len + worker_len)
    assert tuple(mask.shape) == (worker_len, prefix_len + worker_len)
    assert mask.dtype == torch.bool
    # Every branch token attends to the full prefix.
    assert bool(mask[:, :prefix_len].all())
    # Branch tokens attend causally to themselves: row i sees i+1 branch tokens.
    for i in range(worker_len):
        attended = int(mask[i, prefix_len:].sum())
        assert attended == i + 1


def test_full_decoding_mask_is_causal():
    torch = pytest.importorskip("torch")
    prefix_len, worker_lengths = 2, [2, 3]
    total = prefix_len + sum(worker_lengths)
    mask = InterBranchMaskGenerator.generate_full_decoding_mask(
        prefix_len=prefix_len, worker_lengths=worker_lengths, device="cpu"
    )
    assert tuple(mask.shape) == (total, total)
    # Lower-triangular causal mask.
    assert bool(torch.equal(mask, torch.tril(torch.ones(total, total, dtype=torch.bool))))


def test_rope_aligner_identity_when_no_shift():
    torch = pytest.importorskip("torch")
    aligner = RoPEAligner(head_dim=8, rope_base=10000.0)
    key = torch.randn(2, 4, 8)
    out = aligner.realign(key, from_pos=3, to_pos=3)
    assert torch.equal(out, key)


def test_rope_aligner_preserves_shape_and_norm():
    torch = pytest.importorskip("torch")
    aligner = RoPEAligner(head_dim=8, rope_base=10000.0)
    key = torch.randn(2, 4, 8)
    out = aligner.realign(key, from_pos=0, to_pos=5)
    assert out.shape == key.shape
    # RoPE is a rotation → preserves per-vector norm.
    assert torch.allclose(out.norm(dim=-1), key.norm(dim=-1), atol=1e-4)


def test_complexity_info_linear_in_branch_count():
    """BPA reduces attention complexity O(N^2)→O(N) in branch count N."""
    pytest.importorskip("torch")
    mgr = create_bpp_manager(num_layers=1, num_heads=1, head_dim=2, device="cpu")
    mgr.set_supervisor_context(SupervisorContext(tokens=list(range(8))))

    for wid in ["w1", "w2", "w3"]:
        mgr.prefill_worker(WorkerOutput(worker_id=wid, tokens=[1, 2, 3, 4]))

    info = mgr.get_complexity_info()
    assert info["num_branches"] == 3
    # BPA collapses the quadratic all-to-all term, so it must beat full attention.
    assert info["bpa_complexity"] <= info["full_complexity"]
    assert info["speedup"] >= 1.0


def test_prefill_and_stitch_roundtrip():
    torch = pytest.importorskip("torch")
    mgr = create_bpp_manager(num_layers=2, num_heads=2, head_dim=4, device="cpu")
    mgr.set_supervisor_context(SupervisorContext(tokens=list(range(4))))

    lengths = {"w1": 3, "w2": 2}
    for wid, n in lengths.items():
        mgr.prefill_worker(WorkerOutput(worker_id=wid, tokens=list(range(n))))

    result = mgr.stitch_and_realign()
    assert isinstance(result, BPAResult)
    assert result.total_tokens == sum(lengths.values())
    assert set(result.worker_order) == set(lengths.keys())
    # Stitched KV sequence length == total branch tokens, per layer.
    for layer in range(2):
        key, value = result.combined_kv[layer]
        assert key.shape[-2] == sum(lengths.values())
        assert value.shape[-2] == sum(lengths.values())

    stats = mgr.get_stats()
    assert stats["total_prefills"] == 2
    assert stats["total_stitches"] == 1
