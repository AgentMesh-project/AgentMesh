"""
Tests for TSD (Topology-aware Semantic Decoupling) mechanism.

Tests the supervisor-worker TDA (Topologically Decoupled Attention):
inter-worker masking, RoPE positional realignment, parallel prefill,
and KV stitching.
"""

import pytest
import torch
import numpy as np
from agentmesh.mechanisms.tsd import (
    WorkerOutput,
    SupervisorContext,
    TDAResult,
    InterWorkerMaskGenerator,
    RoPEAligner,
    TDAManager,
    KVCacheManager,
    create_tda_manager,
    create_tsd_manager,
)


# ---------------------------------------------------------------------------
# WorkerOutput / SupervisorContext data classes
# ---------------------------------------------------------------------------

class TestWorkerOutput:
    """Tests for WorkerOutput dataclass."""

    def test_creation(self):
        """WorkerOutput stores worker_id, tokens, arrival_time, kv_states."""
        wo = WorkerOutput(
            worker_id="w1",
            tokens=[10, 20, 30],
            kv_states={0: (torch.zeros(2, 3, 8), torch.zeros(2, 3, 8))},
        )
        assert wo.worker_id == "w1"
        assert wo.length == 3

    def test_length_property(self):
        wo = WorkerOutput(worker_id="w2", tokens=[1, 2])
        assert wo.length == 2


class TestSupervisorContext:
    """Tests for SupervisorContext dataclass."""

    def test_creation(self):
        ctx = SupervisorContext(
            tokens=[1, 2, 3],
            kv_states={0: (torch.zeros(2, 3, 8), torch.zeros(2, 3, 8))},
        )
        assert ctx.length == 3


# ---------------------------------------------------------------------------
# InterWorkerMaskGenerator — inter-worker masking
# ---------------------------------------------------------------------------

class TestInterWorkerMaskGenerator:
    """Tests for inter-worker attention masking."""

    def test_worker_mask_shape(self):
        """Worker mask should have correct shape (worker_len, prefix_len + worker_len)."""
        gen = InterWorkerMaskGenerator()
        mask = gen.generate_worker_mask(prefix_len=5, worker_len=3)

        assert mask.shape == (3, 8)  # (worker_len, prefix_len + worker_len)

    def test_worker_mask_attends_to_prefix(self):
        """All worker tokens should attend to all prefix tokens."""
        gen = InterWorkerMaskGenerator()
        mask = gen.generate_worker_mask(prefix_len=3, worker_len=2)

        # Worker tokens (rows) should attend to prefix tokens (cols 0-2)
        assert mask[0, 0].item() is True
        assert mask[0, 1].item() is True
        assert mask[0, 2].item() is True
        assert mask[1, 0].item() is True

    def test_worker_mask_causal_self(self):
        """Worker tokens should have causal self-attention."""
        gen = InterWorkerMaskGenerator()
        mask = gen.generate_worker_mask(prefix_len=2, worker_len=3)

        # Worker token 0 can attend to itself (col 2) but not forward
        assert mask[0, 2].item() is True
        assert mask[0, 3].item() is False

        # Worker token 1 can attend to token 0 and itself (cols 2-3)
        assert mask[1, 2].item() is True
        assert mask[1, 3].item() is True
        assert mask[1, 4].item() is False

        # Worker token 2 can attend to all previous worker tokens
        assert mask[2, 2].item() is True
        assert mask[2, 3].item() is True
        assert mask[2, 4].item() is True

    def test_full_decoding_mask_causal(self):
        """Full decoding mask should be lower triangular (causal)."""
        gen = InterWorkerMaskGenerator()
        mask = gen.generate_full_decoding_mask(
            prefix_len=2,
            worker_lengths=[2, 1],
        )

        # Total: prefix(2) + w0(2) + w1(1) = 5
        assert mask.shape == (5, 5)

        # Should be lower triangular
        assert mask[0, 0].item() is True
        assert mask[0, 1].item() is False
        assert mask[4, 0].item() is True  # last can attend to first


# ---------------------------------------------------------------------------
# RoPEAligner — positional realignment  k_j = R_{j-i} · k_i
# ---------------------------------------------------------------------------

class TestRoPEAligner:
    """Tests for RoPE positional realignment."""

    def test_realign_identity_when_same_position(self):
        """With from_pos == to_pos the key should be unchanged."""
        aligner = RoPEAligner(head_dim=8, rope_base=10000.0)
        key = torch.randn(1, 4, 8)  # (num_heads, seq_len, head_dim)
        realigned = aligner.realign(key, from_pos=0, to_pos=0)

        assert realigned.shape == key.shape
        assert torch.allclose(key, realigned, atol=1e-5)

    def test_realign_shape_preserved(self):
        """Realignment should preserve tensor shape."""
        aligner = RoPEAligner(head_dim=16, rope_base=10000.0)
        key = torch.randn(4, 10, 16)
        realigned = aligner.realign(key, from_pos=0, to_pos=5)

        assert realigned.shape == key.shape

    def test_batch_realign_multiple_workers(self):
        """Batch realignment should handle multiple workers."""
        aligner = RoPEAligner(head_dim=8, rope_base=10000.0)
        keys = [torch.randn(2, 3, 8), torch.randn(2, 5, 8)]
        from_positions = [0, 0]
        to_positions = [10, 3]

        results = aligner.batch_realign(keys, from_positions, to_positions)
        assert len(results) == 2
        assert results[0].shape == (2, 3, 8)
        assert results[1].shape == (2, 5, 8)


# ---------------------------------------------------------------------------
# TDAManager — main orchestrator
# ---------------------------------------------------------------------------

class TestTDAManager:
    """Tests for TDAManager (Topologically Decoupled Attention)."""

    NUM_LAYERS = 2
    NUM_HEADS = 4
    HEAD_DIM = 8

    @pytest.fixture
    def manager(self):
        return TDAManager(
            num_layers=self.NUM_LAYERS,
            num_heads=self.NUM_HEADS,
            head_dim=self.HEAD_DIM,
            max_parallel_workers=4,
            rope_base=10000.0,
            device="cpu",
        )

    def _make_supervisor_ctx(self, prefix_len: int = 3) -> SupervisorContext:
        """Helper to create a SupervisorContext."""
        kv = {
            l: (
                torch.randn(self.NUM_HEADS, prefix_len, self.HEAD_DIM),
                torch.randn(self.NUM_HEADS, prefix_len, self.HEAD_DIM),
            )
            for l in range(self.NUM_LAYERS)
        }
        return SupervisorContext(tokens=list(range(prefix_len)), kv_states=kv)

    def _make_worker_output(self, worker_id: str, length: int = 3) -> WorkerOutput:
        """Helper to create a WorkerOutput."""
        return WorkerOutput(worker_id=worker_id, tokens=list(range(length)))

    def test_create_via_factory(self):
        """create_tda_manager should return a valid TDAManager."""
        m = create_tda_manager(
            num_layers=2,
            num_heads=4,
            head_dim=8,
            device="cpu",
        )
        assert isinstance(m, TDAManager)

    def test_backward_compat_alias(self):
        """KVCacheManager should be an alias for TDAManager."""
        assert KVCacheManager is TDAManager

    def test_backward_compat_factory(self):
        """create_tsd_manager should also work."""
        m = create_tsd_manager(
            num_layers=2, num_heads=4, head_dim=8, device="cpu"
        )
        assert isinstance(m, TDAManager)

    def test_set_supervisor_context(self, manager):
        """Setting supervisor context should store prefix info."""
        ctx = self._make_supervisor_ctx(prefix_len=3)
        manager.set_supervisor_context(ctx)
        assert manager.supervisor_ctx is not None
        assert manager.supervisor_ctx.length == 3

    def test_prefill_worker(self, manager):
        """prefill_worker should store KV states for the worker."""
        ctx = self._make_supervisor_ctx(prefix_len=3)
        manager.set_supervisor_context(ctx)

        wo = self._make_worker_output("w0", length=3)
        kv_states = manager.prefill_worker(wo)

        assert isinstance(kv_states, dict)
        assert "w0" in manager.worker_kv_states
        assert manager.worker_lengths["w0"] == 3

    def test_get_parallel_groups(self, manager):
        """Parallel groups should list registered workers."""
        ctx = self._make_supervisor_ctx()
        manager.set_supervisor_context(ctx)

        manager.prefill_worker(self._make_worker_output("w0", 2))
        manager.prefill_worker(self._make_worker_output("w1", 2))

        groups = manager.get_parallel_groups()
        assert isinstance(groups, list)
        # All workers should appear in some group
        flat = [wid for g in groups for wid in g]
        assert "w0" in flat
        assert "w1" in flat

    def test_get_complexity_info(self, manager):
        """Should report complexity reduction info."""
        info = manager.get_complexity_info()
        assert isinstance(info, dict)

    def test_stitch_and_realign(self, manager):
        """stitch_and_realign should produce TDAResult."""
        ctx = self._make_supervisor_ctx(prefix_len=3)
        manager.set_supervisor_context(ctx)

        manager.prefill_worker(self._make_worker_output("w0", 2))
        manager.prefill_worker(self._make_worker_output("w1", 3))

        result = manager.stitch_and_realign()
        assert isinstance(result, TDAResult)
        assert result.total_tokens == 5
        assert result.worker_order == ["w0", "w1"]
        assert 0 in result.combined_kv

    def test_clear(self, manager):
        """Clear should reset manager state."""
        ctx = self._make_supervisor_ctx()
        manager.set_supervisor_context(ctx)
        manager.prefill_worker(self._make_worker_output("w0", 2))
        manager.clear()
        assert len(manager.worker_kv_states) == 0


# ---------------------------------------------------------------------------
# TDAResult
# ---------------------------------------------------------------------------

class TestTDAResult:
    """Tests for TDAResult dataclass."""

    def test_creation(self):
        result = TDAResult(
            combined_kv={},
            total_tokens=5,
            worker_order=["w0", "w1"],
        )
        assert len(result.worker_order) == 2
        assert result.total_tokens == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
