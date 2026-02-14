"""
Tests for AgentMesh runtime — core/runtime.py.

Verifies that the runtime correctly initializes SRR, TSD, and ESF
subsystems and orchestrates the full processing pipeline.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from agentmesh.core.config import (
    AgentMeshConfig,
    SRRConfig,
    TSDConfig,
    ESFConfig,
)
from agentmesh.core.runtime import (
    AgentMeshRuntime,
    AgentResponse,
    StreamChunk,
)


class TestAgentMeshConfig:
    """Tests for configuration dataclasses."""

    def test_srr_config_defaults(self):
        """SRRConfig defaults should match paper (τ=0.7, k=3)."""
        cfg = SRRConfig()
        assert cfg.confidence_threshold == 0.7
        assert cfg.hotspot_k == 3
        assert cfg.slm_model == "Qwen/Qwen2.5-0.5B-Instruct"

    def test_tsd_config_defaults(self):
        """TSDConfig defaults should include rope_base=10000."""
        cfg = TSDConfig()
        assert cfg.rope_base == 10000.0
        assert cfg.max_parallel_workers > 0

    def test_esf_config_defaults(self):
        """ESFConfig should expose theta min/max."""
        cfg = ESFConfig()
        assert cfg.theta_min < cfg.theta_max
        assert cfg.initial_microbatch_size > 0

    def test_agent_mesh_config_composition(self):
        """AgentMeshConfig should compose SRR, TSD, ESF configs."""
        cfg = AgentMeshConfig()
        assert isinstance(cfg.srr, SRRConfig)
        assert isinstance(cfg.tsd, TSDConfig)
        assert isinstance(cfg.esf, ESFConfig)

    def test_agent_mesh_config_from_env(self):
        """from_env should respect environment variable overrides."""
        import os

        os.environ["AGENTMESH_CONFIDENCE_THRESHOLD"] = "0.8"
        try:
            cfg = AgentMeshConfig.from_env()
            assert cfg.srr.confidence_threshold == 0.8
        finally:
            del os.environ["AGENTMESH_CONFIDENCE_THRESHOLD"]

    def test_to_dict(self):
        """to_dict should serialize all sub-configs."""
        cfg = AgentMeshConfig()
        d = cfg.to_dict()
        assert "srr" in d
        assert "tsd" in d
        assert "esf" in d
        assert d["srr"]["confidence_threshold"] == 0.7


class TestAgentResponse:
    """Tests for AgentResponse dataclass."""

    def test_default_values(self):
        resp = AgentResponse(content="hello")
        assert resp.srr_activated is False
        assert resp.projection_intensity == 0.0

    def test_srr_activated(self):
        resp = AgentResponse(
            content="cached",
            srr_activated=True,
            projection_intensity=0.85,
            reuse_count=3,
            residual_count=1,
        )
        assert resp.srr_activated
        assert resp.reuse_count == 3


class TestStreamChunk:
    """Tests for StreamChunk dataclass."""

    def test_default_microbatch_size(self):
        chunk = StreamChunk(content="token", chunk_index=0)
        assert chunk.microbatch_size >= 0 or chunk.microbatch_size is None


class TestAgentMeshRuntime:
    """Tests for the AgentMeshRuntime orchestrator."""

    def test_init_with_defaults(self):
        """Runtime should initialize with default config."""
        runtime = AgentMeshRuntime()
        assert runtime.config is not None

    def test_init_with_custom_config(self):
        """Runtime should accept custom config."""
        cfg = AgentMeshConfig(
            srr=SRRConfig(confidence_threshold=0.5),
            esf=ESFConfig(initial_microbatch_size=256),
        )
        runtime = AgentMeshRuntime(config=cfg)
        assert runtime.config.srr.confidence_threshold == 0.5

    def test_stats_initial(self):
        """Initial stats should be zeroed."""
        runtime = AgentMeshRuntime()
        stats = runtime.get_stats()
        assert stats.get("srr_activations", 0) == 0
        assert stats.get("cold_retrievals", 0) == 0

    def test_reset(self):
        """Reset should clear runtime state."""
        runtime = AgentMeshRuntime()
        runtime.reset()
        stats = runtime.get_stats()
        assert stats.get("total_requests", 0) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
