"""
Tests for AgentMesh configuration management.

Covers the locked defaults (τ=0.8, initial_theta=512, max_parallel_branches=3),
from_dict / to_dict / from_yaml round-trips over the dtr/bpp/des keys, and
environment-variable overrides.
"""

import os

import pytest

from agentmesh.core.config import (
    DTRConfig,
    BPPConfig,
    DESConfig,
    LLMConfig,
    AgentMeshConfig,
    load_config,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
def test_dtr_default_tau():
    assert DTRConfig().confidence_threshold == 0.8
    assert DTRConfig().retrieval_depth == 10
    assert DTRConfig().hotspot_k == 3
    assert DTRConfig().enabled is True


def test_bpp_default_branches():
    assert BPPConfig().max_parallel_branches == 3
    assert BPPConfig().num_layers == 32
    assert BPPConfig().num_heads == 32
    assert BPPConfig().head_dim == 128
    assert BPPConfig().rope_base == 10000.0


def test_des_default_theta():
    cfg = DESConfig()
    assert cfg.initial_theta == 512
    assert cfg.theta_min == 64
    assert cfg.theta_max == 4096


def test_llm_default_endpoint():
    assert LLMConfig().endpoint == "http://localhost:8000/v1"


def test_top_level_config_composes_sub_configs():
    cfg = AgentMeshConfig()
    assert isinstance(cfg.dtr, DTRConfig)
    assert isinstance(cfg.bpp, BPPConfig)
    assert isinstance(cfg.des, DESConfig)
    assert isinstance(cfg.llm, LLMConfig)


# ---------------------------------------------------------------------------
# from_dict / to_dict round-trip
# ---------------------------------------------------------------------------
def test_from_dict_overrides():
    data = {
        "dtr": {"confidence_threshold": 0.9, "hotspot_k": 5},
        "bpp": {"max_parallel_branches": 8},
        "des": {"initial_theta": 1024, "theta_max": 8192},
        "llm": {"endpoint": "http://localhost:9999/v1", "model": "test-model"},
        "log_level": "DEBUG",
        "device": "cpu",
    }
    cfg = AgentMeshConfig.from_dict(data)
    assert cfg.dtr.confidence_threshold == 0.9
    assert cfg.dtr.hotspot_k == 5
    assert cfg.bpp.max_parallel_branches == 8
    assert cfg.des.initial_theta == 1024
    assert cfg.des.theta_max == 8192
    assert cfg.llm.endpoint == "http://localhost:9999/v1"
    assert cfg.log_level == "DEBUG"
    assert cfg.device == "cpu"


def test_to_dict_has_all_mechanism_keys():
    cfg = AgentMeshConfig()
    d = cfg.to_dict()
    assert set(["dtr", "bpp", "des", "llm"]).issubset(d.keys())
    assert d["dtr"]["confidence_threshold"] == 0.8
    assert d["bpp"]["max_parallel_branches"] == 3
    assert d["des"]["initial_theta"] == 512


def test_dict_round_trip_preserves_values():
    cfg = AgentMeshConfig()
    cfg.dtr.confidence_threshold = 0.92
    cfg.bpp.max_parallel_branches = 6
    cfg.des.initial_theta = 256
    restored = AgentMeshConfig.from_dict(cfg.to_dict())
    assert restored.dtr.confidence_threshold == 0.92
    assert restored.bpp.max_parallel_branches == 6
    assert restored.des.initial_theta == 256


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------
def test_yaml_round_trip(tmp_path):
    cfg = AgentMeshConfig()
    cfg.dtr.confidence_threshold = 0.88
    cfg.bpp.max_parallel_branches = 4
    cfg.des.initial_theta = 768

    path = tmp_path / "agentmesh.yaml"
    cfg.to_yaml(str(path))
    assert path.exists()

    loaded = AgentMeshConfig.from_yaml(str(path))
    assert loaded.dtr.confidence_threshold == 0.88
    assert loaded.bpp.max_parallel_branches == 4
    assert loaded.des.initial_theta == 768


# ---------------------------------------------------------------------------
# Environment-variable overrides
# ---------------------------------------------------------------------------
def test_env_overrides(monkeypatch):
    monkeypatch.setenv("AGENTMESH_LLM_ENDPOINT", "http://localhost:7777/v1")
    monkeypatch.setenv("AGENTMESH_LLM_MODEL", "env-model")
    monkeypatch.setenv("AGENTMESH_CONFIDENCE_THRESHOLD", "0.77")
    monkeypatch.setenv("AGENTMESH_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("AGENTMESH_DEVICE", "cpu")

    cfg = AgentMeshConfig.from_env()
    assert cfg.llm.endpoint == "http://localhost:7777/v1"
    assert cfg.llm.model == "env-model"
    assert cfg.dtr.confidence_threshold == 0.77
    assert cfg.log_level == "WARNING"
    assert cfg.device == "cpu"


def test_load_config_defaults_without_file(monkeypatch):
    # Ensure no env overrides leak in.
    for var in [
        "AGENTMESH_LLM_ENDPOINT",
        "AGENTMESH_LLM_MODEL",
        "AGENTMESH_CONFIDENCE_THRESHOLD",
        "AGENTMESH_LOG_LEVEL",
    ]:
        monkeypatch.delenv(var, raising=False)
    cfg = load_config(config_path=None, use_env=True)
    assert cfg.dtr.confidence_threshold == 0.8
    assert cfg.des.initial_theta == 512


def test_load_config_file_takes_priority(tmp_path, monkeypatch):
    cfg = AgentMeshConfig()
    cfg.dtr.confidence_threshold = 0.91
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(str(path))

    monkeypatch.delenv("AGENTMESH_CONFIDENCE_THRESHOLD", raising=False)
    loaded = load_config(config_path=str(path), use_env=False)
    assert loaded.dtr.confidence_threshold == 0.91
