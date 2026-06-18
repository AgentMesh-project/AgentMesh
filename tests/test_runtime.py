"""
Tests for the AgentMesh runtime.

The runtime orchestrates DTR (produce), BPP (consume), and DES (overlap) over
the produce–transfer–consume dataflow abstraction. These tests construct the
runtime with mechanisms enabled/disabled, drive the ``process`` path with a
fake (no-network) LLM client, and check the statistics surface.

BPP construction requires torch (KV tensors), so configurations that enable BPP
are guarded with ``pytest.importorskip("torch")``; DTR and DES are pure-Python.
"""

import pytest

from agentmesh.core.config import AgentMeshConfig, LLMConfig
from agentmesh.core.runtime import (
    AgentMeshRuntime,
    AgentRequest,
    AgentResponse,
    StreamChunk,
)
from agentmesh.mechanisms.bpp import attention as bpp_attention

from tests.conftest import FakeLLMClient


def _cpu_config(dtr=True, bpp=False, des=True):
    cfg = AgentMeshConfig()
    cfg.dtr.enabled = dtr
    cfg.bpp.enabled = bpp
    cfg.des.enabled = des
    cfg.device = "cpu"
    cfg.log_level = "WARNING"
    cfg.llm = LLMConfig(endpoint="http://localhost:8000/v1")
    # Keep BPP tiny if it is enabled (only reachable when torch present).
    cfg.bpp.num_layers = 2
    cfg.bpp.num_heads = 2
    cfg.bpp.head_dim = 4
    return cfg


# ---------------------------------------------------------------------------
# Construction with mechanisms enabled / disabled
# ---------------------------------------------------------------------------
def test_construct_dtr_only(fake_llm_client):
    rt = AgentMeshRuntime(config=_cpu_config(dtr=True, bpp=False, des=False),
                          llm_client=fake_llm_client)
    assert rt.dtr_cache is not None
    assert rt.bpp_manager is None
    assert rt.des_controller is None


def test_construct_des_enabled(fake_llm_client):
    rt = AgentMeshRuntime(config=_cpu_config(dtr=False, bpp=False, des=True),
                          llm_client=fake_llm_client)
    assert rt.dtr_cache is None
    assert rt.des_controller is not None
    assert rt.des_controller.current_theta == 512


def test_construct_all_disabled(fake_llm_client):
    rt = AgentMeshRuntime(config=_cpu_config(dtr=False, bpp=False, des=False),
                          llm_client=fake_llm_client)
    assert rt.dtr_cache is None
    assert rt.bpp_manager is None
    assert rt.des_controller is None


def test_construct_with_bpp_requires_torch(fake_llm_client):
    pytest.importorskip("torch")
    rt = AgentMeshRuntime(config=_cpu_config(dtr=True, bpp=True, des=True),
                          llm_client=fake_llm_client)
    assert rt.bpp_manager is not None


def test_bpp_construction_without_torch_raises(fake_llm_client):
    if bpp_attention._TORCH_AVAILABLE:
        pytest.skip("torch present; guard not exercised")
    with pytest.raises(ImportError):
        AgentMeshRuntime(config=_cpu_config(dtr=True, bpp=True, des=True),
                         llm_client=fake_llm_client)


# ---------------------------------------------------------------------------
# process() path with a fake LLM client (no network)
# ---------------------------------------------------------------------------
def test_process_returns_response(run_async, fake_llm_client):
    rt = AgentMeshRuntime(config=_cpu_config(dtr=True, bpp=False, des=False),
                          llm_client=fake_llm_client)
    req = AgentRequest(request_id="r1", agent_id="a1", prompt="what is machine learning")
    resp = run_async(rt.process(req))
    assert isinstance(resp, AgentResponse)
    assert resp.request_id == "r1"
    assert resp.agent_id == "a1"
    assert "machine learning" in resp.content
    assert resp.latency_ms >= 0.0


def test_process_cold_then_activation(run_async):
    """First lookup is cold; a high-similarity repeat activates DTR."""
    client = FakeLLMClient(reply="cached content body")
    cfg = _cpu_config(dtr=True, bpp=False, des=False)
    cfg.dtr.confidence_threshold = 0.0  # force activation on any populated cache
    rt = AgentMeshRuntime(config=cfg, llm_client=client)

    r1 = run_async(rt.process(AgentRequest(request_id="r1", agent_id="a", prompt="alpha query")))
    assert r1.dtr_activated is False  # empty cache

    r2 = run_async(rt.process(AgentRequest(request_id="r2", agent_id="a", prompt="alpha query two")))
    assert r2.dtr_activated is True
    assert r2.projection_intensity >= 0.0


def test_process_without_dtr_calls_llm(run_async, fake_llm_client):
    rt = AgentMeshRuntime(config=_cpu_config(dtr=False, bpp=False, des=False),
                          llm_client=fake_llm_client)
    resp = run_async(rt.process(AgentRequest(request_id="r", agent_id="a", prompt="hello")))
    assert resp.dtr_activated is False
    assert "machine learning" in resp.content  # fake reply
    assert fake_llm_client.chat.completions.calls  # the LLM was actually called


# ---------------------------------------------------------------------------
# batch_process (BPP scatter-gather; falls back sequentially without manager)
# ---------------------------------------------------------------------------
def test_batch_process_sequential_fallback(run_async, fake_llm_client):
    rt = AgentMeshRuntime(config=_cpu_config(dtr=True, bpp=False, des=False),
                          llm_client=fake_llm_client)
    reqs = [
        AgentRequest(request_id=f"r{i}", agent_id="a", prompt=f"query {i}")
        for i in range(3)
    ]
    resps = run_async(rt.batch_process(reqs))
    assert len(resps) == 3
    assert [r.request_id for r in resps] == ["r0", "r1", "r2"]


# ---------------------------------------------------------------------------
# get_stats surface
# ---------------------------------------------------------------------------
def test_get_stats_keys(run_async, fake_llm_client):
    rt = AgentMeshRuntime(config=_cpu_config(dtr=True, bpp=False, des=True),
                          llm_client=fake_llm_client)
    run_async(rt.process(AgentRequest(request_id="r", agent_id="a", prompt="some query")))
    stats = rt.get_stats()
    assert "total_requests" in stats
    assert "dtr_activations" in stats
    assert "cold_retrievals" in stats
    # DTR sub-stats are surfaced under the "dtr" key when DTR is enabled.
    assert "dtr" in stats
    assert "total_lookups" in stats["dtr"]
    assert "des" in stats


def test_reset_clears_stats(run_async, fake_llm_client):
    rt = AgentMeshRuntime(config=_cpu_config(dtr=True, bpp=False, des=True),
                          llm_client=fake_llm_client)
    run_async(rt.process(AgentRequest(request_id="r", agent_id="a", prompt="q")))
    assert rt.get_stats()["total_requests"] == 1
    rt.reset()
    assert rt.get_stats()["total_requests"] == 0


# ---------------------------------------------------------------------------
# StreamChunk / DES streaming path
# ---------------------------------------------------------------------------
def test_stream_yields_chunks(run_async):
    client = FakeLLMClient(stream_pieces=["a " * 300, "b " * 300, "tail"])
    rt = AgentMeshRuntime(config=_cpu_config(dtr=False, bpp=False, des=True),
                          llm_client=client)

    async def _collect():
        chunks = []
        async for ch in rt.stream(AgentRequest(request_id="s", agent_id="a", prompt="go")):
            chunks.append(ch)
        return chunks

    chunks = run_async(_collect())
    assert chunks
    assert all(isinstance(c, StreamChunk) for c in chunks)
    assert chunks[-1].is_final is True
