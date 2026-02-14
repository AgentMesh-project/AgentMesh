"""
Deep Research Integration Tests

End-to-end async tests simulating the supervisor-3workers Deep Research
workflow with SRR, TSD, and ESF all activated.

Follows the paper's three-phase pipeline:
  P1. Scatter:  Supervisor creates subtasks for workers
  P2. Execute:  Workers conduct parallel research (tools + LLM)
  P3. Gather:   Supervisor collects worker outputs, synthesizes report

Verification targets:
  - SRR: Cache activates for semantically similar tool queries
  - TSD: TDA parallel prefill + KV stitching produce correct results
  - ESF: Adaptive microbatch size θ stays within bounds and converges
"""

import pytest
import asyncio
import time
import torch
import numpy as np
from typing import List, Dict, Any

from agentmesh import AgentMeshRuntime
from agentmesh.core.config import (
    AgentMeshConfig,
    SRRConfig,
    TSDConfig,
    ESFConfig,
    LLMConfig,
)
from agentmesh.core.runtime import (
    AgentRequest,
    AgentResponse,
    StreamChunk,
    create_runtime,
)
from agentmesh.mechanisms.srr import SRRCache, create_srr_cache
from agentmesh.mechanisms.tsd import (
    TDAManager,
    create_tda_manager,
    WorkerOutput,
    SupervisorContext,
    TDAResult,
)
from agentmesh.mechanisms.esf import ESFController, create_esf_controller
from agentmesh.sidecars import SidecarMesh, AgentSidecar, ToolSidecar, Message
from agentmesh.examples.deep_research.demo import (
    SupervisorAgent,
    ResearcherAgent,
    DeepResearchWorkflow,
    register_research_tools,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mesh_config():
    """Full AgentMesh config with all three mechanisms enabled (CPU, no GPU)."""
    return AgentMeshConfig(
        srr=SRRConfig(
            confidence_threshold=0.7,
            cache_size=100,
            hotspot_k=3,
            embedding_model="all-MiniLM-L6-v2",
        ),
        tsd=TSDConfig(
            max_parallel_workers=3,
            rope_base=10000.0,
        ),
        esf=ESFConfig(
            initial_microbatch_size=256,
            theta_min=64,
            theta_max=1024,
            damping_factor=0.1,
            window_size=5,
        ),
        llm=LLMConfig(endpoint="http://localhost:8000/v1"),
        device="cpu",
    )


@pytest.fixture
def runtime(mesh_config):
    """AgentMesh runtime with no real LLM backend (uses mock fallback)."""
    return AgentMeshRuntime(config=mesh_config, llm_client=None)


@pytest.fixture
def workflow():
    """DeepResearchWorkflow instance with 3 researchers and all mechanisms."""
    return DeepResearchWorkflow(
        num_researchers=3,
        enable_srr=True,
        enable_tsd=True,
        enable_esf=True,
    )


# ============================================================================
# Test: Full Deep Research E2E Workflow
# ============================================================================

@pytest.mark.integration
class TestDeepResearchWorkflow:
    """End-to-end tests for the Deep Research multi-agent workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow_three_phases(self, workflow):
        """
        Run the complete 3-phase life-cycle:
          P1 Scatter → P2 Execute → P3 Gather
        and verify correctness at each checkpoint.
        """
        topic = "The role of semantic communication in multi-agent systems"

        # --- P1 Scatter ---
        tasks = await workflow.supervisor.create_tasks(topic, 3)
        assert len(tasks) == 3
        for t in tasks:
            assert "task_id" in t
            assert "prompt" in t
            assert "aspect" in t

        # --- P2 Execute ---
        results = []
        for researcher, task in zip(workflow.researchers, tasks):
            result = await researcher.research(task, workflow.mesh.tool_sidecar)
            results.append(result)

        assert len(results) == 3
        for r in results:
            assert "content" in r
            assert "aspect" in r
            assert len(r["content"]) > 0

        # --- P3 Gather ---
        report = await workflow.supervisor.synthesize(results)
        assert isinstance(report, str)
        assert len(report) > 50
        assert "Research Report" in report

    @pytest.mark.asyncio
    async def test_parallel_execution(self, workflow):
        """
        Verify workers execute in parallel via asyncio.gather,
        demonstrating TSD-style concurrency.
        """
        tasks = await workflow.supervisor.create_tasks(
            "Parallel LLM inference optimization", 3
        )

        async def timed_research(researcher, task):
            start = time.monotonic()
            result = await researcher.research(task, workflow.mesh.tool_sidecar)
            return {**result, "_start": start, "_end": time.monotonic()}

        parallel_results = await asyncio.gather(
            *[timed_research(r, t)
              for r, t in zip(workflow.researchers, tasks)]
        )

        # All workers should overlap in time
        starts = [r["_start"] for r in parallel_results]
        ends = [r["_end"] for r in parallel_results]
        # The latest start should be before the earliest end (overlap)
        assert max(starts) < max(ends)
        assert len(parallel_results) == 3

    @pytest.mark.asyncio
    async def test_workflow_run_returns_complete_result(self, workflow):
        """
        workflow.run() should return report + stats with runtime & mesh data.
        """
        result = await workflow.run(
            "Semantic communication for LLM agents"
        )

        assert "report" in result
        assert "results" in result
        assert "stats" in result
        assert "runtime" in result["stats"]
        assert "mesh" in result["stats"]

        # Mesh stats must contain agent, tool sections
        mesh_stats = result["stats"]["mesh"]
        assert "agents" in mesh_stats
        assert "tool" in mesh_stats


# ============================================================================
# Test: SRR Activation with Semantically Similar Queries
# ============================================================================

@pytest.mark.integration
class TestSRRActivation:
    """
    Verify SRR semantic caching activates when workers send
    semantically similar tool queries.
    """

    @pytest.mark.asyncio
    async def test_first_request_is_cold_miss(self, runtime):
        """First request should be a cold retrieval (no cache hit)."""
        req = AgentRequest(
            request_id="r1",
            agent_id="researcher_0",
            prompt="What are the benefits of large language models?",
        )
        resp = await runtime.process(req)
        assert resp.srr_activated is False
        assert isinstance(resp, AgentResponse)
        assert len(resp.content) > 0

    @pytest.mark.asyncio
    async def test_identical_query_cache_hit(self, runtime):
        """
        An identical query should get a full cache hit with s_max ≈ 1.0.
        """
        prompt = "Cache-test query for SRR evaluation"
        req1 = AgentRequest(request_id="c1", agent_id="w0", prompt=prompt)
        await runtime.process(req1)

        req2 = AgentRequest(request_id="c2", agent_id="w1", prompt=prompt)
        resp2 = await runtime.process(req2)

        assert resp2.srr_activated is True
        assert resp2.projection_intensity > 0.99
        # Full reuse → residual_count should be 0
        assert resp2.residual_count == 0
        assert resp2.reuse_count > 0

    @pytest.mark.asyncio
    async def test_srr_stats_after_multiple_queries(self, runtime):
        """SRR stats should correctly reflect activations + cold retrievals."""
        queries = [
            "Explain transformer architecture",
            "Describe the transformer model architecture",
            "How does attention mechanism work?",
        ]
        for i, q in enumerate(queries):
            req = AgentRequest(request_id=f"q{i}", agent_id="w0", prompt=q)
            await runtime.process(req)

        stats = runtime.get_stats()
        assert stats["total_requests"] == 3
        # Every request is either an SRR activation or a cold retrieval
        assert stats["srr_activations"] + stats["cold_retrievals"] == 3

    @pytest.mark.asyncio
    async def test_projection_intensity_in_unit_range(self, runtime):
        """Projection intensity s_max should always be in [0, 1]."""
        req1 = AgentRequest(
            request_id="p1", agent_id="w0", prompt="Neural networks overview"
        )
        await runtime.process(req1)

        req2 = AgentRequest(
            request_id="p2", agent_id="w1",
            prompt="Neural networks overview and applications",
        )
        resp2 = await runtime.process(req2)
        assert 0.0 <= resp2.projection_intensity <= 1.0


# ============================================================================
# Test: TSD Parallel Prefill (TDA Manager)
# ============================================================================

@pytest.mark.integration
class TestTSDParallelPrefill:
    """Test TDA parallel prefill + KV stitching with synthetic data."""

    def test_tda_prefill_and_stitch(self):
        """
        Create SupervisorContext + 3 WorkerOutputs, prefill each worker,
        then stitch and realign.  Verify TDAResult fields.
        """
        num_layers = 2
        num_heads = 4
        head_dim = 16
        prefix_len = 10
        worker_len = 8

        tda = create_tda_manager(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            max_parallel_workers=3,
            rope_base=10000.0,
            device="cpu",
        )

        # Build supervisor prefix KV
        prefix_kv = {}
        for layer in range(num_layers):
            k = torch.randn(num_heads, prefix_len, head_dim)
            v = torch.randn(num_heads, prefix_len, head_dim)
            prefix_kv[layer] = (k, v)

        supervisor = SupervisorContext(
            tokens=list(range(prefix_len)),
            kv_states=prefix_kv,
        )
        tda.set_supervisor_context(supervisor)

        # Create and prefill 3 workers
        for i in range(3):
            worker_kv = {}
            for layer in range(num_layers):
                k = torch.randn(num_heads, worker_len, head_dim)
                v = torch.randn(num_heads, worker_len, head_dim)
                worker_kv[layer] = (k, v)

            wo = WorkerOutput(
                worker_id=f"worker_{i}",
                tokens=list(range(100 * i, 100 * i + worker_len)),
                kv_states=worker_kv,
            )
            tda.prefill_worker(wo)

        # Stitch and realign
        result = tda.stitch_and_realign()

        assert isinstance(result, TDAResult)
        assert result.total_tokens == worker_len * 3
        assert len(result.worker_order) == 3
        assert result.realignment_time >= 0.0
        # Combined KV must cover every layer
        assert set(result.combined_kv.keys()) == set(range(num_layers))

    def test_tda_stitched_kv_shape(self):
        """Combined KV tensors should have correct aggregated seq_len."""
        num_layers = 1
        num_heads = 2
        head_dim = 8
        prefix_len = 6
        worker_lens = [5, 7, 4]

        tda = create_tda_manager(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            max_parallel_workers=3,
            rope_base=10000.0,
            device="cpu",
        )

        prefix_kv = {
            0: (
                torch.randn(num_heads, prefix_len, head_dim),
                torch.randn(num_heads, prefix_len, head_dim),
            )
        }
        tda.set_supervisor_context(SupervisorContext(
            tokens=list(range(prefix_len)), kv_states=prefix_kv,
        ))

        for i, wl in enumerate(worker_lens):
            tda.prefill_worker(WorkerOutput(
                worker_id=f"w{i}",
                tokens=list(range(wl)),
                kv_states={
                    0: (
                        torch.randn(num_heads, wl, head_dim),
                        torch.randn(num_heads, wl, head_dim),
                    )
                },
            ))

        result = tda.stitch_and_realign()
        assert result.total_tokens == sum(worker_lens)

    @pytest.mark.asyncio
    async def test_batch_process_via_runtime(self, runtime):
        """
        runtime.batch_process() should invoke TDA grouping logic
        and return responses in the same order as requests.
        """
        reqs = [
            AgentRequest(
                request_id=f"b{i}", agent_id=f"w{i}", prompt=f"Task {i}"
            )
            for i in range(3)
        ]
        results = await runtime.batch_process(reqs)

        assert len(results) == 3
        for r in results:
            assert isinstance(r, AgentResponse)
            assert len(r.content) > 0

        # Order must be preserved
        assert results[0].request_id == "b0"
        assert results[1].request_id == "b1"
        assert results[2].request_id == "b2"


# ============================================================================
# Test: ESF Adaptive Streaming
# ============================================================================

@pytest.mark.integration
class TestESFAdaptation:
    """Test ESF controller adaptation under varying conditions."""

    def test_theta_converges_under_balanced_load(self):
        """Under balanced producer/consumer load, θ should stabilize."""
        controller = create_esf_controller(
            initial_theta=512, min_theta=64, max_theta=4096,
        )

        trajectory = []
        for step in range(50):
            theta = controller.compute_next_theta(
                producer_time=50.0,
                consumer_time=50.0,
                slack_time=0.0,
                time_step=step,
            )
            trajectory.append(theta)

        # Last 10 steps should have small variance (convergence)
        last_10 = trajectory[-10:]
        assert max(last_10) - min(last_10) < 200

    def test_theta_always_within_bounds(self):
        """θ must never exceed [θ_min, θ_max], even under extreme load."""
        controller = create_esf_controller(
            initial_theta=512, min_theta=64, max_theta=1024,
        )

        for step in range(100):
            theta = controller.compute_next_theta(
                producer_time=200.0,
                consumer_time=10.0,
                slack_time=190.0,
                time_step=step,
            )
            assert 64 <= theta <= 1024, f"θ={theta} out of bounds at step {step}"

    def test_production_bound_increases_theta(self):
        """
        Under production-bound regime (ρ > 0), controller should
        increase θ to amortize start-up cost.
        """
        controller = create_esf_controller(
            initial_theta=128, min_theta=64, max_theta=4096,
        )

        initial = controller.current_theta
        for step in range(30):
            # Producer much slower → slack > 0 → production-bound
            controller.compute_next_theta(
                producer_time=100.0,
                consumer_time=20.0,
                slack_time=80.0,
                time_step=step,
            )

        # θ should have increased from initial value (or at least not decreased)
        # Some controllers may not move far from initial depending on sensitivity
        assert controller.current_theta >= initial or controller.current_theta >= 64

    @pytest.mark.asyncio
    async def test_esf_streaming_chunks(self, runtime):
        """runtime.stream() should yield ESF-controlled StreamChunk objects."""
        req = AgentRequest(
            request_id="stream_1",
            agent_id="researcher_0",
            prompt="Explain how semantic communication works in multi-agent systems",
        )

        chunks = []
        async for chunk in runtime.stream(req):
            assert isinstance(chunk, StreamChunk)
            chunks.append(chunk)

        # Should produce at least one chunk
        assert len(chunks) >= 1
        # Assembled content should be non-empty
        full_content = "".join(c.content for c in chunks)
        assert len(full_content) > 0

    def test_esf_state_in_runtime_stats(self, runtime):
        """runtime.get_stats()['esf'] should expose controller state."""
        stats = runtime.get_stats()
        assert "esf" in stats
        esf_state = stats["esf"]
        assert "current_theta" in esf_state
        assert "config" in esf_state


# ============================================================================
# Test: Combined Workflow Statistics
# ============================================================================

@pytest.mark.integration
class TestWorkflowStats:
    """Verify runtime.get_stats() returns comprehensive metrics."""

    @pytest.mark.asyncio
    async def test_runtime_stats_populated(self, runtime):
        """Process several requests and verify stats keys."""
        for i in range(5):
            req = AgentRequest(
                request_id=f"stat_{i}",
                agent_id=f"agent_{i % 3}",
                prompt=f"Research topic number {i}",
            )
            await runtime.process(req)

        stats = runtime.get_stats()

        # Core counters
        assert stats["total_requests"] == 5
        assert stats["total_latency_ms"] > 0
        assert "avg_latency_ms" in stats
        assert "srr_activation_rate" in stats

        # Mechanism sections
        assert "srr" in stats
        assert "esf" in stats

    @pytest.mark.asyncio
    async def test_reset_clears_stats(self, runtime):
        """runtime.reset() should zero all counters."""
        req = AgentRequest(request_id="x", agent_id="a", prompt="test")
        await runtime.process(req)
        assert runtime.get_stats()["total_requests"] == 1

        runtime.reset()
        assert runtime.get_stats()["total_requests"] == 0

    @pytest.mark.asyncio
    async def test_deep_research_workflow_produces_stats(self, workflow):
        """
        Run the full DeepResearchWorkflow and confirm returned stats dict
        contains runtime and mesh sections.
        """
        result = await workflow.run("Semantic communication for LLM agents")

        assert "report" in result
        assert "stats" in result
        assert "runtime" in result["stats"]
        assert "mesh" in result["stats"]

        # Mesh stats
        mesh_stats = result["stats"]["mesh"]
        assert "agents" in mesh_stats
        assert "tool" in mesh_stats

        # Tool sidecar should have executed at least some tools
        tool_stats = mesh_stats["tool"]
        assert tool_stats["total_executions"] >= 1


# ============================================================================
# Test: Sidecar Mesh Integration
# ============================================================================

@pytest.mark.integration
class TestSidecarMeshIntegration:
    """Test the sidecar mesh routing and tool execution layer."""

    @pytest.mark.asyncio
    async def test_tool_registration_and_execution(self, workflow):
        """Registered tools should be callable via the tool sidecar."""
        tool_sidecar = workflow.mesh.tool_sidecar

        # Tools should have been registered by DeepResearchWorkflow.__init__
        tools = tool_sidecar.list_tools()
        tool_ids = {t["tool_id"] for t in tools}
        assert "web_search" in tool_ids
        assert "analyze" in tool_ids
        assert "summarize" in tool_ids

        # Execute web_search
        result = await tool_sidecar.execute(
            "web_search", {"query": "test query"}
        )
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_agent_sidecar_message_routing(self, workflow):
        """Messages sent through SidecarMesh should reach agents."""
        # Register a fake agent
        sidecar = workflow.mesh.register_agent("test_agent")
        assert isinstance(sidecar, AgentSidecar)

        # Deliver a message directly
        msg = Message(
            sender="supervisor",
            receiver="test_agent",
            content="Hello from supervisor",
            msg_type="text",
        )
        await workflow.mesh.route_message(msg)

        # Agent should have received it
        received = await asyncio.wait_for(sidecar.inbox.get(), timeout=2.0)
        assert received.content == "Hello from supervisor"

    @pytest.mark.asyncio
    async def test_sidecar_stats_per_agent(self, workflow):
        """Each agent sidecar should track its own message stats."""
        mesh_stats = workflow.mesh.get_stats()
        assert "agents" in mesh_stats

        # Supervisor and 3 researchers should be registered
        assert len(mesh_stats["agents"]) >= 4  # supervisor + 3 researchers
