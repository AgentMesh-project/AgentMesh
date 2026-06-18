"""
AgentMesh Runtime

Main runtime that orchestrates the three core mechanisms over the
produce–transfer–consume dataflow abstraction:
  - DTR (Delta Tool Retrieval)          — the "produce" step (removes REDUNDANCY)
  - BPP (Branch-Parallel Prefill)       — the "consume" step (removes BARRIER)
  - DES (Dynamic-Equilibrium Streaming) — overlaps the stages (drives slack δ→0)

Provides the high-level API for multi-agent workflows built on AutoGen with
vLLM as the LLM backend.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable, AsyncIterator
from dataclasses import dataclass, field
import time

from agentmesh.core.config import AgentMeshConfig, load_config
from agentmesh.mechanisms.dtr import DTRCache, create_dtr_cache
from agentmesh.mechanisms.bpp import BranchParallelManager, create_bpp_manager
from agentmesh.mechanisms.des import DESController, create_des_controller

logger = logging.getLogger(__name__)


@dataclass
class AgentRequest:
    """A request from an agent to the LLM or tool (the producer side)."""
    request_id: str
    agent_id: str
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResponse:
    """A response from the LLM to an agent (the consumer side)."""
    content: str
    request_id: str = ""
    agent_id: str = ""
    tokens_generated: int = 0
    latency_ms: float = 0.0
    dtr_activated: bool = False
    projection_intensity: float = 0.0
    reuse_count: int = 0
    residual_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamChunk:
    """A streaming chunk of a response (DES-controlled)."""
    content: str
    chunk_index: int = 0
    request_id: str = ""
    is_final: bool = False
    tokens: int = 0
    chunk_size: int = 0


class AgentMeshRuntime:
    """
    Main AgentMesh runtime orchestrating DTR, BPP, and DES.

    Provides:
    - Delta tool retrieval via DTR for removing REDUNDANCY in the produce step
    - Branch-parallel KV prefill via BPP for removing the BARRIER in the
      consume step of the supervisor-worker topology
    - Dynamic-equilibrium chunk streaming via DES for overlapping the
      produce/transfer/consume stages

    Example:
        ```python
        runtime = AgentMeshRuntime()

        # Process a single request with DTR
        response = await runtime.process(request)

        # Process streaming response with DES
        async for chunk in runtime.stream(request):
            print(chunk.content, end="", flush=True)

        # Batch process multiple worker branches via BPP
        responses = await runtime.batch_process(requests)
        ```
    """

    def __init__(
        self,
        config: Optional[AgentMeshConfig] = None,
        config_path: Optional[str] = None,
        llm_client: Optional[Any] = None,
    ):
        """
        Initialize AgentMesh runtime.

        Args:
            config: Configuration object (takes priority).
            config_path: Path to config file.
            llm_client: Optional pre-configured LLM client.
        """
        # Load configuration
        if config:
            self.config = config
        else:
            self.config = load_config(config_path)

        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))

        # Initialize mechanisms
        self._init_dtr()
        self._init_bpp()
        self._init_des()

        # LLM client
        self.llm_client = llm_client
        self._init_llm_client()

        # Statistics
        self.stats = {
            "total_requests": 0,
            "dtr_activations": 0,
            "cold_retrievals": 0,
            "total_tokens_processed": 0,
            "total_tokens_reused": 0,
            "total_latency_ms": 0.0,
            "parallel_batches": 0,
        }

        logger.info("AgentMesh runtime initialized")

    def _init_dtr(self):
        """Initialize DTR mechanism (the produce step)."""
        if self.config.dtr.enabled:
            self.dtr_cache = create_dtr_cache(
                confidence_threshold=self.config.dtr.confidence_threshold,
                cache_size=self.config.dtr.cache_size,
                embedding_model=self.config.dtr.embedding_model,
            )
            logger.info(
                f"DTR enabled: τ={self.config.dtr.confidence_threshold}, "
                f"k={self.config.dtr.hotspot_k}"
            )
        else:
            self.dtr_cache = None
            logger.info("DTR disabled")

    def _init_bpp(self):
        """Initialize BPP/BPA mechanism (the consume step)."""
        if self.config.bpp.enabled:
            self.bpp_manager = create_bpp_manager(
                num_layers=self.config.bpp.num_layers,
                num_heads=self.config.bpp.num_heads,
                head_dim=self.config.bpp.head_dim,
                max_parallel_branches=self.config.bpp.max_parallel_branches,
                rope_base=self.config.bpp.rope_base,
                device=self.config.device,
            )
            logger.info(
                f"BPP enabled: max_branches={self.config.bpp.max_parallel_branches}"
            )
        else:
            self.bpp_manager = None
            logger.info("BPP disabled")

    def _init_des(self):
        """Initialize DES mechanism (stage overlap)."""
        if self.config.des.enabled:
            self.des_controller = create_des_controller(
                initial_theta=self.config.des.initial_theta,
                min_theta=self.config.des.theta_min,
                max_theta=self.config.des.theta_max,
                damping_factor=self.config.des.damping_factor,
                window_size=self.config.des.window_size,
            )
            logger.info(f"DES enabled: θ₀={self.config.des.initial_theta}")
        else:
            self.des_controller = None
            logger.info("DES disabled")

    def _init_llm_client(self):
        """Initialize LLM client if not provided."""
        if self.llm_client is None:
            try:
                from openai import AsyncOpenAI

                self.llm_client = AsyncOpenAI(
                    base_url=self.config.llm.endpoint,
                    api_key=self.config.llm.api_key or "EMPTY",
                )
                logger.info(f"LLM client initialized: {self.config.llm.endpoint}")
            except ImportError:
                logger.warning("OpenAI client not available, LLM calls will fail")
                self.llm_client = None

    async def process(self, request: AgentRequest) -> AgentResponse:
        """
        Process a single agent request with DTR delta retrieval.

        DTR flow (the produce step):
        1. Embed query → find nearest cache entry
        2. Compute projection intensity s_max
        3. If s_max ≥ τ: reuse ⌊s_max · n⌋ items (the base), fetch the delta
        4. If s_max < τ: cold retrieval (full tool call)

        Args:
            request: Agent request to process.

        Returns:
            Agent response with DTR metadata.
        """
        start_time = time.time()
        self.stats["total_requests"] += 1

        # DTR: Check cache
        cache_result = None
        if self.dtr_cache:
            cache_result = self.dtr_cache.lookup(request.prompt)

            if cache_result.hit:
                self.stats["dtr_activations"] += 1

                # Get reused items (the cached base) from cache
                reused_items = self.dtr_cache.get_reused_items(cache_result)

                # If high confidence and all items reused, skip LLM call
                if cache_result.residual_count == 0:
                    latency = (time.time() - start_time) * 1000
                    self.stats["total_latency_ms"] += latency
                    self.stats["total_tokens_reused"] += cache_result.reuse_count

                    return AgentResponse(
                        request_id=request.request_id,
                        agent_id=request.agent_id,
                        content="\n".join(reused_items),
                        dtr_activated=True,
                        projection_intensity=cache_result.projection_intensity,
                        reuse_count=cache_result.reuse_count,
                        residual_count=0,
                        latency_ms=latency,
                    )

                # Partial hit: use reformulated query for the semantic delta
                prompt_to_send = cache_result.reformulated_query or request.prompt
            else:
                self.stats["cold_retrievals"] += 1
                prompt_to_send = request.prompt
        else:
            prompt_to_send = request.prompt

        # Process with LLM
        response_content = await self._call_llm(prompt_to_send)

        # Combine reused base + new delta for partial hits
        if cache_result and cache_result.hit and cache_result.reuse_count > 0:
            reused_items = self.dtr_cache.get_reused_items(cache_result)
            response_content = "\n".join(reused_items) + "\n" + response_content

        # Store in cache
        if self.dtr_cache:
            items = response_content.split("\n")
            self.dtr_cache.store(request.prompt, items)

        latency = (time.time() - start_time) * 1000
        self.stats["total_latency_ms"] += latency

        return AgentResponse(
            request_id=request.request_id,
            agent_id=request.agent_id,
            content=response_content,
            dtr_activated=cache_result.hit if cache_result else False,
            projection_intensity=(
                cache_result.projection_intensity if cache_result else 0.0
            ),
            reuse_count=cache_result.reuse_count if cache_result else 0,
            residual_count=cache_result.residual_count if cache_result else 0,
            latency_ms=latency,
        )

    async def stream(
        self,
        request: AgentRequest,
    ) -> AsyncIterator[StreamChunk]:
        """
        Process request with DES-controlled streaming.

        DES adaptively adjusts the chunk size θ based on OSE to drive the
        pipeline to a dynamic equilibrium (slack δ→0) between producer and
        consumer.

        Args:
            request: Agent request to process.

        Yields:
            Stream chunks with adaptive chunk sizing.
        """
        self.stats["total_requests"] += 1

        # Get current chunk size θ from DES
        chunk_size = self.config.des.initial_theta
        if self.des_controller:
            chunk_size = self.des_controller.current_theta

        # Stream from LLM
        chunk_index = 0
        accumulated = ""
        producer_start = time.time()

        async for content in self._stream_llm(request.prompt):
            accumulated += content

            # Emit chunk when size reached
            if len(accumulated) >= chunk_size:
                producer_time = time.time() - producer_start

                yield StreamChunk(
                    request_id=request.request_id,
                    content=accumulated,
                    chunk_index=chunk_index,
                    tokens=len(accumulated.split()),
                    chunk_size=chunk_size,
                )

                # DES: Update controller with observation
                if self.des_controller:
                    consumer_time = 0.01  # Measured from downstream
                    slack_time = producer_time - consumer_time
                    chunk_size = self.des_controller.compute_next_theta(
                        producer_time=producer_time,
                        consumer_time=consumer_time,
                        slack_time=slack_time,
                        time_step=chunk_index,
                    )

                accumulated = ""
                chunk_index += 1
                producer_start = time.time()

        # Emit final chunk
        if accumulated:
            yield StreamChunk(
                request_id=request.request_id,
                content=accumulated,
                chunk_index=chunk_index,
                is_final=True,
                tokens=len(accumulated.split()),
                chunk_size=chunk_size,
            )

    async def batch_process(
        self,
        requests: List[AgentRequest],
    ) -> List[AgentResponse]:
        """
        Process multiple worker requests with BPP branch-parallel prefill.

        Uses BPP's supervisor-worker topology: all worker branches are
        independent and can be prefilled concurrently with inter-branch
        masking (BPA).

        Args:
            requests: List of worker requests to process.

        Returns:
            List of responses in same order as requests.
        """
        if not self.bpp_manager or len(requests) <= 1:
            # Sequential fallback
            return [await self.process(req) for req in requests]

        # BPP: All branches can be prefilled in parallel
        max_parallel = self.config.bpp.max_parallel_branches
        self.stats["parallel_batches"] += 1

        results = {}

        # Process in groups of max_parallel_branches
        for i in range(0, len(requests), max_parallel):
            group = requests[i : i + max_parallel]
            tasks = [self.process(req) for req in group]
            group_results = await asyncio.gather(*tasks)

            for req, resp in zip(group, group_results):
                results[req.request_id] = resp

        # Return in original order
        return [results[req.request_id] for req in requests]

    async def _call_llm(self, prompt: str) -> str:
        """Make an asynchronous LLM call."""
        if self.llm_client is None:
            raise RuntimeError(
                "LLM client not configured. Set llm.endpoint in config "
                "or provide a pre-configured llm_client to AgentMeshRuntime."
            )

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.config.llm.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"[Error: {str(e)}]"

    async def _stream_llm(self, prompt: str) -> AsyncIterator[str]:
        """Make a streaming LLM call."""
        if self.llm_client is None:
            raise RuntimeError(
                "LLM client not configured. Set llm.endpoint in config "
                "or provide a pre-configured llm_client to AgentMeshRuntime."
            )

        try:
            stream = await self.llm_client.chat.completions.create(
                model=self.config.llm.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            yield f"[Error: {str(e)}]"

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive runtime statistics."""
        stats = self.stats.copy()

        if self.dtr_cache:
            stats["dtr"] = self.dtr_cache.get_stats()

        if self.bpp_manager:
            stats["bpp"] = self.bpp_manager.get_stats()

        if self.des_controller:
            stats["des"] = self.des_controller.get_state()

        # Computed metrics
        if stats["total_requests"] > 0:
            stats["avg_latency_ms"] = (
                stats["total_latency_ms"] / stats["total_requests"]
            )
            stats["dtr_activation_rate"] = (
                stats["dtr_activations"] / stats["total_requests"]
            )

        return stats

    def reset(self):
        """Reset runtime state (alias for reset_stats)."""
        self.reset_stats()

    def reset_stats(self):
        """Reset runtime statistics."""
        self.stats = {
            "total_requests": 0,
            "dtr_activations": 0,
            "cold_retrievals": 0,
            "total_tokens_processed": 0,
            "total_tokens_reused": 0,
            "total_latency_ms": 0.0,
            "parallel_batches": 0,
        }

        if self.dtr_cache:
            self.dtr_cache.clear()

        if self.bpp_manager:
            self.bpp_manager.clear()

        if self.des_controller:
            self.des_controller.reset()


# Convenience function
def create_runtime(
    dtr_enabled: bool = True,
    bpp_enabled: bool = True,
    des_enabled: bool = True,
    llm_endpoint: str = "http://localhost:8000/v1",
    **kwargs,
) -> AgentMeshRuntime:
    """
    Create AgentMesh runtime with specified options.

    Args:
        dtr_enabled: Enable delta tool retrieval (DTR, the produce step).
        bpp_enabled: Enable branch-parallel prefill (BPP, the consume step).
        des_enabled: Enable dynamic-equilibrium streaming (DES, stage overlap).
        llm_endpoint: LLM API endpoint.
        **kwargs: Additional configuration overrides.

    Returns:
        Configured AgentMeshRuntime.
    """
    config = AgentMeshConfig()
    config.dtr.enabled = dtr_enabled
    config.bpp.enabled = bpp_enabled
    config.des.enabled = des_enabled
    config.llm.endpoint = llm_endpoint

    # Apply any additional overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return AgentMeshRuntime(config=config)
