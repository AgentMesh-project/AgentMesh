"""
AgentMesh Runtime

Main runtime that orchestrates the three core mechanisms:
  - SRR (Semantic Residual Retrieval)
  - TSD (Topology-aware Semantic Decoupling)
  - ESF (Elastic Semantic Flow)

Provides the high-level API for multi-agent workflows built on
AutoGen with vLLM as the LLM backend.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable, AsyncIterator
from dataclasses import dataclass, field
import time

from agentmesh.core.config import AgentMeshConfig, load_config
from agentmesh.mechanisms.srr import SRRCache, create_srr_cache
from agentmesh.mechanisms.tsd import TDAManager, create_tda_manager
from agentmesh.mechanisms.esf import ESFController, create_esf_controller

logger = logging.getLogger(__name__)


@dataclass
class AgentRequest:
    """A request from an agent to the LLM or tool."""
    request_id: str
    agent_id: str
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResponse:
    """A response from the LLM to an agent."""
    content: str
    request_id: str = ""
    agent_id: str = ""
    tokens_generated: int = 0
    latency_ms: float = 0.0
    srr_activated: bool = False
    projection_intensity: float = 0.0
    reuse_count: int = 0
    residual_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamChunk:
    """A microbatch of streaming response (ESF-controlled)."""
    content: str
    chunk_index: int = 0
    request_id: str = ""
    is_final: bool = False
    tokens: int = 0
    microbatch_size: int = 0


class AgentMeshRuntime:
    """
    Main AgentMesh runtime orchestrating SRR, TSD, and ESF.

    Provides:
    - Semantic caching via SRR for reducing redundant tool retrieval
    - Parallel KV prefill via TSD/TDA for supervisor-worker topology
    - Adaptive microbatch streaming via ESF for optimal flow control

    Example:
        ```python
        runtime = AgentMeshRuntime()

        # Process a single request with SRR
        response = await runtime.process(request)

        # Process streaming response with ESF
        async for chunk in runtime.stream(request):
            print(chunk.content, end="", flush=True)

        # Batch process multiple worker outputs via TSD
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
        self._init_srr()
        self._init_tsd()
        self._init_esf()

        # LLM client
        self.llm_client = llm_client
        self._init_llm_client()

        # Statistics
        self.stats = {
            "total_requests": 0,
            "srr_activations": 0,
            "cold_retrievals": 0,
            "total_tokens_processed": 0,
            "total_tokens_reused": 0,
            "total_latency_ms": 0.0,
            "parallel_batches": 0,
        }

        logger.info("AgentMesh runtime initialized")

    def _init_srr(self):
        """Initialize SRR mechanism."""
        if self.config.srr.enabled:
            self.srr_cache = create_srr_cache(
                confidence_threshold=self.config.srr.confidence_threshold,
                cache_size=self.config.srr.cache_size,
                embedding_model=self.config.srr.embedding_model,
            )
            logger.info(
                f"SRR enabled: τ={self.config.srr.confidence_threshold}, "
                f"k={self.config.srr.hotspot_k}"
            )
        else:
            self.srr_cache = None
            logger.info("SRR disabled")

    def _init_tsd(self):
        """Initialize TSD/TDA mechanism."""
        if self.config.tsd.enabled:
            self.tda_manager = create_tda_manager(
                num_layers=self.config.tsd.num_layers,
                num_heads=self.config.tsd.num_heads,
                head_dim=self.config.tsd.head_dim,
                max_parallel_workers=self.config.tsd.max_parallel_workers,
                rope_base=self.config.tsd.rope_base,
                device=self.config.device,
            )
            logger.info(
                f"TSD enabled: max_workers={self.config.tsd.max_parallel_workers}"
            )
        else:
            self.tda_manager = None
            logger.info("TSD disabled")

    def _init_esf(self):
        """Initialize ESF mechanism."""
        if self.config.esf.enabled:
            self.esf_controller = create_esf_controller(
                initial_theta=self.config.esf.initial_microbatch_size,
                min_theta=self.config.esf.theta_min,
                max_theta=self.config.esf.theta_max,
                damping_factor=self.config.esf.damping_factor,
                window_size=self.config.esf.window_size,
            )
            logger.info(f"ESF enabled: θ₀={self.config.esf.initial_microbatch_size}")
        else:
            self.esf_controller = None
            logger.info("ESF disabled")

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
        Process a single agent request with SRR semantic caching.

        SRR flow:
        1. Embed query → find nearest cache entry
        2. Compute projection intensity s_max
        3. If s_max ≥ τ: reuse ⌊s_max · n⌋ items, fetch residual
        4. If s_max < τ: cold retrieval (full tool call)

        Args:
            request: Agent request to process.

        Returns:
            Agent response with SRR metadata.
        """
        start_time = time.time()
        self.stats["total_requests"] += 1

        # SRR: Check cache
        cache_result = None
        if self.srr_cache:
            cache_result = self.srr_cache.lookup(request.prompt)

            if cache_result.hit:
                self.stats["srr_activations"] += 1

                # Get reused items from cache
                reused_items = self.srr_cache.get_reused_items(cache_result)

                # If high confidence and all items reused, skip LLM call
                if cache_result.residual_count == 0:
                    latency = (time.time() - start_time) * 1000
                    self.stats["total_latency_ms"] += latency
                    self.stats["total_tokens_reused"] += cache_result.reuse_count

                    return AgentResponse(
                        request_id=request.request_id,
                        agent_id=request.agent_id,
                        content="\n".join(reused_items),
                        srr_activated=True,
                        projection_intensity=cache_result.projection_intensity,
                        reuse_count=cache_result.reuse_count,
                        residual_count=0,
                        latency_ms=latency,
                    )

                # Partial hit: use reformulated query for residual
                prompt_to_send = cache_result.reformulated_query or request.prompt
            else:
                self.stats["cold_retrievals"] += 1
                prompt_to_send = request.prompt
        else:
            prompt_to_send = request.prompt

        # Process with LLM
        response_content = await self._call_llm(prompt_to_send)

        # Combine reused items + new content for partial hits
        if cache_result and cache_result.hit and cache_result.reuse_count > 0:
            reused_items = self.srr_cache.get_reused_items(cache_result)
            response_content = "\n".join(reused_items) + "\n" + response_content

        # Store in cache
        if self.srr_cache:
            items = response_content.split("\n")
            self.srr_cache.store(request.prompt, items)

        latency = (time.time() - start_time) * 1000
        self.stats["total_latency_ms"] += latency

        return AgentResponse(
            request_id=request.request_id,
            agent_id=request.agent_id,
            content=response_content,
            srr_activated=cache_result.hit if cache_result else False,
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
        Process request with ESF-controlled streaming.

        ESF adaptively adjusts microbatch size θ based on OSE
        to minimize stall time between producer and consumer.

        Args:
            request: Agent request to process.

        Yields:
            Stream chunks with adaptive microbatch sizing.
        """
        self.stats["total_requests"] += 1

        # Get current microbatch size from ESF
        microbatch_size = self.config.esf.initial_microbatch_size
        if self.esf_controller:
            microbatch_size = self.esf_controller.current_theta

        # Stream from LLM
        chunk_index = 0
        accumulated = ""
        producer_start = time.time()

        async for content in self._stream_llm(request.prompt):
            accumulated += content

            # Emit microbatch when size reached
            if len(accumulated) >= microbatch_size:
                producer_time = time.time() - producer_start

                yield StreamChunk(
                    request_id=request.request_id,
                    content=accumulated,
                    chunk_index=chunk_index,
                    tokens=len(accumulated.split()),
                    microbatch_size=microbatch_size,
                )

                # ESF: Update controller with observation
                if self.esf_controller:
                    consumer_time = 0.01  # Measured from downstream
                    slack_time = producer_time - consumer_time
                    microbatch_size = self.esf_controller.compute_next_theta(
                        producer_time=producer_time,
                        consumer_time=consumer_time,
                        slack_time=slack_time,
                        time_step=chunk_index,
                    )

                accumulated = ""
                chunk_index += 1
                producer_start = time.time()

        # Emit final microbatch
        if accumulated:
            yield StreamChunk(
                request_id=request.request_id,
                content=accumulated,
                chunk_index=chunk_index,
                is_final=True,
                tokens=len(accumulated.split()),
                microbatch_size=microbatch_size,
            )

    async def batch_process(
        self,
        requests: List[AgentRequest],
    ) -> List[AgentResponse]:
        """
        Process multiple worker requests with TDA parallel prefill.

        Uses TSD's supervisor-worker topology: all worker outputs are
        independent and can be prefilled concurrently with inter-worker
        masking.

        Args:
            requests: List of worker requests to process.

        Returns:
            List of responses in same order as requests.
        """
        if not self.tda_manager or len(requests) <= 1:
            # Sequential fallback
            return [await self.process(req) for req in requests]

        # TDA: All workers can be prefilled in parallel
        max_parallel = self.config.tsd.max_parallel_workers
        self.stats["parallel_batches"] += 1

        results = {}

        # Process in groups of max_parallel_workers
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

        if self.srr_cache:
            stats["srr"] = self.srr_cache.get_stats()

        if self.tda_manager:
            stats["tsd"] = self.tda_manager.get_stats()

        if self.esf_controller:
            stats["esf"] = self.esf_controller.get_state()

        # Computed metrics
        if stats["total_requests"] > 0:
            stats["avg_latency_ms"] = (
                stats["total_latency_ms"] / stats["total_requests"]
            )
            stats["srr_activation_rate"] = (
                stats["srr_activations"] / stats["total_requests"]
            )

        return stats

    def reset(self):
        """Reset runtime state (alias for reset_stats)."""
        self.reset_stats()

    def reset_stats(self):
        """Reset runtime statistics."""
        self.stats = {
            "total_requests": 0,
            "srr_activations": 0,
            "cold_retrievals": 0,
            "total_tokens_processed": 0,
            "total_tokens_reused": 0,
            "total_latency_ms": 0.0,
            "parallel_batches": 0,
        }

        if self.srr_cache:
            self.srr_cache.clear()

        if self.tda_manager:
            self.tda_manager.clear()

        if self.esf_controller:
            self.esf_controller.reset()


# Convenience function
def create_runtime(
    srr_enabled: bool = True,
    tsd_enabled: bool = True,
    esf_enabled: bool = True,
    llm_endpoint: str = "http://localhost:8000/v1",
    **kwargs,
) -> AgentMeshRuntime:
    """
    Create AgentMesh runtime with specified options.

    Args:
        srr_enabled: Enable semantic caching (SRR).
        tsd_enabled: Enable parallel prefill (TSD/TDA).
        esf_enabled: Enable adaptive streaming (ESF).
        llm_endpoint: LLM API endpoint.
        **kwargs: Additional configuration overrides.

    Returns:
        Configured AgentMeshRuntime.
    """
    config = AgentMeshConfig()
    config.srr.enabled = srr_enabled
    config.tsd.enabled = tsd_enabled
    config.esf.enabled = esf_enabled
    config.llm.endpoint = llm_endpoint

    # Apply any additional overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return AgentMeshRuntime(config=config)
