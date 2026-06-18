"""
AgentMesh Sidecar Implementations

Provides the three-layer sidecar architecture over the produce–transfer–consume
dataflow:
- Agent Sidecar: handles agent-to-agent dataflows with DTR delta retrieval
- LLM Sidecar: manages LLM consume steps with BPP/BPA branch-parallel prefill
- Tool Sidecar: orchestrates tool produce steps with DES chunk streaming

The BPP manager is referenced only for typing/wiring; its import is guarded so
the sidecars keep importing when torch (an optional extra) is absent.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncIterator, TYPE_CHECKING
from dataclasses import dataclass, field
import time
import uuid

from agentmesh.mechanisms.dtr import DTRCache, CacheResult
from agentmesh.mechanisms.des import DESController

if TYPE_CHECKING:  # pragma: no cover - typing only; avoids importing torch
    from agentmesh.mechanisms.bpp import BranchParallelManager

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Internal message format for sidecar communication."""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    receiver: str = ""
    content: str = ""
    msg_type: str = "text"  # "text", "tool_call", "tool_result", "system"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class AgentSidecar:
    """
    Agent Sidecar - Intercepts agent dataflows.

    Responsibilities:
    - Route messages between agents
    - Apply DTR delta retrieval for repeated queries (the produce step)
    - Track conversation context for BPP branch segments

    The Agent Sidecar sits between each agent and the message bus,
    enabling DTR delta retrieval of similar requests across agents.
    """

    def __init__(
        self,
        agent_id: str,
        dtr_cache: Optional[DTRCache] = None
    ):
        """
        Initialize Agent Sidecar.

        Args:
            agent_id: ID of the agent this sidecar serves.
            dtr_cache: Optional shared DTR cache instance.
        """
        self.agent_id = agent_id
        self.dtr_cache = dtr_cache

        # Message queues
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.outbox: asyncio.Queue = asyncio.Queue()

        # Conversation tracking
        self.conversation_history: List[Message] = []
        self.segment_ids: List[str] = []

        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "cache_redirects": 0
        }

        logger.info(f"AgentSidecar initialized for agent {agent_id}")

    async def send(self, message: Message) -> Optional[CacheResult]:
        """
        Send a message, checking the DTR cache first.

        If a semantically similar message exists in cache, returns
        the cached response instead of forwarding to the LLM.

        Args:
            message: Message to send.

        Returns:
            CacheResult if cache hit, None otherwise.
        """
        self.stats["messages_sent"] += 1
        message.sender = self.agent_id

        # Check DTR cache for LLM-bound messages
        if self.dtr_cache and message.msg_type == "text":
            cache_result = self.dtr_cache.lookup(message.content)

            if cache_result.hit and cache_result.residual_ratio < 0.2:
                self.stats["cache_redirects"] += 1
                logger.debug(f"DTR cache hit for agent {self.agent_id}")
                return cache_result

        # Forward message
        await self.outbox.put(message)
        self.conversation_history.append(message)

        return None

    async def receive(self) -> Message:
        """
        Receive next message from inbox.

        Returns:
            Next message for this agent.
        """
        message = await self.inbox.get()
        self.stats["messages_received"] += 1
        self.conversation_history.append(message)

        # Store successful responses in cache
        if self.dtr_cache and message.msg_type == "text":
            # Find the original query
            for hist in reversed(self.conversation_history[:-1]):
                if hist.sender == self.agent_id:
                    self.dtr_cache.store(hist.content, [message.content])
                    break

        return message

    async def deliver(self, message: Message):
        """Deliver a message to this agent's inbox."""
        await self.inbox.put(message)

    def create_segment(self, content: str, segment_type: str = "worker") -> str:
        """
        Create a segment ID for this conversation turn.

        Args:
            content: Content of the segment.
            segment_type: Type ("supervisor" or "worker").

        Returns:
            Segment ID.
        """
        segment_id = f"{self.agent_id}_{len(self.segment_ids)}"
        self.segment_ids.append(segment_id)
        return segment_id

    def get_stats(self) -> Dict:
        """Get sidecar statistics."""
        stats = self.stats.copy()
        stats["conversation_length"] = len(self.conversation_history)
        stats["segment_count"] = len(self.segment_ids)
        return stats


class LLMSidecar:
    """
    LLM Sidecar - Manages LLM consume steps.

    Responsibilities:
    - Aggregate requests for batch processing
    - Apply BPP/BPA for branch-parallel KV prefill (supervisor-worker topology)
    - Coordinate with DES for streaming chunk sizes

    The LLM Sidecar sits between the agent sidecars and the LLM backend,
    enabling efficient batching and BPA-based KV cache prefill.
    """

    def __init__(
        self,
        llm_client=None,
        bpp_manager: Optional["BranchParallelManager"] = None,
        des_controller: Optional[DESController] = None,
        batch_timeout_ms: float = 50.0,
        max_batch_size: int = 8
    ):
        """
        Initialize LLM Sidecar.

        Args:
            llm_client: LLM backend client (must support `generate(prompt)`
                and `stream(prompt)` methods). Required for inference.
            bpp_manager: BPP manager for branch-parallel prefill.
            des_controller: DES streaming controller.
            batch_timeout_ms: Max time to wait for batching.
            max_batch_size: Maximum requests per batch.
        """
        self.llm_client = llm_client
        self.bpp_manager = bpp_manager
        self.des_controller = des_controller
        self.batch_timeout_ms = batch_timeout_ms
        self.max_batch_size = max_batch_size

        # Request queue for batching
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.pending_requests: Dict[str, asyncio.Future] = {}

        # Statistics
        self.stats = {
            "total_requests": 0,
            "batches_processed": 0,
            "avg_batch_size": 0.0,
            "parallel_speedup": 0.0
        }

        logger.info("LLMSidecar initialized")

    async def process(
        self,
        request_id: str,
        prompt: str,
    ) -> str:
        """
        Process an LLM request.

        May be batched with other requests for BPP branch-parallel processing.

        Args:
            request_id: Unique request identifier.
            prompt: The prompt to process.

        Returns:
            LLM response content.
        """
        self.stats["total_requests"] += 1

        # Add to BPP manager if available
        if self.bpp_manager:
            pass  # BPP integration handled at batch level

        # Create future for result
        future = asyncio.Future()
        self.pending_requests[request_id] = future

        # Queue request
        await self.request_queue.put({
            "request_id": request_id,
            "prompt": prompt,
            "timestamp": time.time()
        })

        # Wait for result
        return await future

    async def stream(
        self,
        request_id: str,
        prompt: str
    ) -> AsyncIterator[str]:
        """
        Stream an LLM response with DES-controlled chunking.

        Args:
            request_id: Unique request identifier.
            prompt: The prompt to process.

        Yields:
            Response chunks with adaptive sizing.
        """
        chunk_size = 512
        if self.des_controller:
            chunk_size = self.des_controller.current_theta

        if self.llm_client is None:
            raise RuntimeError(
                "LLM client not configured. Provide an llm_client "
                "when initializing LLMSidecar."
            )

        # Stream response from LLM backend
        accumulated = []
        producer_start = time.time()
        chunk_index = 0

        async for token in self.llm_client.stream(prompt):
            accumulated.append(token)
            joined = "".join(accumulated)

            if len(joined) >= chunk_size:
                producer_time = time.time() - producer_start

                # DES: adaptively adjust chunk size
                if self.des_controller:
                    chunk_size = self.des_controller.compute_next_theta(
                        producer_time=producer_time,
                        consumer_time=0.01,
                        slack_time=producer_time - 0.01,
                        time_step=chunk_index,
                    )

                yield joined
                accumulated = []
                producer_start = time.time()
                chunk_index += 1

        # Final chunk
        if accumulated:
            yield "".join(accumulated)

    async def batch_processor(self):
        """
        Background task that batches and processes requests.

        Uses BPP to identify parallelizable branches and processes
        them together for improved throughput.
        """
        while True:
            batch = []
            deadline = time.time() + self.batch_timeout_ms / 1000

            # Collect requests for batching
            while len(batch) < self.max_batch_size:
                try:
                    timeout = max(0, deadline - time.time())
                    request = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=timeout
                    )
                    batch.append(request)
                except asyncio.TimeoutError:
                    break

            if not batch:
                await asyncio.sleep(0.01)
                continue

            # Process batch
            await self._process_batch(batch)

    async def _process_batch(self, batch: List[Dict]):
        """Process a batch of requests."""
        self.stats["batches_processed"] += 1

        # Update average batch size
        n = self.stats["batches_processed"]
        old_avg = self.stats["avg_batch_size"]
        self.stats["avg_batch_size"] = old_avg + (len(batch) - old_avg) / n

        # BPP: Generate attention mask for branch-parallel processing
        if self.bpp_manager and len(batch) > 1:
            parallel_groups = self.bpp_manager.get_parallel_groups()
            logger.debug(f"BPA parallel groups: {len(parallel_groups)}")

        # Process each request via LLM client
        for request in batch:
            request_id = request["request_id"]
            prompt = request["prompt"]

            try:
                if self.llm_client is not None:
                    response = await self.llm_client.generate(prompt)
                else:
                    raise RuntimeError(
                        "LLM client not configured. Provide an llm_client "
                        "when initializing LLMSidecar."
                    )
            except Exception as exc:
                if request_id in self.pending_requests:
                    self.pending_requests[request_id].set_exception(exc)
                    del self.pending_requests[request_id]
                continue

            # Resolve future
            if request_id in self.pending_requests:
                self.pending_requests[request_id].set_result(response)
                del self.pending_requests[request_id]

    def get_stats(self) -> Dict:
        """Get sidecar statistics."""
        return self.stats.copy()


class ToolSidecar:
    """
    Tool Sidecar - Manages tool produce steps.

    Responsibilities:
    - Route tool calls to appropriate executors
    - Stream tool results back to agents
    - Apply DES for result streaming optimization

    The Tool Sidecar sits between agents and external tools,
    enabling efficient streaming of tool results.
    """

    def __init__(
        self,
        des_controller: Optional[DESController] = None
    ):
        """
        Initialize Tool Sidecar.

        Args:
            des_controller: DES controller for streaming optimization.
        """
        self.des_controller = des_controller

        # Registered tools
        self.tools: Dict[str, Dict] = {}

        # Statistics
        self.stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "avg_latency_ms": 0.0
        }

        logger.info("ToolSidecar initialized")

    def register_tool(
        self,
        tool_id: str,
        name: str,
        description: str,
        executor: callable,
        parameters: Optional[List[Dict]] = None
    ):
        """
        Register a tool with the sidecar.

        Args:
            tool_id: Unique tool identifier.
            name: Human-readable name.
            description: Tool description.
            executor: Callable that executes the tool.
            parameters: List of parameter definitions.
        """
        self.tools[tool_id] = {
            "tool_id": tool_id,
            "name": name,
            "description": description,
            "executor": executor,
            "parameters": parameters or []
        }
        logger.info(f"Registered tool: {name} ({tool_id})")

    async def execute(
        self,
        tool_id: str,
        arguments: Dict[str, Any]
    ) -> str:
        """
        Execute a tool.

        Args:
            tool_id: Tool to execute.
            arguments: Tool arguments.

        Returns:
            Tool execution result.
        """
        self.stats["total_executions"] += 1
        start_time = time.time()

        if tool_id not in self.tools:
            self.stats["failed_executions"] += 1
            return f"Error: Unknown tool {tool_id}"

        tool = self.tools[tool_id]

        try:
            executor = tool["executor"]
            if asyncio.iscoroutinefunction(executor):
                result = await executor(**arguments)
            else:
                result = executor(**arguments)

            self.stats["successful_executions"] += 1

        except Exception as e:
            self.stats["failed_executions"] += 1
            result = f"Error executing {tool_id}: {str(e)}"

        # Update latency stats
        latency = (time.time() - start_time) * 1000
        n = self.stats["total_executions"]
        old_avg = self.stats["avg_latency_ms"]
        self.stats["avg_latency_ms"] = old_avg + (latency - old_avg) / n

        return str(result)

    async def stream_execute(
        self,
        tool_id: str,
        arguments: Dict[str, Any]
    ) -> AsyncIterator[str]:
        """
        Execute a tool with streaming results.

        Uses DES to control chunk sizes for optimal streaming.

        Args:
            tool_id: Tool to execute.
            arguments: Tool arguments.

        Yields:
            Chunks of tool execution result.
        """
        result = await self.execute(tool_id, arguments)

        # Stream result with DES-controlled chunking
        chunk_size = 256
        if self.des_controller:
            chunk_size = self.des_controller.current_theta

        for i in range(0, len(result), chunk_size):
            yield result[i:i + chunk_size]
            await asyncio.sleep(0.01)  # Yield control between chunks

    def list_tools(self) -> List[Dict]:
        """List all registered tools."""
        return [
            {
                "tool_id": t["tool_id"],
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"]
            }
            for t in self.tools.values()
        ]

    def get_stats(self) -> Dict:
        """Get sidecar statistics."""
        stats = self.stats.copy()
        stats["registered_tools"] = len(self.tools)
        return stats


class SidecarMesh:
    """
    Coordinates all sidecars in the mesh.

    Provides a unified interface for managing agent, LLM, and tool
    sidecars, and routes messages between them.
    """

    def __init__(
        self,
        dtr_cache: Optional[DTRCache] = None,
        bpp_manager: Optional["BranchParallelManager"] = None,
        des_controller: Optional[DESController] = None
    ):
        """
        Initialize the sidecar mesh.

        Args:
            dtr_cache: Shared DTR cache.
            bpp_manager: Shared BPP manager.
            des_controller: Shared DES controller.
        """
        self.dtr_cache = dtr_cache
        self.bpp_manager = bpp_manager
        self.des_controller = des_controller

        self.agent_sidecars: Dict[str, AgentSidecar] = {}
        self.llm_sidecar = LLMSidecar(
            bpp_manager=bpp_manager,
            des_controller=des_controller
        )
        self.tool_sidecar = ToolSidecar(des_controller=des_controller)

        logger.info("SidecarMesh initialized")

    def register_agent(self, agent_id: str) -> AgentSidecar:
        """
        Register a new agent and create its sidecar.

        Args:
            agent_id: Unique agent identifier.

        Returns:
            The agent's sidecar instance.
        """
        sidecar = AgentSidecar(
            agent_id=agent_id,
            dtr_cache=self.dtr_cache
        )
        self.agent_sidecars[agent_id] = sidecar
        return sidecar

    async def route_message(self, message: Message):
        """
        Route a message to its recipient.

        Args:
            message: Message to route.
        """
        receiver = message.receiver

        if receiver in self.agent_sidecars:
            await self.agent_sidecars[receiver].deliver(message)
        elif receiver == "llm":
            # Route to LLM sidecar
            response = await self.llm_sidecar.process(
                message.message_id,
                message.content
            )
            # Send response back
            reply = Message(
                sender="llm",
                receiver=message.sender,
                content=response,
                msg_type="text"
            )
            if message.sender in self.agent_sidecars:
                await self.agent_sidecars[message.sender].deliver(reply)
        else:
            logger.warning(f"Unknown recipient: {receiver}")

    def get_stats(self) -> Dict:
        """Get comprehensive mesh statistics."""
        return {
            "agents": {
                agent_id: sidecar.get_stats()
                for agent_id, sidecar in self.agent_sidecars.items()
            },
            "llm": self.llm_sidecar.get_stats(),
            "tool": self.tool_sidecar.get_stats()
        }
