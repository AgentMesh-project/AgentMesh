"""
AutoGen Adapter for AgentMesh

Provides integration with Microsoft AutoGen multi-agent framework.
Wraps AutoGen agents with AgentMesh sidecars to enable SRR, TSD/TDA, and ESF.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

from agentmesh.core.runtime import AgentMeshRuntime, AgentRequest, AgentResponse
from agentmesh.sidecars import SidecarMesh, AgentSidecar, Message

logger = logging.getLogger(__name__)


@dataclass
class WrappedAgent:
    """An AutoGen agent wrapped with AgentMesh optimization."""
    original_agent: Any
    sidecar: AgentSidecar
    agent_id: str


class AutoGenAdapter:
    """
    Adapter for integrating AgentMesh with AutoGen.
    
    Wraps AutoGen agents with AgentMesh sidecars to enable:
    - SRR semantic caching between agent conversations
    - TDA parallel KV prefill for supervisor-worker topology
    - ESF adaptive microbatch streaming
    
    Example:
        ```python
        from autogen import ConversableAgent
        from agentmesh.adapters import AutoGenAdapter
        
        # Create AutoGen agents
        researcher = ConversableAgent("researcher", ...)
        supervisor = ConversableAgent("supervisor", ...)
        
        # Create AgentMesh adapter
        adapter = AutoGenAdapter()
        
        # Wrap agents
        wrapped = adapter.wrap_agents([researcher, supervisor])
        
        # Run workflow with AgentMesh optimizations
        result = await adapter.run_chat(
            initial_agent=wrapped[0],
            message="Research AI safety",
            max_rounds=10
        )
        ```
    """
    
    def __init__(
        self,
        runtime: Optional[AgentMeshRuntime] = None,
        **runtime_kwargs
    ):
        """
        Initialize AutoGen adapter.
        
        Args:
            runtime: Optional pre-configured AgentMesh runtime.
            **runtime_kwargs: Kwargs passed to AgentMeshRuntime if runtime not provided.
        """
        if runtime:
            self.runtime = runtime
        else:
            from agentmesh.core.runtime import create_runtime
            self.runtime = create_runtime(**runtime_kwargs)
        
        # Create sidecar mesh
        self.mesh = SidecarMesh(
            srr_cache=self.runtime.srr_cache,
            tda_manager=getattr(self.runtime, 'tda_manager', None),
            esf_controller=self.runtime.esf_controller
        )
        
        self.wrapped_agents: Dict[str, WrappedAgent] = {}
        
        logger.info("AutoGenAdapter initialized")
    
    def wrap_agent(self, agent: Any) -> WrappedAgent:
        """
        Wrap a single AutoGen agent with AgentMesh sidecar.
        
        Args:
            agent: AutoGen ConversableAgent or subclass.
            
        Returns:
            Wrapped agent with sidecar.
        """
        # Extract agent name/id
        agent_id = getattr(agent, "name", str(id(agent)))
        
        # Create sidecar
        sidecar = self.mesh.register_agent(agent_id)
        
        # Create wrapped agent
        wrapped = WrappedAgent(
            original_agent=agent,
            sidecar=sidecar,
            agent_id=agent_id
        )
        
        self.wrapped_agents[agent_id] = wrapped
        
        # Monkey-patch the agent's generate_reply method
        original_generate = getattr(agent, "generate_reply", None)
        if original_generate:
            async def patched_generate(messages=None, sender=None, **kwargs):
                return await self._intercept_generate(
                    wrapped, original_generate, messages, sender, **kwargs
                )
            agent.generate_reply = patched_generate
        
        logger.info(f"Wrapped agent: {agent_id}")
        return wrapped
    
    def wrap_agents(self, agents: List[Any]) -> List[WrappedAgent]:
        """
        Wrap multiple AutoGen agents.
        
        Args:
            agents: List of AutoGen agents.
            
        Returns:
            List of wrapped agents.
        """
        return [self.wrap_agent(agent) for agent in agents]
    
    async def _intercept_generate(
        self,
        wrapped: WrappedAgent,
        original_generate: Callable,
        messages: Optional[List] = None,
        sender: Any = None,
        **kwargs
    ) -> str:
        """
        Intercept agent's generate_reply to apply AgentMesh optimizations.
        
        Checks SRR cache first, falls back to original generation if miss.
        """
        if not messages:
            return await original_generate(messages=messages, sender=sender, **kwargs)
        
        # Get the last user message
        last_message = messages[-1] if messages else None
        if not last_message:
            return await original_generate(messages=messages, sender=sender, **kwargs)
        
        content = last_message.get("content", "")
        
        # Check SRR cache via sidecar
        msg = Message(
            sender=wrapped.agent_id,
            receiver="llm",
            content=content,
            msg_type="text"
        )
        
        cache_result = await wrapped.sidecar.send(msg)
        
        if cache_result and cache_result.hit and cache_result.residual_ratio < 0.2:
            # Full cache hit - return cached response
            logger.debug(f"SRR cache hit for agent {wrapped.agent_id}")
            return "\n".join(cache_result.cached_entry.items)
        
        # Cache miss or partial hit - call original
        response = await original_generate(messages=messages, sender=sender, **kwargs)
        
        # Store response in cache
        if wrapped.sidecar.srr_cache:
            wrapped.sidecar.srr_cache.store(content, [response])
        
        return response
    
    async def run_chat(
        self,
        initial_agent: WrappedAgent,
        message: str,
        recipient: Optional[WrappedAgent] = None,
        max_rounds: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a chat workflow between wrapped agents.
        
        Args:
            initial_agent: Agent to start the conversation.
            message: Initial message.
            recipient: Optional recipient agent.
            max_rounds: Maximum conversation rounds.
            **kwargs: Additional kwargs for AutoGen.
            
        Returns:
            Chat result with statistics.
        """
        agent = initial_agent.original_agent
        
        # Use AutoGen's chat methods if available
        if hasattr(agent, "initiate_chat") and recipient:
            result = await agent.initiate_chat(
                recipient=recipient.original_agent,
                message=message,
                max_turns=max_rounds,
                **kwargs
            )
        elif hasattr(agent, "generate_reply"):
            # Simple single-turn
            result = await agent.generate_reply(
                messages=[{"role": "user", "content": message}]
            )
        else:
            result = {"error": "Agent does not support chat"}
        
        # Add AgentMesh statistics
        return {
            "result": result,
            "agentmesh_stats": self.get_stats()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive adapter statistics."""
        return {
            "runtime": self.runtime.get_stats(),
            "mesh": self.mesh.get_stats(),
            "wrapped_agents": list(self.wrapped_agents.keys())
        }


# Convenience function
def create_autogen_adapter(
    srr_enabled: bool = True,
    tsd_enabled: bool = True,
    esf_enabled: bool = True,
    **kwargs
) -> AutoGenAdapter:
    """
    Create an AutoGen adapter with specified options.
    
    Args:
        srr_enabled: Enable semantic caching.
        tsd_enabled: Enable parallel processing.
        esf_enabled: Enable adaptive streaming.
        **kwargs: Additional runtime configuration.
        
    Returns:
        Configured AutoGenAdapter.
    """
    from agentmesh.core.runtime import create_runtime
    
    runtime = create_runtime(
        srr_enabled=srr_enabled,
        tsd_enabled=tsd_enabled,
        esf_enabled=esf_enabled,
        **kwargs
    )
    
    return AutoGenAdapter(runtime=runtime)
