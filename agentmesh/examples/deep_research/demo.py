"""
Deep Research Demo

A multi-agent research workflow demonstrating AgentMesh optimization.

This demo implements a supervisor-researcher pattern where:
- 1 Supervisor agent coordinates the research
- N Researcher agents conduct parallel research tasks
- Tools provide web search and data analysis capabilities

AgentMesh optimizations (over the produce–transfer–consume dataflow):
- DTR: serves the cached base of similar research queries and fetches only the
       semantic delta (the produce step)
- BPP: enables branch-parallel KV prefill for concurrent researchers via BPA
       (the consume step)
- DES: adapts streaming chunk sizes for varying network conditions
       (stage overlap)
"""

import asyncio
import argparse
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# AgentMesh imports
from agentmesh import AgentMeshRuntime
from agentmesh.core.config import AgentMeshConfig
from agentmesh.mechanisms import DTRCache, DESController
from agentmesh.sidecars import SidecarMesh, AgentSidecar, ToolSidecar, Message

logger = logging.getLogger(__name__)


# =============================================================================
# Agent Definitions
# =============================================================================

@dataclass
class Agent:
    """Base agent class for the demo."""
    agent_id: str
    role: str
    system_prompt: str
    sidecar: Optional[AgentSidecar] = None

    async def think(self, context: str) -> str:
        """Generate agent's response given context."""
        raise NotImplementedError


@dataclass
class SupervisorAgent(Agent):
    """
    Supervisor agent that coordinates researchers.

    Responsibilities:
    - Break down research topic into subtasks
    - Assign tasks to researchers
    - Synthesize final research report
    """

    def __init__(self, agent_id: str = "supervisor"):
        super().__init__(
            agent_id=agent_id,
            role="supervisor",
            system_prompt="""You are a research supervisor coordinating a team of researchers.
Your responsibilities:
1. Break down complex research topics into focused subtasks
2. Assign clear, specific tasks to each researcher
3. Review and synthesize findings into a coherent report

Be concise and actionable in your task assignments."""
        )
        self.assigned_tasks: List[Dict] = []
        self.collected_results: List[Dict] = []

    async def create_tasks(self, topic: str, num_researchers: int) -> List[Dict]:
        """Break down topic into research tasks."""
        # In production, this would call the LLM
        tasks = []
        aspects = [
            "historical background and evolution",
            "current state and key developments",
            "challenges and open problems",
            "future directions and implications"
        ]

        for i in range(min(num_researchers, len(aspects))):
            tasks.append({
                "task_id": f"task_{i}",
                "researcher_id": f"researcher_{i}",
                "aspect": aspects[i],
                "prompt": f"Research the {aspects[i]} of: {topic}"
            })

        self.assigned_tasks = tasks
        return tasks

    async def synthesize(self, results: List[Dict]) -> str:
        """Synthesize researcher findings into final report."""
        self.collected_results = results

        # Build synthesis prompt
        findings = "\n\n".join([
            f"## {r['aspect']}\n{r['content']}"
            for r in results
        ])

        return f"""# Research Report

## Topic Overview
{findings}

## Synthesis
Based on the above findings, here is the integrated analysis...

## Conclusions
Key takeaways from this research include...
"""


@dataclass
class ResearcherAgent(Agent):
    """
    Researcher agent that conducts focused research.

    Each researcher handles a specific aspect of the research topic
    and uses tools to gather information.
    """

    def __init__(self, agent_id: str, specialty: str = ""):
        super().__init__(
            agent_id=agent_id,
            role="researcher",
            system_prompt=f"""You are a research assistant specializing in {specialty or 'general research'}.
Your task is to:
1. Thoroughly research the assigned topic
2. Use available tools to gather relevant information
3. Synthesize findings into a clear, well-structured report

Focus on accuracy, relevance, and clarity."""
        )
        self.specialty = specialty
        self.findings: List[str] = []

    async def research(self, task: Dict, tool_sidecar: ToolSidecar) -> Dict:
        """Conduct research on assigned task."""
        aspect = task["aspect"]
        prompt = task["prompt"]

        # Use tools to gather information
        search_result = await tool_sidecar.execute(
            "web_search",
            {"query": prompt}
        )

        # Analyze gathered information
        analysis = await tool_sidecar.execute(
            "analyze",
            {"text": search_result, "focus": aspect}
        )

        self.findings.append(analysis)

        return {
            "task_id": task["task_id"],
            "researcher_id": self.agent_id,
            "aspect": aspect,
            "content": analysis
        }


# =============================================================================
# Tool Definitions
# =============================================================================

def register_research_tools(tool_sidecar: ToolSidecar):
    """Register research tools with the tool sidecar."""

    def web_search(query: str) -> str:
        """Web search tool (placeholder).

        Replace with a real search API (e.g., Tavily, Bing, SerpAPI)
        for production use.
        """
        return f"[Search results for: {query}]\n" + \
               "- Finding 1: Important information related to the query\n" + \
               "- Finding 2: Additional relevant data points\n" + \
               "- Finding 3: Expert opinions and analysis"

    def analyze(text: str, focus: str = "") -> str:
        """Analysis tool (placeholder).

        Replace with an actual analytical backend for production use.
        """
        return f"[Analysis of {focus}]\n" + \
               f"Based on the gathered information:\n{text[:200]}...\n\n" + \
               "Key insights:\n" + \
               "1. Main point one\n" + \
               "2. Main point two\n" + \
               "3. Main point three"

    def summarize(text: str, max_length: int = 500) -> str:
        """Summarization tool (placeholder).

        Replace with a dedicated summarization model or API
        for production use.
        """
        return f"[Summary]\n{text[:max_length]}..."

    tool_sidecar.register_tool(
        tool_id="web_search",
        name="Web Search",
        description="Search the web for information",
        executor=web_search,
        parameters=[{"name": "query", "type": "string", "required": True}]
    )

    tool_sidecar.register_tool(
        tool_id="analyze",
        name="Analyze",
        description="Analyze text and extract insights",
        executor=analyze,
        parameters=[
            {"name": "text", "type": "string", "required": True},
            {"name": "focus", "type": "string", "required": False}
        ]
    )

    tool_sidecar.register_tool(
        tool_id="summarize",
        name="Summarize",
        description="Summarize text content",
        executor=summarize,
        parameters=[
            {"name": "text", "type": "string", "required": True},
            {"name": "max_length", "type": "int", "required": False}
        ]
    )


# =============================================================================
# Workflow Orchestration
# =============================================================================

class DeepResearchWorkflow:
    """
    Orchestrates the deep research multi-agent workflow.

    This class demonstrates how AgentMesh optimizations apply:
    - DTR: When researchers query similar topics, the cached base is reused and
      only the semantic delta is fetched (the produce step)
    - BPP: Researcher agents' KV prefill runs in parallel via BPA (the consume
      step)
    - DES: Streaming responses adapt chunk sizes to dynamics (stage overlap)
    """

    def __init__(
        self,
        num_researchers: int = 3,
        enable_dtr: bool = True,
        enable_bpp: bool = True,
        enable_des: bool = True,
        llm_endpoint: str = "http://localhost:8000/v1"
    ):
        """
        Initialize the workflow.

        Args:
            num_researchers: Number of researcher agents.
            enable_dtr: Enable DTR delta retrieval.
            enable_bpp: Enable BPP branch-parallel prefill.
            enable_des: Enable DES adaptive chunk streaming.
            llm_endpoint: LLM API endpoint.
        """
        self.num_researchers = num_researchers

        # Initialize AgentMesh runtime
        config = AgentMeshConfig()
        config.dtr.enabled = enable_dtr
        config.bpp.enabled = enable_bpp
        config.des.enabled = enable_des
        config.llm.endpoint = llm_endpoint

        self.runtime = AgentMeshRuntime(config=config)

        # Create sidecar mesh
        self.mesh = SidecarMesh(
            dtr_cache=self.runtime.dtr_cache,
            bpp_manager=getattr(self.runtime, 'bpp_manager', None),
            des_controller=self.runtime.des_controller
        )

        # Create agents
        self.supervisor = SupervisorAgent()
        self.supervisor.sidecar = self.mesh.register_agent(self.supervisor.agent_id)

        self.researchers: List[ResearcherAgent] = []
        specialties = ["historical analysis", "current trends", "challenges", "future outlook"]

        for i in range(num_researchers):
            researcher = ResearcherAgent(
                agent_id=f"researcher_{i}",
                specialty=specialties[i % len(specialties)]
            )
            researcher.sidecar = self.mesh.register_agent(researcher.agent_id)
            self.researchers.append(researcher)

        # Register tools
        register_research_tools(self.mesh.tool_sidecar)

        # Statistics
        self.stats = {
            "start_time": None,
            "end_time": None,
            "tasks_completed": 0,
            "parallel_executions": 0
        }

        logger.info(f"Initialized DeepResearchWorkflow with {num_researchers} researchers")

    async def run(self, topic: str) -> Dict[str, Any]:
        """
        Execute the research workflow.

        Args:
            topic: Research topic to investigate.

        Returns:
            Workflow results including report and statistics.
        """
        self.stats["start_time"] = time.time()

        print(f"\n{'='*60}")
        print(f"Starting Deep Research on: {topic}")
        print(f"{'='*60}\n")

        # Phase 1: Task Creation
        print("Phase 1: Creating research tasks...")
        tasks = await self.supervisor.create_tasks(topic, self.num_researchers)
        print(f"  Created {len(tasks)} tasks")

        # Phase 2: Parallel Research (BPP optimization applies here)
        print("\nPhase 2: Conducting parallel research...")
        research_tasks = []

        for i, (researcher, task) in enumerate(zip(self.researchers, tasks)):
            print(f"  Researcher {i}: {task['aspect']}")
            research_tasks.append(
                researcher.research(task, self.mesh.tool_sidecar)
            )

        # Execute in parallel
        results = await asyncio.gather(*research_tasks)
        self.stats["tasks_completed"] = len(results)
        self.stats["parallel_executions"] = 1  # All executed in one parallel batch

        print(f"\n  Completed {len(results)} research tasks")

        # Phase 3: Synthesis
        print("\nPhase 3: Synthesizing findings...")
        report = await self.supervisor.synthesize(results)

        self.stats["end_time"] = time.time()

        # Print results
        print(f"\n{'='*60}")
        print("RESEARCH REPORT")
        print(f"{'='*60}")
        print(report)

        # Print statistics
        print(f"\n{'='*60}")
        print("WORKFLOW STATISTICS")
        print(f"{'='*60}")

        duration = self.stats["end_time"] - self.stats["start_time"]
        runtime_stats = self.runtime.get_stats()

        print(f"  Total duration: {duration:.2f}s")
        print(f"  Tasks completed: {self.stats['tasks_completed']}")
        print(f"  Parallel batches: {self.stats['parallel_executions']}")

        if "dtr" in runtime_stats:
            dtr = runtime_stats["dtr"]
            print(f"  DTR cache hit rate: {dtr.get('hit_rate', 0):.1%}")
            print(f"  Item reuse rate: {dtr.get('item_reuse_rate', 0):.1%}")

        return {
            "report": report,
            "results": results,
            "stats": {
                **self.stats,
                "runtime": runtime_stats,
                "mesh": self.mesh.get_stats()
            }
        }


# =============================================================================
# Programmatic Entry Point
# =============================================================================

async def run_demo(
    topic: str = "The impact of large language models on software engineering",
    num_workers: int = 3,
    llm_endpoint: str = "http://localhost:8000/v1",
    enable_dtr: bool = True,
    enable_bpp: bool = True,
    enable_des: bool = True,
) -> Dict[str, Any]:
    """
    Run the deep research demo programmatically.

    Args:
        topic: Research topic to investigate.
        num_workers: Number of researcher agents.
        llm_endpoint: LLM API endpoint.
        enable_dtr: Enable DTR delta retrieval.
        enable_bpp: Enable BPP branch-parallel prefill.
        enable_des: Enable DES adaptive chunk streaming.

    Returns:
        Dict containing the research report and statistics.
    """
    workflow = DeepResearchWorkflow(
        num_researchers=num_workers,
        enable_dtr=enable_dtr,
        enable_bpp=enable_bpp,
        enable_des=enable_des,
        llm_endpoint=llm_endpoint
    )

    result = await workflow.run(topic)
    return result


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    """Main entry point for the demo."""
    parser = argparse.ArgumentParser(
        description="Deep Research Demo with AgentMesh optimization"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="The impact of large language models on software engineering",
        help="Research topic"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=3,
        help="Number of researcher agents"
    )
    parser.add_argument(
        "--llm-backend",
        type=str,
        default="http://localhost:8000/v1",
        help="LLM API endpoint"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="LLM model name"
    )
    parser.add_argument(
        "--enable-dtr",
        action="store_true",
        default=True,
        help="Enable DTR delta retrieval"
    )
    parser.add_argument(
        "--enable-bpp",
        action="store_true",
        default=True,
        help="Enable BPP branch-parallel prefill"
    )
    parser.add_argument(
        "--enable-des",
        action="store_true",
        default=True,
        help="Enable DES adaptive chunk streaming"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run workflow
    workflow = DeepResearchWorkflow(
        num_researchers=args.num_workers,
        enable_dtr=args.enable_dtr,
        enable_bpp=args.enable_bpp,
        enable_des=args.enable_des,
        llm_endpoint=args.llm_backend
    )

    result = await workflow.run(args.topic)

    return result


if __name__ == "__main__":
    asyncio.run(main())
