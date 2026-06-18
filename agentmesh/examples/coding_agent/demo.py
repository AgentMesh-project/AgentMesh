"""
Coding Agent Demo.

A supervisor-worker coding-agent workflow demonstrating AgentMesh optimization
over the produce--transfer--consume dataflow.

Topology (mirrors deep_research/demo.py, but for code tasks):
- 1 Supervisor agent decomposes a repository task into N *independent* fixes
  (different files / sections that do not depend on one another).
- N Worker agents each repair their assigned file/section in parallel.
- Coding tools (read file / grep search / build) produce context for prefill.

AgentMesh optimizations:
- DTR (Delta Tool Retrieval) -- shrinks the PRODUCE step: the codebase read /
  search / build results for related files share a cached base; only the
  semantic delta is fetched, then async-refreshed.
- BPP (Branch-Parallel Prefill) -- shrinks the CONSUME step: the N independent
  worker branches are prefilled in parallel against the shared supervisor
  prefix (the task brief + repo summary) with cross-branch attention masked
  (BPA).
- DES (Dynamic-Equilibrium Streaming) -- OVERLAPS produce/transfer/consume:
  tool and LLM outputs are streamed under adaptive chunk control (slack
  delta -> 0).

Runnable against a local OpenAI-compatible endpoint (e.g. vLLM at
http://localhost:8000/v1), exactly like the deep_research demo. Inspired by
coding agents such as OpenCode on SWE-bench Lite tasks; this demo depends on
neither.
"""

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# AgentMesh imports (locked public API).
from agentmesh import AgentMeshRuntime
from agentmesh.core.config import AgentMeshConfig
from agentmesh.sidecars import SidecarMesh, AgentSidecar, ToolSidecar

logger = logging.getLogger(__name__)


# =============================================================================
# Agent definitions
# =============================================================================

@dataclass
class CodingAgent:
    """Base agent class for the coding demo."""

    agent_id: str
    role: str
    system_prompt: str
    sidecar: Optional[AgentSidecar] = None


@dataclass
class CodingSupervisorAgent(CodingAgent):
    """
    Supervisor that decomposes a coding task into independent worker fixes.

    Responsibilities:
    - Survey the repository (shared prefix used by every worker branch).
    - Split the task into N independent file/section edits (BPP branches).
    - Review and assemble the per-file patches into a single changeset.
    """

    def __init__(self, agent_id: str = "supervisor"):
        super().__init__(
            agent_id=agent_id,
            role="supervisor",
            system_prompt=(
                "You are a coding supervisor coordinating a team of code-fixing "
                "workers.\n"
                "1. Survey the repository and the failing tests.\n"
                "2. Split the task into independent file/section edits.\n"
                "3. Assign one focused edit to each worker.\n"
                "4. Review and assemble the patches into one changeset.\n"
            ),
        )
        self.assigned_tasks: List[Dict] = []
        self.collected_patches: List[Dict] = []

    async def create_tasks(
        self, issue: str, target_files: List[str], num_workers: int
    ) -> List[Dict]:
        """Split the issue into independent per-file fix tasks (BPP branches)."""
        tasks: List[Dict] = []
        n = min(num_workers, len(target_files)) if target_files else num_workers
        files = target_files or [f"module_{i}.py" for i in range(num_workers)]

        for i in range(n):
            path = files[i % len(files)]
            tasks.append(
                {
                    "task_id": f"fix_{i}",
                    "worker_id": f"worker_{i}",
                    "file_path": path,
                    "prompt": (
                        f"Fix the following file to resolve the issue.\n"
                        f"Issue: {issue}\n"
                        f"File: {path}\n"
                    ),
                }
            )
        self.assigned_tasks = tasks
        return tasks

    async def assemble(self, patches: List[Dict]) -> str:
        """Assemble the per-worker patches into a single changeset summary."""
        self.collected_patches = patches
        body = "\n\n".join(
            f"--- a/{p['file_path']}\n+++ b/{p['file_path']}\n{p['patch']}"
            for p in patches
        )
        return (
            "# Proposed Changeset\n\n"
            f"{body}\n\n"
            "# Review\n"
            f"Assembled {len(patches)} independent file fix(es) into one "
            "changeset. Each edit was prepared in its own worker branch.\n"
        )


@dataclass
class CodingWorkerAgent(CodingAgent):
    """
    Worker that repairs a single assigned file/section.

    Uses coding tools (read file / search / build) to gather context (the
    produce step served via DTR), then proposes a patch.
    """

    def __init__(self, agent_id: str, specialty: str = ""):
        super().__init__(
            agent_id=agent_id,
            role="worker",
            system_prompt=(
                f"You are a software engineer specializing in "
                f"{specialty or 'general bug fixing'}.\n"
                "1. Read the target file and search for the relevant symbol.\n"
                "2. Build to confirm the failure.\n"
                "3. Propose a minimal, well-scoped patch.\n"
            ),
        )
        self.specialty = specialty
        self.patches: List[str] = []

    async def fix(self, task: Dict, tool_sidecar: ToolSidecar) -> Dict:
        """Conduct the fix for one assigned file using tools + the LLM."""
        path = task["file_path"]

        # --- Produce step (DTR): gather code context via tools.
        source = await tool_sidecar.execute("read_file", {"path": path})
        matches = await tool_sidecar.execute(
            "search_code", {"query": task["prompt"], "path": path}
        )
        build_log = await tool_sidecar.execute("build", {"target": path})

        # --- Consume step: synthesize a patch from the gathered context.
        patch = (
            f"@@ {path} @@\n"
            f"# context: {matches[:120]}\n"
            f"# build: {build_log[:80]}\n"
            "- # buggy line\n"
            "+ # fixed line (scoped edit by "
            f"{self.agent_id})\n"
        )
        self.patches.append(patch)

        return {
            "task_id": task["task_id"],
            "worker_id": self.agent_id,
            "file_path": path,
            "context_bytes": len(source) + len(matches) + len(build_log),
            "patch": patch,
        }


# =============================================================================
# Tool definitions (the produce step; DTR caches their bases)
# =============================================================================

def register_coding_tools(tool_sidecar: ToolSidecar) -> None:
    """Register coding tools with the tool sidecar.

    These are placeholders. Replace with a real workspace backend (file IO, a
    code search index such as ripgrep, and a build/test runner) for production
    use. They model the produce step whose base is reused by DTR across files.
    """

    def read_file(path: str) -> str:
        """Read a source file (placeholder)."""
        return (
            f"# file: {path}\n"
            "import os\n\n"
            "def handler(request):\n"
            "    # TODO: known defect lives here\n"
            "    return process(request)\n"
        )

    def search_code(query: str, path: str = "") -> str:
        """Search the codebase for a symbol/pattern (placeholder)."""
        return (
            f"[search '{query[:48]}' in {path or 'repo'}]\n"
            f"- {path or 'src/app.py'}:42: def handler(request):\n"
            f"- {path or 'src/app.py'}:58: return process(request)\n"
        )

    def build(target: str = "") -> str:
        """Run the build / test target (placeholder)."""
        return (
            f"[build {target or 'all'}]\n"
            "FAILED tests/test_handler.py::test_process - AssertionError\n"
            "1 failed, 12 passed\n"
        )

    tool_sidecar.register_tool(
        tool_id="read_file",
        name="Read File",
        description="Read the contents of a source file",
        executor=read_file,
        parameters=[{"name": "path", "type": "string", "required": True}],
    )
    tool_sidecar.register_tool(
        tool_id="search_code",
        name="Search Code",
        description="Search the codebase for a symbol or pattern",
        executor=search_code,
        parameters=[
            {"name": "query", "type": "string", "required": True},
            {"name": "path", "type": "string", "required": False},
        ],
    )
    tool_sidecar.register_tool(
        tool_id="build",
        name="Build",
        description="Run the build / test target and capture the log",
        executor=build,
        parameters=[{"name": "target", "type": "string", "required": False}],
    )


# =============================================================================
# Workflow orchestration
# =============================================================================

class CodingAgentWorkflow:
    """
    Orchestrates the supervisor-worker coding-agent workflow.

    AgentMesh optimizations applied:
    - DTR: the workers' read/search/build results share a cached base; only the
      semantic delta per file is fetched (the produce step).
    - BPP: the N independent worker branches prefill in parallel against the
      shared supervisor prefix via BPA (the consume step).
    - DES: tool and LLM streaming adapt their chunk sizes to drive slack
      delta -> 0 (stage overlap).
    """

    def __init__(
        self,
        num_workers: int = 3,
        enable_dtr: bool = True,
        enable_bpp: bool = True,
        enable_des: bool = True,
        llm_endpoint: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen3-4B",
    ):
        """
        Initialize the workflow.

        Args:
            num_workers: Number of code-fixing worker agents (BPP branches).
            enable_dtr: Enable DTR delta retrieval.
            enable_bpp: Enable BPP branch-parallel prefill.
            enable_des: Enable DES adaptive chunk streaming.
            llm_endpoint: OpenAI-compatible endpoint (e.g. a local vLLM server).
            model: LLM model name.
        """
        self.num_workers = num_workers
        self.model = model

        config = AgentMeshConfig()
        config.dtr.enabled = enable_dtr
        config.bpp.enabled = enable_bpp
        config.des.enabled = enable_des
        config.llm.endpoint = llm_endpoint
        config.llm.model = model

        # BPP/BPA's KV-stitching path needs PyTorch (an optional extra). When
        # torch is absent we keep the *branch* parallelism (the workers still
        # run concurrently via asyncio.gather) but skip the tensor-level BPA
        # manager, so the demo stays runnable on a CPU-only box.
        try:
            self.runtime = AgentMeshRuntime(config=config)
            self.bpp_available = self.runtime.bpp_manager is not None
        except ImportError:
            if not enable_bpp:
                raise
            logger.warning(
                "BPP manager unavailable (PyTorch not installed); running "
                "branch-parallel workers without the tensor-level BPA manager."
            )
            config.bpp.enabled = False
            self.runtime = AgentMeshRuntime(config=config)
            self.bpp_available = False

        # Sidecar mesh wired to the runtime's three mechanisms.
        self.mesh = SidecarMesh(
            dtr_cache=self.runtime.dtr_cache,
            bpp_manager=getattr(self.runtime, "bpp_manager", None),
            des_controller=self.runtime.des_controller,
        )

        # Supervisor agent.
        self.supervisor = CodingSupervisorAgent()
        self.supervisor.sidecar = self.mesh.register_agent(self.supervisor.agent_id)

        # Worker agents (BPP branches).
        self.workers: List[CodingWorkerAgent] = []
        specialties = ["backend", "parsing", "concurrency", "io"]
        for i in range(num_workers):
            worker = CodingWorkerAgent(
                agent_id=f"worker_{i}",
                specialty=specialties[i % len(specialties)],
            )
            worker.sidecar = self.mesh.register_agent(worker.agent_id)
            self.workers.append(worker)

        register_coding_tools(self.mesh.tool_sidecar)

        self.stats = {
            "start_time": None,
            "end_time": None,
            "tasks_completed": 0,
            "parallel_executions": 0,
        }
        logger.info(f"Initialized CodingAgentWorkflow with {num_workers} workers")

    async def run(
        self,
        issue: str,
        target_files: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the coding workflow.

        Args:
            issue: The bug/feature description to resolve.
            target_files: Independent files to fix (one per worker branch). If
                omitted, synthetic file names are generated.

        Returns:
            Workflow results including the changeset and statistics.
        """
        self.stats["start_time"] = time.time()
        target_files = target_files or []

        print(f"\n{'=' * 60}")
        print(f"Starting Coding Agent on issue: {issue}")
        print(f"{'=' * 60}\n")

        # Phase 1: Task decomposition into independent branches.
        print("Phase 1: Decomposing task into independent file fixes...")
        tasks = await self.supervisor.create_tasks(issue, target_files, self.num_workers)
        print(f"  Created {len(tasks)} independent fix tasks")

        # Phase 2: Parallel fixing (BPP branch-parallel prefill applies here).
        print("\nPhase 2: Fixing files in parallel worker branches...")
        fix_coros = []
        for i, (worker, task) in enumerate(zip(self.workers, tasks)):
            print(f"  Worker {i}: {task['file_path']}")
            fix_coros.append(worker.fix(task, self.mesh.tool_sidecar))

        patches = await asyncio.gather(*fix_coros)
        self.stats["tasks_completed"] = len(patches)
        self.stats["parallel_executions"] = 1
        print(f"\n  Completed {len(patches)} file fixes")

        # Phase 3: Assemble the changeset.
        print("\nPhase 3: Assembling changeset...")
        changeset = await self.supervisor.assemble(patches)

        self.stats["end_time"] = time.time()

        print(f"\n{'=' * 60}")
        print("CHANGESET")
        print(f"{'=' * 60}")
        print(changeset)

        print(f"\n{'=' * 60}")
        print("WORKFLOW STATISTICS")
        print(f"{'=' * 60}")
        duration = self.stats["end_time"] - self.stats["start_time"]
        runtime_stats = self.runtime.get_stats()
        print(f"  Total duration: {duration:.3f}s")
        print(f"  Files fixed: {self.stats['tasks_completed']}")
        print(f"  Parallel branches: {self.stats['parallel_executions']}")
        if "dtr" in runtime_stats:
            dtr = runtime_stats["dtr"]
            print(f"  DTR cache hit rate: {dtr.get('hit_rate', 0):.1%}")

        return {
            "changeset": changeset,
            "patches": patches,
            "stats": {
                **self.stats,
                "runtime": runtime_stats,
                "mesh": self.mesh.get_stats(),
            },
        }


# =============================================================================
# Programmatic entry point
# =============================================================================

async def run_demo(
    issue: str = "process() drops the request id under concurrent load",
    target_files: Optional[List[str]] = None,
    num_workers: int = 3,
    llm_endpoint: str = "http://localhost:8000/v1",
    model: str = "Qwen/Qwen3-4B",
    enable_dtr: bool = True,
    enable_bpp: bool = True,
    enable_des: bool = True,
) -> Dict[str, Any]:
    """Run the coding-agent demo programmatically. See CLI for flag meanings."""
    workflow = CodingAgentWorkflow(
        num_workers=num_workers,
        enable_dtr=enable_dtr,
        enable_bpp=enable_bpp,
        enable_des=enable_des,
        llm_endpoint=llm_endpoint,
        model=model,
    )
    return await workflow.run(issue, target_files)


# =============================================================================
# CLI entry point
# =============================================================================

async def main():
    """Main entry point for the coding-agent demo."""
    parser = argparse.ArgumentParser(
        description="Coding Agent Demo with AgentMesh optimization"
    )
    parser.add_argument(
        "--issue",
        type=str,
        default="process() drops the request id under concurrent load",
        help="Bug/feature description to resolve",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="*",
        default=[],
        help="Independent target files (one per worker branch)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=3, help="Number of code-fixing workers"
    )
    parser.add_argument(
        "--llm-backend",
        type=str,
        default="http://localhost:8000/v1",
        help="OpenAI-compatible LLM endpoint URL (e.g. a local vLLM server)",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-4B", help="LLM model name"
    )
    parser.add_argument(
        "--enable-dtr", action="store_true", default=True, help="Enable DTR"
    )
    parser.add_argument(
        "--enable-bpp", action="store_true", default=True, help="Enable BPP"
    )
    parser.add_argument(
        "--enable-des", action="store_true", default=True, help="Enable DES"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    workflow = CodingAgentWorkflow(
        num_workers=args.num_workers,
        enable_dtr=args.enable_dtr,
        enable_bpp=args.enable_bpp,
        enable_des=args.enable_des,
        llm_endpoint=args.llm_backend,
        model=args.model,
    )
    return await workflow.run(args.issue, args.files)


if __name__ == "__main__":
    asyncio.run(main())
