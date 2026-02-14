"""
AgentMesh Command Line Interface

Provides CLI commands for running demos, benchmarks, and utilities.
"""

import argparse
import asyncio
import logging
import sys
from typing import Optional

from agentmesh import __version__


def setup_logging(level: str = "INFO"):
    """Configure logging format and level."""
    import coloredlogs
    coloredlogs.install(
        level=level,
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


def cmd_demo(args):
    """Run the deep research demo."""
    setup_logging(args.log_level)
    
    from agentmesh.examples.deep_research.demo import run_demo
    asyncio.run(run_demo(
        topic=args.topic,
        num_workers=args.workers,
    ))


def cmd_benchmark(args):
    """Run benchmarks."""
    setup_logging(args.log_level)
    
    if args.component == "srr":
        from benchmarks.benchmark_srr import main as run_srr
        run_srr()
    elif args.component == "tsd":
        from benchmarks.benchmark_tsd import main as run_tsd
        run_tsd()
    elif args.component == "esf":
        from benchmarks.benchmark_esf import main as run_esf
        run_esf()
    elif args.component == "all":
        from benchmarks.benchmark_srr import main as run_srr
        from benchmarks.benchmark_tsd import main as run_tsd
        from benchmarks.benchmark_esf import main as run_esf
        print("=== SRR Benchmark ===")
        run_srr()
        print("\n=== TSD Benchmark ===")
        run_tsd()
        print("\n=== ESF Benchmark ===")
        run_esf()
    else:
        print(f"Unknown component: {args.component}")
        sys.exit(1)


def cmd_version(args):
    """Print version information."""
    print(f"AgentMesh v{__version__}")


def main(argv: Optional[list] = None):
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="agentmesh",
        description="AgentMesh: Semantic Communication for Multi-Agent Systems",
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run deep research demo")
    demo_parser.add_argument(
        "--topic", "-t",
        default="Advances in Large Language Models",
        help="Research topic for the demo",
    )
    demo_parser.add_argument(
        "--workers", "-w",
        type=int,
        default=3,
        help="Number of worker agents",
    )
    demo_parser.add_argument(
        "--log-level", "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    demo_parser.set_defaults(func=cmd_demo)
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument(
        "component",
        choices=["srr", "tsd", "esf", "all"],
        help="Component to benchmark",
    )
    bench_parser.add_argument(
        "--log-level", "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    bench_parser.set_defaults(func=cmd_benchmark)
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Show version")
    version_parser.set_defaults(func=cmd_version)
    
    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    args.func(args)


if __name__ == "__main__":
    main()
