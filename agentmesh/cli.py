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
    """Run benchmarks for a given mechanism."""
    setup_logging(args.log_level)

    if args.component == "dtr":
        from benchmarks.benchmark_dtr import main as run_dtr
        run_dtr()
    elif args.component == "bpp":
        from benchmarks.benchmark_bpp import main as run_bpp
        run_bpp()
    elif args.component == "des":
        from benchmarks.benchmark_des import main as run_des
        run_des()
    elif args.component == "all":
        from benchmarks.benchmark_dtr import main as run_dtr
        from benchmarks.benchmark_bpp import main as run_bpp
        from benchmarks.benchmark_des import main as run_des
        print("=== DTR Benchmark ===")
        run_dtr()
        print("\n=== BPP Benchmark ===")
        run_bpp()
        print("\n=== DES Benchmark ===")
        run_des()
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
        description=(
            "AgentMesh: Toward Redundancy- and Barrier-Free Dataflow in "
            "Multi-Agent Systems"
        ),
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
        choices=["dtr", "bpp", "des", "all"],
        help="Mechanism to benchmark (dtr, bpp, des, or all)",
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
