"""
Entry point for running deep_research as a module.

Usage:
    python -m agentmesh.examples.deep_research --topic "Your topic" --num-workers 3
"""

import asyncio
from .demo import main

if __name__ == "__main__":
    asyncio.run(main())
