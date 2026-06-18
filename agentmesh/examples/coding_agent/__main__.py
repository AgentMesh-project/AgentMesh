"""
Entry point for running coding_agent as a module.

Usage:
    python -m agentmesh.examples.coding_agent \
        --issue "fix the race in process()" --num-workers 3 \
        --llm-backend http://localhost:8000/v1 --model Qwen/Qwen3-4B
"""

import asyncio

from .demo import main

if __name__ == "__main__":
    asyncio.run(main())
