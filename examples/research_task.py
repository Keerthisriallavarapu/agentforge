"""Run a sample task end-to-end. Requires ANTHROPIC_API_KEY in env.

Usage:
    python examples/research_task.py
"""
from __future__ import annotations

import asyncio
import logging

from agentforge import Runtime


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    runtime = Runtime(max_revisions=2)

    goal = (
        "Calculate the compound annual growth rate (CAGR) for a portfolio that grew "
        "from $10,000 to $34,500 over 7 years. Show your work."
    )

    result = await runtime.run(goal)

    print("\n" + "=" * 60)
    print(f"Status: {result.status.value}")
    print(f"Revisions: {result.state.revision_count}")
    print(f"Cost: ${result.cost_usd:.4f}")
    print("=" * 60)
    print(result.final_output)
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
