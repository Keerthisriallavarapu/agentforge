"""Command-line interface. Mainly useful for one-off testing and CI runs."""
from __future__ import annotations

import asyncio
import logging

import typer
from rich.console import Console
from rich.panel import Panel

from .runtime import Runtime

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def run(
    goal: str = typer.Argument(..., help="What you want the agents to do."),
    max_revisions: int = typer.Option(3, help="Max critic revision loops."),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Execute a goal end-to-end and print the result."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )

    async def _go():
        runtime = Runtime(max_revisions=max_revisions)
        result = await runtime.run(goal)
        console.print(Panel.fit(
            result.final_output or "(no output)",
            title=f"Run {result.state.id} — {result.status.value}",
        ))
        console.print(
            f"\nTokens: in={result.state.total_input_tokens} "
            f"out={result.state.total_output_tokens}  "
            f"Cost: ${result.cost_usd:.4f}  "
            f"Revisions: {result.state.revision_count}"
        )

    asyncio.run(_go())


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind address."),
    port: int = typer.Option(8000, help="Port."),
):
    """Start the HTTP API server."""
    import uvicorn

    uvicorn.run("agentforge.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
