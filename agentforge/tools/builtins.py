"""Built-in tools.

Note: PythonReplTool runs code in a subprocess with resource limits. For
production multi-tenant deployments you'd want stronger isolation (gVisor,
Firecracker). The subprocess approach is fine for single-tenant or trusted
contexts. See docs/DECISIONS.md D-005.
"""
from __future__ import annotations

import asyncio
import logging
import os
import resource
import shutil
import sys
import tempfile
from pathlib import Path

import httpx

from ..types import ToolResult
from .base import ToolSpec

log = logging.getLogger(__name__)


# ---- Python REPL ----------------------------------------------------------

PYTHON_REPL_DESCRIPTION = (
    "Execute Python code in a sandboxed subprocess and return stdout/stderr. "
    "Limited to 30 seconds and 512MB memory. Useful for calculations, "
    "data manipulation, and small scripts. No persistent state between calls."
)

PYTHON_REPL_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": "Python source to execute. Use print() to return values.",
        },
    },
    "required": ["code"],
}


def _set_subprocess_limits() -> None:
    """Called in the child process before exec. Sets RLIMITs.
    Only works on Unix; on Windows the limits are not enforced."""
    if sys.platform == "win32":
        return
    # 512 MB memory cap
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    # 30 seconds CPU
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    # No new processes
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))


class PythonReplTool:
    spec = ToolSpec(
        name="python_repl",
        description=PYTHON_REPL_DESCRIPTION,
        parameters=PYTHON_REPL_SCHEMA,
        requires_sandbox=True,
        timeout_seconds=35,  # slightly more than CPU limit to allow clean termination
    )

    async def run(self, code: str) -> ToolResult:
        # Write code to a tempfile so we don't have to deal with shell escaping
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmppath = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",  # isolated mode: no user site-packages, no PYTHON* env vars
                tmppath,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=_set_subprocess_limits if sys.platform != "win32" else None,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ToolResult(
                    tool_call_id="",
                    output="Error: code execution timed out after 30s",
                    is_error=True,
                )

            output = stdout.decode("utf-8", errors="replace")
            err = stderr.decode("utf-8", errors="replace")
            if proc.returncode != 0:
                return ToolResult(
                    tool_call_id="",
                    output=f"Exit code {proc.returncode}\nstdout:\n{output}\nstderr:\n{err}",
                    is_error=True,
                    metadata={"returncode": proc.returncode},
                )
            return ToolResult(
                tool_call_id="",
                output=output if output else "(no output)",
                metadata={"stderr": err} if err else {},
            )
        finally:
            try:
                os.unlink(tmppath)
            except OSError:
                pass


# ---- Web search -----------------------------------------------------------

# Uses Tavily by default because it's free for low volume and returns clean
# results. Falls back to a no-op if no key configured so demos don't crash.
# To swap to Brave/SerpAPI/etc, replace _do_search().

WEB_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query."},
        "max_results": {
            "type": "integer",
            "description": "Number of results (default 5, max 10).",
            "default": 5,
        },
    },
    "required": ["query"],
}


class WebSearchTool:
    spec = ToolSpec(
        name="web_search",
        description="Search the web and return a list of titles, URLs, and snippets.",
        parameters=WEB_SEARCH_SCHEMA,
    )

    async def run(self, query: str, max_results: int = 5) -> ToolResult:
        max_results = min(max_results, 10)
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            return ToolResult(
                tool_call_id="",
                output=(
                    "Web search is not configured. Set TAVILY_API_KEY to enable. "
                    "Skipping search; answer from training knowledge."
                ),
                is_error=False,
                metadata={"reason": "no_api_key"},
            )

        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                },
            )
            r.raise_for_status()
            data = r.json()

        lines = []
        for i, result in enumerate(data.get("results", []), 1):
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("content", "")[:300]
            lines.append(f"[{i}] {title}\n    {url}\n    {content}")
        return ToolResult(
            tool_call_id="",
            output="\n\n".join(lines) if lines else "No results found.",
            metadata={"num_results": len(lines)},
        )


# ---- File IO --------------------------------------------------------------

# Both file tools are restricted to a workspace directory to prevent path
# traversal attacks. The workspace is per-run and cleaned up at run end.

WORKSPACE_ROOT = Path(os.environ.get("AGENTFORGE_WORKSPACE", "/tmp/agentforge"))


def _safe_path(filename: str) -> Path:
    """Resolve filename under WORKSPACE_ROOT, rejecting path traversal."""
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    p = (WORKSPACE_ROOT / filename).resolve()
    if not str(p).startswith(str(WORKSPACE_ROOT.resolve())):
        raise ValueError(f"Path escapes workspace: {filename}")
    return p


class ReadFileTool:
    spec = ToolSpec(
        name="read_file",
        description="Read a file from the agent workspace. Returns the file contents as text.",
        parameters={
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Path relative to workspace."},
            },
            "required": ["filename"],
        },
    )

    async def run(self, filename: str) -> ToolResult:
        path = _safe_path(filename)
        if not path.exists():
            return ToolResult(tool_call_id="", output=f"File not found: {filename}", is_error=True)
        if path.stat().st_size > 1_000_000:
            return ToolResult(
                tool_call_id="",
                output=f"File too large ({path.stat().st_size} bytes). Use a tool to chunk.",
                is_error=True,
            )
        return ToolResult(tool_call_id="", output=path.read_text(encoding="utf-8", errors="replace"))


class WriteFileTool:
    spec = ToolSpec(
        name="write_file",
        description="Write text to a file in the agent workspace, creating directories as needed.",
        parameters={
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Path relative to workspace."},
                "content": {"type": "string", "description": "Text to write."},
            },
            "required": ["filename", "content"],
        },
    )

    async def run(self, filename: str, content: str) -> ToolResult:
        path = _safe_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolResult(
            tool_call_id="",
            output=f"Wrote {len(content)} bytes to {filename}",
            metadata={"path": str(path), "bytes": len(content)},
        )


def cleanup_workspace() -> None:
    """Call this between runs to keep the workspace from growing unbounded."""
    if WORKSPACE_ROOT.exists():
        shutil.rmtree(WORKSPACE_ROOT)
