"""Write-through for coding agents: an engine (engine.Session) and adapters for Claude Code
(claude_code), any hook system that can run a command (protocol) and any MCP client (mcp_server)."""

from .engine import Session  # noqa: F401
