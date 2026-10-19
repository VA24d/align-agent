"""Write-through for coding agents: an engine (engine.Session) and adapters for Claude Code
(claude_code), any hook system that can run a command (protocol) and any MCP client (mcp_server)."""

try:
    from .engine import Session  # noqa: F401
except Exception:  # noqa: BLE001  the hook adapters report it and fail open; a broken engine must not block an agent
    pass
