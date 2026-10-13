# MCP: any client

Run the shell through `stalecheck-mcp` and turn the agent's built-in shell off, so the model uses
`run_command` (and `read_file`). Server entry (the key name varies by client):

```json
{"mcpServers": {"stalecheck": {"command": "stalecheck-mcp", "env": {"STALECHECK_ROOT": "/path/to/repo"}}}}
```

Turning the built-in shell off:

| Client | Setting |
|---|---|
| Claude Code | `--disallowedTools Bash` or `permissions.deny: ["Bash"]` |
| Codex CLI | `[features] shell_tool = false` in `config.toml` |
| Gemini CLI | a policy rule `toolName = "run_shell_command"`, `decision = "deny"` in `~/.gemini/policies/*.toml` |
| Copilot CLI | `--excluded-tools=bash` |
| OpenCode | `"tools": {"bash": false}` |
| Goose | disable the `developer` extension in `~/.config/goose/config.yaml` |
| Cline (SDK/CLI) | `toolPolicies: {run_commands: {enabled: false}}` |
| Cursor | no switch documented; refuse `Shell` in `beforeShellExecution` with an `agent_message` pointing to `run_command` |
