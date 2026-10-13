"""Native hook adapters: `stalecheck-hook <agent> [event]` reads one hook event as JSON on stdin and
writes the agent's own reply format on stdout. Field names and reply formats follow each agent's
documentation as of October 2026 (see tool/integrations/README.md for the sources and configs).

    agent      events handled                                    after a tool            before a shell command
    claude     SessionStart, PreToolUse, PostToolUse              additionalContext       permissionDecision deny
    codex      SessionStart, PostCompact, PreToolUse, PostToolUse additionalContext       permissionDecision deny
    gemini     SessionStart, PreCompress, BeforeTool, AfterTool   additionalContext       decision deny
    cursor     beforeShellExecution, beforeReadFile, postToolUse, additional_context      permission deny + agent_message
               afterFileEdit
    copilot    sessionStart, preToolUse, postToolUse,             additionalContext       permissionDecision deny
               postToolUseFailure (event given as an argument)
    cline      PreToolUse, PostToolUse (file hooks; event given   contextModification     (none: cancelling ends the task)
               as an argument)

Agents without a hook that can add text after a tool (Goose, Antigravity) use the MCP server
(stalecheck-mcp) instead. Every adapter fails open: an error is logged and the agent continues.
"""

import json
import os
import re
import sys
import time

from .claude_code import handle as claude_handle
from .engine import Session, default_state_dir

PATCH_FILE = re.compile(r"(?m)^\*\*\* (?:Update|Add|Delete) File: (.+)$")


def _out_text(resp):
    if isinstance(resp, dict):
        for k in ("output", "stdout", "llmContent", "textResultForLlm", "result"):
            if resp.get(k):
                v = resp[k]
                return v if isinstance(v, str) else json.dumps(v)
        return json.dumps(resp)
    if isinstance(resp, str):
        try:
            return _out_text(json.loads(resp))
        except ValueError:
            return resp
    return str(resp or "")


def _args(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except ValueError:
            return {"command": a}
    return a or {}


# ---------------------------------------------------------------- Codex CLI

def codex(p, event=None):
    """Claude Code's schema, but unknown output fields fail the hook, there is no read tool (reads
    go through the shell), edits arrive as apply_patch text, and added context over about 2,500
    tokens is spilled to a file (raise it with additionalContextLimit on the handler)."""
    event = event or p.get("hook_event_name")
    root = p.get("cwd") or os.getcwd()
    with Session(root, p.get("session_id") or "codex", budget=6000) as s:
        if event in ("SessionStart", "PostCompact"):
            s.start(fresh=event == "PostCompact" or p.get("source") != "resume")
            return None
        tool, inp = p.get("tool_name") or "", _args(p.get("tool_input"))
        if event == "PreToolUse":
            reason = s.before(inp.get("command", "")) if tool == "Bash" else None
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                           "permissionDecisionReason": reason}} if reason else None
        if event != "PostToolUse":
            return None
        command = inp.get("command", "") if tool == "Bash" else ""
        authored = []
        if tool in ("apply_patch", "Edit", "Write"):
            text = inp.get("command") or inp.get("patch") or ""
            authored = PATCH_FILE.findall(text) or [inp.get("file_path") or inp.get("path")]
        ctx = s.after(command, _out_text(p.get("tool_response")) if command else "", authored=[a for a in authored if a])
        return {"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": ctx}} if ctx else None


# ---------------------------------------------------------------- Gemini CLI

GEMINI_READ = {"read_file": ("file_path", "offset", "limit"), "read_many_files": None}
GEMINI_EDIT = {"write_file", "replace", "edit"}


def gemini(p, event=None):
    """BeforeTool/AfterTool. read_file's offset is 0-based. Added context is appended to the tool's
    result inside <hook_context>; a deny reaches the model as 'Tool execution blocked: <reason>'."""
    event = event or p.get("hook_event_name")
    root = os.environ.get("GEMINI_PROJECT_DIR") or p.get("cwd") or os.getcwd()
    with Session(root, p.get("session_id") or "gemini") as s:
        if event in ("SessionStart", "PreCompress"):
            s.start(fresh=event == "PreCompress" or p.get("source") not in ("resume",))
            return None
        tool, inp = p.get("tool_name") or "", _args(p.get("tool_input"))
        if event == "BeforeTool":
            reason = s.before(inp.get("command", "")) if tool == "run_shell_command" else None
            return {"decision": "deny", "reason": reason} if reason else None
        if event != "AfterTool":
            return None
        command = inp.get("command", "") if tool == "run_shell_command" else ""
        read = rng = None
        if tool == "read_file":
            read = inp.get("file_path") or inp.get("absolute_path")
            if inp.get("offset") is not None or inp.get("limit") is not None:
                lo = int(inp.get("offset") or 0) + 1
                rng = (lo, lo + int(inp.get("limit") or 2000) - 1)
        authored = inp.get("file_path") if tool in GEMINI_EDIT else None
        ctx = s.after(command, _out_text(p.get("tool_response")) if command else "", read=read, read_range=rng,
                      authored=authored)
        return {"hookSpecificOutput": {"additionalContext": ctx}} if ctx else None


# ---------------------------------------------------------------- Cursor

def cursor(p, event=None):
    """beforeShellExecution refuses with agent_message (the model reads it; user_message is for the
    person). afterShellExecution cannot add text, so the diff goes through postToolUse. A permission
    hook must always print valid JSON, or Cursor blocks the action."""
    event = event or p.get("hook_event_name")
    roots = p.get("workspace_roots") or []
    root = (roots[0] if roots else None) or p.get("cwd") or os.getcwd()
    with Session(root, p.get("conversation_id") or "cursor") as s:
        if event == "beforeShellExecution":
            reason = s.before(p.get("command", ""))
            if reason:
                return {"permission": "deny", "agent_message": reason,
                        "user_message": "stalecheck: line numbers in this command are out of date; the agent was shown the current lines."}
            return {"permission": "allow"}
        if event == "beforeReadFile":
            s.read(p.get("file_path"))
            return {"permission": "allow"}
        if event == "afterFileEdit":
            s.authored(p.get("file_path"))
            return {}
        if event != "postToolUse":
            return {}
        tool, inp = p.get("tool_name") or "", _args(p.get("tool_input"))
        command = inp.get("command", "") if tool == "Shell" else ""
        path = inp.get("file_path") or inp.get("path") or inp.get("target_file")
        ctx = s.after(command, _out_text(p.get("tool_output")) if command else "",
                      read=path if tool == "Read" else None, authored=path if tool in ("Write", "Edit") else None)
        return {"additional_context": ctx} if ctx else {}


# ---------------------------------------------------------------- GitHub Copilot CLI

def copilot(p, event=None):
    """camelCase events: {sessionId, cwd, toolName, toolArgs, toolResult}. postToolUse fires only on
    success, so register postToolUseFailure too. Added context is capped at 10 KB."""
    root = p.get("cwd") or os.getcwd()
    with Session(root, p.get("sessionId") or p.get("session_id") or "copilot") as s:
        if event == "sessionStart":
            s.start(fresh=True)
            return None
        tool, inp = p.get("toolName") or p.get("tool_name") or "", _args(p.get("toolArgs") or p.get("tool_input"))
        if event == "preToolUse":
            reason = s.before(inp.get("command", "")) if tool in ("bash", "Bash") else None
            return {"permissionDecision": "deny", "permissionDecisionReason": reason} if reason else None
        if event not in ("postToolUse", "postToolUseFailure"):
            return None
        command = inp.get("command", "") if tool in ("bash", "Bash") else ""
        path = inp.get("path") or inp.get("file_path")
        ctx = s.after(command, _out_text(p.get("toolResult") or p.get("tool_response")) if command else "",
                      read=path if tool in ("view", "Read") else None,
                      authored=path if tool in ("edit", "create", "Edit", "Write") else None)
        return {"additionalContext": ctx} if ctx else None


# ---------------------------------------------------------------- Cline

def cline(p, event=None):
    """File hooks. PostToolUse's contextModification reaches the model as <hook_context> after the
    tool results. PreToolUse can only cancel the whole task, so the line-number guard is not used."""
    event = event or ("PostToolUse" if "postToolUse" in p else "PreToolUse" if "preToolUse" in p else None)
    roots = p.get("workspaceRoots") or []
    root = (roots[0] if roots else None) or os.getcwd()
    with Session(root, p.get("taskId") or "cline") as s:
        if event != "PostToolUse":
            return {"cancel": False}
        post = p.get("postToolUse") or {}
        tool, prm = post.get("toolName") or "", post.get("parameters") or {}
        commands = prm.get("commands") or ([prm["command"]] if prm.get("command") else [])
        if isinstance(commands, str):
            commands = [commands]
        read = None
        if tool in ("read_file", "read_files"):
            files = prm.get("files")
            read = prm.get("path") or (files[0].get("path") if isinstance(files, list) and files else None)
        authored = prm.get("path") if tool in ("editor", "write_to_file", "replace_in_file", "apply_patch") else None
        ctx = s.after(" && ".join(commands), _out_text(post.get("result")) if commands else "", read=read, authored=authored)
        return {"cancel": False, "contextModification": ctx} if ctx else {"cancel": False}


AGENTS = {"claude": lambda p, e=None: claude_handle(p), "codex": codex, "gemini": gemini, "cursor": cursor,
          "copilot": copilot, "cline": cline}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in AGENTS:
        sys.stderr.write("usage: stalecheck-hook {" + ",".join(AGENTS) + "} [event] < event.json\n")
        return 2
    agent = argv[0]
    event = argv[1] if len(argv) > 1 else None
    try:
        payload = json.load(sys.stdin)
        event = event or payload.get("hook_event_name") or payload.get("hookName")
        reply = AGENTS[agent](payload, event)
    except Exception as e:  # noqa: BLE001  fail open
        log = os.path.join(default_state_dir(), "errors.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {agent} {type(e).__name__}: {e}\n")
        # Cursor blocks an action when a permission hook prints no valid JSON
        reply = {"permission": "allow"} if agent == "cursor" and str(event or "").startswith("before") else None
    if reply is not None:
        json.dump(reply, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
