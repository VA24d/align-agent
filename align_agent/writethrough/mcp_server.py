"""An MCP server whose shell keeps the model's view of its files current: `align-agent-mcp`.

For any MCP client (an agent, an IDE, a desktop app). Tools:
  run_command(command, timeout?)  runs a shell command in the workspace and returns its output,
                                  followed by the diff of every file the model has read that
                                  changed since (and line-shift notes); a `sed -i` on line
                                  numbers the model has not seen as they are now is refused once,
                                  with those lines shown.
  read_file(path, offset?, limit?) returns numbered lines and records what the model saw.

The workspace is $ALIGN_ROOT or the server's working directory; one session per server
process. MCP stdio transport: one JSON-RPC message per line on stdin and stdout. No dependencies.
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import uuid

from .engine import Session, read_text, rel_path

PROTOCOL = "2025-06-18"
SUPPORTED = (PROTOCOL, "2025-03-26", "2024-11-05")  # versions whose tools/* messages this server speaks
MAX_OUTPUT = 30000
TOOLS = [
    {"name": "run_command",
     "description": "Run a shell command in the workspace. After the output, lists the changes to files you have "
                    "read that this or any earlier command made, with current line numbers.",
     "inputSchema": {"type": "object", "properties": {
         "command": {"type": "string", "description": "the shell command"},
         "timeout": {"type": "number", "description": "seconds (default 120)"}}, "required": ["command"]}},
    {"name": "read_file",
     "description": "Read a file with line numbers. Prefer this to printing files with run_command.",
     "inputSchema": {"type": "object", "properties": {
         "path": {"type": "string"}, "offset": {"type": "integer", "description": "first line, 1-based"},
         "limit": {"type": "integer", "description": "number of lines"}}, "required": ["path"]}},
]


def kill_group(proc):
    """Stop a timed-out command with everything it started."""
    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    except OSError:
        pass
    try:
        proc.kill()
    except OSError:
        pass
    proc.wait()


class Server:
    def __init__(self, root=None, session=None):
        self.root = os.path.realpath(root or os.environ.get("ALIGN_ROOT") or os.getcwd())
        self.session = session or f"mcp-{uuid.uuid4().hex[:12]}"
        with Session(self.root, self.session) as s:
            s.start(fresh=True)

    def run_command(self, command, timeout=120):
        with Session(self.root, self.session) as s:
            refusal = s.before(command)
        if refusal:
            return refusal, True
        # output goes to a file, not a pipe: a command that leaves a process in the background
        # (`npm run dev &`) returns when the shell exits instead of when the pipe closes
        with tempfile.TemporaryFile() as f:
            group = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            proc = subprocess.Popen(command, shell=True, cwd=self.root, stdout=f, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, **group)  # stdin carries the MCP protocol; never hand it to a command
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                kill_group(proc)
                code = None
            f.seek(0)
            out = f.read().decode("utf-8", "replace")
        out = (out[:MAX_OUTPUT] + f"\n... output cut at {MAX_OUTPUT} characters") if len(out) > MAX_OUTPUT else out
        if code is None:
            out, err = out + f"\n[timed out after {timeout} s; the command and its processes were stopped]", True
        else:
            out, err = out + f"\n[exit code {code}]", code != 0
        with Session(self.root, self.session) as s:
            extra = s.after(command, out)
        return out + ("\n\n" + extra if extra else ""), err

    def read_file(self, path, offset=None, limit=None):
        rel = rel_path(self.root, path)
        text = read_text(self.root, rel) if rel else None
        if text is None:
            return f"cannot read {path} (outside the workspace, missing, binary or too large)", True
        lines = text.splitlines()
        lo = max(1, int(offset or 1))
        hi = min(len(lines), lo + int(limit) - 1) if limit else len(lines)
        with Session(self.root, self.session) as s:
            extra = s.after(read=rel, read_range=None if (offset is None and limit is None) else (lo, hi))
        body = "\n".join(f"{k:6d}\t{lines[k - 1]}" for k in range(lo, hi + 1))
        return body + ("\n\n" + extra if extra else ""), False

    def call(self, name, args):
        if name == "run_command":
            return self.run_command(args["command"], args.get("timeout", 120))
        if name == "read_file":
            return self.read_file(args["path"], args.get("offset"), args.get("limit"))
        return f"unknown tool {name}", True

    def handle(self, msg):
        method, mid = msg.get("method"), msg.get("id")
        if method == "initialize":
            asked = (msg.get("params") or {}).get("protocolVersion")
            ver = asked if asked in SUPPORTED else PROTOCOL  # the spec: answer with a version the server supports
            result = {"protocolVersion": ver, "capabilities": {"tools": {}},
                      "serverInfo": {"name": "align-agent", "version": "0.4.10"}}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            p = msg.get("params") or {}
            try:
                text, err = self.call(p.get("name"), p.get("arguments") or {})
            except Exception as e:  # noqa: BLE001
                text, err = f"{type(e).__name__}: {e}", True
            result = {"content": [{"type": "text", "text": text}], "isError": bool(err)}
        elif method == "ping":
            result = {}
        elif mid is None:
            return None  # a notification
        else:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}


def main():
    server = Server()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            reply = server.handle(json.loads(line))
        except ValueError:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
