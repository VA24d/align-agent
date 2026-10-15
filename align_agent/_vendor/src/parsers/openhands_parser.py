"""Parser for OpenHands trajectories."""

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .base_parser import BaseTrajectoryParser, TrajectoryStep, ActionKind
from .file_events import (_SCRIPT_SUFFIXES, command_cwd, output_views, script_write_events,
                          shell_events_with_outcome)

_EXIT_RE = re.compile(r"\[The command completed with exit code (-?\d+)\.\]")
_ECHO = "Here's the result of running `cat -n`"
_CWD_BANNER = re.compile(r"\[Current working directory: ([^\]\n]+)\]")
_EDITOR_COMMANDS = ("view", "open", "str_replace", "insert", "create", "undo_edit")


def _text(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(x.get("text", "") for x in content if isinstance(x, dict))
    return "" if content is None else str(content)


def _signature(message: Dict[str, Any]) -> str:
    """Identity of a message as it recurs in later prompts. An assistant turn
    is known by its tool-call ids, which are unique to the call; other turns
    by their text."""
    calls = message.get("tool_calls") or []
    ids = ",".join(sorted(str(c.get("id")) for c in calls if isinstance(c, dict)))
    body = f"{message.get('role')}|{ids}|{'' if ids else _text(message.get('content'))}"
    return hashlib.sha1(body.encode("utf-8", "replace"), usedforsecurity=False).hexdigest()


def _response_message(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    resp = item.get("response") if isinstance(item, dict) else None
    choices = resp.get("choices") if isinstance(resp, dict) else None
    return choices[0].get("message") if choices else None


def assign_sessions(items: List[Dict[str, Any]]) -> List[int]:
    """Which conversation each LLM call belongs to.

    Some CodeTraceBench OpenHands logs hold several conversations run side by
    side, stored as one trajectory; they share a system prompt and task, so a
    shared prefix proves nothing. A call continues an earlier call when its
    prompt starts with that call's prompt plus that call's own reply, and a
    reply is identified by its tool-call ids. Without this, one session's
    edits were scored against another session's views of its own copy of the
    file.
    """
    extended: List[List[str]] = []
    out: List[int] = []
    n_sessions = 0
    for item in items:
        msgs = item.get("messages") if isinstance(item, dict) else None
        if not isinstance(msgs, list):
            # An event-stream item (action or observation) carries no prompt,
            # so it cannot open a conversation of its own: without this, every
            # event became its own session and no edit was ever scored against
            # an earlier view.
            out.append(out[-1] if out else 0)
            extended.append([])
            continue
        sig = [_signature(m) for m in msgs if isinstance(m, dict)]
        best, best_len = None, -1
        for j, ext in enumerate(extended):
            if ext and len(ext) <= len(sig) and sig[:len(ext)] == ext and len(ext) > best_len:
                best, best_len = j, len(ext)
        if best is None:
            session = n_sessions
            n_sessions += 1
        else:
            session = out[best]
        out.append(session)
        reply = _response_message(item) if isinstance(item, dict) else None
        extended.append(sig + ([_signature(reply)] if reply else []))
    return out


def tool_results(items: List[Dict[str, Any]]) -> Dict[str, str]:
    """Tool output by call id, gathered from every later prompt."""
    out: Dict[str, str] = {}
    for item in items:
        for m in (item.get("messages") or []) if isinstance(item, dict) else []:
            if isinstance(m, dict) and m.get("role") == "tool" and m.get("tool_call_id"):
                out.setdefault(m["tool_call_id"], _text(m.get("content")))
    return out


def editor_events(sub_cmd: str, path: str, result: Optional[str]) -> List[Tuple[str, str, str, bool]]:
    """Events for one str_replace_editor call, read from its reply.

    An edit lands when the editor says so; its reply then shows a snippet of
    the file as it now stands, which is a view. With no reply on record the
    edit is taken to have landed and nothing is taken to have been shown."""
    if not path:
        return []
    known = result is not None
    result = result or ""
    if sub_cmd in ("view", "open"):
        return [] if result.startswith(("ERROR", "Invalid")) else [("R", path, "view", True)]
    if sub_cmd in ("str_replace", "insert"):
        landed = (not known) or "has been edited" in result
        events = [("W", path, "anchored", landed)]
        if landed and _ECHO in result:
            events.append(("R", path, "echo", True))
        return events
    if sub_cmd == "create":
        return [("W", path, "overwrite", (not known) or "created successfully" in result)]
    if sub_cmd == "undo_edit":
        landed = (not known) or "undone successfully" in result
        events = [("W", path, "restore", landed)]
        if landed and _ECHO in result:
            events.append(("R", path, "echo", True))
        return events
    return []


class OpenHandsParser(BaseTrajectoryParser):
    """Parses tool-calling trajectories generated by OpenHands."""

    def _event_stream_step(self, item: Dict[str, Any], obs: Optional[Any], step_counter: int,
                           incorrect_steps, unuseful_steps, session: int = 0,
                           cwd: Optional[str] = None,
                           scripts: Optional[Dict[str, str]] = None) -> Optional[TrajectoryStep]:
        """A step from OpenHands' event-stream format (action + args)."""
        action = item.get("action")
        args = item.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        reads, writes, events = [], [], []
        kind = ActionKind.UNKNOWN
        action_str = f"{action}: {args}"
        path = self._clean_path(args.get("path", "") or "")

        obs_str = ""
        if isinstance(obs, str):
            obs_str = obs
        elif isinstance(obs, dict):
            content = obs.get("content")
            if isinstance(content, str):
                obs_str = content
            elif content is not None:
                obs_str = str(content)
            elif isinstance(obs.get("message"), str):
                obs_str = obs["message"]

        is_err = bool(obs_str.startswith(("ERROR", "Invalid"))
                      or "No replacement was performed" in obs_str
                      or "[Errno " in obs_str)
        landed = not is_err if obs_str else True

        if action == "run":
            cmd = args.get("command", "")
            action_str = cmd
            kind, reads, writes = self.extract_bash_file_operations(cmd)
            rc = None
            rc_m = _EXIT_RE.search(obs_str)
            if rc_m:
                rc = int(rc_m.group(1))
            elif isinstance(obs, dict):
                meta = obs.get("extras", {}).get("metadata", {})
                if isinstance(meta, dict) and "exit_code" in meta:
                    rc = meta.get("exit_code")
            # The tool refuses some commands outright; then nothing ran.
            refused = obs_str.lstrip().startswith("ERROR: Cannot execute multiple commands")
            events = [] if refused else shell_events_with_outcome(cmd, obs_str, rc, cwd=cwd, scripts=scripts)
        elif action == "run_ipython":
            # A Jupyter cell: its file writes are read from the code as for an
            # inline `python -c` script. A cell that raised is taken to have
            # written nothing, as a script that exits on a traceback is.
            action_str = args.get("code", "") or ""
            kind = ActionKind.EXECUTE
            landed = "Traceback (most recent call last)" not in obs_str
            for p, wkind in script_write_events(action_str):
                writes.append(p)
                events.append(("W", p, wkind, landed))
        elif action == "read" and path:
            reads.append(path)
            kind = ActionKind.FILE_READ
            if not is_err:
                events = [("R", path, "view", True)]
        elif action in ("write", "edit") and path:
            cmd_type = args.get("command")
            if scripts is not None and cmd_type == "create" and path.endswith(_SCRIPT_SUFFIXES) \
                    and isinstance(args.get("file_text"), str):
                scripts[path] = args["file_text"]
            if cmd_type in _EDITOR_COMMANDS:
                # The same rule as the tool-call format: the editor's own
                # reply decides whether the edit landed and whether it echoed
                # the file. With no reply linked, nothing is taken as shown.
                events = editor_events(cmd_type, path, obs_str if obs is not None else None)
                if cmd_type in ("view", "open"):
                    reads.append(path)
                    kind = ActionKind.FILE_READ
                else:
                    writes.append(path)
                    kind = ActionKind.FILE_WRITE
            else:
                # Older FileWriteAction / LLM-based edit without an editor command.
                writes.append(path)
                kind = ActionKind.FILE_WRITE
                events = [("W", path, "overwrite" if action == "write" else "anchored", landed)]
                if action == "edit" and landed and _ECHO in obs_str:
                    events.append(("R", path, "echo", True))
        elif action in ("finish", "finish_task", "submit"):
            kind = ActionKind.TERMINATE

        if kind == ActionKind.UNKNOWN and not reads and not writes:
            return None
        meta = ((obs.get("extras") or {}).get("metadata") or {}) if isinstance(obs, dict) else {}
        after = meta.get("working_dir") if isinstance(meta, dict) and meta.get("working_dir") else (
            command_cwd(args.get("command", "") or "", cwd) if action == "run" else cwd)

        return TrajectoryStep(
            step_id=step_counter, action_raw=action_str, action_kind=kind,
            target_files_read=reads, target_files_written=writes,
            observation_raw=obs_str,
            is_incorrect=(step_counter in incorrect_steps),
            is_unuseful=(step_counter in unuseful_steps),
            token_count_estimate=self.estimate_tokens(action_str + obs_str),
            metadata={"framework": "OpenHands", "action": action},
            events=events, session=session,
            output_reads=output_views(obs_str, cwd=after))

    def _tool_call_step(self, tc: Dict[str, Any], result: Optional[str], step_counter: int,
                        session: int, incorrect_steps, unuseful_steps, cwd: Optional[str] = None,
                        scripts: Optional[Dict[str, str]] = None) -> TrajectoryStep:
        func = tc.get("function", {})
        fname = func.get("name", "")
        fargs_raw = func.get("arguments", "{}")
        try:
            fargs = json.loads(fargs_raw) if isinstance(fargs_raw, str) else fargs_raw
        except Exception:
            fargs = {}
        if not isinstance(fargs, dict):
            fargs = {}

        reads, writes, events, shown = [], [], [], []
        kind = ActionKind.UNKNOWN
        action_str = f"{fname}: {fargs_raw}"

        if fname in ["execute_bash", "bash", "terminal", "cmd"]:
            cmd = fargs.get("command", "")
            action_str = cmd
            kind, reads, writes = self.extract_bash_file_operations(cmd)
            rc_m = _EXIT_RE.search(result or "")
            events = shell_events_with_outcome(cmd, result or "", int(rc_m.group(1)) if rc_m else None, cwd=cwd,
                                               scripts=scripts)
            banner = _CWD_BANNER.findall(result or "")
            shown = output_views(result or "", cwd=banner[-1].strip() if banner else command_cwd(cmd, cwd))
        elif "editor" in fname or fname in ["str_replace_editor", "file_editor"]:
            sub_cmd = fargs.get("command", "")
            path = self._clean_path(fargs.get("path", "") or "")
            if scripts is not None and sub_cmd == "create" and path.endswith(_SCRIPT_SUFFIXES) \
                    and isinstance(fargs.get("file_text"), str):
                scripts[path] = fargs["file_text"]
            if sub_cmd in ["view", "open"]:
                reads.append(path)
                kind = ActionKind.FILE_READ
            # undo_edit restores the previous version, which is a
            # write like any other.
            elif sub_cmd in ["str_replace", "create", "insert", "undo_edit"]:
                writes.append(path)
                kind = ActionKind.FILE_WRITE
            else:
                kind = ActionKind.FILE_READ
            events = editor_events(sub_cmd, path, result)
        elif fname in ["finish", "finish_task", "submit"]:
            kind = ActionKind.TERMINATE

        return TrajectoryStep(
            step_id=step_counter, action_raw=action_str, action_kind=kind,
            target_files_read=reads, target_files_written=writes,
            observation_raw=result or "",
            is_incorrect=(step_counter in incorrect_steps),
            is_unuseful=(step_counter in unuseful_steps),
            token_count_estimate=self.estimate_tokens(action_str),
            metadata={"framework": "OpenHands", "tool_name": fname},
            events=events, session=session, output_reads=shown)

    def parse_trajectory(self, raw_data: Any, manifest_item: Dict[str, Any]) -> List[TrajectoryStep]:
        if not raw_data:
            return []

        steps: List[TrajectoryStep] = []
        step_dicts = []

        if isinstance(raw_data, list):
            step_dicts = raw_data
        elif isinstance(raw_data, dict):
            # Might be a single combined dict with messages or steps
            if "steps" in raw_data:
                step_dicts = raw_data["steps"]
            elif "messages" in raw_data:
                step_dicts = [{"messages": raw_data["messages"]}]

        # If raw_data contains items with integer IDs (OpenHands event stream format),
        # deduplicate and sort by event ID so nested checkpoint lists and individual
        # step files do not duplicate events.
        has_int_ids = any(
            isinstance(x, dict) and isinstance(x.get("id"), int)
            for it in step_dicts
            for x in (it if isinstance(it, list) else [it])
        )
        if has_int_ids:
            events_by_id = {}
            for it in step_dicts:
                items = it if isinstance(it, list) else [it]
                for x in items:
                    if isinstance(x, dict) and isinstance(x.get("id"), int):
                        events_by_id.setdefault(x["id"], x)
            step_dicts = [events_by_id[k] for k in sorted(events_by_id.keys())]
        else:
            flattened = []
            for it in step_dicts:
                if isinstance(it, list):
                    flattened.extend(x for x in it if isinstance(x, dict))
                elif isinstance(it, dict):
                    flattened.append(it)
            step_dicts = flattened

        manifest_item = manifest_item or {}
        # Collect ground-truth failure annotations
        incorrect_steps = set()
        unuseful_steps = set()
        for stage in manifest_item.get("incorrect_stages", []):
            incorrect_steps.update(stage.get("incorrect_step_ids", []))
            unuseful_steps.update(stage.get("unuseful_step_ids", []))

        sessions = assign_sessions(step_dicts)
        results = tool_results(step_dicts)

        obs_by_cause: Dict[Any, Dict[str, Any]] = {}
        for it in step_dicts:
            if isinstance(it, dict) and it.get("cause") is not None:
                cause_id = it["cause"]
                if cause_id not in obs_by_cause:
                    obs_by_cause[cause_id] = it
                    obs_by_cause[str(cause_id)] = it

        step_counter = 0
        # The shell persists between commands, and OpenHands records where each
        # left it: `working_dir` in an event-stream observation's metadata, a
        # "[Current working directory: ...]" line in a tool-call result.
        cwd: Optional[str] = None
        last_session = None
        scripts: Dict[str, str] = {}

        for idx, (item, session) in enumerate(zip(step_dicts, sessions)):
            if session != last_session:
                cwd, last_session = None, session
            response = item.get("response", {})
            choices = response.get("choices", []) if isinstance(response, dict) else []
            if not choices:
                # OpenHands' event-stream format carries action and args.
                if item.get("action") and (item.get("source") == "agent" or item.get("action") in ["run", "read", "write", "edit", "call", "finish"]):
                    obs = None
                    act_id = item.get("id")
                    if act_id is not None and (act_id in obs_by_cause or str(act_id) in obs_by_cause):
                        obs = obs_by_cause.get(act_id) or obs_by_cause.get(str(act_id))
                    else:
                        # Sequential next observation in the event stream
                        for nxt in step_dicts[idx + 1:]:
                            if not isinstance(nxt, dict):
                                continue
                            if nxt.get("observation") or nxt.get("cause") is not None or (nxt.get("action") is None and "content" in nxt):
                                obs = nxt
                                break
                            if nxt.get("action") and (nxt.get("source") == "agent" or nxt.get("action") in ["run", "read", "write", "edit", "call", "finish"]):
                                break

                    step = self._event_stream_step(
                        item, obs, step_counter + 1, incorrect_steps, unuseful_steps, session=session, cwd=cwd,
                        scripts=scripts,
                    )
                    meta = ((obs.get("extras") or {}).get("metadata") or {}) if isinstance(obs, dict) else {}
                    if isinstance(meta, dict) and isinstance(meta.get("working_dir"), str) and meta["working_dir"]:
                        cwd = meta["working_dir"]
                    if step is not None:
                        step_counter += 1
                        steps.append(step)
                continue

            msg = choices[0].get("message", {})
            tool_calls = msg.get("tool_calls", [])

            if not tool_calls:
                # A reply with no tool call runs nothing. Its prose is kept for
                # the older counters but yields no file events.
                content = msg.get("content", "") or ""
                if content.strip():
                    step_counter += 1
                    b_kind, b_reads, b_writes = self.extract_bash_file_operations(content)
                    steps.append(TrajectoryStep(
                        step_id=step_counter, action_raw=content, action_kind=b_kind,
                        target_files_read=b_reads, target_files_written=b_writes,
                        observation_raw="",
                        is_incorrect=(step_counter in incorrect_steps),
                        is_unuseful=(step_counter in unuseful_steps),
                        token_count_estimate=self.estimate_tokens(content),
                        metadata={"framework": "OpenHands"}, session=session))
                continue

            for tc in tool_calls:
                step_counter += 1
                result = results.get(tc.get("id"))
                steps.append(self._tool_call_step(
                    tc, result, step_counter, session, incorrect_steps, unuseful_steps, cwd=cwd, scripts=scripts))
                banner = _CWD_BANNER.findall(result or "")
                if banner:
                    cwd = banner[-1].strip()

        return self.canonicalize_paths(steps, manifest_item)
