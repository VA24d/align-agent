"""Parser for SWE-agent trajectories."""

import json
import re
import shlex
from typing import List, Dict, Any, Optional
from .base_parser import BaseTrajectoryParser, TrajectoryStep, ActionKind
from .file_events import _SCRIPT_SUFFIXES, command_cwd, output_views, resolve_path, shell_events_with_outcome
from .openhands_parser import editor_events

# SWE-agent's str_replace_editor tool (the same editor OpenHands uses, with
# the same replies): `str_replace_editor <command> <path> [--flags ...]`.
_EDITOR_RE = re.compile(r"str_replace_editor\s+(\w+)\s+(\S+)")
_DIR_LISTING = "Here's the files and directories"


def unwrap_shell(action: str) -> Optional[str]:
    """The command inside `bash -lc "..."` / `sh -c "..."`, if the action is
    wrapped that way; the file events are those of the inner command."""
    if not re.match(r"(ba)?sh\s+(-\w+\s+)*-\w*c\b", action):
        return None
    try:
        parts = shlex.split(action)
    except ValueError:
        return None
    rest = [p for p in parts[1:] if not p.startswith("-")]
    return rest[0] if rest else None


class SWEAgentParser(BaseTrajectoryParser):
    """Parses SWE-agent trajectory files (.traj JSON lists)."""

    # Commands that display lines of the currently open file.
    _VIEWER_READS = {"goto", "scroll_up", "scroll_down", "search_file"}
    # An edit SWE-agent refused: malformed, or reverted by its linter.
    _EDIT_REJECTED = ("Usage: edit", "Your proposed edit has introduced new syntax error",
                      "This is how your edit would have looked if applied")

    def _viewer_target(self, action_raw, verb, current_open_file):
        """The file a viewer command displays: `search_file TERM FILE` names
        its file, and every other form shows the open one."""
        if verb == "search_file":
            try:
                parts = shlex.split(action_raw)
            except ValueError:
                parts = action_raw.split()
            if len(parts) > 2:
                return self._clean_path(parts[2])
        return current_open_file

    def parse_trajectory(self, raw_data: Any, manifest_item: Dict[str, Any]) -> List[TrajectoryStep]:
        steps: List[TrajectoryStep] = []
        if isinstance(raw_data, dict) and "trajectory" in raw_data:
            entries = raw_data["trajectory"]
        elif isinstance(raw_data, list):
            entries = raw_data
        else:
            return steps

        # Collect ground-truth failure annotations
        incorrect_steps = set()
        unuseful_steps = set()
        for stage in manifest_item.get("incorrect_stages", []):
            incorrect_steps.update(stage.get("incorrect_step_ids", []))
            unuseful_steps.update(stage.get("unuseful_step_ids", []))

        current_open_file = None
        # Every command runs in a fresh `bash -lc` at the session's working
        # directory, which each entry's state records.
        working_dir = None
        scripts: Dict[str, str] = {}

        for idx, entry in enumerate(entries, start=1):
            action_raw = entry.get("action", "").strip()
            obs_raw = entry.get("observation", "")
            state_raw = entry.get("state", "")

            # Attempt to extract open_file from state json if available
            step_cwd = working_dir
            if isinstance(state_raw, str) and ("open_file" in state_raw or "working_dir" in state_raw):
                try:
                    s_data = json.loads(state_raw)
                    if s_data.get("open_file") and s_data["open_file"] != "n/a":
                        current_open_file = self._clean_path(s_data["open_file"])
                    if isinstance(s_data.get("working_dir"), str) and s_data["working_dir"]:
                        working_dir = s_data["working_dir"]
                        step_cwd = step_cwd or working_dir
                except Exception:
                    pass

            reads = []
            writes = []
            kind = ActionKind.UNKNOWN
            obs_text = obs_raw if isinstance(obs_raw, str) else str(obs_raw)
            events, shown = [], []

            # Check SWE-agent specific tools
            open_match = self.swe_open_re.match(action_raw)
            edit_match = self.swe_edit_re.match(action_raw)

            verb = action_raw.split(None, 1)[0] if action_raw else ""
            editor = _EDITOR_RE.match(action_raw)
            if editor:
                # The editor's reply decides whether an edit landed and whether
                # it echoed the file, exactly as for OpenHands. A view of a
                # directory lists names, not file content.
                sub_cmd, target = editor.group(1), self._clean_path(editor.group(2))
                listing = sub_cmd == "view" and obs_text.lstrip().startswith(_DIR_LISTING)
                events = [] if listing else editor_events(sub_cmd, target, obs_text)
                if sub_cmd == "create" and target.endswith(_SCRIPT_SUFFIXES):
                    try:
                        argv = shlex.split(action_raw)
                        if "--file_text" in argv:
                            scripts[resolve_path(step_cwd, target)] = argv[argv.index("--file_text") + 1]
                    except (ValueError, IndexError):
                        pass
                if sub_cmd in ("view", "open"):
                    reads.extend([target] if events else [])
                    kind = ActionKind.FILE_READ
                else:
                    writes.append(target)
                    kind = ActionKind.FILE_WRITE
            elif open_match:
                filepath = self._clean_path(open_match.group(1))
                current_open_file = filepath
                reads.append(filepath)
                kind = ActionKind.FILE_READ
                events = [("R", filepath, "view", True)]
            elif verb in self._VIEWER_READS:
                # Navigation redraws the open file's window from disk, so it is
                # a read of that file just as `open` is. Counting only `open`
                # scored every edit after a `goto` as editing blind.
                target = self._viewer_target(action_raw, verb, current_open_file)
                if target:
                    reads.append(target)
                    events = [("R", target, "view", True)]
                kind = ActionKind.FILE_READ
            elif edit_match or action_raw.startswith("edit"):
                if current_open_file:
                    writes.append(current_open_file)
                    # A landed edit redraws the window around it, which is a
                    # view of the file as it now stands.
                    landed = not any(r in obs_text for r in self._EDIT_REJECTED)
                    events = [("W", current_open_file, "anchored", landed)]
                    if landed:
                        events.append(("R", current_open_file, "echo", True))
                kind = ActionKind.FILE_WRITE
            elif action_raw.startswith("create "):
                filepath = self._clean_path(action_raw.split(None, 1)[1])
                writes.append(filepath)
                kind = ActionKind.FILE_WRITE
                events = [("W", filepath, "overwrite", True), ("R", filepath, "view", True)]
            else:
                # Fallback to bash command extractor, on the inner command when
                # the action is wrapped in `bash -lc "..."`.
                cmd = unwrap_shell(action_raw) or action_raw
                b_kind, b_reads, b_writes = self.extract_bash_file_operations(cmd)
                kind = b_kind
                reads.extend(b_reads)
                writes.extend(b_writes)
                events = shell_events_with_outcome(cmd, obs_text, cwd=step_cwd, scripts=scripts)
                shown = output_views(obs_text, cwd=command_cwd(cmd, step_cwd))

            step = TrajectoryStep(
                step_id=idx,
                action_raw=action_raw,
                action_kind=kind,
                target_files_read=list(dict.fromkeys(reads)),
                target_files_written=list(dict.fromkeys(writes)),
                observation_raw=obs_raw,
                is_incorrect=(idx in incorrect_steps),
                is_unuseful=(idx in unuseful_steps),
                token_count_estimate=self.estimate_tokens(action_raw + obs_raw),
                metadata={"framework": "SWE-agent", "entry_index": idx},
                events=events,
                output_reads=shown,
            )
            steps.append(step)

        return self.canonicalize_paths(steps, manifest_item)
