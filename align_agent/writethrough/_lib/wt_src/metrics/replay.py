"""The stale-act replay: one definition, applied to both research questions.

A step is a list of file events in the order the agent's action performed
them, each (op, path, kind, landed); see src/parsers/file_events.py for the
kinds. The replay keeps, per file, when the agent last saw it and when it last
changed, and scores every *anchored* edit (one located by existing text or
line numbers) against those two times:

    opportunity  an anchored edit to a file the agent had seen before
    hazard       an opportunity where the file changed after the agent last
                 saw it

Counts are in (step, file) units: several anchored edits to one file in one
action are one act, composed against one view. The rate is pooled, hazards
over opportunities summed across tasks, so a task with one edit weighs one
edit and a task with none contributes nothing.

What counts as seeing a file: any output that prints its content, whole or in
part (cat, sed -n, grep lines, an editor's view, the snippet an editor echoes
after an edit). A whole-file overwrite with text the command itself carries
also leaves the agent with a current view, since it wrote every byte. What
counts as a change: a write that landed. A failed command, a rejected editor
call and a `git checkout` of a file the agent never modified change nothing.

Arms act on the harness's own record of views, not on ours. An arm that puts
current content in the prompt (store-forwarding appends it, in-place
replacement writes it over the old view) supplies a file only when the harness
had bound that file; the credited count treats a supplied file as seen at the
start of the step, the strict count does not. The coherent-prompt rate is
measured the same way: a stale view is withdrawn only if the arm withdraws and
the harness had bound that view.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from wt_src.parsers.file_events import (ALL_PATHS, CONTAINER_ROOTS, command_cwd, copy_source, map_kind_path,
                                     output_views, shell_events_with_outcome)

Event = Tuple[str, str, str, bool]


@dataclass
class ReplayResult:
    steps: int = 0
    opportunities: int = 0
    hazards_strict: int = 0
    hazards_credited: int = 0
    coherent_steps: int = 0
    agent_views: int = 0
    harness_bound_views: int = 0
    hazard_steps: List[int] = field(default_factory=list)
    hazard_files: List[str] = field(default_factory=list)
    hazard_pairs: List[Tuple[int, str]] = field(default_factory=list)
    # Secondary, outside the headline definition: a whole-file overwrite or
    # replace of a file the agent had seen, and how many of those landed on a
    # version it had not seen -- a rewrite that may silently discard a change.
    overwrite_opportunities: int = 0
    lost_updates: int = 0
    # Opportunities on a file some write had changed since the agent first saw
    # it: the only ones that can go stale when nothing else writes the file
    # (in the live runs, repeat edits). Every hazard is one of these.
    at_risk_opportunities: int = 0
    at_risk_hazards: int = 0


@dataclass
class _View:
    version: int
    bound: bool


class _Replayer:
    """Per-file state for one trajectory, advanced one step at a time.

    Staleness follows content, as the model in Section II defines it: every
    landed write gives the file a new version, a restore returns it to the
    committed version (0), and the agent holds a valid view when any view it
    has seen shows the file's current version. A `git checkout` back to
    content the agent had already seen therefore leaves it with a valid view.

    An agent composes a whole action before it sees any of that action's
    output, so an edit is scored against the views held when the step began.
    What the step prints (a `cat`, an editor's echo, a compiler quoting a
    line) counts from the next step on.
    """

    def __init__(self, withdraws: bool, forwards: bool, refreshes: bool, known_substitutions: bool = False):
        self.withdraws, self.forwards, self.refreshes = withdraws, forwards, refreshes
        # Sensitivity: a substitution on a version the agent knew leaves a
        # version it knows, as a whole-file write does.
        self.known_substitutions = known_substitutions
        self.out = ReplayResult()
        self.next_version = 1
        self.version: Dict[Tuple, int] = {}      # current content version
        self.seen: Dict[Tuple, Set[int]] = {}    # versions the agent has seen
        self.supplied: Dict[Tuple, int] = {}     # version the harness last supplied
        self.views: Dict[Tuple, List[_View]] = {}
        self.modified: Set[Tuple] = set()
        self.changed_after_seen: Set[Tuple] = set()
        self.h_bound: Set[Tuple] = set()         # the harness has a binding
        self.h_invalid: Set[Tuple] = set()       # ... and has marked it invalid
        self.h_kept: Set[Tuple] = set()          # in-place can refresh it
        self.hazard_files: Set[str] = set()

    def current(self, k: Tuple) -> int:
        return self.version.get(k, 0)

    def supply(self, sess) -> None:
        """What the arm puts in front of the agent before it acts."""
        if not (self.forwards or self.refreshes):
            return
        for k in self.h_invalid:
            if k[0] == sess and (self.forwards or k in self.h_kept):
                self.supplied[k] = self.current(k)

    def prompt_is_coherent(self) -> bool:
        """No stale view left in the prompt after the arm's withdrawals."""
        for k, vs in self.views.items():
            cur = self.current(k)
            if any(v.version != cur and not (self.withdraws and v.bound) for v in vs):
                return False
        return True

    def score_edit(self, k: Tuple, held: Dict[Tuple, Set[int]]) -> bool:
        """Score an anchored edit against the views held at step start;
        True when none of them shows the file as it now stands."""
        if k not in held:
            return False
        self.out.opportunities += 1
        at_risk = k in self.changed_after_seen
        self.out.at_risk_opportunities += at_risk
        cur = self.current(k)
        if cur not in held[k] and cur != self.supplied.get(k):
            self.out.hazards_credited += 1
        if cur not in held[k]:
            self.out.hazards_strict += 1
            self.out.at_risk_hazards += at_risk
            self.hazard_files.add(k[1])
            return True
        return False

    def land(self, k: Tuple, kind: str) -> None:
        if k in self.seen:
            self.changed_after_seen.add(k)
        if kind == "restore":
            if k in self.modified:
                self.version[k] = 0
            return
        self.version[k] = self.next_version
        self.next_version += 1
        self.modified.add(k)

    def land_all(self, sess, kind: str) -> None:
        for k in list(self.modified):
            if k[0] == sess:
                self.land(k, kind)

    def step(self, step: Dict[str, Any]) -> List[str]:
        """Advance one step; return the list of file paths hazarded (empty if none)."""
        sess = step.get("session", 0)
        self.supply(sess)
        # Files whose current contents the prompt itself carried (the synced-
        # context arm): the agent sees their current version as it acts.
        for path in step.get("presented") or ():
            k = (sess, path)
            self.seen.setdefault(k, set()).add(self.current(k))
        self.out.coherent_steps += self.prompt_is_coherent()
        harness_reads = set(step.get("harness_reads") or ())
        held = {k: set(v) for k, v in self.seen.items()}
        acted: Set[Tuple] = set()
        overwrote: Set[Tuple] = set()
        last_op: Dict[Tuple, str] = {}
        shown: List[Tuple[Tuple, int, bool]] = []
        hazarded_paths: List[str] = []
        for op, path, kind, landed in step.get("events", []):
            k = (sess, path)
            if op == "R":
                shown.append((k, self.current(k), path in harness_reads))
                last_op[k] = "R"
            elif path == ALL_PATHS:
                if landed:
                    self.land_all(sess, kind)
            else:
                last_op[k] = "W"
                src = copy_source(kind)
                if kind in ("anchored", "subst") and k not in acted:
                    acted.add(k)
                    if self.score_edit(k, held):
                        hazarded_paths.append(path)
                elif (kind in ("overwrite", "replace") or src is not None) and k not in overwrote and k in held:
                    overwrote.add(k)
                    self.out.overwrite_opportunities += 1
                    if self.current(k) not in held[k]:
                        self.out.lost_updates += 1
                if landed:
                    # The agent knows what a write leaves when it supplied
                    # every byte, or copied a file whose current version it
                    # had seen: later edits in this same action are composed
                    # against that, not against the views held at step start.
                    s_k = (sess, src) if src is not None else None
                    known = kind == "overwrite" or (
                        s_k is not None and self.current(s_k) in self.seen.get(s_k, ())) or (
                        kind == "subst" and self.known_substitutions and self.current(k) in self.seen.get(k, ()))
                    self.land(k, "replace" if src is not None else kind)
                    if known:
                        self.seen.setdefault(k, set()).add(self.current(k))
                        held.setdefault(k, set()).add(self.current(k))
        for k, ver, bound in shown:
            self.seen.setdefault(k, set()).add(ver)
            self.views.setdefault(k, []).append(_View(ver, bound))
            self.out.agent_views += 1
            self.out.harness_bound_views += bound
        for path in step.get("output_reads") or ():
            k = (sess, path)
            self.seen.setdefault(k, set()).add(self.current(k))
        self.harness_update(sess, step, harness_reads, last_op)
        return hazarded_paths

    def harness_update(self, sess, step, harness_reads, last_op) -> None:
        """The harness's own bookkeeping, which is what the arms act on."""
        for p in step.get("harness_writes") or ():
            if (sess, p) in self.h_bound:
                self.h_invalid.add((sess, p))
        kept = step.get("harness_kept")
        for p in harness_reads:
            k = (sess, p)
            self.h_bound.add(k)
            if p == kept:
                self.h_kept.add(k)
            if last_op.get(k, "R") == "R":
                self.h_invalid.discard(k)


def replay(steps: Sequence[Dict[str, Any]], *, withdraws: bool = False,
           forwards: bool = False, refreshes: bool = False, known_substitutions: bool = False) -> ReplayResult:
    """Score a trajectory. Each step dict carries `events`, optionally
    `session` (interleaved sessions keep separate state), and for live runs
    `harness_reads`, `harness_writes`, `harness_kept`."""
    r = _Replayer(withdraws, forwards, refreshes, known_substitutions)
    for idx, step in enumerate(steps):
        paths = r.step(step)
        if paths:
            r.out.hazard_steps.append(idx)
            for p in paths:
                r.out.hazard_pairs.append((idx, p))
    r.out.steps = len(steps)
    r.out.hazard_files = sorted(r.hazard_files)
    return r.out


# --- live (RQ2) traces --------------------------------------------------

_CONTAINER_ROOTS = CONTAINER_ROOTS


def _canon(path: str) -> Optional[str]:
    """Workspace-relative spelling, or None for a path outside the workspace."""
    if path == ALL_PATHS:
        return path
    for root in _CONTAINER_ROOTS:
        if path.startswith(root):
            path = path[len(root):]
            break
    else:
        if path.startswith("/"):
            return None
    path = os.path.normpath(path)
    return None if path.startswith("..") else path


def _as_list(v: Any) -> List[str]:
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.startswith("["):
        import ast
        try:
            return list(ast.literal_eval(v))
        except (ValueError, SyntaxError):
            return []
    return []


def _live_events(s: Dict[str, Any], scripts: Optional[Dict[str, str]] = None) -> List[Event]:
    cmd = s.get("command") or ""
    try:
        rc = int(s.get("returncode"))
    except (TypeError, ValueError):
        rc = None
    detected = [p for p in (_canon(x) for x in _as_list(s.get("writes_detected"))) if p]
    # mini-SWE-agent runs each command in a fresh `docker exec` at the
    # repository root: a `cd` lasts only for the command that makes it.
    raw_events = shell_events_with_outcome(cmd, str(s.get("observation") or ""), rc, cwd="", scripts=scripts)

    events: List[Event] = []
    parsed_w: Set[str] = set()
    restore_all = False
    restored_dirs: List[str] = []
    for op, raw, kind, landed in raw_events:
        p = _canon(raw)
        if p is None:
            continue
        src = copy_source(kind)
        if src is not None:
            # A source outside the workspace carries content nothing tracks.
            kind = map_kind_path(kind, _canon) if _canon(src) else "replace"
        if op == "R":
            events.append(("R", p, kind, True))
        elif p == ALL_PATHS:
            restore_all = True
        else:
            parsed_w.add(p)
            if kind == "restore" and landed:
                restored_dirs.append(p)
            # The harness recorded a write it parsed or saw change on disk;
            # a target it did not record did not change.
            events.append(("W", p, kind, landed and p in detected))
    if not raw_events and not cmd.strip():
        return events
    # Changes the command did not name: a restore, a script that named the
    # file, or something else that touched it.
    unparsed: List[Event] = []
    for p in detected:
        if p in parsed_w:
            continue
        # `git checkout -- dir/` names a directory: every file it rewrote
        # under that directory went back to the committed content, not to a
        # new version.
        if restore_all or any(p.startswith(d + "/") for d in restored_dirs):
            unparsed.append(("W", p, "restore", True))
        elif p in cmd or os.path.basename(p) in cmd:
            unparsed.append(("W", p, "anchored", True))
        else:
            unparsed.append(("W", p, "replace", True))

    if unparsed:
        # Event chronology: unparsed harness-detected writes (e.g. from scripts or
        # build steps) that precede inspection reads in the same turn must be placed
        # before those reads, so the view reflects the landed write.
        read_paths = {p for op, p, _, _ in events if op == "R"}
        if any(p in read_paths for _, p, _, _ in unparsed):
            new_events: List[Event] = []
            unparsed_placed: Set[str] = set()
            for ev in events:
                op, p, kind, landed = ev
                if op == "R" and p in {up[1] for up in unparsed} and p not in unparsed_placed:
                    for up in unparsed:
                        if up[1] == p:
                            new_events.append(up)
                            unparsed_placed.add(p)
                new_events.append(ev)
            for up in unparsed:
                if up[1] not in unparsed_placed:
                    new_events.append(up)
            events = new_events
        else:
            events.extend(unparsed)
    return events


_DEPENDENCY_NOTICE = "\n\n[DEPENDENCY NOTICE:"


def _command_output(s: Dict[str, Any]) -> str:
    """The step's observation without the dependency notice.

    The notice lists `file:line: text` for uses in other files. Read as
    command output, each line would count as a view of that file, and the
    dependency arm would be credited with views it was only pointed at. The
    write-through viewport and diff stay in: they show the written file itself.
    """
    obs = str(s.get("observation") or "")
    cut = obs.find(_DEPENDENCY_NOTICE)
    return obs if cut < 0 else obs[:cut]


def live_steps(trace_steps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Replay steps for a trace recorded by the live harness.

    Events come from the command text. The harness's `writes_detected` (parsed
    targets plus files whose size or modification time changed) decides which
    writes landed, and its `reads_detected` is the record the arms acted on.
    """
    out = []
    scripts: Dict[str, str] = {}
    for s in trace_steps:
        reads_detected = [p for p in (_canon(x) for x in _as_list(s.get("reads_detected"))) if p]
        out.append({
            "events": _live_events(s, scripts),
            "output_reads": [p for p in (_canon(x) for x in output_views(
                _command_output(s), cwd=command_cwd(str(s.get("command") or s.get("action") or ""), ""))) if p],
            "harness_reads": reads_detected,
            "harness_writes": [p for p in (_canon(x) for x in _as_list(s.get("writes_detected"))) if p],
            "harness_kept": reads_detected[-1] if reads_detected else None,
            "presented": [p for p in (_canon(x) for x in _as_list(s.get("synced_files"))) if p],
        })
    return out
