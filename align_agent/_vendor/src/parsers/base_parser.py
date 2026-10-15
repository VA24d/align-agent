"""Base parser classes and utilities for coding agent trajectories."""

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
import re
import shlex
from .file_events import map_kind_path, _grep, script_writes


class ActionKind(Enum):
    """High-level classification of an agent action."""
    FILE_READ = "READ"          # Inspecting/reading file content into context
    FILE_WRITE = "WRITE"        # Modifying, creating, or applying diff to a file
    EXECUTE = "EXECUTE"          # Running tests, builds, or scripts
    INSPECT_ENV = "INSPECT"      # Listing directories, git status, search/grep
    TERMINATE = "TERMINATE"      # Submitting patch, exit, finish
    UNKNOWN = "UNKNOWN"


@dataclass
class TrajectoryStep:
    """Normalized single step in an agent trajectory."""
    step_id: int                           # 1-indexed step number matching benchmark
    action_raw: str                        # Raw command or tool call string
    action_kind: ActionKind                # Semantic category of the action
    target_files_read: List[str] = field(default_factory=list)    # Files read by this action
    target_files_written: List[str] = field(default_factory=list) # Files mutated by this action
    observation_raw: str = ""              # Output or response from the environment
    is_incorrect: bool = False             # Human ground-truth annotation: failure step
    is_unuseful: bool = False              # Human ground-truth annotation: useless step
    token_count_estimate: int = 0          # Estimated token count for action + observation
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Ordered (op, path, kind, landed) file events for the stale-act replay;
    # see src/parsers/file_events.py. `session` separates interleaved runs
    # that a corpus stored as one trajectory.
    events: List[Tuple[str, str, str, bool]] = field(default_factory=list)
    session: int = 0
    # Files whose lines the action's output quoted (a traceback, a compiler
    # diagnostic, a grep hit); views from the next step on.
    output_reads: List[str] = field(default_factory=list)


class BaseTrajectoryParser(ABC):
    """Abstract base parser for extracting normalized TrajectoryStep lists."""

    def __init__(self):
        # A single '<' only: the lookarounds prevent matching either '<' of a '<<'
        # heredoc operator, which would otherwise capture the delimiter (EOF,
        # ENDFILE, ...) as though it were a filename.
        self.redirect_read_re = re.compile(r'(?<!<)<(?!<)\s*([^\s;&|<>]+)')
        # `>>` is tried first: with `>` first, an append captured its own
        # second `>` as the filename and the real target was lost.
        self.redirect_write_re = re.compile(r'(?:>>|>)\s*([^\s;&|<>]+)')
        self.heredoc_re = re.compile(r"""cat\s+<<\s*["']?([A-Za-z0-9_-]+)["']?\s*>\s*([^\s;&|]+)""")
        # Any heredoc form, for locating and removing the body before scanning.
        self.heredoc_any_re = re.compile(r"""<<-?\s*["']?([A-Za-z0-9_]+)["']?""")
        self.swe_open_re = re.compile(r'^\s*open\s+([^\s]+)')
        self.swe_edit_re = re.compile(r'^\s*edit\s+([0-9]+:[0-9]+)')
        self.git_apply_re = re.compile(r'git\s+apply\s+([^\s;&|]+)')

    @abstractmethod
    def parse_trajectory(self, raw_data: Any, manifest_item: Dict[str, Any]) -> List[TrajectoryStep]:
        """Parse raw trajectory data into a sequence of TrajectoryStep objects."""
        pass


    # --- command-aware file extraction -------------------------------------
    #
    # The regex sweep below it was written for whole command strings and
    # matched anything shaped like a path. Applied to real agent commands it
    # read `sed -i 's/{all-args}/}/' f.rs` as a write to `{all-args}/}`, and a
    # pipeline's grep pattern as a filename. Shell words respect quoting, so
    # tokenising first and then applying each utility's own argument grammar
    # removes that whole class of error.

    _SED_SCRIPT_RE = re.compile(r"^[0-9,~$+]*\s*[a-zA-Z]([/,;|!].*)?$|^[/$^].*[/]")
    _READ_VERBS = {"cat", "head", "tail", "nl", "less", "more", "od", "wc", "strings", "view"}


    _OPERATORS = {"|", "||", "&&", ";", ";;", "&", "(", ")"}
    _SEPARATOR_CHARS = set("();|&\n")
    _REDIRECT_CHARS = set("<>&")

    def _simple_commands(self, cmd: str):
        """Tokenise once, then cut the token stream on shell operators.

        A newline separates commands exactly as `;` does, so it is kept as a
        token instead of being dropped as whitespace; dropping it merged
        `cd repo` and the next line's `sed -n 1,20p f` into one command whose
        verb was `cd`, and the read was lost. A redirection and its target are
        dropped from argv: the redirection scan records the target, and left in
        place it read as an operand, so `cat > f` counted a read of f.
        """
        try:
            lexer = shlex.shlex(cmd, posix=True, punctuation_chars="();<>|&\n")
            lexer.whitespace = " \t\r"
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError:
            return []
        out, cur, skip_target = [], [], False
        for t in tokens:
            if t in self._OPERATORS or (set(t) <= self._SEPARATOR_CHARS and "\n" in t):
                if cur:
                    out.append(cur)
                cur, skip_target = [], False
            elif skip_target:
                skip_target = False
            elif set(t) <= self._REDIRECT_CHARS and ("<" in t or ">" in t):
                skip_target = True
            else:
                cur.append(t)
        if cur:
            out.append(cur)
        return out

    def _plausible_path(self, tok: str) -> bool:
        """Whether a shell word can be a file path rather than an expression."""
        if not tok or tok.startswith("-"):
            return False
        if tok in ("/dev/null", "/dev/stdout", "/dev/stderr", "null", ".", ".."):
            return False
        # Shell/regex expression debris, variables, and process substitution.
        if any(ch in tok for ch in "{}$*?!()<>\"'`"):
            return False
        if tok.isdigit():
            return False
        return "/" in tok or bool(re.search(r"\.[A-Za-z0-9_]{1,6}$", tok))

    def _files_from_tokens(self, argv):
        """(reads, writes) implied by one simple command's argv."""
        reads, writes = [], []
        if not argv:
            return reads, writes
        verb = argv[0].split("/")[-1]
        flags = [a for a in argv[1:] if a.startswith("-")]
        operands = [a for a in argv[1:] if not a.startswith("-")]

        if verb == "sed":
            inplace = any(f == "-i" or f.startswith("-i") for f in flags)
            # sed [opts] script file...: drop the script, which is an
            # expression and not a path, then treat the rest as operands.
            if operands and self._SED_SCRIPT_RE.match(operands[0]):
                operands = operands[1:]
            targets = [o for o in operands if self._plausible_path(o)]
            (writes if inplace else reads).extend(targets)
        elif verb in ("grep", "rg", "egrep", "fgrep", "ack"):
            # grep [opts] pattern file...
            r, _w = _grep(argv, flags, operands, None, False)
            reads.extend(o for o in r if self._plausible_path(o))
        elif verb == "awk":
            inplace = "-i" in flags
            if inplace and operands and operands[0] == "inplace":
                operands = operands[1:]  # -i's own argument
            if operands:
                operands = operands[1:]  # the program text
            targets = [o for o in operands if self._plausible_path(o)]
            (writes if inplace else reads).extend(targets)
        elif verb in ("perl", "ruby"):
            inplace = any(f.startswith("-i") for f in flags)
            if operands:
                operands = operands[1:]  # the program text
            targets = [o for o in operands if self._plausible_path(o)]
            (writes if inplace else reads).extend(targets)
        elif verb in self._READ_VERBS:
            reads.extend([o for o in operands if self._plausible_path(o)])
        elif verb in ("cp", "mv", "install"):
            targets = [o for o in operands if self._plausible_path(o)]
            if len(targets) >= 2:
                reads.append(targets[0])
                writes.append(targets[-1])
        elif verb in ("tee", "touch", "truncate"):
            writes.extend([o for o in operands if self._plausible_path(o)])
        elif verb == "patch":
            writes.extend([o for o in operands if self._plausible_path(o)])
        elif verb == "python" or verb.startswith("python"):
            for i, a in enumerate(argv):
                if a in ("-c", "-e") and i + 1 < len(argv):
                    writes.extend(script_writes(argv[i + 1]))
                    break

        return reads, writes

    def extract_bash_file_operations(self, cmd: str) -> Tuple[ActionKind, List[str], List[str]]:
        """Extract read and written files from bash commands using heuristics and regex."""
        cmd = cmd.strip()
        reads = []
        writes = []

        # Check heredoc writes: cat << 'EOF' > file.py
        heredoc_match = self.heredoc_re.search(cmd)
        if heredoc_match:
            clean_p = self._clean_path(heredoc_match.group(2).strip())
            if clean_p and clean_p not in ['/dev/null', '/dev/stdout', '/dev/stderr', 'null']:
                writes.append(clean_p)

        # Heredoc payloads are file content, not shell syntax: remove them so
        # their contents cannot masquerade as redirections below.
        cmd = self._strip_heredoc_bodies(cmd)

        # Check redirections: echo ... > file.py
        for match in self.redirect_write_re.finditer(cmd):
            clean_p = self._clean_path(match.group(1).strip())
            if clean_p and not clean_p.startswith('&') and not clean_p.isdigit() and clean_p not in ['/dev/null', '/dev/stdout', '/dev/stderr', 'null']:
                writes.append(clean_p)

        # Check input redirections: command < file.py
        for match in self.redirect_read_re.finditer(cmd):
            filepath = match.group(1).strip()
            clean_p = self._clean_path(filepath)
            if clean_p and not clean_p.startswith('&') and not clean_p.isdigit() and clean_p not in ['/dev/null', '/dev/stdout', '/dev/stderr', 'null']:
                reads.append(clean_p)

        # Check git apply
        for match in self.git_apply_re.finditer(cmd):
            clean_p = self._clean_path(match.group(1).strip())
            if clean_p:
                writes.append(clean_p)

        # Per-utility argument grammar, over shell words rather than raw text.
        # Splitting on separators must happen after tokenising, not before: a
        # grep pattern like "sweep\|free" contains a pipe, and cutting the raw
        # string on it tears the command in half.
        for argv in self._simple_commands(cmd):
            r, w = self._files_from_tokens(argv)
            reads.extend(r)
            writes.extend(w)

        # Only keep redirection and sed-style targets that can be paths.
        writes = [x for x in writes if self._plausible_path(x)]
        reads = [x for x in reads if self._plausible_path(x)]

        # Categorize action kind
        if writes:
            kind = ActionKind.FILE_WRITE
        elif reads:
            kind = ActionKind.FILE_READ
        elif any(term in cmd for term in ['pytest', 'python ', 'cargo test', 'npm test', 'make']):
            kind = ActionKind.EXECUTE
        elif any(term in cmd for term in ['ls', 'find', 'grep', 'pwd', 'git status', 'git diff']):
            kind = ActionKind.INSPECT_ENV
        elif 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' in cmd or 'exit' in cmd:
            kind = ActionKind.TERMINATE
        else:
            kind = ActionKind.UNKNOWN

        return kind, list(dict.fromkeys(reads)), list(dict.fromkeys(writes))

    # --- one spelling per file ---------------------------------------------
    #
    # An agent reaches the same file by more than one spelling: an editor tool
    # reports /home/clap/src/lib.rs while a shell command run from the
    # repository root says src/lib.rs. Compared as raw strings, a write never
    # matched the read it invalidated, which zeroed SWE-agent's count outright
    # (its viewer reports absolute paths, its `open` command relative ones) and
    # undercounted every other framework. A trajectory's repository root is its
    # most common /home/<name>/ or /workspace/<name>/ prefix; that prefix, any
    # other root with the same <name>, /testbed/ (SWE-bench) and /app/
    # (Terminal-Bench and SWE-Bench Pro, whose tasks run there) are stripped.

    _ROOT_RE = re.compile(r"^/(?:home|workspace)/([^/]+)/|^/testbed/|^/app/")

    def canonicalize_paths(self, steps: List[TrajectoryStep],
                           manifest_item: Optional[Dict[str, Any]] = None) -> List[TrajectoryStep]:
        """Rewrite every path in `steps` to its repository-relative spelling.

        After the root is stripped, a path that still carries the repository
        directory (`ponyc/src/x.c` from a shell in /home, or
        `/home/sandbox/clap/src/x.rs`) is cut back to the part after it when
        that shorter spelling occurs elsewhere in the trajectory."""
        def paths(s):
            return (s.target_files_read + s.target_files_written
                    + [e[1] for e in s.events] + s.output_reads)

        names = Counter()
        for s in steps:
            for p in paths(s):
                m = self._ROOT_RE.match(p)
                if m and m.group(1):
                    names[m.group(1)] += 1
        repo = names.most_common(1)[0][0] if names else None
        # The task names its repository (`clap-rs__clap-2501` -> clap). When
        # /workspace is the repository itself, `/workspace/lib/...` names lib/
        # inside it, not a repository called lib.
        task = str((manifest_item or {}).get("task_name") or "")
        hint = task.split("__", 1)[1].rsplit("-", 1)[0].lower() if "__" in task else None

        def strip_root(p: str) -> str:
            m = self._ROOT_RE.match(p)
            if m and hint and m.group(1) and p.startswith("/workspace/") and m.group(1).lower() != hint:
                return p[len("/workspace/"):]
            if m and (m.group(1) is None or m.group(1) == repo):
                return p[m.end():]
            return p

        stripped = {p: strip_root(p) for s in steps for p in paths(s)}
        known = set(stripped.values())
        marker = f"{repo}/" if repo else None

        def canon(p: str) -> str:
            q = stripped.get(p, strip_root(p))
            if marker and marker in q:
                tail = q[q.index(marker) + len(marker):]
                if tail in known and (q.startswith(marker) or q.startswith("/")):
                    return tail
            return q

        for s in steps:
            s.target_files_read = list(dict.fromkeys(canon(p) for p in s.target_files_read))
            s.target_files_written = list(dict.fromkeys(canon(p) for p in s.target_files_written))
            s.events = [(op, canon(p) if p != "*" else p, map_kind_path(kind, canon), landed)
                        for op, p, kind, landed in s.events]
            s.output_reads = list(dict.fromkeys(canon(p) for p in s.output_reads))
        return steps

    def _clean_path(self, p: str) -> str:
        """Clean file paths of quotes and leading/trailing noise."""
        p = p.strip("'\"` ,;()")
        if p.startswith('./'):
            p = p[2:]
        return p

    def _strip_heredoc_bodies(self, cmd: str) -> str:
        """Remove heredoc operators and their payloads from a command string.

        Heredoc bodies are arbitrary file content, so scanning them for
        redirections yields phantom paths (a payload line such as `if a < b:`
        looks like an input redirection). The write target itself is recovered
        separately by heredoc_re before this runs.
        """
        m = self.heredoc_any_re.search(cmd)
        if not m:
            return cmd
        delim = m.group(1)
        head = cmd[:m.start()]
        rest = cmd[m.end():]
        # Drop everything up to the terminating delimiter on its own line.
        for line_end in re.finditer(r'^\s*' + re.escape(delim) + r'\s*$', rest, re.MULTILINE):
            return head + rest[line_end.end():]
        return head

    def estimate_tokens(self, text: str) -> int:
        """Approximate token count (1 token ~= 4 characters / 0.75 words)."""
        if not text:
            return 0
        return max(1, len(text) // 4)
