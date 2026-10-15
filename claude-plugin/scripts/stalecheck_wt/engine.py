"""Agent-agnostic write-through engine: keep a coding agent's view of its files current.

An agent edits files through what it last saw of them, and its own edits (a `sed -i`, a
script, a formatter) make those views out of date without telling it. The engine keeps, per
session, what the model last saw of each file and which line numbers it has seen as they are
now. An adapter tells it three things, and appends what it returns to what the model sees:

    s.read(path, lo, hi)          the model was shown a file (or lines lo..hi of it)
    s.authored(path)              the model changed a file with its own edit tool
    s.before(command) -> refusal  a shell command is about to run (None: let it run)
    s.after(command, output)      any tool call finished: text to append (may be empty)

`after` returns the net diff of each file the model saw that changed since (with current line
numbers; text from src/harness/tail_diff.py), a note per stretch of lines that moved without
being shown, and the remaining uses in other files of any definition a change removed or
re-signatured (src/harness/dependency_notice.py). `before` refuses once a `sed -i` addressed by
line number whose lines the model has not seen numbered as they are now, and the refusal shows
them. Nothing is rewritten in what the model saw before, so the prompt cache is kept.

Adapters: claude_code (hooks), protocol (a JSON command for any hook system), mcp_server (an
MCP shell tool for any MCP client). State: <state_dir>/sessions/<session>/, see State.
"""

import difflib
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.join(HERE, "_lib") not in sys.path:
    sys.path.insert(0, os.path.join(HERE, "_lib"))

from wt_src.harness.dependency_notice import dependency_notice  # noqa: E402
from wt_src.harness.lean import redundant_writes  # noqa: E402
from wt_src.metrics.replay import _live_events  # noqa: E402
from wt_src.harness.tail_diff import format_line_numbered_diff, line_number_note  # noqa: E402

BUDGET = 9000        # Claude Code caps hook context at 10,000 characters; others are similar
PER_FILE = 3000
MAX_BYTES = 2_000_000
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", "dist", "build"}
CONTEXT = 3          # context lines around a hunk, as in src/harness/tail_diff.py
SHIFTS = os.environ.get("STALECHECK_SHIFTS", "1") != "0"
GUARD = os.environ.get("STALECHECK_GUARD", "1") != "0"
KEEP_DAYS = float(os.environ.get("STALECHECK_KEEP_DAYS", "7"))  # sessions idle longer are deleted
# files whose text is never stored or diffed: they may hold credentials
SECRET = re.compile(r"""(?ix)^(?:
    \.env(?:\..+)? | .+\.(?:pem|key|p12|pfx|jks|keystore|secret|kdbx) | id_(?:rsa|dsa|ecdsa|ed25519)(?:\..+)?
  | \.(?:netrc|npmrc|pypirc|pgpass) | credentials(?:\..+)? | secrets?(?:\..+)? )$""")
NOT_SECRET = re.compile(r"(?i)\.(?:example|sample|template|dist)$|\.(?:py|js|ts|go|rs|java|rb|md)$")


def is_secret(rel):
    """A file stalecheck never copies: keys, .env files, credential stores (by name)."""
    name = os.path.basename(rel)
    return bool(SECRET.match(name)) and not NOT_SECRET.search(name)


# ---------------------------------------------------------------- state

def default_state_dir():
    return (os.environ.get("STALECHECK_STATE") or os.environ.get("CLAUDE_PLUGIN_DATA")
            or os.path.expanduser("~/.cache/stalecheck-wt"))


class State:
    """snapshot.json: path -> [size, mtime_ns] of each workspace file at the last call;
    ledger.json: path -> {"sha", "regions", "base"}: the version the model last saw, the line ranges it
    has seen numbered as they are now, and the version of its last whole-file view; written.json: files the model changed;
    echoed.json: files the host's own editor wrote last, whose later shell changes the host reports itself; blobs/<sha>: what it saw."""

    def __init__(self, session_id, state_dir=None):
        sessions = os.path.join(state_dir or default_state_dir(), "sessions")
        self.dir = os.path.join(sessions, session_id or "default")
        os.makedirs(os.path.join(self.dir, "blobs"), mode=0o700, exist_ok=True)
        prune_sessions(sessions, keep=self.dir)
        self._lock = open(os.path.join(self.dir, ".lock"), "w")
        fcntl.flock(self._lock, fcntl.LOCK_EX)  # parallel tool calls run their hooks in parallel
        self.snapshot = self._load("snapshot.json", None)
        self.ledger = self._load("ledger.json", {})
        self.written = set(self._load("written.json", []))
        self.echoed = set(self._load("echoed.json", []))

    def _load(self, name, default):
        try:
            with open(os.path.join(self.dir, name)) as f:
                return json.load(f)
        except (OSError, ValueError):
            return default

    def save(self):
        for name, obj in (("snapshot.json", self.snapshot), ("ledger.json", self.ledger), ("written.json", sorted(self.written)), ("echoed.json", sorted(self.echoed))):
            tmp = os.path.join(self.dir, name + ".tmp")
            with open(tmp, "w") as f:
                json.dump(obj, f)
            os.replace(tmp, os.path.join(self.dir, name))
        self.drop_unused_blobs()
        fcntl.flock(self._lock, fcntl.LOCK_UN)

    def drop_unused_blobs(self):
        """Delete stored file copies the ledger no longer points to (old versions, cleared views)."""
        used = {v for e in self.ledger.values() for v in (e.get("sha"), e.get("base")) if v}
        bdir = os.path.join(self.dir, "blobs")
        try:
            names = os.listdir(bdir)
        except OSError:
            return
        for name in names:
            if name not in used:
                try:
                    os.remove(os.path.join(bdir, name))
                except OSError:
                    pass

    def put_blob(self, text):
        sha = hashlib.sha256(text.encode("utf-8", "surrogateescape")).hexdigest()
        path = os.path.join(self.dir, "blobs", sha)
        if not os.path.exists(path):
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # readable by the user only
            with open(fd, "w", encoding="utf-8", errors="surrogateescape") as f:
                f.write(text)
        return sha

    def blob(self, sha):
        try:
            with open(os.path.join(self.dir, "blobs", sha), encoding="utf-8", errors="surrogateescape") as f:
                return f.read()
        except OSError:
            return None


def prune_sessions(sessions, keep=None):
    """Delete session state (with its stored file copies) idle for more than KEEP_DAYS."""
    cutoff = time.time() - KEEP_DAYS * 86400
    try:
        names = os.listdir(sessions)
    except OSError:
        return
    for name in names:
        d = os.path.join(sessions, name)
        if d == keep or not os.path.isdir(d):
            continue
        try:
            last = max(os.path.getmtime(os.path.join(d, f)) for f in os.listdir(d))
        except (OSError, ValueError):
            continue
        if last < cutoff:
            import shutil
            shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- workspace

def list_files(root):
    """Workspace files, relative to root: git's view when it is a repository, else a walk."""
    try:
        out = subprocess.run(["git", "-C", root, "ls-files", "-co", "--exclude-standard", "-z"],
                             capture_output=True, timeout=10)
        if out.returncode == 0:
            return [p for p in out.stdout.decode("utf-8", "surrogateescape").split("\0") if p]
    except (OSError, subprocess.SubprocessError):
        pass
    files = []
    for d, dirs, names in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
        files += [os.path.relpath(os.path.join(d, n), root) for n in names]
    return files


def scan(root):
    snap = {}
    for rel in list_files(root):
        try:
            st = os.stat(os.path.join(root, rel))
        except OSError:
            continue
        snap[rel] = [st.st_size, st.st_mtime_ns]
    return snap


def read_text(root, rel):
    path = os.path.join(root, rel)
    try:
        if os.path.getsize(path) > MAX_BYTES:
            return None
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    if b"\0" in data[:8192]:
        return None  # binary
    return data.decode("utf-8", "surrogateescape")


def rel_path(root, path):
    if not path:
        return None
    full = os.path.realpath(path if os.path.isabs(path) else os.path.join(root, path))
    r = os.path.relpath(full, os.path.realpath(root))
    return None if r.startswith("..") else r


# ---------------------------------------------------------------- line numbers

def merge(ranges):
    out = []
    for lo, hi in sorted((int(a), int(b)) for a, b in ranges if int(b) >= int(a)):
        if out and lo <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def covered(ranges, lo, hi):
    return all(any(a <= x <= b for a, b in ranges) for x in range(lo, hi + 1))


def remap(old, new, known):
    """Line ranges (1-based, in `new`'s numbering) the model still knows by number after old -> new:
    lines it knew whose number did not change."""
    keep = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old.splitlines(), new.splitlines(), autojunk=False).get_opcodes():
        if tag == "equal" and i1 == j1:
            keep += [[max(a, i1 + 1), min(b, i2)] for a, b in known if max(a, i1 + 1) <= min(b, i2)]
    return merge(keep)


def hunks_and_shifts(old, new):
    """The live line ranges a diff block prints, and the moved stretches it does not print."""
    a, b = old.splitlines(), new.splitlines()
    shown, shifts = [], []
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    for tag, i1, i2, j1, j2 in ops:
        if tag != "equal":
            shown.append([max(0, j1 - CONTEXT) + 1, min(len(b), j2 + CONTEXT)])
    shown = merge(shown)
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal" and j1 != i1 and not covered(shown, j1 + 1, j2):
            shifts.append((i1 + 1, i2, j1 + 1, j2, j1 - i1))
    return shown, shifts


# ---------------------------------------------------------------- blocks

def clip_block(text, limit):
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit("\n", 1)[0]
    more = text[len(cut):].count("\n")
    return cut + f"\n... {more} more changed lines not shown; re-read the file before a line-anchored edit."


def blocks_for(state, root, changed, skip, command="", output="", budget=None):
    """The write-through text for the seen files among `changed`, within the budget."""
    budget = budget or BUDGET
    seen = [p for p in sorted(changed) if p in state.ledger and p not in skip]
    if command and seen:
        try:
            seen = [p for p in seen if p not in redundant_writes(command, output, seen)]
        except Exception:  # noqa: BLE001  the lean rule is an optimisation; never fail the hook on it
            pass
    parts, before, shown = [], {}, {}
    used = 0
    for p in seen:
        old = state.blob(state.ledger[p]["sha"])
        new = read_text(root, p)
        if old is None:
            continue
        if new is None:
            text = f"[TAIL DIFF: File '{p}' was deleted, or is no longer text, since your last view]"
        else:
            text = format_line_numbered_diff(old, new, p)
        if not text:
            continue
        before[p] = old
        room = min(PER_FILE, budget - used)
        if room < 200:
            parts.append(f"[TAIL DIFF: File '{p}' also changed; re-read it before a line-anchored edit]")
            shown[p] = []
            continue
        hunks, _ = hunks_and_shifts(old, new or "")
        if SHIFTS and new is not None:
            base = state.blob(state.ledger[p].get("base") or "") if state.ledger[p].get("base") else None
            note = line_number_note(p, base if base is not None else old, new)
            if note:
                text += "\n" + note
        clipped = clip_block(text, room)
        shown[p] = hunks if clipped == text else []  # a cut block shows only part of its hunks
        parts.append(clipped)
        used += len(clipped)
    if before:
        try:
            notice, _ = dependency_notice(root, list(before), before, viewed=set(state.ledger), written=state.written)
        except Exception:  # noqa: BLE001
            notice = ""
        if notice and used + len(notice) <= budget:
            parts.append(notice)
            used += len(notice)
    unseen = sorted(p for p in changed if p not in state.ledger and p not in skip)
    if unseen and parts:
        names = ", ".join(unseen[:8]) + (f" and {len(unseen) - 8} more" if len(unseen) > 8 else "")
        parts.append(f"[Also changed, not viewed: {names}]")
    return "\n\n".join(parts), shown


def see(state, root, rel, regions=None, after_write=False):
    """Record that the model now knows `rel` as it is on disk.

    `regions` are line ranges shown with their current numbers (None: the whole file). After a
    write the model learned of (an Edit it made, a diff block), the lines it knew keep their
    numbers only where the write did not move them."""
    text = read_text(root, rel) if not is_secret(rel) else None
    if text is None:  # gone, binary, too large, or a secrets file: nothing stored, no diffs later
        state.ledger.pop(rel, None)
        return
    sha = state.put_blob(text)
    n = text.count("\n") + (0 if text.endswith("\n") or not text else 1)
    shown = [[1, max(n, 1)]] if regions is None else [list(r) for r in regions]
    old = state.ledger.get(rel)
    keep = []
    if old and old["sha"] == sha:
        keep = old.get("regions") or []
    elif old and after_write:
        prev = state.blob(old["sha"])
        if prev is not None:
            keep = remap(prev, text, old.get("regions") or [])
    # base: the version the model last saw whole; line-number notes count from it
    base = sha if (regions is None and not after_write) else ((old or {}).get("base") or (old or {}).get("sha"))
    state.ledger[rel] = {"sha": sha, "regions": merge(keep + shown), "base": base}


# ---------------------------------------------------------------- reads through the shell

SED_PRINT = re.compile(r"""sed\s+-n\s+['"]?(\d+)(?:\s*,\s*(\d+))?p['"]?""")
HEAD_N = re.compile(r"\bhead\s+(?:-n\s*|-)(\d+)")
NUMBERED = re.compile(r"(?m)^(?:([^\s:]+):)?(\d+)[:-]")


def shell_reads(root, command, output):
    """(path, line ranges or None for the whole file) for each file a shell command printed.

    Which files: the replay's command parser (src/metrics/replay.py). Which lines: `cat`/`nl`/`bat`
    print the whole file, `sed -n 'A,Bp'` lines A-B, `head -n N` lines 1-N, `grep -n`/`rg -n` the
    numbered lines in the output; anything else (tail, awk, a pipe) leaves the version seen but no
    line number known."""
    try:
        events = _live_events({"command": command, "observation": output, "writes_detected": []}, {})
    except Exception:  # noqa: BLE001
        return []
    out = []
    for op, path, kind, landed in events:
        if op != "R" or not landed:
            continue
        rel = rel_path(root, path)
        if not rel or not os.path.isfile(os.path.join(root, rel)):
            continue
        regions = []
        m = SED_PRINT.search(command)
        h = HEAD_N.search(command)
        if m:
            regions = [[int(m.group(1)), int(m.group(2) or m.group(1))]]
        elif h:
            regions = [[1, int(h.group(1))]]
        elif re.search(r"\b(grep|rg|ag)\b.*\s-\w*n", command):
            nums = [int(n) for f, n in NUMBERED.findall(output) if not f or f.endswith(os.path.basename(rel))]
            regions = merge([[n, n] for n in nums])
        elif re.match(r"\s*(?:cd\s+\S+\s*&&\s*)?(cat|nl|bat)\b", command) and "|" not in command:
            regions = None
        out.append((rel, regions))
    return out


# ---------------------------------------------------------------- guard

SED_LINE = re.compile(r"^\s*(\d+)\s*(?:,\s*(\d+|\$)\s*)?!?\s*([acdisy])")


def shell_segments(command):
    """Simple commands of a shell line, split at `&&`, `||`, `;`, `|`, `&` and unquoted newlines."""
    lex = shlex.shlex(command, posix=True, punctuation_chars="();<>|&\n")
    lex.whitespace = " \t\r"
    lex.whitespace_split = True
    seg, out = [], []
    try:
        for tok in lex:
            if tok and set(tok) <= set(";|&\n()"):
                if seg:
                    out.append(seg)
                seg = []
            else:
                seg.append(tok)
    except ValueError:  # unbalanced quotes: nothing safe to say
        return []
    if seg:
        out.append(seg)
    return out


def sed_line_edits(command):
    """(file, first line, last line) for each `sed -i` script command addressed by line number."""
    out = []
    for words in shell_segments(command):
        while words and (words[0] in ("sudo", "command") or "=" in words[0]):
            words = words[1:]
        if not words or os.path.basename(words[0]) not in ("sed", "gsed"):
            continue
        inplace, scripts, files, explicit, i = False, [], [], "-e" in words or "--expression" in words, 1
        while i < len(words):
            w = words[i]
            if w in ("-e", "--expression") and i + 1 < len(words):
                scripts.append(words[i + 1]); i += 2; continue
            if w.startswith("-i") or w.startswith("--in-place"):
                inplace = True
            elif w.startswith("-") and len(w) > 1:
                pass
            elif not scripts and not explicit:
                scripts.append(w)
            elif w not in (">", ">>", "<"):
                files.append(w)
            i += 1
        if not inplace:
            continue
        for script in scripts:
            for cmd in re.split(r"[;\n]", script):
                m = SED_LINE.match(cmd)
                if not m:
                    continue
                lo = int(m.group(1))
                hi = lo if not m.group(2) or m.group(2) == "$" else int(m.group(2))
                out += [(f, lo, max(lo, hi)) for f in files]
    return out


def guard(state, root, command):
    """A refusal for a line-numbered edit whose lines the model has not seen numbered as they are now."""
    for f, lo, hi in sed_line_edits(command):
        rel = rel_path(root, f)
        entry = state.ledger.get(rel) if rel else None
        if not entry:
            continue  # a file the model never saw: not a stale view
        text = read_text(root, rel)
        if text is None:
            continue
        sha = hashlib.sha256(text.encode("utf-8", "surrogateescape")).hexdigest()
        known = entry.get("regions") or [] if entry["sha"] == sha else []
        if covered(known, lo, hi):
            continue
        lines = text.splitlines()
        a, b = max(1, lo - 5), min(len(lines), hi + 5, lo + 80)
        body = "\n".join(f"{k:6d}\t{lines[k - 1]}" for k in range(a, b + 1))
        see(state, root, rel, [[a, b]])  # the refusal shows them: a retry with these numbers passes
        why = (f"'{rel}' changed since you last saw lines {lo}-{hi} numbered" if entry["sha"] != sha
               else f"you have not seen lines {lo}-{hi} of '{rel}' with their line numbers")
        return (f"[STALE LINE NUMBERS] {why}, so they "
                f"may not be the lines you mean. Here they are as the file stands now (lines {a}-{b}):\n{body}\n"
                f"Reissue the command with these line numbers if this is what you meant to change.")
    return None


# ---------------------------------------------------------------- session API

class Session:
    """One agent session's write-through state. Use as a context manager: the state is locked
    while open (parallel tool calls run their hooks in parallel) and saved on exit."""

    def __init__(self, root, session_id="default", state_dir=None, budget=None, host_echoes=False):
        self.root = os.path.realpath(root)
        # host_echoes: the host re-sends the changed lines of a file its own editor wrote last when
        # a shell command changes it (Claude Code does), so a block for that change would repeat it
        self.host_echoes = host_echoes
        self.state = State(session_id, state_dir)
        self.budget = budget or BUDGET  # characters an agent accepts per tool call
        if self.state.snapshot is None:
            # no session-start hook (Cursor, Cline) or the first call after install: changes are
            # counted from now on
            self.state.snapshot = scan(self.root)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.state.save()
        return False

    def start(self, fresh=True):
        """A session (re)starts. fresh: the model holds none of its earlier views (a new session,
        a cleared or compacted context)."""
        if fresh or self.state.snapshot is None:
            self.state.ledger, self.state.written, self.state.echoed = {}, set(), set()
        self.state.snapshot = scan(self.root)

    def read(self, path, lo=None, hi=None):
        rel = rel_path(self.root, path)
        if rel:
            see(self.state, self.root, rel, None if lo is None else [[int(lo), int(hi if hi is not None else lo)]])
        return rel

    def authored(self, path):
        rel = rel_path(self.root, path)
        if rel:
            see(self.state, self.root, rel, [], after_write=True)
            self.state.written.add(rel)
            if self.host_echoes:
                self.state.echoed.add(rel)
        return rel

    def before(self, command):
        return guard(self.state, self.root, command) if GUARD and command else None

    def after(self, command="", output="", read=None, read_range=None, authored=None):
        """Text to append after a tool call. `read` / `authored` name a file this very call showed
        or changed through the agent's own tools, so it is not reported back."""
        now = scan(self.root)
        prev = self.state.snapshot if self.state.snapshot is not None else now
        changed = {p for p in set(now) | set(prev) if now.get(p) != prev.get(p)}
        changed = {p for p in changed if not (p in self.state.ledger and (t := read_text(self.root, p)) is not None
                                             and hashlib.sha256(t.encode("utf-8", "surrogateescape")).hexdigest()
                                             == self.state.ledger[p]["sha"])}  # same content: not a change
        skip = set()
        if read:
            rel = self.read(read, *(read_range or (None, None)))
            skip.add(rel)
        for a in ([authored] if isinstance(authored, str) else authored or []):
            skip.add(self.authored(a))
        reads = shell_reads(self.root, command, output) if command else []
        skip |= {p for p, regions in reads if regions is None}  # printed whole: nothing to add
        if command and self.host_echoes:
            for p in changed & self.state.echoed:  # the host reports this change; the shell is now the last writer
                see(self.state, self.root, p, [], after_write=True)
                skip.add(p)
            self.state.echoed -= changed
        text, shown = blocks_for(self.state, self.root, changed, skip, command or "", output if command else "", self.budget)
        for p, hunks in shown.items():
            see(self.state, self.root, p, hunks, after_write=True)
            self.state.written.add(p)
        for p, regions in reads:  # what the command printed is now what the model saw
            see(self.state, self.root, p, regions)
        self.state.snapshot = now
        return text
