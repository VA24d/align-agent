"""Independent trajectory extractor for the RQ1 audit. Usage: python3 scripts/audit/extract.py [--only TRAJ_ID]

Reads raw trajectories straight from the cache (never src/parsers, never the
downloader) and writes one normalised record per trajectory to
docs/audit/rq1/work/extracted.jsonl (gitignored; about 60 MB):

  steps[i] = {i, unit, conv, bench, tool, action, obs, rc, cwd, ops, tok_a, tok_o}

`unit` groups steps composed together (a Terminus2 episode, an OpenHands
reply with several tool calls); `bench` is the CodeTraceBench step id under
the step definition that reproduces step_count for that framework (None when
the trajectory does not align). ops come from scripts/audit/shellparse.py for
shell text and from the editor arguments for editor tools; every write op
carries `landed` (True/False/None) and the evidence used.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import WORK, load_manifest, load_raw  # noqa: E402
import shellparse as sp  # noqa: E402

TOK = 4  # chars per token


def tok(s):
    return len(s or "") // TOK


# ---------------------------------------------------------------- outcome evidence

SED_ERR = re.compile(r"^sed: (-e expression #\d+, char \d+: .*|can't read ([^:\n]+): .*|couldn't .*|cannot rename .*|no input files)$", re.M)
NOT_FOUND = re.compile(r"(?:^|[\s:])([\w./+-]+): (?:command )?not found", re.M)
PY_TB_FRAME = re.compile(r'File "([^"]+)", line (\d+)')
NO_SUCH = re.compile(r"([^\s:'\"`]+): No such file or directory")
PY_SYNTAX = re.compile(r"^\s*(SyntaxError|IndentationError|TabError): ", re.M)
TRACEBACK = re.compile(r"Traceback \(most recent call last\)")
BASH_SYNTAX = re.compile(r"syntax error near unexpected token|syntax error: unexpected end of file|Syntax error: ")
HEREDOC_EOF = re.compile(r"here-document at line \d+ delimited by end-of-file")
UNTERMINATED = re.compile(r"unexpected EOF while looking for matching|Unterminated quoted string")
TOOL_REFUSED = re.compile(r"^\s*ERROR: Cannot execute multiple commands at once")
PATCH_FAIL = re.compile(r"No valid patches in input|corrupt patch|Hunk #\d+ FAILED|hunks? FAILED|patch: \*\*\*\* |error: patch failed|does not apply|malformed patch|Only garbage was found")
EDIT_OK = re.compile(r"has been edited|created successfully|File created successfully|undone successfully|Text inserted|successfully inserted", re.I)
EDIT_FAIL = re.compile(r"No replacement was performed|did not appear verbatim|Multiple occurrences of old_str|is not a valid|does not exist|already exists|Invalid `|ERROR:|Parameter `\w+` is required|No edit history found|cannot be empty|must be different|should be different", re.I)


SH_SYNTAX_AT = re.compile(r"line (\d+): syntax error([^\n]*)|(?:^|\n)\S*sh: (\d+): Syntax error:([^\n]*)|line (\d+): unexpected EOF")


def unit_lines(cmd_text, ops):
    """Line (1-based) on which each simple command's verb first appears,
    found by a sequential search through the command text."""
    lines = {}
    pos = 0
    text = sp.mask_heredocs(cmd_text or "")
    for ci in sorted({o["cmd_i"] for o in ops}):
        verb = next((o.get("verb") for o in ops if o["cmd_i"] == ci), "") or ""
        f = text.find(verb, pos) if verb else -1
        if f < 0:
            f = pos
        lines[ci] = text.count("\n", 0, f) + 1
        pos = f + max(1, len(verb))
    return lines


def shell_outcome(ops, obs, rc, cmd_text, truncated=False):
    """Mark each op's `ran` and each write's `landed` (True/False/None) from
    the command's own output. Rules, recorded in op['ev']:
      shell syntax error: an unterminated construct (EOF) runs nothing; an
        unexpected token at line N runs the commands on earlier lines
      verb not found ("applypatch: not found"): that command ran nothing
      sed: an error line; `can't read F` names the file
      No such file or directory naming the target
      python: SyntaxError with no traceback -> nothing ran; a traceback ->
        landed iff the script's first write line precedes the raising line
        (unknown when the frames cannot be matched)
      patch: FAILED hunks / does not apply
      && chains: a command after a failed command on an && guard did not run
    Otherwise landed=True (no failure evidence)."""
    obs = obs or ""
    for o in ops:
        o["ran"] = True
    if TOOL_REFUSED.search(obs):
        # OpenHands refused the multi-command input: nothing was executed
        for o in ops:
            o["ran"] = False
            o["landed"], o["ev"] = False, "tool refused: multiple commands"
        return
    m = SH_SYNTAX_AT.search(obs)
    if m or UNTERMINATED.search(obs):
        if m and (m.group(1) or m.group(3)):
            n = int(m.group(1) or m.group(3))
            msg = (m.group(2) or m.group(4) or "")
            eof = "end of file" in msg or "EOF" in msg
        else:
            n, eof = 1, True
        ul = unit_lines(cmd_text, ops)
        for o in ops:
            if eof or ul.get(o["cmd_i"], 1) >= n:
                o["ran"] = False
                o["landed"], o["ev"] = False, "shell syntax error: did not run" + (" (EOF)" if eof else f" (line {n})")
    sed_errs = SED_ERR.findall(obs)
    sed_units = sorted({o["cmd_i"] for o in ops if o.get("mech") == "sed" and o["op"] == "W" and o["ran"]})
    # A sed expression error is a compile error: that sed touched nothing,
    # and with one error line the failing sed is the first whose script is at
    # least as long as the reported character offset.
    failing_sed = None
    char_errs = re.findall(r"^sed: -e expression #(\d+), char (\d+): ", obs, re.M)
    if len(char_errs) == 1 and len(sed_units) > 1:
        e_no, ch = int(char_errs[0][0]), int(char_errs[0][1])
        for ci in sed_units:
            op0 = next(o for o in ops if o["cmd_i"] == ci and o.get("mech") == "sed" and o["op"] == "W")
            raw = ((op0.get("anchor") or {}).get("raw") or [])
            if len(raw) >= e_no and len(raw[e_no - 1]) >= ch:
                failing_sed = ci
                break
    missing = set(mm.strip() for mm in NO_SUCH.findall(obs))
    notfound = set(NOT_FOUND.findall(obs))
    failed_units = set()
    for o in ops:
        if not o["ran"]:
            failed_units.add(o["cmd_i"])
            continue
        ev, landed = "no failure evidence", True
        if o.get("verb") in notfound:
            landed, ev = False, f"{o.get('verb')}: not found"
            o["ran"] = False
        elif o.get("mech") == "sed" and o["op"] == "W" and sed_errs:
            named = [x[1] for x in sed_errs if x[1]]
            if named:
                if any(o["path"].endswith(nm.lstrip("./")) or nm.endswith(os.path.basename(o["path"])) for nm in named):
                    landed, ev = False, "sed: can't read target"
            elif len(sed_units) == 1:
                landed, ev = False, "sed error (single sed unit)"
            elif failing_sed is not None:
                if o["cmd_i"] == failing_sed:
                    landed, ev = False, "sed expression error (identified by char offset)"
                elif o["cmd_i"] < failing_sed:
                    landed, ev = True, "sed before the failing sed"
                else:
                    landed, ev = None, "sed after the failing sed"
            else:
                landed, ev = None, "sed error (ambiguous unit)"
        elif o.get("mech") in ("python",) and o["op"] == "W":
            if PY_SYNTAX.search(obs) and not TRACEBACK.search(obs):
                landed, ev = False, "python SyntaxError: nothing ran"
                o["ran"] = False
            elif TRACEBACK.search(obs):
                src = o.get("script_src") or ""
                base = os.path.basename(src[5:]) if src.startswith("file:") else None
                fl = [int(nn) for f, nn in PY_TB_FRAME.findall(obs)
                      if (f in ("<stdin>", "<string>", "-") and not base) or (base and (f.endswith("/" + base) or f == base))]
                wl = o.get("write_line")
                if fl and wl:
                    if wl > max(fl):
                        landed, ev = False, f"python raised at line {max(fl)} before its write (line {wl})"
                    else:
                        landed, ev = True, f"python wrote (line {wl}) before raising at line {max(fl)}"
                else:
                    landed, ev = None, "python traceback, write order unknown"
        elif o.get("mech") in ("patch", "git-apply", "applypatch") and PATCH_FAIL.search(obs):
            landed, ev = False, "patch failed"
        if landed is not False and o["op"] == "W":
            base = os.path.basename(o["path"])
            for mm in missing:
                if mm.endswith(base) and o["kind"] in ("anchored",) and o.get("mech") != "python":
                    landed, ev = False, "No such file or directory"
        if o.get("guard") == "&&" and (o["cmd_i"] - 1) in failed_units:
            landed, ev = False, "&& after a failed command"
            o["ran"] = False
        if landed is False:
            failed_units.add(o["cmd_i"])
        if o["op"] == "W":
            o["landed"], o["ev"] = landed, ev
        else:
            o["landed"] = landed is not False and o["ran"]
            if o["path"] in missing or any(mm.endswith(os.path.basename(o["path"])) and o["path"].endswith(mm.lstrip("./"))
                                            for mm in missing):
                o["landed"], o["ev"] = False, "No such file (view failed)"
    if HEREDOC_EOF.search(obs):
        for o in ops:
            o.setdefault("flags", []).append("heredoc_eof")


def editor_outcome(sub, result):
    if result is None:
        return None, "no result on record"
    if EDIT_FAIL.search(result) and not EDIT_OK.search(result):
        return False, "editor rejected: " + EDIT_FAIL.search(result).group(0)
    if EDIT_OK.search(result):
        return True, "editor confirmed"
    return None, "editor result unclear"


# ---------------------------------------------------------------- editor ops

def editor_ops(sub, path, args, result, cwd="/"):
    """Ops for one str_replace_editor-style call."""
    if not path:
        return []
    p = sp.resolve(path, cwd)
    if sub in ("view", "open"):
        if result is not None and re.match(r"\s*(ERROR|Invalid|The path .* does not exist)", result or ""):
            return []
        if result is not None and ("files and directories up to" in (result or "") or "is a directory" in (result or "")):
            return [{"op": "R", "path": p, "kind": "listing", "mech": "editor-view", "view": {}, "landed": True}]
        vr = args.get("view_range")
        if isinstance(vr, str):
            try:
                vr = json.loads(vr)
            except Exception:
                vr = None
        view = {"range": [int(vr[0]), int(vr[1]) if int(vr[1]) > 0 else 10 ** 9]} if isinstance(vr, (list, tuple)) and len(vr) == 2 else {"range": [1, 10 ** 9]}
        view["numbered"] = True
        kind = "full" if view["range"] == [1, 10 ** 9] else "range"
        return [{"op": "R", "path": p, "kind": kind, "mech": "editor-view", "view": view, "landed": True}]
    landed, ev = editor_outcome(sub, result)
    if sub == "str_replace":
        op = {"op": "W", "path": p, "kind": "anchored", "mech": "editor",
              "anchor": {"type": "str_replace", "old": args.get("old_str"), "new": args.get("new_str") or ""},
              "landed": landed, "ev": ev}
    elif sub == "insert":
        op = {"op": "W", "path": p, "kind": "anchored", "mech": "editor",
              "anchor": {"type": "insert", "line": args.get("insert_line"), "new": args.get("new_str") or ""},
              "landed": landed, "ev": ev}
    elif sub == "create":
        op = {"op": "W", "path": p, "kind": "overwrite", "mech": "editor-create",
              "text": args.get("file_text"), "landed": landed, "ev": ev}
    elif sub == "undo_edit":
        op = {"op": "W", "path": p, "kind": "undo", "mech": "editor-undo", "landed": landed, "ev": ev}
    else:
        return []
    ops = [op]
    if landed and result and "result of running `cat -n`" in result:
        ops.append({"op": "R", "path": p, "kind": "echo", "mech": "editor-echo", "view": {"numbered": True},
                    "landed": True})
    return ops


# ---------------------------------------------------------------- mini-SWE-agent

MINI_ACTION = re.compile(r"```(?:mswea_bash_command|bash)\s*\n(.*?)\n```", re.S)
MINI_CWD = {"mini_swe_agent__multi": "/home", "mini_swe_agent__poly": "/testbed",
            "mini_swe_agent__pro": "/app", "mini_swe_agent__verified": "/testbed"}


def subset_of(item):
    sr = item.get("source_relpath") or ""
    return sr.split("/")[1] if sr.startswith("swe_raw/") else "tbench"


def mini_obs(text):
    rc_m = re.search(r"<returncode>(-?\d+)</returncode>", text or "")
    rc = int(rc_m.group(1)) if rc_m else None
    trunc = "<output_head>" in (text or "") or "<warning>" in (text or "")
    m = re.search(r"<output>\n?(.*?)</output>", text or "", re.S)
    if m:
        out = m.group(1)
        # Blocks a write-through hook appended after the command's output
        # (src/harness): part of what the agent was shown.
        tail = (text or "")[m.end():]
        if re.search(r"(?m)^\[(?:TAIL DIFF|VIEWPORT|DEPENDENCY NOTICE)", tail):
            out += tail
    else:
        h = re.search(r"<output_head>\n?(.*?)</output_head>", text or "", re.S)
        t = re.search(r"<output_tail>\n?(.*?)</output_tail>", text or "", re.S)
        out = ((h.group(1) if h else "") + "\n[...elided...]\n" + (t.group(1) if t else "")) if (h or t) else (text or "")
    return rc, out, trunc


def extract_mini(raw, item):
    msgs = raw.get("messages", [])
    cwd0 = item.get("cwd") or MINI_CWD.get(subset_of(item), "/app")
    steps, known = [], {}
    k = 0
    final_diff = None
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        k += 1
        content = m.get("content") or ""
        blocks = MINI_ACTION.findall(content)
        nxt = msgs[i + 1] if i + 1 < len(msgs) else None
        obs_raw = nxt.get("content", "") if nxt and nxt.get("role") in ("user", "tool") else ""
        obs_raw = obs_raw if isinstance(obs_raw, str) else str(obs_raw)
        rc, out, trunc = mini_obs(obs_raw)
        ran = len(blocks) == 1 and "Please always provide EXACTLY ONE action" not in obs_raw
        cmd = blocks[0] if ran else ""
        ops = []
        if ran:
            ops, _, info = sp.extract(cmd, cwd0, known, persist_cd=False)
            shell_outcome(ops, out, rc, cmd, trunc)
        steps.append({"i": len(steps), "unit": len(steps), "conv": 0, "bench": k, "tool": "shell",
                      "action": cmd if ran else content[-2000:], "thought": content.split("```")[0][-1500:],
                      "obs": out, "rc": rc, "truncated": trunc, "cwd": cwd0, "ran": ran, "ops": ops,
                      "tok_a": tok(content), "tok_o": tok(obs_raw)})
    last = msgs[-1] if msgs else {}
    if last.get("role") == "user" and (last.get("content") or "").lstrip().startswith("diff --git"):
        final_diff = last["content"]
    return steps, {"n_conv": 1, "final_diff": final_diff, "bench_status": "exact" if k == item["step_count"] else "none"}


# ---------------------------------------------------------------- SWE-agent

def _state(e):
    st = e.get("state")
    if isinstance(st, str):
        try:
            st = json.loads(st)
        except Exception:
            st = {}
    return st if isinstance(st, dict) else {}


def parse_sre(action):
    """Parse a `str_replace_editor <sub> <path> --flag value ...` action."""
    try:
        toks = shlex.split(action, posix=True)
    except ValueError:
        toks = None
    if not toks or toks[0] != "str_replace_editor" or len(toks) < 3:
        m = re.match(r"str_replace_editor\s+(\w+)\s+(\S+)", action)
        return (m.group(1), m.group(2), {"_unparsed": True}) if m else (None, None, {})
    sub, path, args = toks[1], toks[2], {}
    i = 3
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            key = t[2:]
            if key == "view_range" and i + 2 < len(toks):
                args["view_range"] = [toks[i + 1], toks[i + 2]]
                i += 3
                continue
            args[key] = toks[i + 1] if i + 1 < len(toks) else ""
            i += 2
        else:
            i += 1
    if "insert_line" in args:
        try:
            args["insert_line"] = int(args["insert_line"])
        except ValueError:
            pass
    return sub, path, args


SWE_OLD_VIEW = re.compile(r"^\[File: (\S+) \((\d+) lines total\)\]", re.M)


def extract_swe(raw, item):
    tr = raw.get("trajectory", [])
    steps, known = [], {}
    cwd = "/testbed"
    st0 = _state(tr[0]) if tr else {}
    open_file = None
    for idx, e in enumerate(tr, 1):
        a = (e.get("action") or "").strip()
        obs = e.get("observation") or ""
        obs = obs if isinstance(obs, str) else str(obs)
        st = _state(e)
        cwd_before = cwd
        ops = []
        verb = a.split(None, 1)[0] if a else ""
        tool = "shell"
        if verb == "str_replace_editor":
            tool = "editor"
            sub, path, args = parse_sre(a)
            if sub:
                ops = editor_ops(sub, path, args, obs, cwd)
        elif verb in ("open", "goto", "scroll_up", "scroll_down", "search_file", "create", "edit"):
            tool = "editor-old"
            m = SWE_OLD_VIEW.search(obs)
            shown = m.group(1) if m else None
            if verb == "open":
                parts = a.split()
                if len(parts) > 1:
                    open_file = sp.resolve(parts[1].strip('"\''), cwd)
                if shown:
                    open_file = sp.resolve(shown, cwd)
                    ops = [{"op": "R", "path": open_file, "kind": "range", "mech": "swe-open", "view": {"window": True}, "landed": True}]
            elif verb in ("goto", "scroll_up", "scroll_down"):
                if shown:
                    open_file = sp.resolve(shown, cwd)
                    ops = [{"op": "R", "path": open_file, "kind": "range", "mech": "swe-" + verb, "view": {"window": True}, "landed": True}]
            elif verb == "search_file":
                try:
                    parts = shlex.split(a)
                except ValueError:
                    parts = a.split()
                f = sp.resolve(parts[2], cwd) if len(parts) > 2 else open_file
                if f and "Line " in obs:
                    ops = [{"op": "R", "path": f, "kind": "lines", "mech": "swe-search", "view": {"grep": parts[1:2]}, "landed": True}]
            elif verb == "create":
                parts = a.split()
                if len(parts) > 1:
                    f = sp.resolve(parts[1], cwd)
                    ok = "already exists" not in obs
                    ops = [{"op": "W", "path": f, "kind": "overwrite", "mech": "swe-create", "text": "", "landed": ok,
                            "ev": "create"}]
                    if ok:
                        open_file = f
            elif verb == "edit":
                m2 = re.match(r"edit\s+(?:(\S+)\s+)?(\d+):(\d+)", a)
                if m2:
                    f = sp.resolve(m2.group(1), cwd) if m2.group(1) else open_file
                    body = a.split("\n", 1)[1] if "\n" in a else ""
                    body = re.sub(r"\n?end_of_edit\s*$", "", body)
                    rejected = any(x in obs for x in ("Usage: edit", "Your proposed edit has introduced new syntax error",
                                                      "This is how your edit would have looked if applied"))
                    if f:
                        ops = [{"op": "W", "path": f, "kind": "anchored", "mech": "swe-edit",
                                "anchor": {"type": "lines", "start": int(m2.group(2)), "end": int(m2.group(3)), "new": body},
                                "landed": not rejected, "ev": "swe edit " + ("rejected" if rejected else "applied")}]
                        if not rejected and shown:
                            ops.append({"op": "R", "path": f, "kind": "echo", "mech": "swe-edit-echo", "view": {"window": True}, "landed": True})
        elif verb in ("submit", "exit_cost", "exit_format", "find_file", "search_dir", ""):
            tool = "other"
        else:
            ops, cwd, _ = sp.extract(a, cwd, known, persist_cd=True)
            shell_outcome(ops, obs, None, a)
        if st.get("working_dir"):
            cwd = st["working_dir"]
        if st.get("open_file") and st.get("open_file") != "n/a":
            open_file = sp.resolve(st["open_file"], cwd)
        steps.append({"i": len(steps), "unit": len(steps), "conv": 0, "bench": idx, "tool": tool, "action": a,
                      "thought": (e.get("thought") or "")[-1500:], "obs": obs, "rc": None, "cwd": cwd_before,
                      "ran": True, "ops": ops, "tok_a": tok((e.get("response") or "") or a), "tok_o": tok(obs)})
    sub = raw.get("info", {}).get("submission")
    return steps, {"n_conv": 1, "final_diff": sub, "bench_status": "exact" if len(tr) == item["step_count"] else "none"}


# ---------------------------------------------------------------- OpenHands completion logs

def _txt(c):
    if isinstance(c, list):
        return " ".join(x.get("text", "") for x in c if isinstance(x, dict))
    return "" if c is None else str(c)


def _reply(x):
    ch = ((x.get("response") or {}).get("choices") or [])
    return ch[0].get("message", {}) if ch else {}


def oh_conversations(calls):
    conv_of, all_ids = [], []
    for x in calls:
        pids = set()
        for m in x.get("messages") or []:
            if not isinstance(m, dict):
                continue
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict) and tc.get("id"):
                    pids.add(tc["id"])
            if m.get("role") == "tool" and m.get("tool_call_id"):
                pids.add(m["tool_call_id"])
        best, best_n = None, 0
        for c, ids in enumerate(all_ids):
            n = len(ids & pids)
            if n > best_n:
                best, best_n = c, n
        if best is None:
            best = len(all_ids)
            all_ids.append(set())
        conv_of.append(best)
        all_ids[best] |= {tc.get("id") for tc in _reply(x).get("tool_calls") or [] if isinstance(tc, dict)}
    return conv_of


OH_CWD = re.compile(r"\[Current working directory: ([^\]]+)\]")
OH_RC = re.compile(r"\[The command completed with exit code (-?\d+)\.\]|\[Command finished with exit code (-?\d+)\]")


def extract_oh_log(raw, item):
    calls = [x for x in raw if isinstance(x, dict) and "response" in x]
    conv = oh_conversations(calls)
    results = {}
    for x in calls:
        for m in x.get("messages") or []:
            if isinstance(m, dict) and m.get("role") == "tool" and m.get("tool_call_id"):
                results.setdefault(m["tool_call_id"], _txt(m.get("content")))
    first_user = ""
    for x in calls[:1]:
        for m in x.get("messages") or []:
            if isinstance(m, dict) and m.get("role") == "user":
                first_user = _txt(m.get("content"))
                break
    m0 = re.search(r"(/workspace/[\w.\-]+)", first_user)
    cwd_conv = defaultdict(lambda: m0.group(1) if m0 else "/workspace")
    known = defaultdict(dict)
    steps = []
    per_conv_nonfinish = Counter()
    for ci, (x, c) in enumerate(zip(calls, conv)):
        msg = _reply(x)
        tcs = msg.get("tool_calls") or []
        if not tcs:
            steps.append({"i": len(steps), "unit": ci, "conv": c, "bench": None, "tool": "message",
                          "action": _txt(msg.get("content"))[:2000], "thought": "", "obs": "", "rc": None,
                          "cwd": cwd_conv[c], "ran": False, "ops": [], "tok_a": tok(_txt(msg.get("content"))), "tok_o": 0})
            continue
        for tc in tcs:
            fn = (tc.get("function") or {}).get("name", "")
            try:
                args = json.loads((tc.get("function") or {}).get("arguments") or "{}")
            except Exception:
                args = {}
            if not isinstance(args, dict):
                args = {}
            res = results.get(tc.get("id"))
            cwd_before = cwd_conv[c]
            ops, rc, tool, action = [], None, "other", f"{fn}: {json.dumps(args)[:1500]}"
            if fn in ("execute_bash", "bash", "terminal"):
                tool = "shell"
                action = args.get("command", "")
                m = OH_RC.search(res or "")
                rc = int(m.group(1) or m.group(2)) if m else None
                ops, _, _ = sp.extract(action, cwd_before, known[c], persist_cd=True)
                shell_outcome(ops, res or "", rc, action)
                mc = OH_CWD.search(res or "")
                if mc:
                    cwd_conv[c] = mc.group(1).strip()
            elif fn == "str_replace_editor":
                tool = "editor"
                sub = args.get("command", "")
                ops = editor_ops(sub, args.get("path", ""), args, res, cwd_before)
                action = f"str_replace_editor {sub} {args.get('path', '')}"
            if fn != "finish":
                per_conv_nonfinish[c] += 1
            steps.append({"i": len(steps), "unit": ci, "conv": c, "bench": None, "tool": tool, "fn": fn,
                          "args": {k: v for k, v in args.items() if k in ("old_str", "new_str", "file_text", "insert_line", "view_range", "command", "path")},
                          "action": action, "thought": _txt(msg.get("content"))[-1500:], "obs": res or "",
                          "rc": rc, "cwd": cwd_before, "ran": res is not None, "ops": ops,
                          "tok_a": tok(json.dumps(args)), "tok_o": tok(res)})
    # benchmark alignment: non-finish tool calls of the conversation whose count equals step_count
    sc = item["step_count"]
    match = [c for c, n in per_conv_nonfinish.items() if n == sc]
    status, annotated_conv = "none", None
    if len(match) == 1:
        status, annotated_conv = "exact", match[0]
    elif len(set(conv)) == 1 and sc == 48 and per_conv_nonfinish[0] > 48:
        status, annotated_conv = "prefix48", 0
    elif len(match) > 1:
        status = "ambiguous"
    if annotated_conv is not None:
        k = 0
        for s in steps:
            if s["conv"] == annotated_conv and s.get("fn") and s.get("fn") != "finish":
                k += 1
                s["bench"] = k if k <= sc else None
    return steps, {"n_conv": len(set(conv)), "final_diff": None, "bench_status": status,
                   "annotated_conv": annotated_conv}


# ---------------------------------------------------------------- OpenHands event streams

def extract_oh_events(raw, item):
    byid = {}
    for it in raw:
        for e in (it if isinstance(it, list) else [it]):
            if isinstance(e, dict) and isinstance(e.get("id"), int) and ("action" in e or "observation" in e):
                byid.setdefault(e["id"], e)
    evs = [byid[k] for k in sorted(byid)]
    obs_by_cause = {}
    for e in evs:
        if "observation" in e and e.get("cause") is not None:
            obs_by_cause.setdefault(e["cause"], e)
    steps, known = [], {}
    cwd = "/app"
    k = 0
    unit_ids = {}
    for e in evs:
        if e.get("source") != "agent" or not e.get("action"):
            continue
        tcm = e.get("tool_call_metadata")
        if tcm is None:
            continue
        k += 1
        mr = (tcm.get("model_response") or {}).get("id") or f"ev{e['id']}"
        unit = unit_ids.setdefault(mr, len(unit_ids))
        a = e.get("action")
        args = e.get("args") or {}
        o = obs_by_cause.get(e["id"])
        res = None if o is None else _txt(o.get("content"))
        extras = (o or {}).get("extras") or {}
        cwd_before = cwd
        ops, rc, tool, action = [], None, "other", f"{a}: {json.dumps(args)[:800]}"
        if a == "run":
            tool = "shell"
            action = args.get("command", "")
            md = extras.get("metadata") or {}
            rc = md.get("exit_code")
            ops, _, _ = sp.extract(action, cwd_before, known, persist_cd=True)
            shell_outcome(ops, res or "", rc, action)
            if md.get("working_dir"):
                cwd = md["working_dir"]
        elif a == "read":
            tool = "editor"
            ops = editor_ops("view", args.get("path", ""), {"view_range": args.get("view_range")}, res, cwd_before)
            if o is not None and o.get("observation") == "error":
                ops = []
            action = f"read {args.get('path')} {args.get('view_range') or ''}"
        elif a == "edit":
            tool = "editor"
            sub = args.get("command") or ("create" if args.get("file_text") else "str_replace")
            ops = editor_ops(sub, args.get("path", ""), args, res if o is not None and o.get("observation") != "error" else (res or "ERROR: error observation"), cwd_before)
            if o is not None and o.get("observation") == "edit":
                for op in ops:
                    if op["op"] == "W":
                        op["oracle"] = {"old_content": extras.get("old_content"), "new_content": extras.get("new_content"),
                                        "prev_exist": extras.get("prev_exist")}
            action = f"edit {sub} {args.get('path')}"
        elif a == "run_ipython":
            tool = "ipython"
            code = args.get("code", "")
            action = code
            w, _ = sp.python_script_ops(code, cwd_before)
            for op in w:
                op["landed"], op["ev"] = (None, "ipython traceback") if res and TRACEBACK.search(res) else (True, "no failure evidence")
            ops = w
        steps.append({"i": len(steps), "unit": unit, "conv": 0, "bench": k, "tool": tool, "fn": a,
                      "args": {kk: vv for kk, vv in args.items() if kk in ("old_str", "new_str", "file_text", "insert_line", "view_range", "command", "path")},
                      "action": action, "thought": (args.get("thought") or "")[-1500:], "obs": res or "", "rc": rc,
                      "cwd": cwd_before, "ran": o is not None, "ops": ops,
                      "tok_a": tok(json.dumps(args)), "tok_o": tok(res)})
    status = "exact" if k == item["step_count"] else "none"
    return steps, {"n_conv": 1, "final_diff": None, "bench_status": status}


# ---------------------------------------------------------------- Terminus2

T2_PROMPT = re.compile(r"root@[\w.-]+:(/[^#\n]*|~)#")


def t2_commands(resp):
    """(commands, thought, strict): strict is True when the response parses
    as JSON as a whole; the benchmark counts only those commands."""
    obj, strict = None, False
    for n, text in enumerate((resp, (re.search(r"\{.*\}", resp or "", re.S) or [None])[0] if resp else None)):
        if not text:
            continue
        try:
            obj = json.loads(text)
            strict = n == 0
            break
        except Exception:
            continue
    if not isinstance(obj, dict):
        return None, "", False
    cmds = [c for c in obj.get("commands") or [] if isinstance(c, dict)]
    return cmds, (obj.get("analysis") or "") + "\n" + (obj.get("plan") or ""), strict


def extract_t2(raw, item):
    eps = raw if isinstance(raw, list) else []
    steps, known = [], {}
    cwd = "/app"
    bench = 0
    for ei, ep in enumerate(eps):
        cmds, thought, strict = t2_commands(ep.get("response") or "")
        nxt = eps[ei + 1].get("prompt", "") if ei + 1 < len(eps) else ""
        obs = nxt.split("New Terminal Output:", 1)[1] if "New Terminal Output:" in (nxt or "") else (nxt or "")
        if cmds is None:
            steps.append({"i": len(steps), "unit": ei, "conv": 0, "bench": None, "tool": "unparsed",
                          "action": (ep.get("response") or "")[:1500], "thought": "", "obs": obs, "rc": None,
                          "cwd": cwd, "ran": False, "ops": [], "tok_a": tok(ep.get("response")), "tok_o": tok(obs),
                          "episode": ei})
            continue
        # Join the keystrokes into one shell text, remembering which command
        # each character came from; heredocs may span entries.
        text, owner = "", []
        for j, c in enumerate(cmds):
            ks = str(c.get("keystrokes", ""))
            if re.fullmatch(r"(C-[a-z]|\^[A-Z]|Escape|Enter|q|:q!?|:wq)\n?", ks.strip() + ("\n" if ks.endswith("\n") else "")):
                ks = "\n"  # control keys, pager exits: no shell text
            text += ks
            owner += [j] * len(ks)
        ops, cwd_after, info = sp.extract(text, cwd, known, persist_cd=True)
        # attribute each op to the command whose keystrokes contain its unit
        units = sp.simple_commands(text)
        # map unit index -> char offset (approximate: search the unit's first word)
        pos, unit_cmd = 0, {}
        masked = sp.mask_heredocs(text)
        for ui, u in enumerate(units):
            w = u["argv"][0] if u["argv"] else ""
            if u.get("paren") or not w:
                unit_cmd[ui] = unit_cmd.get(ui - 1, 0)
                continue
            f = masked.find(w, pos)
            if f >= 0:
                unit_cmd[ui] = min(owner[f], len(cmds) - 1) if f < len(owner) else len(cmds) - 1
                pos = f + len(w)
            else:
                unit_cmd[ui] = unit_cmd.get(ui - 1, 0)
        shell_outcome(ops, obs, None, text)
        for o in ops:
            o["cmd_j"] = unit_cmd.get(o["cmd_i"], 0)
        mprompt = list(T2_PROMPT.finditer(obs or ""))
        cwd_before_ep = cwd
        cwd = cwd_after
        if mprompt and mprompt[-1].group(1) not in ("~",):
            cwd = mprompt[-1].group(1).strip()
        for j, c in enumerate(cmds):
            if strict:
                bench += 1
            steps.append({"i": len(steps), "unit": ei, "conv": 0, "bench": bench if strict else None, "tool": "shell",
                          "action": str(c.get("keystrokes", "")), "thought": thought[-1500:] if j == 0 else "",
                          "obs": obs if j == len(cmds) - 1 else "", "rc": None, "cwd": cwd_before_ep, "ran": True,
                          "ops": [o for o in ops if o["cmd_j"] == j],
                          "tok_a": tok(ep.get("response")) if j == 0 else 0,
                          "tok_o": tok(obs) if j == len(cmds) - 1 else 0, "episode": ei})
        if not cmds:
            steps.append({"i": len(steps), "unit": ei, "conv": 0, "bench": None, "tool": "none",
                          "action": "", "thought": thought[-1500:], "obs": obs, "rc": None, "cwd": cwd,
                          "ran": False, "ops": [], "tok_a": tok(ep.get("response")), "tok_o": tok(obs), "episode": ei})
    status = "exact" if bench == item["step_count"] else "none"
    return steps, {"n_conv": 1, "final_diff": None, "bench_status": status}


# ---------------------------------------------------------------- canonical paths

ROOT_RE = re.compile(r"^(/testbed|/app|/workspace/[^/]+|/home/[^/]+|/repo)(/|$)")


def canonicalize(steps):
    """Strip the trajectory's repository root; keep other paths absolute."""
    cnt = Counter()
    for s in steps:
        for o in s["ops"]:
            m = ROOT_RE.match(o["path"])
            if m:
                cnt[m.group(1)] += 1
    root = cnt.most_common(1)[0][0] if cnt else None
    for s in steps:
        for o in s["ops"]:
            o["abs"] = o["path"]
            if root and (o["path"] == root or o["path"].startswith(root + "/")):
                o["path"] = o["path"][len(root) + 1:] or "."
            if o.get("src") and root and o["src"].startswith(root + "/"):
                o["src"] = o["src"][len(root) + 1:]
    return root


def extract_one(item):
    raw = load_raw(item["traj_id"])
    if raw is None:
        return {"traj_id": item["traj_id"], "missing": True}
    a = item["agent"]
    if a == "mini-SWE-agent":
        steps, meta = extract_mini(raw, item)
    elif a == "SWE-agent":
        steps, meta = extract_swe(raw, item)
    elif a == "Terminus2":
        steps, meta = extract_t2(raw, item)
    elif a == "OpenHands":
        if isinstance(raw, list) and any(isinstance(x, dict) and "response" in x for x in raw):
            steps, meta = extract_oh_log(raw, item)
            meta["layout"] = "completion_log"
        else:
            steps, meta = extract_oh_events(raw, item)
            meta["layout"] = "event_stream"
    else:
        return {"traj_id": item["traj_id"], "missing": True}
    root = canonicalize(steps)
    return {"traj_id": item["traj_id"], "agent": a, "model": item["model"], "subset": subset_of(item),
            "step_count": item["step_count"], "root": root, "n_steps": len(steps), **meta, "steps": steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    rows = load_manifest()
    if args.only:
        rows = [r for r in rows if r["traj_id"] == args.only]
    out_path = os.path.join(WORK, "extracted.jsonl" if not args.only else "extracted_one.jsonl")
    stats = Counter()
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            rec = extract_one(r)
            if rec.get("missing"):
                stats["missing"] += 1
            else:
                stats[f"{r['agent']}:{rec.get('bench_status')}"] += 1
            f.write(json.dumps(rec, default=str) + "\n")
    print(dict(stats))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
