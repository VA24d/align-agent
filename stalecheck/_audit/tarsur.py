"""Recount on tarsur385/swebench-verified-trajectories (mini-swe-agent tool-call format). Usage: python3 scripts/audit/tarsur.py

Adapter: each assistant message's tool_calls (function 'bash', arguments.command)
is one step; its reply is the role:'tool' message with the same
tool_call_id ('Exit code: N' header gives the return code). Commands are
parsed with scripts/audit/shellparse.py, outcomes with extract.shell_outcome,
scored with score.Replay and classified with materialize.classify (automated
classifier only; no human labels exist for this corpus). Reports event
coverage first (steps with a tool call, ops per step, share of steps with no
op, unparsed commands), then per model: opportunities, hazards, structural
rate (Clopper-Pearson), automated classes, and hazards vs info.resolved.
Reads docs/audit/rq1/work/tarsur_raw/ (fetch_tarsur.py); writes work/tarsur.json.
"""

import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import WORK, dump  # noqa: E402
import shellparse as sp  # noqa: E402
from extract import shell_outcome, canonicalize, tok  # noqa: E402
from score import Replay, cp_ci  # noqa: E402
from materialize import classify  # noqa: E402

RC = re.compile(r"^Exit code: (-?\d+)")


def adapt(d, cwd="/testbed"):
    msgs = d.get("messages", [])
    replies = {m.get("tool_call_id"): m for m in msgs if m.get("role") == "tool"}
    steps, known, cov = [], {}, Counter()
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        tcs = m.get("tool_calls") or []
        if not tcs:
            cov["assistant_without_tool_call"] += 1
            continue
        for tc in tcs:
            cov["tool_calls"] += 1
            fn = tc.get("function")
            fn = fn.get("name") if isinstance(fn, dict) else fn
            args = tc.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            cmd = args.get("command", "") if fn == "bash" else ""
            cov[f"fn:{fn}"] += 1
            r = replies.get(tc.get("id"))
            obs = str((r or {}).get("content") or "")
            mm = RC.match(obs)
            rc = int(mm.group(1)) if mm else None
            cov["reply_found"] += r is not None
            ops, _, info = sp.extract(cmd, cwd, known, persist_cd=False) if cmd else ([], cwd, {})
            cov["unparsed"] += bool(info.get("unparsed"))
            if ops:
                shell_outcome(ops, obs, rc, cmd)
            steps.append({"i": len(steps), "unit": len(steps), "conv": 0, "bench": None, "tool": "shell",
                          "action": cmd, "thought": str(m.get("content") or "")[-1500:], "obs": obs, "rc": rc,
                          "cwd": cwd, "ran": True, "ops": ops, "tok_a": tok(cmd), "tok_o": tok(obs)})
    return steps, cov


def main():
    base = os.path.join(WORK, "tarsur_raw")
    per_model = defaultdict(Counter)
    classes = defaultdict(Counter)
    resolved = defaultdict(lambda: defaultdict(list))
    cov_all = defaultdict(Counter)
    for f in sorted(glob.glob(os.path.join(base, "*", "*.traj.json"))):
        model = f.split(os.sep)[-2]
        d = json.load(open(f))
        steps, cov = adapt(d)
        cov_all[model].update(cov)
        t = {"traj_id": os.path.basename(f), "agent": "mini-SWE-agent-2.1", "model": model, "steps": steps,
             "final_diff": (d.get("info") or {}).get("submission") or None}
        t["root"] = canonicalize(steps)
        r = Replay(t).run()
        c = per_model[model]
        c["trajectories"] += 1
        c["steps"] += len(steps)
        c["zero_op_steps"] += sum(1 for s in steps if not s["ops"])
        c["ops"] += sum(len(s["ops"]) for s in steps)
        c["anchored_writes"] += sum(1 for s in steps for o in s["ops"] if o["op"] == "W" and o["kind"] == "anchored")
        c["opportunities"] += len(r.opps)
        c["hazards"] += len(r.hazards)
        c["hazard_trajectories"] += bool(r.hazards)
        for h in r.hazards:
            cls, why, ev = classify(h, t)
            classes[model][cls] += 1
        res = bool((d.get("info") or {}).get("resolved"))
        c["resolved"] += res
        resolved[model]["haz" if r.hazards else "nohaz"].append(res)
    out = {}
    tot = Counter()
    for model, c in sorted(per_model.items()):
        tot.update(c)
        rv = resolved[model]
        out[model] = dict(c, rate=round(c["hazards"] / c["opportunities"], 4) if c["opportunities"] else None,
                          ci=cp_ci(c["hazards"], c["opportunities"]), classes=dict(classes[model]),
                          resolved_with_hazard=[sum(rv["haz"]), len(rv["haz"])],
                          resolved_without_hazard=[sum(rv["nohaz"]), len(rv["nohaz"])],
                          coverage=dict(cov_all[model]))
        print(f"{model:24s} traj {c['trajectories']:4d} steps {c['steps']:6d} zero-op {c['zero_op_steps'] / max(1, c['steps']):.2f} "
              f"opp {c['opportunities']:5d} haz {c['hazards']:4d} rate {out[model]['rate']} {out[model]['ci']} "
              f"cls {dict(classes[model])} res(haz) {out[model]['resolved_with_hazard']} res(no) {out[model]['resolved_without_hazard']}")
    pool(out)
    print("ALL", out["ALL"])
    dump(out, "tarsur.json")


# Configurations the dataset publishes twice: every trajectory of the first is
# an exact copy of the second's. Kept in the per-model rows, left out of ALL.
DUPLICATES = {"gpt-5.2-high": "gpt-5.2-codex"}

POOLED = ("trajectories", "steps", "zero_op_steps", "ops", "anchored_writes", "opportunities", "hazards",
          "hazard_trajectories", "resolved")


def pool(out):
    """Pool the per-model rows into ALL, skipping published duplicates."""
    tot, allcls = Counter(), Counter()
    for model, o in out.items():
        if model == "ALL":
            continue
        if model in DUPLICATES:
            o["duplicate_of"] = DUPLICATES[model]
            continue
        tot.update({k: o.get(k, 0) for k in POOLED})
        allcls.update(o.get("classes", {}))
    out["ALL"] = dict(tot, rate=round(tot["hazards"] / tot["opportunities"], 4) if tot["opportunities"] else None,
                      ci=cp_ci(tot["hazards"], tot["opportunities"]), classes=dict(allcls),
                      models=sum(1 for k in out if k != "ALL" and k not in DUPLICATES),
                      excluded_duplicates=sorted(DUPLICATES))


if __name__ == "__main__":
    if sys.argv[1:] == ["--repool"]:  # re-pool the saved per-model rows without the raw data
        from common import WORK as _W
        _out = json.load(open(os.path.join(_W, "tarsur.json")))
        _out.pop("ALL", None)
        pool(_out)
        print("ALL", _out["ALL"])
        dump(_out, "tarsur.json")
    else:
        main()
