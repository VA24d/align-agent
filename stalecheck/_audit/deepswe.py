"""Recount on the DeepSWE leaderboard trajectories (mini-swe-agent, converted to ATIF). Usage: python3 scripts/audit/deepswe.py [--limit N]

DeepSWE (deepswe.datacurve.ai) is a long-horizon benchmark: 113 original tasks
in 91 repositories, reference solutions of 668 added lines over 7 files on
average, graded by hand-written verifiers. Its leaderboard dataset
(datacurve/deep-swe-leaderboard, CC-BY-4.0) holds every official trial run
with mini-swe-agent, stored as ATIF v1.7 (Harbor's trajectory format).

Adapter, as scripts/audit/tarsur.py: each agent step's bash tool call is one
step; its observation is the JSON mini-swe-agent returns ({returncode,
output} or, when elided, {returncode, output_head, output_tail}). Commands are
parsed with scripts/audit/shellparse.py, outcomes with extract.shell_outcome,
scored with score.Replay and classified with materialize.classify (automated
classifier only). Reward comes from the trial's verifier/reward.txt.

Reads data/deepswe/leaderboard/raw/jobs/*/*/; writes work/deepswe.json and
docs/audit/rq1/deepswe.md.
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import OUT, ROOT, WORK, dump  # noqa: E402
import shellparse as sp  # noqa: E402
from extract import shell_outcome, canonicalize, tok  # noqa: E402
from score import Replay, cp_ci  # noqa: E402
from materialize import classify, describe  # noqa: E402

BASE = os.path.join(ROOT, "data", "deepswe", "leaderboard", "raw", "jobs")
TASKS = os.path.join(ROOT, "data", "deepswe", "tasks", "tasks")


def observation(step, k):
    """(text, return code, elided) of the k-th tool result of an ATIF step."""
    results = (step.get("observation") or {}).get("results") or []
    if k >= len(results):
        return "", None, False
    raw = results[k].get("content")
    raw = raw if isinstance(raw, str) else json.dumps(raw)
    try:
        d = json.loads(raw)
    except ValueError:
        return raw, None, False
    if not isinstance(d, dict):
        return raw, None, False
    rc = d.get("returncode")
    if "output" in d:
        return str(d.get("output") or ""), rc, False
    head, tail = str(d.get("output_head") or ""), str(d.get("output_tail") or "")
    return head + "\n[... output elided ...]\n" + tail, rc, True


def adapt(d, cwd="/testbed"):
    steps, known, cov = [], {}, Counter()
    for s in d.get("steps") or []:
        if s.get("source") != "agent":
            continue
        tcs = s.get("tool_calls") or []
        if not tcs:
            cov["agent_step_without_tool_call"] += 1
            continue
        for k, tc in enumerate(tcs):
            cov["tool_calls"] += 1
            fn = tc.get("function_name")
            args = tc.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            cmd = args.get("command", "") if fn == "bash" else ""
            cov[f"fn:{fn}"] += 1
            obs, rc, elided = observation(s, k)
            cov["elided"] += elided
            ops, _, info = sp.extract(cmd, cwd, known, persist_cd=False) if cmd else ([], cwd, {})
            cov["unparsed"] += bool(info.get("unparsed"))
            if ops:
                shell_outcome(ops, obs, rc, cmd, truncated=elided)
            steps.append({"i": len(steps), "unit": len(steps), "conv": 0, "bench": None, "tool": "shell",
                          "action": cmd, "thought": str(s.get("message") or "")[-1500:], "obs": obs, "rc": rc,
                          "cwd": cwd, "ran": True, "ops": ops, "tok_a": tok(cmd), "tok_o": tok(obs)})
    return steps, cov


def _result(trial_dir):
    try:
        return json.load(open(os.path.join(trial_dir, "result.json")))
    except (OSError, ValueError):
        return {}


def trial_model(d, trial_dir):
    """Model and reasoning effort, as the leaderboard labels a configuration."""
    model = ((d.get("agent") or {}).get("model_name") or "unknown").split("/")[-1]
    cfg = (_result(trial_dir).get("config") or {}).get("agent") or {}
    kw = cfg.get("kwargs") or {}
    effort = kw.get("reasoning_effort") or kw.get("effort") or (kw.get("model_kwargs") or {}).get("reasoning_effort")
    return f"{model} [{effort}]" if effort else model


def reward(trial_dir):
    """The verifier's binary reward (result.json, else verifier/reward.txt); None if the trial has none."""
    rw = ((_result(trial_dir).get("verifier_result") or {}).get("rewards") or {}).get("reward")
    if rw is not None:
        return float(rw) >= 1.0
    try:
        return open(os.path.join(trial_dir, "verifier", "reward.txt")).read().strip() == "1"
    except OSError:
        return None


DEV_CSV = os.path.join(OUT, "deepswe_m3_handcheck.csv")
HELDOUT_CSV = os.path.join(OUT, "deepswe_m3_heldout.csv")


def handcheck_sections():
    """The hand checks of the automated label, rendered from their CSVs."""
    import csv
    L = []
    if os.path.exists(DEV_CSV):
        rows = list(csv.DictReader(open(DEV_CSV)))
        c = Counter(r["label"] for r in rows)
        L += ["", f"## Development check of the automated \"materialized\" label ({len(rows)} cases, classifier v1)", "",
              "`docs/audit/rq1/deepswe_m3_handcheck.csv`: a seeded sample of the first classifier's 277 labels, one reader. "
              "Its spurious cases (different text sharing a line, a re-insert after a restore, prepends at line 1, inserts "
              "where a block was deleted, runs of sequential inserts) became the fixtures in `tests/test_materialize.py`, "
              "and the classifier was revised against them. Its precision is therefore not an estimate for the revised "
              "classifier; the held-out check below is.", "",
              "| Label | Cases |", "|---|---|"] + [f"| {k} | {c[k]} |" for k in ("real", "plausible", "ambiguous", "spurious")]
    hj = os.path.join(WORK, "deepswe_heldout.json")
    if os.path.exists(hj):
        h = json.load(open(hj))
        a, o, e = h["auto_m3"], h["auto_other"], h.get("estimate", {})
        pc = lambda x: f"{100 * x:.0f}%"  # noqa: E731
        L += ["", "## Held-out check (classifier v2.1, frozen before any held-out label was read)", "",
              "`docs/audit/rq1/deepswe_m3_heldout.csv`, drawn by `scripts/audit/deepswe_heldout.py` (seed 20261001) from trials the "
              "development check did not touch: 60 edits the classifier labelled materialized, round-robin by model, and 40 line-anchored "
              "or inserting stale edits it did not. Each was labelled blind to stratum and class from a census-format excerpt with the "
              f"audit's rubric by the census labeller (a language model). "
              + (f"A second, independent run of blind model labellers on the same packets agreed on {h['agree_census_second']} of "
                 f"{h['pairs_census_second']} (coarse kappa {h['kappa_census_second_coarse']}); the {h.get('disagreements_undecided', 0)} items they "
                 f"disagree on count as undecided. " if h.get("pairs_census_second") else "")
              + f"Human labels: {'none yet' if h.get('kappa_census_human_coarse') is None else 'kappa ' + str(h['kappa_census_human_coarse'])}. "
              f"{h.get('drawn_m3_no_longer_m3', 0)} edits drawn as materialized are not materialized under the frozen classifier and are left out.", "",
              "| Stratum | Labelled | Materialized | Undecided | Not | Rate (95% CI) |", "|---|---|---|---|---|---|",
              f"| flagged materialized | {a['n']} | {a['counts'].get('MATERIALIZED', 0)} | {a['counts'].get('INDETERMINATE', 0)} | {a['counts'].get('NOT', 0)} | "
              f"{pc(a['rate_low'])} ({pc(a['ci_low'][0])}–{pc(a['ci_low'][1])}) |",
              f"| not flagged (line-anchored or insert) | {o['n']} | {o['counts'].get('MATERIALIZED', 0)} | {o['counts'].get('INDETERMINATE', 0)} | {o['counts'].get('NOT', 0)} | "
              f"{pc(o['rate_low'])} ({pc(o['ci_low'][0])}–{pc(o['ci_low'][1])}) |", ""]
        if e:
            L += [f"Edits that went wrong in the corpus: {e['flagged']} flagged x precision + {e['pool']} unflagged line-anchored or inserting "
                  f"stale edits x miss rate = {e['went_wrong']:.0f} of {e['opportunities']:,} anchored edits "
                  f"({100 * e['went_wrong'] / e['opportunities']:.2f}%; {100 * e['went_wrong_low'] / e['opportunities']:.2f}% to "
                  f"{100 * e['went_wrong_high'] / e['opportunities']:.2f}% from the intervals)."]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(BASE, "*", "*", "agent", "trajectory.json")))
    if args.limit:
        files = files[:args.limit]
    per_model, classes = defaultdict(Counter), defaultdict(Counter)
    resolved = defaultdict(lambda: defaultdict(list))
    cov_all = defaultdict(Counter)
    hazards_out = []
    for n, f in enumerate(files, 1):
        trial_dir = os.path.dirname(os.path.dirname(f))
        try:
            d = json.load(open(f))
        except ValueError:
            continue
        model = trial_model(d, trial_dir)
        steps, cov = adapt(d)
        cov_all[model].update(cov)
        t = {"traj_id": os.path.basename(trial_dir), "agent": "mini-SWE-agent (DeepSWE)", "model": model, "steps": steps,
             "final_diff": None}
        t["root"] = canonicalize(steps)
        r = Replay(t).run()
        c = per_model[model]
        c["trajectories"] += 1
        c["steps"] += len(steps)
        c["zero_op_steps"] += sum(1 for s in steps if not s["ops"])
        c["anchored_writes"] += sum(1 for s in steps for o in s["ops"] if o["op"] == "W" and o["kind"] == "anchored")
        c["opportunities"] += len(r.opps)
        c["hazards"] += len(r.hazards)
        c["hazard_trajectories"] += bool(r.hazards)
        for h in r.hazards:
            cls, why, ev = classify(h, t)
            classes[model][cls] += 1
            e = steps[h["step"]]["ops"][h["op"]]
            de = describe(e, steps[h["step"]].get("action"))
            inserts = any(c.get("cmd") in ("a", "i") for c in ((e.get("anchor") or {}).get("cmds") or [])) or \
                (e.get("anchor") or {}).get("type") == "insert" or e.get("kind") == "append"
            hazards_out.append({**{k: h.get(k) for k in ("step", "op", "path", "mech", "last_view")}, "traj_id": t["traj_id"],
                                "job": os.path.basename(os.path.dirname(trial_dir)), "model": model, "class": cls, "why": why,
                                "line_anchored": bool(de["line_anchors"]), "inserts": inserts, "task": t["traj_id"].split("__")[0]})
        res = reward(trial_dir)
        c["scored"] += res is not None
        c["resolved"] += bool(res)
        if res is not None:
            resolved[model]["haz" if r.hazards else "nohaz"].append(res)
        if n % 500 == 0:
            print(f"{n}/{len(files)}", flush=True)
    out, tot, allcls = {}, Counter(), Counter()
    for model, c in sorted(per_model.items()):
        tot.update(c)
        allcls.update(classes[model])
        rv = resolved[model]
        out[model] = dict(c, rate=round(c["hazards"] / c["opportunities"], 4) if c["opportunities"] else None,
                          ci=cp_ci(c["hazards"], c["opportunities"]), classes=dict(classes[model]),
                          resolved_with_hazard=[sum(rv["haz"]), len(rv["haz"])],
                          resolved_without_hazard=[sum(rv["nohaz"]), len(rv["nohaz"])], coverage=dict(cov_all[model]))
    out["ALL"] = dict(tot, rate=round(tot["hazards"] / tot["opportunities"], 4) if tot["opportunities"] else None,
                      ci=cp_ci(tot["hazards"], tot["opportunities"]), classes=dict(allcls))
    dump(out, "deepswe.json")
    dump(hazards_out, "deepswe_hazards.json")
    L = ["# File-level stale edits on DeepSWE (published mini-swe-agent trials)", "",
         "Generated by `scripts/audit/deepswe.py`, the RQ1 audit's recount applied to the DeepSWE leaderboard trajectories "
         "(datacurve/deep-swe-leaderboard). Classes are the automated classifier's; no human labels exist for this corpus.", "",
         "| Model | Trials | Mean steps | Opportunities | Stale edits | SAR | 95% CI | Trials with one | Resolved, with / without a stale edit | Automated classes |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for model, o in sorted(out.items(), key=lambda kv: (kv[0] == "ALL", kv[0])):
        if model == "ALL":
            continue
        ci = o["ci"]
        L.append(f"| {model} | {o['trajectories']} | {o['steps'] // max(1, o['trajectories'])} | {o['opportunities']} | {o['hazards']} | "
                 f"{100 * (o['rate'] or 0):.1f}% | {100 * ci[0]:.1f}–{100 * ci[1]:.1f}% | {o['hazard_trajectories']} | "
                 f"{o['resolved_with_hazard'][0]}/{o['resolved_with_hazard'][1]} / {o['resolved_without_hazard'][0]}/{o['resolved_without_hazard'][1]} | "
                 f"{', '.join(f'{k} {v}' for k, v in sorted(o['classes'].items(), key=lambda kv: -kv[1]))} |")
    a = out["ALL"]
    L.append(f"| **All** | {a['trajectories']} | {a['steps'] // max(1, a['trajectories'])} | {a['opportunities']} | {a['hazards']} | "
             f"{100 * (a['rate'] or 0):.1f}% | | {a['hazard_trajectories']} | | {', '.join(f'{k} {v}' for k, v in sorted(a['classes'].items(), key=lambda kv: -kv[1]))} |")
    L += handcheck_sections()
    open(os.path.join(OUT, "deepswe.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def rewrite_handcheck():
    """Replace the hand-check sections of deepswe.md without recounting the corpus."""
    path = os.path.join(OUT, "deepswe.md")
    text = open(path).read()
    cut = text.find("\n## Development check")
    if cut < 0:
        cut = text.find("\n## Held-out check")
    head = text[:cut].rstrip("\n").split("\n") if cut >= 0 else text.rstrip("\n").split("\n")
    open(path, "w").write("\n".join(head + handcheck_sections()) + "\n")


if __name__ == "__main__":
    rewrite_handcheck() if sys.argv[1:] == ["--handcheck-only"] else main()
