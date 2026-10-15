"""Build blind trajectory excerpts for hazard labeling. Usage: python3 scripts/audit/excerpts.py [--show ITEM_ID] [--write]

For every structural hazard in docs/audit/rq1/work/hazards.jsonl (and every
intra-unit repeated edit in intra_unit.jsonl, marked as such), writes a
plain-text excerpt: the last view of the file before the edit, every write to
the file between that view and the edit, the edit itself with the agent's
stated reasoning, the next five steps, and the final diff when the trajectory
has one. Outputs are trimmed around the anchor text.

Blindness: excerpts never include the manifest's annotations, `solved`,
OpenHands metadata (is_resolved, parser_results, test results), or any
automated classification. Items are identified by an opaque id (H###/I###)
whose mapping to trajectories is kept in work/excerpt_index.json.
With --write, excerpts go to docs/audit/rq1/census/items/<id>.txt.
"""

import argparse
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import OUT, WORK  # noqa: E402

CENSUS = os.path.join(OUT, "census")
MAXOBS = 1800


def load_trajs(ids):
    out = {}
    with open(os.path.join(WORK, "extracted.jsonl"), encoding="utf-8") as f:
        for line in f:
            tid = line[len('{"traj_id": "'):line.index('"', len('{"traj_id": "'))]
            if tid in ids:
                out[tid] = json.loads(line)
    return out


def anchor_hint(op):
    a = op.get("anchor") or {}
    t = a.get("type")
    if t == "str_replace":
        return a.get("old") or ""
    if t == "py":
        vals = [r.get("old") for r in a.get("replaces") or [] if r.get("old")] + [r.get("pat") for r in a.get("regex") or []]
        vals += a.get("finds") or []
        return max(vals, key=len) if vals else ""
    if t in ("sed", "perl"):
        pats = [c.get("pat") or "" for c in a.get("cmds") or []] + [x.get("pat") or "" for x in a.get("subs") or []]
        pats += [ad.get("re", "") for c in a.get("cmds") or [] for ad in (c.get("addr") or []) if isinstance(ad, dict) and ad.get("re")]
        return max(pats, key=len) if pats else ""
    if t == "patch":
        lines = [ln[1:] for h in a.get("hunks") or [] for ln in h.get("lines", []) if ln[:1] in ("-", " ")]
        return "\n".join(lines[:6])
    return ""


def trim(text, keys, limit=MAXOBS):
    """Keep the part of text around the first occurrence of any key."""
    text = text or ""
    if len(text) <= limit:
        return text
    for k in keys:
        for frag in [k] + [x for x in re.split(r"\\[.*+?^$()\[\]{}|]|\n", k or "") if len(x) >= 6]:
            if frag and frag in text:
                i = text.index(frag)
                lo = max(0, i - limit // 2)
                return ("[...]" if lo else "") + text[lo:lo + limit] + ("[...]" if lo + limit < len(text) else "")
    return text[:limit // 2] + "\n[... output trimmed ...]\n" + text[-limit // 2:]


def episode_obs(t, s):
    obs = s.get("obs") or ""
    if t["agent"] == "Terminus2" and not obs:
        same = [x for x in t["steps"] if x.get("episode") == s.get("episode") and x.get("obs")]
        if same:
            obs = same[-1]["obs"]
    return obs


def step_block(t, s, keys, label, show_thought=False, cmd_lim=900, obs_lim=900, printed_eps=None):
    head = f"--- {label}: step {s['i']}" + (f" (episode {s['episode']})" if s.get("episode") is not None else "")
    out = [head]
    if show_thought and (s.get("thought") or "").strip():
        out.append("AGENT REASONING (excerpt):\n" + trim(s["thought"], keys, 600))
    out.append("COMMAND:\n" + trim(s.get("action") or "", keys, cmd_lim))
    obs = episode_obs(t, s)
    if t["agent"] == "Terminus2" and printed_eps is not None:
        if s.get("episode") in printed_eps:
            out.append("OUTPUT: (terminal output for this episode shown above)")
            return "\n".join(out)
        printed_eps.add(s.get("episode"))
        obs = "(terminal screen after the whole episode)\n" + obs if obs else obs
    out.append("OUTPUT:\n" + (trim(obs, keys, obs_lim) if obs.strip() else "(no output)"))
    return "\n".join(out)


def final_diff_for(t, path):
    d = t.get("final_diff") or ""
    if not d:
        return None
    base = path.rsplit("/", 1)[-1]
    parts = re.split(r"(?m)^(?=diff --git )", d)
    keep = [p for p in parts if base in p.split("\n", 1)[0]]
    return "\n".join(keep)[:3000] if keep else "(final diff does not touch this file)"


def build(item, t):
    s_e = t["steps"][item["step"]]
    e_op = s_e["ops"][item["op"]]
    keys = [anchor_hint(e_op)]
    eps = set()
    parts = [f"ITEM {item['id']}", f"Framework: {t['agent']}", f"File: {item['path']}",
             f"Edit mechanism: {e_op.get('mech')}"]
    if item.get("kind_of_item") == "intra_unit":
        parts.append("Note: the edit under review was issued in the same action as an earlier write to this file.")
    lv = item.get("last_view")
    shown = set()
    if lv:
        s_v = t["steps"][lv[0]]
        parts.append(step_block(t, s_v, keys, "LAST VIEW OF THE FILE BEFORE THE EDIT", cmd_lim=500, obs_lim=1500, printed_eps=eps))
        shown.add(s_v["i"])
    else:
        parts.append("--- LAST VIEW: none on record before the edit")
    ws = item.get("W") or item.get("prior_same_unit_writes") or []
    for w in ws[-4:]:
        s_w = t["steps"][w[0]]
        if s_w["i"] in shown or s_w["i"] == s_e["i"]:
            continue
        shown.add(s_w["i"])
        parts.append(step_block(t, s_w, keys, "WRITE TO THE FILE AFTER THAT VIEW", cmd_lim=900, obs_lim=400, printed_eps=eps))
    if len(ws) > 4:
        parts.append(f"({len(ws) - 4} earlier writes to the file after the view are not shown)")
    parts.append(step_block(t, s_e, keys, "THE EDIT UNDER REVIEW", show_thought=True, cmd_lim=1400, obs_lim=600, printed_eps=eps))
    later = [x for x in t["steps"] if x["i"] > s_e["i"]][:5]
    for x in later:
        parts.append(step_block(t, x, keys + [anchor_hint(o) for o in x["ops"]], "FOLLOWING STEP", cmd_lim=300, obs_lim=450, printed_eps=eps))
    fd = final_diff_for(t, item["path"])
    if fd:
        parts.append("--- FINAL SUBMITTED DIFF FOR THIS FILE:\n" + fd[:1500])
    return "\n\n".join(parts)


def items():
    out = []
    for fname, prefix in (("hazards.jsonl", "H"), ("intra_unit.jsonl", "I")):
        with open(os.path.join(WORK, fname), encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                it = json.loads(line)
                it["kind_of_item"] = "hazard" if prefix == "H" else "intra_unit"
                raw_id = f"{it['traj_id']}|{it['step']}|{it['op']}|{it['path']}"
                it["id"] = prefix + hashlib.sha1(raw_id.encode()).hexdigest()[:6]
                out.append(it)
    return out


def v1_keys():
    """(traj_id, step, path) of the items in the census's frozen v1 set."""
    p = os.path.join(WORK, "excerpt_index_v1.json")
    if not os.path.exists(p):
        return set()
    return {(v["traj_id"], v["step"], v["path"]) for v in json.load(open(p)).values()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", default="")
    ap.add_argument("--write", action="store_true",
                    help="write excerpts of items NOT in the frozen v1 set to census/items_v2/ "
                         "(census/items/ is frozen: the census labeler read it)")
    args = ap.parse_args()
    its = items()
    trajs = load_trajs({it["traj_id"] for it in its})
    index = {}
    frozen = v1_keys()
    outdir = os.path.join(CENSUS, "items_v2")
    if args.write:
        os.makedirs(outdir, exist_ok=True)
    n_new = 0
    for it in its:
        index[it["id"]] = {k: it[k] for k in ("traj_id", "step", "op", "path", "kind_of_item", "agent", "model")}
        index[it["id"]]["in_v1"] = (it["traj_id"], it["step"], it["path"]) in frozen
        if args.show and it["id"] != args.show:
            continue
        if args.show:
            print(build(it, trajs[it["traj_id"]]))
        if args.write and not index[it["id"]]["in_v1"] and it["kind_of_item"] == "hazard":
            n_new += 1
            with open(os.path.join(outdir, it["id"] + ".txt"), "w", encoding="utf-8") as f:
                f.write(build(it, trajs[it["traj_id"]]))
    with open(os.path.join(WORK, "excerpt_index.json"), "w") as f:
        json.dump(index, f, indent=1)
    if args.write:
        print("new hazard excerpts written:", n_new)
    if not args.show:
        print(len(its), "items;", sum(1 for i in its if i["kind_of_item"] == "hazard"), "hazards")


if __name__ == "__main__":
    main()
