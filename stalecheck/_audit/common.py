"""Shared read-only helpers for the RQ1 audit. Usage: imported by the other scripts/audit/*.py files.

Reads the manifest from the worktree and raw trajectories straight from the
shared cache. Never downloads, never writes into the cache.
"""

import json
import os
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MANIFEST = os.path.join(ROOT, "data", "manifests", "bench_manifest.verified.jsonl")
# The extracted CodeTraceBench trajectories (one JSON per traj_id) are gitignored;
# point RQ1_CACHE at them, or place them under data/cache/extracted.
CACHE = os.environ.get("RQ1_CACHE", os.path.join(ROOT, "data", "cache", "extracted"))
OUT = os.path.join(ROOT, "docs", "audit", "rq1")
WORK = os.path.join(OUT, "work")  # intermediate machine-readable outputs


def load_manifest(path: str = MANIFEST) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cache_path(traj_id: str) -> str:
    return os.path.join(CACHE, f"{traj_id}.json")


def load_raw(traj_id: str) -> Optional[Any]:
    """Raw trajectory from the cache, or None when the cache has no file."""
    p = cache_path(traj_id)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def annotated_ids(item: Dict[str, Any]):
    inc, unu = set(), set()
    for st in item.get("incorrect_stages") or []:
        inc.update(st.get("incorrect_step_ids") or [])
        unu.update(st.get("unuseful_step_ids") or [])
    return inc, unu


def short_model(m: str) -> str:
    return {"Anthropic/Claude-Sonnet-4-20250514-Thinking": "Claude-Sonnet-4",
            "DeepSeek/DeepSeek-V3.2": "DeepSeek-V3.2",
            "OpenAI/GPT-5": "GPT-5"}.get(m, m)


def dump(obj: Any, name: str) -> str:
    os.makedirs(WORK, exist_ok=True)
    p = os.path.join(WORK, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
    return p
