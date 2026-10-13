"""Observation layer freshness tracking hook for agent scaffolds.

Monitors file observations and detects on-disk mutations before prompt re-serialization.
"""

import os
import hashlib
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field


@dataclass
class ObservationBinding:
    """Binds an observation message index in the conversation history to a file path and hash."""
    message_index: int
    step_id: int
    file_path: str
    content_hash: str
    raw_content: str
    # The file as it stood on disk when it was read. `raw_content` is the whole
    # observation message, which also carries the command's exit code and any
    # surrounding harness text, so it cannot be diffed against the file itself:
    # hunk-level invalidation compared the two and found every line changed,
    # collapsing that arm into whole-observation eviction.
    #
    # None means "never read" (e.g. a binding constructed without going
    # through FreshnessHook.register_read); "" means "read, and the file is
    # genuinely empty." Collapsing these to the same falsy value previously
    # made a real empty file fall back to raw_content, reintroducing the
    # whole-observation-eviction bug for that one case.
    file_content: Optional[str] = None
    is_invalid: bool = False
    invalidation_step: Optional[int] = None


class FreshnessHook:
    """Manages file dependency tracking and invalidation detection for agent message history."""

    def __init__(self, workspace_root: str = ".", use_mtime_cache: bool = True):
        self.workspace_root = os.path.abspath(workspace_root)
        self.use_mtime_cache = use_mtime_cache
        self.bindings: List[ObservationBinding] = []
        self.file_hashes: Dict[str, str] = {}
        # Stores abs_path -> (mtime, size, sha256_hash)
        self.stat_cache: Dict[str, Tuple[float, int, str]] = {}

    def _hash_bytes(self, abs_path: str) -> str:
        """Chunked SHA-256 read of the file at abs_path, truncated to 16 hex chars."""
        hasher = hashlib.sha256()
        with open(abs_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]

    def compute_file_hash(self, file_path: str) -> Optional[str]:
        """Compute real SHA256 hash of file on disk with mtime gating, or simulated hash."""
        abs_path = os.path.join(self.workspace_root, file_path) if not os.path.isabs(file_path) else file_path
        if os.path.exists(abs_path) and os.path.isfile(abs_path):
            try:
                if self.use_mtime_cache:
                    st = os.stat(abs_path)
                    cached = self.stat_cache.get(abs_path)
                    if cached and cached[0] == st.st_mtime and cached[1] == st.st_size:
                        return cached[2]

                    h = self._hash_bytes(abs_path)
                    self.stat_cache[abs_path] = (st.st_mtime, st.st_size, h)
                    return h
                else:
                    return self._hash_bytes(abs_path)
            except Exception:
                pass
        return self.file_hashes.get(file_path)

    def _read_file(self, file_path: str) -> Optional[str]:
        """The file's current text, or None if it cannot be read.

        A genuinely empty file returns "", distinct from None -- callers that
        need to tell "read, and empty" apart from "never read" depend on that.
        """
        abs_path = os.path.join(self.workspace_root, file_path) if not os.path.isabs(file_path) else file_path
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError:
            return None

    def register_read(self, msg_idx: int, step_id: int, file_path: str, content: str) -> ObservationBinding:
        """Register a new observation buffer for a read file."""
        h = self.compute_file_hash(file_path)
        if not h:
            h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
            self.file_hashes[file_path] = h

        binding = ObservationBinding(
            message_index=msg_idx,
            step_id=step_id,
            file_path=file_path,
            content_hash=h,
            raw_content=content,
            file_content=self._read_file(file_path),
            is_invalid=False
        )
        self.bindings.append(binding)
        return binding

    def register_write(self, step_id: int, file_path: str, new_content: str = "") -> List[ObservationBinding]:
        """Record file modification and invalidate all prior observation bindings for this file."""
        # Evict from stat cache to force SHA-256 recomputation and prevent sub-millisecond mtime cache collisions
        abs_path = os.path.join(self.workspace_root, file_path) if not os.path.isabs(file_path) else file_path
        self.stat_cache.pop(abs_path, None)

        # Update current hash
        new_h = self.compute_file_hash(file_path)
        if not new_h:
            hasher = hashlib.sha256()
            hasher.update(f"{file_path}::step{step_id}::{new_content}".encode("utf-8"))
            new_h = hasher.hexdigest()[:16]
            self.file_hashes[file_path] = new_h

        invalidated = []
        for b in self.bindings:
            if b.file_path == file_path and not b.is_invalid:
                b.is_invalid = True
                b.invalidation_step = step_id
                invalidated.append(b)
        return invalidated

    def get_stale_bindings(self) -> List[ObservationBinding]:
        """Return all observation bindings that have been invalidated by subsequent writes."""
        return [b for b in self.bindings if b.is_invalid]
