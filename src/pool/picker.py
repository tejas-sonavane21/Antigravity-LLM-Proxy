"""
src/pool/picker.py — PoolPicker Singleton
==========================================
Stateful singleton that manages the round-robin pool:
  - Selects the least-used available entry on each request (pick())
  - Tracks and persists cooldown state on errors (release())
  - Provides a read-only preview of the next entry (peek() / advance_peek())
    for the fetchAvailableModels interceptor
  - Writes config.json atomically whenever cooldown state changes

Design notes:
  - No asyncio locks — all methods are called from a single async context
    (FastAPI request handlers). Python's GIL protects the simple field updates.
  - in_flight_count and last_used_at are runtime-only (reset on restart).
  - cooldown_until / cooldown_reason are persisted to config.json via
    write_config_atomic() so the proxy resumes correctly after a crash.

Reference: pool_implementation_plan.md Phase 2 Steps 2.1-2.5
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.pool.entry import PoolEntry
from src.pool.cooldown import resolve_cooldown
from src.config import PoolSettings

_log = logging.getLogger("pool.picker")


# ---------------------------------------------------------------------------
# Required top-level sections for config.json structure guard
# ---------------------------------------------------------------------------

_REQUIRED_SECTIONS = ("proxy", "upstream", "tls", "providers", "patcher", "model_pool")


# ---------------------------------------------------------------------------
# Atomic config writer — 4-layer anti-corruption protocol
# ---------------------------------------------------------------------------

def write_config_atomic(config_path: str, raw_config: dict) -> None:
    """
    Write raw_config to config_path using a 4-layer atomic safety protocol.

    LAYER 1 - Structural guard
        Validates that raw_config contains all required top-level sections
        BEFORE touching the filesystem. A partial or wrong-level dict is
        rejected immediately with a clear error. This is the exact failure
        that caused the Phase 2 test corruption (pool sub-dict passed instead
        of full config dict) -- this layer makes that impossible going forward.

    LAYER 2 - Temp file in same directory
        The temp file is written to the same directory as config.json. This
        guarantees that the final os.replace() is an intra-filesystem rename
        (a single syscall). Cross-device moves are NOT atomic and would
        defeat the entire purpose of this approach.

    LAYER 3 - fsync before rename
        f.flush() drains Python's userspace write buffer into the OS kernel.
        os.fsync(fd) forces the OS kernel buffer to physical storage.
        Without this, a power cut between write() and rename() could leave
        the temp file with partially-written content, which would then get
        renamed over the good original file on restart (silent data loss).

    LAYER 4 - Read-back verification before commit
        After writing and syncing, we re-read and parse the temp file as JSON.
        If the parse fails (disk full mid-write, FS corruption, encoding error),
        the temp file is deleted and the original config.json is NEVER touched.
        Only after a successful verified parse do we call os.replace().

    Error behaviour:
        Any failure at any layer raises a RuntimeError. The original config.json
        is NEVER modified if any layer fails. PoolPicker._persist_cooldown_state
        catches and logs RuntimeError without crashing the proxy -- cooldown
        state is still correct in memory; it will be persisted on the next
        successful write cycle.

    Args:
        config_path: Path to config.json (absolute or relative to CWD).
        raw_config:  The FULL raw config dict. Must contain all required
                     top-level sections (proxy, upstream, tls, providers,
                     patcher, model_pool). Passing a sub-section dict is
                     rejected by Layer 1.
    """
    # ── LAYER 1: Structural guard ──────────────────────────────────────────
    missing = [s for s in _REQUIRED_SECTIONS if s not in raw_config]
    if missing:
        raise RuntimeError(
            f"write_config_atomic() rejected: raw_config is missing required "
            f"top-level sections: {missing}. "
            f"The FULL config dict must be passed, not a sub-section. "
            f"Keys present: {list(raw_config.keys())}"
        )

    abs_path = os.path.abspath(config_path)
    dir_     = os.path.dirname(abs_path)
    tmp_path: str | None = None

    try:
        # ── LAYER 2: Temp file in same directory ──────────────────────────
        # mkstemp returns (fd, path); we own the fd until fdopen or os.close.
        fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp", prefix=".cfg_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(raw_config, f, indent=2, ensure_ascii=False)
                # ── LAYER 3: fsync before rename ───────────────────────────
                # flush() moves Python's buffer to the OS kernel.
                # fsync() moves the kernel's buffer to physical storage.
                # Both are needed for true durability against power loss.
                f.flush()
                os.fsync(f.fileno())
            # fd is now closed by the context manager exit
        except Exception:
            # fdopen may have transferred ownership of fd. If it failed before
            # that, close fd manually to avoid a descriptor leak.
            try:
                os.close(fd)
            except OSError:
                pass
            raise

        # ── LAYER 4: Read-back verification ───────────────────────────────
        # Re-open and parse the temp file. If any write corruption occurred
        # (disk full, truncated write, FS error) this parse will raise and
        # the original file is never touched.
        with open(tmp_path, "r", encoding="utf-8") as verify_f:
            verified = json.load(verify_f)

        # Verify the written data still has all required sections
        missing_after = [s for s in _REQUIRED_SECTIONS if s not in verified]
        if missing_after:
            raise RuntimeError(
                f"Read-back verification failed: written JSON is missing "
                f"sections {missing_after}. Aborting -- original config is safe."
            )

        # ── COMMIT: atomic rename ──────────────────────────────────────────
        # Reached only if all 4 layers passed.
        # os.replace() is atomic on NTFS (MoveFileExW + MOVEFILE_REPLACE_EXISTING)
        # and POSIX (rename(2) syscall). The old file is never partially visible.
        os.replace(tmp_path, abs_path)
        tmp_path = None  # rename succeeded; file is now owned by abs_path

        pool_entry_count = len(
            verified.get("model_pool", {}).get("entries", [])
        )
        _log.debug(
            f"Config written atomically to {abs_path} "
            f"({pool_entry_count} pool entries)"
        )

    except Exception as exc:
        # Clean up the orphaned temp file if rename never happened
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
                _log.debug(f"Cleaned up orphaned temp file: {tmp_path}")
            except OSError:
                pass  # best-effort; temp files are harmless but messy
        raise RuntimeError(f"write_config_atomic failed: {exc}") from exc


# ---------------------------------------------------------------------------
# PoolPicker
# ---------------------------------------------------------------------------

class PoolPicker:
    """
    Manages entry selection, cooldown tracking, and metadata pre-announcement.

    Attributes:
        entries:        The full list of PoolEntry objects (all states).
        pool_settings:  Global pool settings (fallback model, buffer size).
        config_path:    Path to config.json for atomic writes.
        raw_config:     Reference to the live FULL raw config dict.
                        Cooldown writes update raw_config["model_pool"]["entries"]
                        in-place, then write the complete dict atomically.
        pending_advance: Set to True when a pool request fires the on-demand
                        fetchAvailableModels trigger. Cleared when the interceptor
                        serves the patched response.
    """

    def __init__(
        self,
        entries: list[PoolEntry],
        pool_settings: PoolSettings,
        config_path: str,
        raw_config: dict,
    ) -> None:
        # Validate at construction time -- fail early if wrong dict passed
        missing = [s for s in _REQUIRED_SECTIONS if s not in raw_config]
        if missing:
            raise ValueError(
                f"PoolPicker __init__: raw_config is missing required sections: "
                f"{missing}. Pass the full config dict, not a sub-section."
            )

        self.entries = entries
        self.pool_settings = pool_settings
        self.config_path = config_path
        self.raw_config = raw_config

        # Metadata pre-announcement state (Phase 4)
        self.pending_advance: bool = False
        self._cached_model_response: bytes | None = None

        # Internal peek cursor — tracks which entry was last announced via
        # fetchAvailableModels. Stored as entry ID (not index) so it remains
        # correct even when _sorted_candidates() reorders as last_used_at updates.
        # None = start-of-list (peek returns candidates[0]).
        self._peek_entry_id: str | None = None

        total = len(entries)
        enabled = sum(1 for e in entries if e.enabled)
        _log.info(
            f"PoolPicker initialized: {total} entries total, "
            f"{enabled} enabled"
        )
        for e in entries:
            status = "enabled" if e.enabled else "DISABLED"
            cooled = f"cooled until {e.cooldown_until}" if e.is_cooled else "ready"
            _log.debug(
                f"  [{e.id}] {status} | {e.label} | "
                f"ctx={e.context_window} usable={e.usable_tokens} | {cooled}"
            )

    # ------------------------------------------------------------------
    # pick() -- select next entry for an outgoing request
    # ------------------------------------------------------------------

    def pick(self) -> PoolEntry | None:
        """
        Select the best available pool entry for the next request.

        Algorithm:
          1. Clear expired cooldowns (may make entries available again).
          2. Filter: enabled=True AND is_cooled=False.
          3. If no entries available -> return None (all-cooled fallback).
          4. Expand by weight: entry with weight=2 appears twice in list.
          5. Sort by: (in_flight_count ASC, last_used_at ASC nulls-first).
          6. Select the first (least-busy, least-recently-used) candidate.
          7. Increment in_flight_count, set last_used_at = now.

        Returns:
            PoolEntry to use, or None if all entries are cooled/disabled.
        """
        # Step 1: clear expired cooldowns (side effect: updates entry state)
        cleared = [e for e in self.entries if e.clear_cooldown_if_expired()]
        if cleared:
            self._persist_cooldown_state()

        # Step 2: available candidates
        available = [e for e in self.entries if e.enabled and not e.is_cooled]
        if not available:
            _log.warning("PoolPicker.pick(): ALL entries cooled/disabled -- returning None")
            return None

        # Steps 3+4: expand by weight
        candidates = []
        for e in available:
            for _ in range(max(1, e.weight)):
                candidates.append(e)

        # Step 5: sort -- least in-flight first, then least-recently-used
        _epoch = datetime.min.replace(tzinfo=timezone.utc)
        candidates.sort(key=lambda e: (
            e.in_flight_count,
            e.last_used_at or _epoch,
        ))

        # Step 6: select
        selected = candidates[0]

        # Step 7: update runtime counters
        selected.in_flight_count += 1
        selected.last_used_at = datetime.now(timezone.utc)

        _log.info(
            f"PoolPicker.pick() -> [{selected.id}] {selected.label} | "
            f"in_flight={selected.in_flight_count} usable={selected.usable_tokens}"
        )
        return selected

    # ------------------------------------------------------------------
    # earliest_cooldown_seconds() -- ETA for next key availability
    # ------------------------------------------------------------------

    def earliest_cooldown_seconds(self) -> float:
        """
        Return how many seconds until the soonest cooled entry becomes
        available, or 0.0 if any entry is already available right now.

        Used by the keep-alive wait loop to report accurate ETAs.
        """
        now = datetime.now(timezone.utc)
        min_wait: float | None = None

        for e in self.entries:
            if not e.enabled:
                continue
            if not e.is_cooled:
                return 0.0  # already available
            if e.cooldown_until is not None:
                remaining = (e.cooldown_until - now).total_seconds()
                if remaining > 0:
                    if min_wait is None or remaining < min_wait:
                        min_wait = remaining

        return min_wait if min_wait is not None else 0.0

    # ------------------------------------------------------------------
    # release() -- called after request completes (success or error)
    # ------------------------------------------------------------------

    def release(
        self,
        entry_id: str,
        http_status: int,
        error_body: bytes | str | dict | None,
    ) -> None:
        """
        Signal that a request using entry_id has completed.

        On success (http_status < 400):
          - Decrements in_flight_count only.

        On retriable error (429, 5xx):
          - Decrements in_flight_count.
          - Resolves cooldown duration from status + error body.
          - Sets cooldown_until and cooldown_reason on the entry.
          - Persists state to config.json atomically.

        On non-retriable error (4xx except 429):
          - Decrements in_flight_count.
          - Applies a short cooldown (same as 5xx = 30s).

        Args:
            entry_id:    The pool entry ID returned by pick().
            http_status: HTTP status from provider response.
            error_body:  Raw error response body (None on success or network error).
        """
        entry = self._find_entry(entry_id)
        if entry is None:
            _log.error(f"PoolPicker.release(): unknown entry_id={entry_id!r}")
            return

        # Always decrement -- even on error
        entry.in_flight_count = max(0, entry.in_flight_count - 1)

        if http_status < 400:
            _log.debug(
                f"PoolPicker.release(): [{entry_id}] success "
                f"(in_flight now={entry.in_flight_count})"
            )
            return

        # Apply cooldown
        cooldown_until, reason = resolve_cooldown(http_status, error_body, entry_id)
        entry.cooldown_until = cooldown_until
        entry.cooldown_reason = reason

        _log.warning(f"PoolPicker.release(): [{entry_id}] -> {reason}")

        # Persist to disk
        self._persist_cooldown_state()

    # ------------------------------------------------------------------
    # peek() / advance_peek() -- for fetchAvailableModels interceptor
    # ------------------------------------------------------------------

    def peek(self) -> PoolEntry | None:
        """
        Return the entry that will be announced next via fetchAvailableModels,
        WITHOUT consuming it or affecting pick() selection.

        Returns None if no entries are available (all cooled/disabled).
        """
        candidates = self._sorted_candidates()
        if not candidates:
            return None
        if self._peek_entry_id is None:
            # No prior peek — return the first (least-used) candidate
            return candidates[0]
        # Find current peek entry in the current sorted list
        for c in candidates:
            if c.id == self._peek_entry_id:
                return c
        # Peek entry was cooled/disabled — fall back to first candidate
        return candidates[0]

    def advance_peek(self) -> PoolEntry | None:
        """
        Advance the peek cursor to the next candidate and return the new peek entry.

        Called by the fetchAvailableModels interceptor after serving a patched
        response. Ensures consecutive on-demand refreshes cycle through entries.

        Returns the newly selected peek entry, or None if pool is exhausted.

        Implementation note:
            Uses entry ID tracking (not an integer index) to avoid the
            cursor-sort-mismatch bug: if the sorted list reorders between
            calls (because last_used_at changed after a pick()), an integer
            index would point to a DIFFERENT entry than intended.
            By finding the current peek entry by ID and advancing from there,
            we always return the entry that logically follows the last-announced
            entry in the current sort order.
        """
        candidates = self._sorted_candidates()
        if not candidates:
            self._peek_entry_id = None
            return None

        if self._peek_entry_id is None:
            # First advance — start from the second candidate
            self._peek_entry_id = candidates[0].id
            idx = 0
        else:
            # Find current peek position by ID
            idx = next(
                (i for i, c in enumerate(candidates) if c.id == self._peek_entry_id),
                0,  # fallback: start from top if peek entry was removed
            )

        # Advance to next
        next_idx = (idx + 1) % len(candidates)
        next_entry = candidates[next_idx]
        self._peek_entry_id = next_entry.id
        _log.debug(
            f"PoolPicker.advance_peek() -> [{next_entry.id}] "
            f"usable={next_entry.usable_tokens}"
        )
        return next_entry

    # ------------------------------------------------------------------
    # Cached model response (for fetchAvailableModels interceptor)
    # ------------------------------------------------------------------

    def set_cached_model_response(self, body: bytes) -> None:
        """Store the last real Google fetchAvailableModels response body."""
        self._cached_model_response = body

    def get_cached_model_response(self) -> bytes | None:
        """Return the cached body, or None if not yet populated."""
        return self._cached_model_response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_entry(self, entry_id: str) -> PoolEntry | None:
        for e in self.entries:
            if e.id == entry_id:
                return e
        return None

    def _sorted_candidates(self) -> list[PoolEntry]:
        """
        Return the weight-expanded, sorted candidate list -- same algorithm
        as pick() but read-only (no state changes).
        """
        available = [e for e in self.entries if e.enabled and not e.is_cooled]
        if not available:
            return []

        candidates = []
        for e in available:
            for _ in range(max(1, e.weight)):
                candidates.append(e)

        _epoch = datetime.min.replace(tzinfo=timezone.utc)
        candidates.sort(key=lambda e: (
            e.in_flight_count,
            e.last_used_at or _epoch,
        ))
        return candidates

    def _persist_cooldown_state(self) -> None:
        """
        Write all current cooldown_until / cooldown_reason values back into
        config.json, updating ONLY those two fields per entry.

        IMPORTANT — Live re-read strategy:
            We re-read config.json from disk immediately before writing instead
            of using self.raw_config (which was loaded once at startup).
            This prevents the "stale snapshot" bug: if the user edits config.json
            while the proxy is running (thinking.enabled, new keys, pool_settings
            changes), a naive write of the startup snapshot would silently
            overwrite those edits. Re-reading ensures we only ever touch
            cooldown_until / cooldown_reason, leaving everything else on disk
            exactly as it is.
        """
        import json as _json
        from pathlib import Path as _Path

        abs_path = str(_Path(self.config_path).resolve())

        # Re-read live config from disk
        try:
            live_raw = _json.loads(_Path(abs_path).read_text(encoding="utf-8"))
        except Exception as exc:
            _log.error(
                f"PoolPicker: Failed to re-read config.json before cooldown "
                f"write: {exc}. Cooldown state NOT persisted."
            )
            return

        # Validate structure before touching anything
        missing = [s for s in _REQUIRED_SECTIONS if s not in live_raw]
        if missing:
            _log.error(
                f"PoolPicker: Re-read config.json is missing sections "
                f"{missing}. Cooldown state NOT persisted."
            )
            return

        # Update ONLY cooldown fields in the live dict
        raw_entries = live_raw.get("model_pool", {}).get("entries", [])
        raw_by_id: dict[str, dict] = {e.get("id"): e for e in raw_entries}

        for entry in self.entries:
            raw = raw_by_id.get(entry.id)
            if raw is None:
                continue  # entry added to config after startup — skip safely
            raw["cooldown_until"] = (
                entry.cooldown_until.isoformat()
                if entry.cooldown_until is not None
                else None
            )
            raw["cooldown_reason"] = entry.cooldown_reason

        # Keep in-memory reference current so next re-read sees a clean base
        self.raw_config = live_raw

        try:
            write_config_atomic(self.config_path, live_raw)
        except RuntimeError as exc:
            _log.error(
                f"PoolPicker: Failed to persist cooldown state: {exc}. "
                f"State is correct in memory but will be lost on restart."
            )

    def status_summary(self) -> list[dict]:
        """
        Return a list of dicts summarising each entry's current state.
        Used for logging at startup and for any future status endpoint.
        """
        result = []
        for e in self.entries:
            result.append({
                "id": e.id,
                "label": e.label,
                "enabled": e.enabled,
                "is_cooled": e.is_cooled,
                "cooldown_until": e.cooldown_until.isoformat() if e.cooldown_until else None,
                "cooldown_reason": e.cooldown_reason,
                "in_flight_count": e.in_flight_count,
                "usable_tokens": e.usable_tokens,
                "response_thinking_field": e.response_thinking_field,
            })
        return result


# ---------------------------------------------------------------------------
# Singleton interface
# ---------------------------------------------------------------------------

_picker_instance: PoolPicker | None = None


def get_picker() -> PoolPicker | None:
    """Return the global PoolPicker instance, or None if not yet initialized."""
    return _picker_instance


def init_picker(
    entries: list[PoolEntry],
    pool_settings: PoolSettings,
    config_path: str,
    raw_config: dict,
) -> PoolPicker:
    """
    Initialize the global PoolPicker singleton.
    Must be called once at startup from main.py, after config is loaded.

    Args:
        entries:       Parsed PoolEntry list from parse_pool_entries().
        pool_settings: Parsed PoolSettings from AppConfig.pool_settings.
        config_path:   Path to config.json for atomic cooldown writes.
        raw_config:    The FULL raw dict loaded from config.json.
                       Must contain all top-level sections (proxy, upstream,
                       tls, providers, patcher, model_pool). The PoolPicker
                       constructor validates this and raises ValueError if wrong.

    Returns:
        The initialized PoolPicker instance.

    Raises:
        ValueError: If raw_config is missing required top-level sections.
    """
    global _picker_instance
    _picker_instance = PoolPicker(entries, pool_settings, config_path, raw_config)
    return _picker_instance
