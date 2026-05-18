"""
src/pool/trigger.py — On-Demand fetchAvailableModels Refresh Trigger
=====================================================================
Writes "1" to the shared ag_proxy_refresh.flag signal file to tell the
patched Antigravity IDE's main.js to call refreshUserStatus() immediately,
which causes the IDE to issue a fresh fetchAvailableModels request that our
proxy intercepts and serves from cache (patched with the next pool entry's
context window as maxTokens).

Architecture:
  - Python proxy writes "1" to the flag file (this module)
  - main.js fs.watch() detects the change in <10ms (OS ReadDirectoryChangesW)
  - main.js reads "1", writes "0" back, calls refreshUserStatus()
  - IDE sends fetchAvailableModels → proxy intercepts (pending_advance=True)
  - Proxy serves cached body patched with next pool entry's usable_tokens

This is fire-and-forget via asyncio.create_task() — it NEVER blocks the
streaming response to the IDE. If the flag file is missing or not writable,
the error is logged at WARNING level and pending_advance stays True, so the
next natural fetchAvailableModels request (IDE's 60s cycle) will be
intercepted instead.

Replaces the broken HTTP server approach (port 9528) that failed because
http.createServer().listen() was running in a sandboxed Electron context
without network stack access.

Reference: pool_implementation_plan.md Phase 3 Step 3.6
           file_trigger_design.md (full design document)
"""

import asyncio
import logging
from pathlib import Path

_log = logging.getLogger("pool.trigger")

# Flag values — single byte, NTFS-atomic for 1-byte writes
_FLAG_TRIGGER = "1"   # proxy → JS: call refreshUserStatus() now
_FLAG_IDLE    = "0"   # default / reset value


async def trigger_model_refresh(flag_path: str | None) -> None:
    """
    Signal the patched IDE to call refreshUserStatus() by writing "1"
    to the shared flag file.

    Args:
        flag_path: Absolute path to ag_proxy_refresh.flag.
                   If None or empty, the trigger is disabled (logs a warning
                   the first time, then silently skips on subsequent calls).

    All errors are caught and logged — this is best-effort. The proxy
    continues serving the stream regardless of trigger success/failure.
    pending_advance stays True so the next natural FAMS call is intercepted.
    """
    if not flag_path:
        _log.debug("trigger_model_refresh: flag_file not configured — trigger disabled")
        return

    try:
        # Path.write_text() is a blocking file write.
        # For a 1-byte local file on SSD this takes <1ms.
        # Running inside asyncio.create_task() so the event loop is not blocked
        # during the await yield points of the streaming response.
        Path(flag_path).write_text(_FLAG_TRIGGER, encoding="utf-8")
        _log.debug(f"trigger_model_refresh: flag set → {flag_path}")

    except OSError as exc:
        # Fail silently — pending_advance stays True.
        # IDE's natural 60s fetchAvailableModels cycle will pick it up.
        _log.warning(
            f"trigger_model_refresh: could not write flag file: {exc} "
            f"(on-demand trigger disabled for this turn; "
            f"natural 60s FAMS cycle will announce next key instead)"
        )
    except Exception as exc:
        _log.debug(f"trigger_model_refresh: unexpected error: {exc}")


def init_flag_file(flag_path: str | None) -> None:
    """
    Initialize the flag file at proxy startup.

    Rules:
      - If flag_path is None/empty: log and skip (trigger disabled).
      - If flag file does NOT exist: create parent dirs + write "0".
      - If flag file EXISTS with value "1": reset to "0"
        (stale signal from a previous session).
      - If flag file EXISTS with value "0": leave it (already idle).

    Called once from main.py before the server starts accepting connections.
    Guarantees the flag file is always in a clean "0" state when the IDE opens.
    """
    if not flag_path:
        _log.info("[POOL] Refresh flag: not configured (trigger disabled)")
        return

    p = Path(flag_path)
    try:
        # Create parent directory (scratchpad/) if it doesn't exist
        p.parent.mkdir(parents=True, exist_ok=True)

        if not p.exists():
            p.write_text(_FLAG_IDLE, encoding="utf-8")
            _log.info(f"[POOL] Refresh flag created: {flag_path}")
        else:
            current = p.read_text(encoding="utf-8").strip()
            if current == _FLAG_TRIGGER:
                # Stale "1" from a previous run — reset to idle
                p.write_text(_FLAG_IDLE, encoding="utf-8")
                _log.info(f"[POOL] Refresh flag reset (stale '1' cleared): {flag_path}")
            else:
                _log.info(f"[POOL] Refresh flag ready (value='{current}'): {flag_path}")

    except OSError as exc:
        _log.warning(
            f"[POOL] Could not initialize refresh flag file: {exc}. "
            f"On-demand trigger will be disabled. "
            f"Check that the scratchpad directory is writable."
        )
