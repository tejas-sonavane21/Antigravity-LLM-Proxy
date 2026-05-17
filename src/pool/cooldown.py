"""
src/pool/cooldown.py — Cooldown Duration Resolver
===================================================
Determines the appropriate cooldown duration for a pool entry based on
the HTTP status code and error response body from the provider.

Cooldown durations (from pool_decisions_next_steps.md):
  429 RPM  → 60 seconds
  429 TPM  → 120 seconds
  429 RPD  → until next UTC midnight
  503      → 90 seconds
  5xx      → 30 seconds

The resolver parses the provider's error body to distinguish RPM vs TPM vs RPD.
If the body is unparseable or the message doesn't match known patterns, a
conservative default (120 seconds) is used.

Reference: pool_implementation_plan.md Phase 2 Steps 2.1, 2.2
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

_log = logging.getLogger("pool.cooldown")


# ---------------------------------------------------------------------------
# Duration constants (seconds)
# ---------------------------------------------------------------------------

_COOLDOWN_RPM_S  = 60     # per-minute rate limit
_COOLDOWN_TPM_S  = 120    # per-minute token limit
_COOLDOWN_5XX_S  = 30     # generic 5xx server error
_COOLDOWN_503_S  = 90     # overloaded / service unavailable
_COOLDOWN_DEFAULT_429_S = 120  # unknown 429 sub-type → conservative fallback


# ---------------------------------------------------------------------------
# Keyword patterns for classifying 429 sub-types
# ---------------------------------------------------------------------------

# Matches RPD (daily limit) signals:  "daily", "per day", "requests per day", "day limit"
_RPD_PATTERN = re.compile(
    r"\b(daily|per[\s_-]?day|day[\s_-]?limit|requests[\s_-]?per[\s_-]?day)\b",
    re.IGNORECASE,
)

# Matches TPM signals:  "token", "tokens per minute", "tpm", "token rate"
_TPM_PATTERN = re.compile(
    r"\b(token|tokens?[\s_-]?per[\s_-]?minute|tpm|token[\s_-]?rate)\b",
    re.IGNORECASE,
)

# Matches RPM signals:  "request", "requests per minute", "rpm", "rate limit"
_RPM_PATTERN = re.compile(
    r"\b(request|requests?[\s_-]?per[\s_-]?minute|rpm|rate[\s_-]?limit)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_utc_midnight() -> datetime:
    """Return the next UTC midnight as a timezone-aware datetime."""
    now = datetime.now(timezone.utc)
    return (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )


def _extract_error_text(error_body: bytes | str | dict | None) -> str:
    """
    Best-effort extraction of a human-readable error message from the
    provider's error response, regardless of format.

    Handles:
      - dict (already parsed JSON)
      - bytes / str JSON
      - raw text
      - None
    """
    if error_body is None:
        return ""

    # Already a dict (e.g. passed in pre-parsed)
    if isinstance(error_body, dict):
        return _dict_to_text(error_body)

    # Bytes → str
    if isinstance(error_body, bytes):
        text = error_body.decode("utf-8", errors="replace")
    else:
        text = str(error_body)

    # Try JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return _dict_to_text(data)
    except (json.JSONDecodeError, ValueError):
        pass

    # Raw text fallback
    return text


def _dict_to_text(data: dict) -> str:
    """
    Extract a meaningful string from a parsed error dict.
    Tries common provider error formats in priority order.
    """
    # OpenAI format: {"error": {"message": "..."}}
    if isinstance(data.get("error"), dict):
        return str(data["error"].get("message", ""))

    # SiliconFlow / generic: {"message": "..."}
    if "message" in data:
        return str(data["message"])

    # Fallback: dump everything as a string for pattern matching
    return json.dumps(data)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_cooldown(
    http_status: int,
    error_body: bytes | str | dict | None,
    entry_id: str = "",
) -> tuple[datetime, str]:
    """
    Determine the cooldown expiry datetime and a human-readable reason string
    for a pool entry that returned an error.

    Args:
        http_status: The HTTP status code from the provider (e.g. 429, 503, 500).
        error_body:  The raw error response body. Accepts bytes, str, dict, or None.
        entry_id:    Pool entry ID for log context only.

    Returns:
        (cooldown_until, cooldown_reason) where cooldown_until is UTC-aware.

    Usage:
        cooldown_dt, reason = resolve_cooldown(429, resp_body, entry_id="sf-key-1")
        entry.cooldown_until = cooldown_dt
        entry.cooldown_reason = reason
    """
    now = datetime.now(timezone.utc)
    error_text = _extract_error_text(error_body)

    if http_status == 429:
        cooldown_until, reason = _resolve_429(now, error_text)
    elif http_status == 503:
        cooldown_until = now + timedelta(seconds=_COOLDOWN_503_S)
        reason = f"503: Service overloaded ({_COOLDOWN_503_S}s cooldown)"
    elif http_status >= 500:
        cooldown_until = now + timedelta(seconds=_COOLDOWN_5XX_S)
        reason = f"{http_status}: Server error ({_COOLDOWN_5XX_S}s cooldown)"
    else:
        # Non-retryable error (4xx except 429) — short cooldown to avoid
        # hammering a broken key but don't lock it out permanently
        cooldown_until = now + timedelta(seconds=_COOLDOWN_5XX_S)
        reason = f"{http_status}: Unexpected error ({_COOLDOWN_5XX_S}s cooldown)"

    _log.warning(
        f"  [{entry_id}] Cooldown applied: {reason} → "
        f"until {cooldown_until.strftime('%H:%M:%S UTC')}"
    )
    return cooldown_until, reason


def _resolve_429(now: datetime, error_text: str) -> tuple[datetime, str]:
    """
    Classify a 429 as RPM, TPM, or RPD and return the appropriate cooldown.
    Priority: RPD > TPM > RPM > default.
    """
    # Daily limit — longest cooldown, check first
    if _RPD_PATTERN.search(error_text):
        midnight = _next_utc_midnight()
        hours_left = (midnight - now).total_seconds() / 3600
        return (
            midnight,
            f"429 RPD: Daily limit reached "
            f"(cooldown until midnight UTC, ~{hours_left:.1f}h)"
        )

    # Token-per-minute limit
    if _TPM_PATTERN.search(error_text):
        return (
            now + timedelta(seconds=_COOLDOWN_TPM_S),
            f"429 TPM: Token rate limit ({_COOLDOWN_TPM_S}s cooldown)"
        )

    # Request-per-minute limit
    if _RPM_PATTERN.search(error_text):
        return (
            now + timedelta(seconds=_COOLDOWN_RPM_S),
            f"429 RPM: Request rate limit ({_COOLDOWN_RPM_S}s cooldown)"
        )

    # Unknown 429 sub-type — conservative fallback
    preview = error_text[:120].replace("\n", " ") if error_text else "(no body)"
    return (
        now + timedelta(seconds=_COOLDOWN_DEFAULT_429_S),
        f"429: Rate limited ({_COOLDOWN_DEFAULT_429_S}s cooldown) | body: {preview}"
    )
