"""
src/pool/config_parser.py — Pool Config Parser
================================================
Converts the raw `model_pool` dict from config.json into typed PoolEntry objects.

The parser is deliberately lenient on optional fields and strict only on
required fields (id, base_url, api_key, model, context_window).

Reference: pool_implementation_plan.md Phase 1 Step 1.3
"""

from datetime import datetime, timezone
from typing import Optional

from src.pool.entry import LimitsConfig, PoolEntry, ThinkingConfig


# Default safety buffer if not set globally or per-entry
_GLOBAL_DEFAULT_SAFETY_BUFFER = 8192


# ---------------------------------------------------------------------------
# Public parser
# ---------------------------------------------------------------------------

def parse_pool_entries(raw_pool: dict) -> tuple[list[PoolEntry], dict]:
    """
    Parse the `model_pool` section of config.json into PoolEntry objects.

    Args:
        raw_pool: The dict at config["model_pool"].

    Returns:
        (entries, pool_settings_raw) where:
          - entries: list of PoolEntry (may be empty if no entries configured)
          - pool_settings_raw: the raw pool_settings dict for use by PoolPicker

    Raises:
        ValueError: If a required field is missing from any entry.
    """
    pool_settings = raw_pool.get("pool_settings", {})
    global_safety = int(pool_settings.get("safety_buffer_tokens", _GLOBAL_DEFAULT_SAFETY_BUFFER))

    entries: list[PoolEntry] = []
    for i, e in enumerate(raw_pool.get("entries", [])):
        prefix = f"model_pool.entries[{i}]"

        # --- Required fields ---
        for req in ("id", "base_url", "api_key", "model", "context_window"):
            if req not in e:
                raise ValueError(f"Missing required field: {prefix}.{req}")

        # --- Resolve safety buffer: per-entry overrides global ---
        entry_safety_raw = e.get("safety_buffer_tokens")
        safety = int(entry_safety_raw) if entry_safety_raw is not None else global_safety

        # --- Limits (all optional) ---
        limits_raw = e.get("limits", {})
        limits = LimitsConfig(
            rpm=limits_raw.get("rpm"),       # None = unconstrained
            tpm=limits_raw.get("tpm"),
            rpd=limits_raw.get("rpd"),
        )

        # --- Thinking config ---
        t = e.get("thinking", {})
        thinking = ThinkingConfig(
            enabled=bool(t.get("enabled", True)),
            budget=t.get("budget"),           # None = use provider default
            enable_param=t.get("enable_param", "thinking"),     # standard OpenAI default
            budget_param=t.get("budget_param", "thinking_budget"),
        )

        # --- Cooldown state ---
        cooldown_dt: Optional[datetime] = None
        cooldown_str = e.get("cooldown_until")
        if cooldown_str:
            try:
                # Handle both "Z" suffix and "+00:00" offset
                cooldown_dt = datetime.fromisoformat(
                    cooldown_str.replace("Z", "+00:00")
                )
            except ValueError:
                # Malformed timestamp — treat as no cooldown (don't crash on startup)
                cooldown_dt = None

        entries.append(PoolEntry(
            id=e["id"],
            label=e.get("label", e["id"]),
            base_url=e["base_url"].rstrip("/"),
            api_key=e["api_key"],
            model=e["model"],
            streaming=bool(e.get("streaming", True)),
            weight=max(1, int(e.get("weight", 1))),   # minimum weight = 1
            enabled=bool(e.get("enabled", True)),
            limits=limits,
            context_window=int(e["context_window"]),
            safety_buffer_tokens=safety,
            thinking=thinking,
            response_thinking_field=e.get("response_thinking_field"),
            cooldown_until=cooldown_dt,
            cooldown_reason=e.get("cooldown_reason"),
            # Runtime state starts fresh every process launch
            in_flight_count=0,
            last_used_at=None,
        ))

    return entries, pool_settings
