"""
src/pool/config_parser.py — Pool Config Parser
================================================
Converts the raw `model_pool` dict from config.json into typed PoolEntry objects.

New schema (DRY / provider-model-keys hierarchy):
  model_pool.providers[]
    └── provider: id, name, base_url, streaming, enabled, defaults{thinking, response_thinking_field}
        ├── models[]: id, model, context_window, weight, limits, max_output_tokens?, thinking?
        └── keys[]:   id, model_ref, api_key, label?, enabled?

Inheritance chain (resolved at parse time, fully flat at runtime):
    provider.defaults → model settings → key (api_key + model_ref only)

Thinking inheritance rules:
  - Provider defaults defines: enabled, budget, enable_param, budget_param
  - Model can override: enabled and budget ONLY
  - enable_param, budget_param, response_thinking_field are provider-level constants
  - If resolved thinking.enabled == False → no thinking params applied at all
    (budget, enable_param, budget_param are NOT inherited for non-thinking models)

Cooldown state:
  - Lives in a SEPARATE cooldowns.json file (not inside config.json)
  - Loaded at startup and merged into entries; written back on every cooldown change
  - config.json is NEVER written by the proxy at runtime

Backward compatibility:
  - If model_pool.entries[] is present (old flat format), it is parsed using the
    legacy path so existing configs keep working without migration.

Reference: config_dry_redesign.md (finalized schema)
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.pool.entry import LimitsConfig, PoolEntry, ThinkingConfig


_log = logging.getLogger("pool.config_parser")

# Default safety buffer if not set globally or per-entry
_GLOBAL_DEFAULT_SAFETY_BUFFER = 8192


# ---------------------------------------------------------------------------
# Cooldown file loader
# ---------------------------------------------------------------------------

def load_cooldowns(cooldowns_path: str) -> dict[str, dict]:
    """
    Load entry cooldown state from the separate cooldowns.json file.

    Returns a dict mapping entry_id → {"cooldown_until": str|None, "cooldown_reason": str|None}.
    Returns {} if the file does not exist (fresh start, no cooldowns).
    Logs and returns {} if the file is malformed (never crashes startup).
    """
    path = Path(cooldowns_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            _log.warning(
                f"[pool.config_parser] cooldowns.json is not a dict (got {type(data).__name__}). "
                f"Ignoring — all entries start with no cooldown."
            )
            return {}
        return data
    except Exception as exc:
        _log.warning(
            f"[pool.config_parser] Failed to load cooldowns.json: {exc}. "
            f"All entries start with no cooldown."
        )
        return {}


def _parse_cooldown_dt(value: str | None) -> Optional[datetime]:
    """Parse an ISO-8601 cooldown string; return None if absent or malformed."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Thinking inheritance helper
# ---------------------------------------------------------------------------

def _resolve_thinking(provider_defaults: dict, model_override: dict | None) -> ThinkingConfig:
    """
    Merge provider defaults with optional model-level thinking overrides.

    Rules:
      - provider_defaults may set: enabled, budget, enable_param, budget_param
      - model_override may only set: enabled, budget  (not enable_param/budget_param)
      - If the resolved enabled == False → return ThinkingConfig(enabled=False, ...)
        with no meaningful budget/param values (they won't be used)
      - enable_param, budget_param are always taken from the provider level
    """
    pdef_t = provider_defaults.get("thinking", {})

    # Provider-level constants (not overridable at model level)
    enable_param = pdef_t.get("enable_param", "thinking")
    budget_param = pdef_t.get("budget_param", "thinking_budget")

    # Resolve enabled: model can override provider default
    prov_enabled = bool(pdef_t.get("enabled", True))
    if model_override is not None and "enabled" in model_override:
        resolved_enabled = bool(model_override["enabled"])
    else:
        resolved_enabled = prov_enabled

    if not resolved_enabled:
        # Non-thinking model — return minimal config, params are irrelevant
        return ThinkingConfig(
            enabled=False,
            budget=None,
            enable_param=enable_param,
            budget_param=budget_param,
        )

    # Resolve budget: model can override provider default
    prov_budget = pdef_t.get("budget")
    if model_override is not None and "budget" in model_override:
        resolved_budget = model_override["budget"]
    else:
        resolved_budget = prov_budget

    return ThinkingConfig(
        enabled=True,
        budget=resolved_budget,
        enable_param=enable_param,
        budget_param=budget_param,
    )


# ---------------------------------------------------------------------------
# Public parser — new provider/model/keys format
# ---------------------------------------------------------------------------

def parse_pool_entries(raw_pool: dict, cooldowns_path: str = "scratchpad/cooldowns.json") -> tuple[list[PoolEntry], dict]:
    """
    Parse the `model_pool` section of config.json into PoolEntry objects.

    Supports both the new provider/model/keys hierarchy (recommended) and
    the legacy flat entries[] format (backward compat).

    Args:
        raw_pool:       The dict at config["model_pool"].
        cooldowns_path: Path to the separate cooldowns.json runtime state file.

    Returns:
        (entries, pool_settings_raw) where:
          - entries: list of PoolEntry (may be empty)
          - pool_settings_raw: the raw pool_settings dict for use by PoolPicker

    Raises:
        ValueError: If a required field is missing.
    """
    pool_settings = raw_pool.get("pool_settings", {})
    global_safety = int(pool_settings.get("safety_buffer_tokens", _GLOBAL_DEFAULT_SAFETY_BUFFER))

    # Load runtime cooldown state from separate file
    cooldowns = load_cooldowns(cooldowns_path)

    # Detect format: new (providers[]) vs legacy (entries[])
    if "providers" in raw_pool:
        entries = _parse_providers(raw_pool["providers"], global_safety, cooldowns)
    elif "entries" in raw_pool:
        _log.warning(
            "[pool.config_parser] Using legacy flat 'entries[]' format. "
            "Consider migrating to the new provider/model/keys hierarchy."
        )
        entries = _parse_legacy_entries(raw_pool["entries"], global_safety, cooldowns)
    else:
        entries = []

    return entries, pool_settings


# ---------------------------------------------------------------------------
# New format parser
# ---------------------------------------------------------------------------

def _parse_providers(providers_raw: list, global_safety: int, cooldowns: dict) -> list[PoolEntry]:
    """Expand provider→model→keys hierarchy into a flat PoolEntry list."""
    entries: list[PoolEntry] = []

    for pi, prov in enumerate(providers_raw):
        pprefix = f"model_pool.providers[{pi}]"

        # Required provider fields
        for req in ("id", "base_url", "keys"):
            if req not in prov:
                raise ValueError(f"Missing required field: {pprefix}.{req}")

        # Provider is disabled → skip ALL its keys
        if not prov.get("enabled", True):
            _log.debug(f"[pool.config_parser] Provider '{prov['id']}' is disabled — skipping all keys.")
            continue

        provider_id   = prov["id"]
        provider_name = prov.get("name", provider_id)
        base_url      = prov["base_url"].rstrip("/")
        streaming     = bool(prov.get("streaming", True))
        prov_defaults = prov.get("defaults", {})
        response_thinking_field = prov_defaults.get("response_thinking_field")

        # Build model lookup: model_id → model_dict
        models_by_id: dict[str, dict] = {}
        for mi, mdl in enumerate(prov.get("models", [])):
            mprefix = f"{pprefix}.models[{mi}]"
            for req in ("id", "model", "context_window"):
                if req not in mdl:
                    raise ValueError(f"Missing required field: {mprefix}.{req}")
            models_by_id[mdl["id"]] = mdl

        # Expand keys
        for ki, key in enumerate(prov["keys"]):
            kprefix = f"{pprefix}.keys[{ki}]"

            for req in ("id", "model_ref", "api_key"):
                if req not in key:
                    raise ValueError(f"Missing required field: {kprefix}.{req}")

            key_id    = key["id"]
            model_ref = key["model_ref"]

            # Resolve model
            if model_ref not in models_by_id:
                raise ValueError(
                    f"{kprefix}.model_ref={model_ref!r} not found in "
                    f"provider '{provider_id}' models. "
                    f"Available: {list(models_by_id.keys())}"
                )
            mdl = models_by_id[model_ref]

            # Key disabled?
            if not key.get("enabled", True):
                _log.debug(f"[pool.config_parser] Key '{key_id}' is disabled — skipping.")
                continue

            # Auto-generate label if not explicitly set
            label = key.get("label") or f"{provider_name} / {model_ref} [{key_id}]"

            # Safety buffer (global only — no per-entry override in new format)
            safety = global_safety

            # Limits from model
            limits_raw = mdl.get("limits", {})
            limits = LimitsConfig(
                rpm=limits_raw.get("rpm"),
                tpm=limits_raw.get("tpm"),
                rpd=limits_raw.get("rpd"),
            )

            # Thinking — resolve inheritance
            model_thinking_override = mdl.get("thinking")  # may be None (fully inherits)
            thinking = _resolve_thinking(prov_defaults, model_thinking_override)

            # Cooldown state from cooldowns.json
            cd_entry  = cooldowns.get(key_id, {})
            cooldown_until  = _parse_cooldown_dt(cd_entry.get("cooldown_until"))
            cooldown_reason = cd_entry.get("cooldown_reason")

            entries.append(PoolEntry(
                id=key_id,
                label=label,
                base_url=base_url,
                api_key=key["api_key"],
                model=mdl["model"],
                streaming=streaming,
                weight=max(1, int(mdl.get("weight", 1))),
                enabled=True,  # already filtered disabled keys above
                limits=limits,
                context_window=int(mdl["context_window"]),
                safety_buffer_tokens=safety,
                thinking=thinking,
                response_thinking_field=response_thinking_field,
                cooldown_until=cooldown_until,
                cooldown_reason=cooldown_reason,
                max_output_tokens=(
                    int(mdl["max_output_tokens"])
                    if mdl.get("max_output_tokens") is not None
                    else None
                ),
                in_flight_count=0,
                last_used_at=None,
            ))

    return entries


# ---------------------------------------------------------------------------
# Legacy flat entries[] format (backward compat)
# ---------------------------------------------------------------------------

def _parse_legacy_entries(entries_raw: list, global_safety: int, cooldowns: dict) -> list[PoolEntry]:
    """Parse the old flat entries[] format. Cooldowns loaded from cooldowns.json if present,
    otherwise falls back to inline cooldown_until/cooldown_reason fields."""
    entries: list[PoolEntry] = []

    for i, e in enumerate(entries_raw):
        prefix = f"model_pool.entries[{i}]"

        for req in ("id", "base_url", "api_key", "model", "context_window"):
            if req not in e:
                raise ValueError(f"Missing required field: {prefix}.{req}")

        entry_safety_raw = e.get("safety_buffer_tokens")
        safety = int(entry_safety_raw) if entry_safety_raw is not None else global_safety

        limits_raw = e.get("limits", {})
        limits = LimitsConfig(
            rpm=limits_raw.get("rpm"),
            tpm=limits_raw.get("tpm"),
            rpd=limits_raw.get("rpd"),
        )

        t = e.get("thinking", {})
        thinking = ThinkingConfig(
            enabled=bool(t.get("enabled", True)),
            budget=t.get("budget"),
            enable_param=t.get("enable_param", "thinking"),
            budget_param=t.get("budget_param", "thinking_budget"),
        )

        # Cooldown: prefer cooldowns.json, fall back to inline fields
        key_id = e["id"]
        cd_entry = cooldowns.get(key_id, {})
        if cd_entry:
            cooldown_until  = _parse_cooldown_dt(cd_entry.get("cooldown_until"))
            cooldown_reason = cd_entry.get("cooldown_reason")
        else:
            cooldown_until  = _parse_cooldown_dt(e.get("cooldown_until"))
            cooldown_reason = e.get("cooldown_reason")

        entries.append(PoolEntry(
            id=key_id,
            label=e.get("label", key_id),
            base_url=e["base_url"].rstrip("/"),
            api_key=e["api_key"],
            model=e["model"],
            streaming=bool(e.get("streaming", True)),
            weight=max(1, int(e.get("weight", 1))),
            enabled=bool(e.get("enabled", True)),
            limits=limits,
            context_window=int(e["context_window"]),
            safety_buffer_tokens=safety,
            thinking=thinking,
            response_thinking_field=e.get("response_thinking_field"),
            cooldown_until=cooldown_until,
            cooldown_reason=cooldown_reason,
            max_output_tokens=(
                int(e["max_output_tokens"])
                if e.get("max_output_tokens") is not None
                else None
            ),
            in_flight_count=0,
            last_used_at=None,
        ))

    return entries
