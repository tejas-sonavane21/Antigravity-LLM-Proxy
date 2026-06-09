"""
src/pool/config_parser.py — Pool Config Parser
================================================
Converts the raw `model_pool` dict from config.json into typed PoolEntry objects.

Schema (DRY / provider-model-keys hierarchy, v3):
  model_pool.providers[]
    └── provider: id, name, base_url, streaming, enabled,
                  defaults{thinking, response_thinking_field},
                  default_models[]          ← provider-level model list (optional)
        ├── models[]: id, model, context_window, weight, limits, max_output_tokens?, thinking?
        └── keys[]:   id, api_key,
                      model_ref: list[str]  ← ALWAYS a list, even for one model
                      label?                ← auto-generated if absent
                      enabled?              ← defaults True

Entry ID convention (always composite):
    "{key_id}:{model_id}"
    e.g.  "or-key-1:gpt-oss-120b-free"
          "sf-key-3:kimi-k2.6"

Inheritance chain (resolved fully at parse time, runtime stays flat):
    provider.defaults → model settings → per-key model_ref list
    provider.default_models ← inherited by any key with no model_ref

Thinking inheritance:
  - Provider defaults: enabled, budget, enable_param, budget_param
  - Model can override: enabled and budget ONLY
  - enable_param, budget_param, response_thinking_field are provider constants
  - If resolved thinking.enabled == False → no thinking params applied at all

Cooldown state:
  - Lives in a SEPARATE cooldowns.json file (path from pool_settings.cooldowns_file)
  - Loaded at startup, merged into entries; written by PoolPicker._persist_cooldown_state
  - config.json is NEVER written by the proxy at runtime
  - Cooldown keys use the same composite entry ID: "or-key-1:gpt-oss-120b-free"

Reference: config_dry_redesign.md v3 final decisions
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.pool.entry import LimitsConfig, PoolEntry, ThinkingConfig


_log = logging.getLogger("pool.config_parser")

_GLOBAL_DEFAULT_SAFETY_BUFFER = 8192


# ---------------------------------------------------------------------------
# SQLite key loader
# ---------------------------------------------------------------------------

def load_keys_from_db(db_path: str, provider_id: str) -> list[dict]:
    """
    Load all enabled keys for a provider from keys.db.

    Returns a list of dicts matching the old config.json keys[] format:
        [{"id": "cl-key-1", "api_key": "..."}, ...]

    Returns [] silently if:
      - The DB file does not exist yet (fresh setup, no keys added).
      - The provider has no enabled keys in the DB.
    """
    path = Path(db_path)
    if not path.exists():
        _log.debug(
            f"[pool.config_parser] keys.db not found at '{db_path}' "
            f"— provider '{provider_id}' will have 0 keys from DB."
        )
        return []
    try:
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT key_id, api_key FROM pool_keys "
            "WHERE provider_id = ? AND enabled = 1 ORDER BY id",
            (provider_id,)
        ).fetchall()
        conn.close()
        return [{"id": r["key_id"], "api_key": r["api_key"]} for r in rows]
    except Exception as exc:
        _log.warning(
            f"[pool.config_parser] Failed to load keys from '{db_path}' "
            f"for provider '{provider_id}': {exc}. Treating as 0 keys."
        )
        return []


# ---------------------------------------------------------------------------
# Cooldown file loader
# ---------------------------------------------------------------------------

def load_cooldowns(cooldowns_path: str) -> dict[str, dict]:
    """
    Load per-(key, model) cooldown state from cooldowns.json.

    Returns a dict mapping composite_entry_id → {cooldown_until, cooldown_reason}.
    Returns {} if the file does not exist (fresh start) or is malformed (never crashes).
    """
    path = Path(cooldowns_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            _log.warning(
                f"[pool.config_parser] cooldowns.json is not a dict "
                f"(got {type(data).__name__}). Ignoring — all entries start uncooled."
            )
            return {}
        return data
    except Exception as exc:
        _log.warning(
            f"[pool.config_parser] Failed to load cooldowns.json: {exc}. "
            f"All entries start uncooled."
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
# Thinking inheritance
# ---------------------------------------------------------------------------

def _resolve_thinking(provider_defaults: dict, model_override: dict | None) -> ThinkingConfig:
    """
    Merge provider-level thinking defaults with optional model-level override.

    Provider-level constants (NOT overridable at model level):
        enable_param, budget_param, response_thinking_field

    Model can only override:
        enabled, budget

    If resolved enabled == False → return disabled ThinkingConfig
    (budget / params will never be used, but we keep the provider params
    on the object so the runtime can inspect them if needed).
    """
    pdef_t = provider_defaults.get("thinking", {})

    # Provider-level constants
    enable_param = pdef_t.get("enable_param", "thinking")
    budget_param = pdef_t.get("budget_param", "thinking_budget")

    # Resolve enabled
    prov_enabled = bool(pdef_t.get("enabled", True))
    if model_override is not None and "enabled" in model_override:
        resolved_enabled = bool(model_override["enabled"])
    else:
        resolved_enabled = prov_enabled

    if not resolved_enabled:
        return ThinkingConfig(
            enabled=False,
            budget=None,
            enable_param=enable_param,
            budget_param=budget_param,
        )

    # Resolve budget
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
# Public entry point
# ---------------------------------------------------------------------------

def parse_pool_entries(
    raw_pool: dict,
    cooldowns_path: str = "scratchpad/cooldowns.json",
    keys_db_path: str = "scratchpad/keys.db",
) -> tuple[list[PoolEntry], dict]:
    """
    Parse model_pool section of config.json into flat PoolEntry objects.

    Args:
        raw_pool:       The dict at config["model_pool"].
        cooldowns_path: Path to the separate cooldowns.json runtime state file.
        keys_db_path:   Path to the SQLite keys.db file. Keys for each provider
                        are loaded from here unless the provider still has a
                        'keys' array in config.json (backward-compat mode).

    Returns:
        (entries, pool_settings_raw)

    Raises:
        ValueError: If a required field is missing or a model_ref references
                    a model ID that does not exist in the provider's models list.
    """
    pool_settings = raw_pool.get("pool_settings", {})
    global_safety = int(pool_settings.get("safety_buffer_tokens", _GLOBAL_DEFAULT_SAFETY_BUFFER))
    cooldowns = load_cooldowns(cooldowns_path)

    if "providers" in raw_pool:
        entries = _parse_providers(raw_pool["providers"], global_safety, cooldowns, keys_db_path)
    else:
        entries = []
        _log.warning("[pool.config_parser] model_pool has no 'providers' list — pool is empty.")

    return entries, pool_settings


# ---------------------------------------------------------------------------
# Provider / model / keys hierarchy parser
# ---------------------------------------------------------------------------

def _parse_providers(
    providers_raw: list,
    global_safety: int,
    cooldowns: dict,
    keys_db_path: str = "scratchpad/keys.db",
) -> list[PoolEntry]:
    entries: list[PoolEntry] = []

    for pi, prov in enumerate(providers_raw):
        pprefix = f"model_pool.providers[{pi}]"

        for req in ("id", "base_url"):
            if req not in prov:
                raise ValueError(f"Missing required field: {pprefix}.{req}")

        if not prov.get("enabled", True):
            _log.debug(
                f"[pool.config_parser] Provider '{prov['id']}' is disabled — skipping."
            )
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

        # Provider-level default models (inherited by all keys)
        provider_default_models: list[str] = prov.get("default_models", [])

        # ── Key source: config.json (backward compat) OR keys.db ──────────
        config_keys: list[dict] = prov.get("keys", [])
        if config_keys:
            # keys[] still present in config.json — use them but warn the user
            _log.warning(
                f"[pool] Provider '{provider_id}' has a 'keys' array in config.json "
                f"({len(config_keys)} key(s)). Using config.json keys directly."
            )
            _log.warning(
                f"[pool] To move keys to the database and stop this warning:"
            )
            _log.warning(
                f"[pool]   1. python keys_db.py import --from config.json"
            )
            _log.warning(
                f"[pool]   2. Remove the 'keys' array from provider '{provider_id}' in config.json"
            )
            _log.warning(
                f"[pool]   3. Restart the proxy"
            )
            raw_keys = config_keys
        else:
            # Load from keys.db
            raw_keys = load_keys_from_db(keys_db_path, provider_id)
            if not raw_keys:
                _log.warning(
                    f"[pool.config_parser] Provider '{provider_id}' has no enabled keys "
                    f"in keys.db — skipping provider."
                )
                continue

        for ki, key in enumerate(raw_keys):
            kprefix = f"{pprefix}.keys[{ki}]"

            for req in ("id", "api_key"):
                if req not in key:
                    raise ValueError(f"Missing required field: {kprefix}.{req}")

            if not key.get("enabled", True):
                _log.debug(
                    f"[pool.config_parser] Key '{key['id']}' is disabled — skipping."
                )
                continue

            key_id  = key["id"]
            api_key = key["api_key"]

            # Resolve model list: key-level model_ref overrides provider default_models
            model_refs: list[str]
            if "model_ref" in key:
                raw_ref = key["model_ref"]
                # model_ref MUST be a list
                if not isinstance(raw_ref, list):
                    raise ValueError(
                        f"{kprefix}.model_ref must be a list of model IDs, "
                        f"got {type(raw_ref).__name__!r}. "
                        f"Use [\"model-id\"] even for a single model."
                    )
                model_refs = raw_ref
            elif provider_default_models:
                model_refs = provider_default_models
            else:
                raise ValueError(
                    f"{kprefix} has no model_ref and provider '{provider_id}' "
                    f"has no default_models. Every key must reference at least one model."
                )

            if not model_refs:
                raise ValueError(
                    f"{kprefix}.model_ref is an empty list. "
                    f"Provide at least one model ID."
                )

            # Expand: one PoolEntry per (key, model)
            for model_id in model_refs:
                entry_id = f"{key_id}:{model_id}"  # always composite

                if model_id not in models_by_id:
                    raise ValueError(
                        f"{kprefix} references model_ref={model_id!r} "
                        f"which is not defined in provider '{provider_id}' models. "
                        f"Available: {list(models_by_id.keys())}"
                    )
                mdl = models_by_id[model_id]

                # Auto-generate label
                label = key.get("label") or f"{provider_name} / {model_id} [{key_id}]"

                limits_raw = mdl.get("limits", {})
                limits = LimitsConfig(
                    rpm=limits_raw.get("rpm"),
                    tpm=limits_raw.get("tpm"),
                    rpd=limits_raw.get("rpd"),
                )

                thinking = _resolve_thinking(prov_defaults, mdl.get("thinking"))

                # Cooldown from cooldowns.json (composite key)
                cd_entry        = cooldowns.get(entry_id, {})
                cooldown_until  = _parse_cooldown_dt(cd_entry.get("cooldown_until"))
                cooldown_reason = cd_entry.get("cooldown_reason")

                entries.append(PoolEntry(
                    id=entry_id,
                    label=label,
                    base_url=base_url,
                    api_key=api_key,
                    model=mdl["model"],
                    streaming=streaming,
                    weight=max(1, int(mdl.get("weight", 1))),
                    enabled=True,
                    limits=limits,
                    context_window=int(mdl["context_window"]),
                    safety_buffer_tokens=global_safety,
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
