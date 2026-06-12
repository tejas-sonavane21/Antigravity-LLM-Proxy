"""
src/config.py — Configuration Loader
=====================================
Reads config.json, validates structure, and exposes typed dataclasses.

Reference: Brainstorming Section 11.9 (Configuration Format)
Reference: models.rs L58-L66 (AiProvider struct) for ProviderConfig shape
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    """Raised when config.json is missing required fields or is malformed."""
    pass


# ---------------------------------------------------------------------------
# Dataclasses — mirrors the config.json structure
# ---------------------------------------------------------------------------

@dataclass
class ProxyConfig:
    host: str
    port: int
    log_level: str          # "DEBUG" | "INFO" | "WARNING" | "ERROR"
    include_thoughts: bool  # whether to include thought parts in upstream requests
    dump_model_responses: bool = False  # when True: Q3-DUMP runs on fetchAvailableModels
    dump_requests: bool = False  # when True: print every intercepted raw request body to terminal
    dump_pool_io: bool = False   # when True: print outgoing OpenAI body + raw provider response chunks


@dataclass
class UpstreamConfig:
    hosts: list  # ordered list of Google upstream hostnames (fallback chain)


@dataclass
class TlsConfig:
    cert_dir: str   # absolute path (~ already expanded)
    cert_file: str  # filename only, e.g. "proxy-ca.crt"
    key_file: str   # filename only, e.g. "proxy-ca.key"

    @property
    def cert_path(self) -> str:
        """Full absolute path to the certificate file."""
        return os.path.join(self.cert_dir, self.cert_file)

    @property
    def key_path(self) -> str:
        """Full absolute path to the private key file."""
        return os.path.join(self.cert_dir, self.key_file)


@dataclass
class ProviderConfig:
    name: str               # human-readable identifier, e.g. "opencode-zen"
    base_url: str           # e.g. "https://opencode.ai/zen/v1"
    api_key: str            # user's API key for this provider
    protocol: str           # "openai" (only supported protocol in v1)
    enabled: bool           # if False, this provider is skipped during routing
    streaming: bool         # True = stream=True SSE; False = collect full body
    model_map: dict         # { "gpt-oss-120b-medium": "minimax-m2.5-free", ... }


@dataclass
class PatcherConfig:
    target_url: str         # e.g. "https://127.0.0.1:9527"
    ide_path: str | None = None   # optional override for IDE install location
    flag_file: str | None = None  # path to ag_proxy_refresh.flag signal file


@dataclass
class FallbackModelLimits:
    """
    Actual capability limits for the all_cooled_fallback model.
    Used by _handle_all_cooled() to patch the Gemini request body fields
    (maxOutputTokens, thinkingBudget) so they match the fallback model's real
    limits before forwarding to Google.

    Why needed: FAMS announces pool entry usable_tokens (e.g. 188808) as the
    mapped model's maxTokens. The IDE includes that value as
    generationConfig.maxOutputTokens in subsequent requests. When the fallback
    model has a lower limit (e.g. claude-sonnet-4-6 = 64000 maxOutputTokens),
    Google rejects the request with 500.

    Lives at config.json `model_pool.pool_settings.fallback_model_limits`.
    """
    max_output_tokens: int | None = None   # e.g. 64000 for claude-sonnet-4-6
    thinking_budget:   int | None = None   # e.g. 1024  for claude-sonnet-4-6


@dataclass
class PoolSettings:
    """
    Global settings that apply to the entire model pool.
    Lives at config.json `model_pool.pool_settings`.
    """
    safety_buffer_tokens: int      # default 8192 — subtracted from usable_tokens
    all_cooled_fallback: str       # "passthrough" | "keep-alive" | "<model-id>"
    mapped_model: str              # the IDE model name we intercept, e.g. "gpt-oss-120b-medium"
    fallback_limits: FallbackModelLimits = None  # real limits of fallback model
    flag_file: str | None = None   # path to ag_proxy_refresh.flag (from patcher config)
    keep_alive_timeout_minutes: int = 10  # max minutes to wait in keep-alive mode before passthrough
    # Path to the separate runtime cooldown state file (never mixed into config.json).
    # Written by _persist_cooldown_state(); loaded at startup to restore cooldowns.
    cooldowns_file: str = "scratchpad/cooldowns.json"
    # Path to the SQLite key database. Keys are loaded from here at startup
    # when a provider has no 'keys' array in config.json.
    keys_db: str = "scratchpad/keys.db"

    def __post_init__(self):
        if self.fallback_limits is None:
            self.fallback_limits = FallbackModelLimits()


@dataclass
class FeaturesConfig:
    """
    Optional feature toggles for clod.io-era band-aids that are dormant by
    default. Lives at config.json `features`. Both default to a lean state, so a
    config WITHOUT a "features" section runs without these extras and nothing
    breaks; flip either to True to re-enable the behavior with zero code change.

      strict_tool_contract: append the <tool_calling_contract> block to the
                            pool/mapped-model system prompt (Changes F+G+H).
      harmony_sanitizer:    run the Layer A harmony residue cleaner on provider
                            output (Change I). A no-op on clean providers, but
                            gated here for peace of mind / future leaky hosts.
    """
    strict_tool_contract: bool = False
    harmony_sanitizer: bool = False


@dataclass
class AppConfig:
    proxy: ProxyConfig
    upstream: UpstreamConfig
    tls: TlsConfig
    providers: list         # list[ProviderConfig] — kept for backward compat
    patcher: PatcherConfig
    pool_settings: PoolSettings | None = None  # None = no pool configured
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    # list[PoolEntry] is NOT stored here; it lives in PoolPicker after startup
    # The raw model_pool dict is stored separately so PoolPicker can write back
    # cooldown state changes to the correct location in config.json.
    raw_model_pool: dict | None = None         # raw dict from config.json["model_pool"]


# ---------------------------------------------------------------------------
# Template — written to disk if config.json is missing
# ---------------------------------------------------------------------------

_TEMPLATE = {
    "proxy": {
        "host": "127.0.0.1",
        "port": 9527,
        "log_level": "DEBUG",
        "include_thoughts": False,
        "dump_model_responses": False,
        "dump_requests": False,   # set True to print every intercepted request body
        "dump_pool_io": False,    # set True to print what we send/receive from pool providers
    },
    "features": {
        "strict_tool_contract": False,  # F+G+H: inject <tool_calling_contract> into pool system prompt
        "harmony_sanitizer": False,     # I: run Layer A harmony residue cleaner on provider output
    },
    "upstream": {
        "hosts": [
            "daily-cloudcode-pa.sandbox.googleapis.com",
            "daily-cloudcode-pa.googleapis.com",
            "cloudcode-pa.googleapis.com",
        ]
    },
    "tls": {
        "cert_dir": "./ag_proxy",
        "cert_file": "proxy-ca.crt",
        "key_file": "proxy-ca.key",
    },
    "providers": [
        {
            "name": "opencode-zen",
            "base_url": "https://opencode.ai/zen/v1",
            "api_key": "YOUR_API_KEY_HERE",
            "protocol": "openai",
            "enabled": False,
            "streaming": True,
            "model_map": {
                "gpt-oss-120b-medium": "minimax-m2.5-free"
            },
        }
    ],
    "patcher": {
        "target_url": "https://127.0.0.1:9527",
        "ide_path": "D:\\Anti_Gravity\\Antigravity IDE\\resources\\app\\out",
        "flag_file": "C:\\Path\\To\\Antigravity_model_api_extension\\scratchpad\\ag_proxy_refresh.flag",
    },
    "model_pool": {
        "mapped_model": "gpt-oss-120b-medium",
        "pool_settings": {
            "safety_buffer_tokens": 8192,
            "all_cooled_fallback": "keep-alive",
            "keep_alive_timeout_minutes": 10,
            "cooldowns_file": "scratchpad/cooldowns.json",
            "fallback_model_limits": {
                "max_output_tokens": 64000,
                "thinking_budget": 1024,
            },
        },
        "providers": [
            {
                "id": "my-provider",
                "name": "My Provider",
                "base_url": "https://api.example.com/v1",
                "streaming": True,
                "enabled": True,
                "defaults": {
                    "thinking": {
                        "enabled": True,
                        "budget": 4096,
                        "enable_param": "thinking",
                        "budget_param": "thinking_budget"
                    },
                    "response_thinking_field": "reasoning"
                },
                "default_models": ["my-model"],  # keys inherit this list if no model_ref
                "models": [
                    {
                        "id": "my-model",
                        "model": "your-model-name",
                        "context_window": 200000,
                        "weight": 1,
                        "limits": {"rpm": None, "tpm": None, "rpd": None}
                    }
                ],
                # Keys with no model_ref use default_models automatically.
                # model_ref is ALWAYS a list; use ["model-id"] for one model.
                "keys": [
                    {"id": "provider-key-1", "api_key": "YOUR_API_KEY_HERE"},
                    {"id": "provider-key-2", "model_ref": ["my-model"], "api_key": "YOUR_API_KEY_HERE"}
                ]
            }
        ],
    },
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_config(raw: dict) -> None:
    """
    Validate that all required fields are present in the raw config dict.
    Raises ConfigError with a precise message on the first missing field.
    """
    # --- Top-level sections ---
    for section in ("proxy", "upstream", "tls", "providers", "patcher"):
        if section not in raw:
            raise ConfigError(f"Missing required section: '{section}'")
    # model_pool is optional — validated separately below if present

    # --- proxy ---
    proxy = raw["proxy"]
    for key, expected_type in (("host", str), ("port", int), ("log_level", str)):
        if key not in proxy:
            raise ConfigError(f"Missing required field: proxy.{key}")
        if not isinstance(proxy[key], expected_type):
            raise ConfigError(
                f"Invalid type for proxy.{key}: expected {expected_type.__name__}, "
                f"got {type(proxy[key]).__name__}"
            )

    valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR")
    if proxy["log_level"].upper() not in valid_levels:
        raise ConfigError(
            f"Invalid proxy.log_level: '{proxy['log_level']}'. "
            f"Must be one of: {', '.join(valid_levels)}"
        )

    # --- upstream ---
    upstream = raw["upstream"]
    if "hosts" not in upstream:
        raise ConfigError("Missing required field: upstream.hosts")
    if not isinstance(upstream["hosts"], list) or len(upstream["hosts"]) == 0:
        raise ConfigError("upstream.hosts must be a non-empty list of hostnames")

    # --- tls ---
    tls = raw["tls"]
    for key in ("cert_dir", "cert_file", "key_file"):
        if key not in tls:
            raise ConfigError(f"Missing required field: tls.{key}")
        if not isinstance(tls[key], str) or not tls[key].strip():
            raise ConfigError(f"tls.{key} must be a non-empty string")

    # --- providers ---
    if not isinstance(raw["providers"], list):
        raise ConfigError("'providers' must be a list")

    for i, provider in enumerate(raw["providers"]):
        prefix = f"providers[{i}]"
        for key in ("name", "base_url", "api_key", "protocol", "enabled", "model_map"):
            if key not in provider:
                raise ConfigError(f"Missing required field: {prefix}.{key}")
        if not isinstance(provider["model_map"], dict):
            raise ConfigError(f"{prefix}.model_map must be a dict")
        if not isinstance(provider["enabled"], bool):
            raise ConfigError(f"{prefix}.enabled must be a boolean")
        # streaming is optional (defaults True) — validate type if present
        if "streaming" in provider and not isinstance(provider["streaming"], bool):
            raise ConfigError(f"{prefix}.streaming must be a boolean")

    # --- patcher ---
    patcher = raw["patcher"]
    if "target_url" not in patcher:
        raise ConfigError("Missing required field: patcher.target_url")
    if not isinstance(patcher["target_url"], str) or not patcher["target_url"].strip():
        raise ConfigError("patcher.target_url must be a non-empty string")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_config(raw: dict) -> AppConfig:
    """Convert raw dict into typed AppConfig dataclasses."""
    proxy_raw = raw["proxy"]
    proxy = ProxyConfig(
        host=proxy_raw["host"],
        port=proxy_raw["port"],
        log_level=proxy_raw["log_level"].upper(),
        include_thoughts=proxy_raw.get("include_thoughts", False),
        dump_model_responses=bool(proxy_raw.get("dump_model_responses", False)),
        dump_requests=bool(proxy_raw.get("dump_requests", False)),
        dump_pool_io=bool(proxy_raw.get("dump_pool_io", False)),
    )

    upstream = UpstreamConfig(
        hosts=raw["upstream"]["hosts"]
    )

    tls_raw = raw["tls"]
    # Resolve cert_dir: expand ~ first, then resolve relative paths
    # relative paths are anchored to the project root (where main.py is run from)
    cert_dir_raw = tls_raw["cert_dir"]
    cert_dir_expanded = os.path.expanduser(cert_dir_raw)
    cert_dir_resolved = str(Path(cert_dir_expanded).resolve())
    tls = TlsConfig(
        cert_dir=cert_dir_resolved,
        cert_file=tls_raw["cert_file"],
        key_file=tls_raw["key_file"],
    )

    providers = []
    for p in raw["providers"]:
        providers.append(ProviderConfig(
            name=p["name"],
            base_url=p["base_url"].rstrip("/"),   # normalize: no trailing slash
            api_key=p["api_key"],
            protocol=p["protocol"].lower(),
            enabled=p["enabled"],
            streaming=p.get("streaming", True),   # default True: almost all providers support it
            model_map=p["model_map"],
        ))

    patcher_raw = raw["patcher"]
    patcher = PatcherConfig(
        target_url=patcher_raw["target_url"],
        ide_path=patcher_raw.get("ide_path"),     # optional field
        flag_file=patcher_raw.get("flag_file"),   # optional signal file path
    )

    # --- model_pool (optional) ---
    pool_settings: PoolSettings | None = None
    raw_model_pool: dict | None = raw.get("model_pool")
    if raw_model_pool is not None:
        ps = raw_model_pool.get("pool_settings", {})
        fb_raw = ps.get("fallback_model_limits", {})
        fallback_limits = FallbackModelLimits(
            max_output_tokens=(
                int(fb_raw["max_output_tokens"])
                if fb_raw.get("max_output_tokens") is not None
                else None
            ),
            thinking_budget=(
                int(fb_raw["thinking_budget"])
                if fb_raw.get("thinking_budget") is not None
                else None
            ),
        )
        pool_settings = PoolSettings(
            safety_buffer_tokens=int(ps.get("safety_buffer_tokens", 8192)),
            all_cooled_fallback=str(ps.get("all_cooled_fallback", "passthrough")),
            mapped_model=str(raw_model_pool.get("mapped_model", "gpt-oss-120b-medium")),
            fallback_limits=fallback_limits,
            flag_file=patcher_raw.get("flag_file"),  # threaded from patcher section
            keep_alive_timeout_minutes=int(ps.get("keep_alive_timeout_minutes", 10)),
            cooldowns_file=str(ps.get("cooldowns_file", "scratchpad/cooldowns.json")),
            keys_db=str(ps.get("keys_db", "scratchpad/keys.db")),
        )

    # --- features (optional) ---
    features_raw = raw.get("features", {})
    if not isinstance(features_raw, dict):
        features_raw = {}
    features = FeaturesConfig(
        strict_tool_contract=bool(features_raw.get("strict_tool_contract", False)),
        harmony_sanitizer=bool(features_raw.get("harmony_sanitizer", False)),
    )

    return AppConfig(
        proxy=proxy,
        upstream=upstream,
        tls=tls,
        providers=providers,
        patcher=patcher,
        pool_settings=pool_settings,
        raw_model_pool=raw_model_pool,
        features=features,
    )


# ---------------------------------------------------------------------------
# Template generator
# ---------------------------------------------------------------------------

def _generate_template(config_path: str) -> None:
    """Write a template config.json and exit with instructions."""
    path = Path(config_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_TEMPLATE, f, indent=2)
    print(
        f"\n[CONFIG] Config file not found.\n"
        f"[CONFIG] Template created at: {path.resolve()}\n"
        f"[CONFIG] Please edit it — add your API key(s) and adjust model mappings.\n"
        f"[CONFIG] Then run the proxy again.\n"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.json") -> AppConfig:
    """
    Load and validate config.json.

    Args:
        config_path: Path to the config file. Defaults to 'config.json'
                     in the current working directory.

    Returns:
        AppConfig: Fully validated, typed configuration object.

    Raises:
        ConfigError: If required fields are missing or have wrong types.
        SystemExit(1): If config file not found (template is auto-generated).
    """
    path = Path(config_path)

    # --- File not found: generate template and exit ---
    if not path.exists():
        _generate_template(config_path)

    # --- Load JSON ---
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError(
            f"Invalid JSON in {config_path}:\n"
            f"  Line {e.lineno}, Column {e.colno}: {e.msg}"
        )

    # --- Validate ---
    _validate_config(raw)

    # --- Parse into dataclasses ---
    return _parse_config(raw)
