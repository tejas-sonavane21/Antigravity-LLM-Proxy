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
    ide_path: str | None = None  # optional override for IDE install location


@dataclass
class AppConfig:
    proxy: ProxyConfig
    upstream: UpstreamConfig
    tls: TlsConfig
    providers: list         # list[ProviderConfig]
    patcher: PatcherConfig


# ---------------------------------------------------------------------------
# Template — written to disk if config.json is missing
# ---------------------------------------------------------------------------

_TEMPLATE = {
    "proxy": {
        "host": "127.0.0.1",
        "port": 9527,
        "log_level": "DEBUG",
        "include_thoughts": False
    },
    "upstream": {
        "hosts": [
            "daily-cloudcode-pa.sandbox.googleapis.com",
            "daily-cloudcode-pa.googleapis.com",
            "cloudcode-pa.googleapis.com"
        ]
    },
    "tls": {
        "cert_dir": "./ag_proxy",
        "cert_file": "proxy-ca.crt",
        "key_file": "proxy-ca.key"
    },
    "providers": [
        {
            "name": "opencode-zen",
            "base_url": "https://opencode.ai/zen/v1",
            "api_key": "YOUR_API_KEY_HERE",
            "protocol": "openai",
            "enabled": True,
            "streaming": True,
            "model_map": {
                "gpt-oss-120b-medium": "minimax-m2.5-free"
            }
        }
    ],
    "patcher": {
        "target_url": "https://127.0.0.1:9527",
        "ide_path": "C:\\Path\\To\\Antigravity\\resources\\app\\out"
    }
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
    )

    return AppConfig(
        proxy=proxy,
        upstream=upstream,
        tls=tls,
        providers=providers,
        patcher=patcher,
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
