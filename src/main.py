"""
src/main.py — Entry Point
==========================
Phase 1: Loads config, sets up structured logging, prints config summary.
Phase 2: Initializes TLS certificate manager, ensures certs exist.
Phase 6: Will start the HTTPS proxy server (uvicorn + FastAPI).

Reference: Brainstorming Section 11.1, 11.2 (logging requirements)
Reference: Brainstorming Section 9.2 (TLS/HTTPS handling)
"""

import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path so 'from src.x import y' works
# when running as: python src/main.py from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig, ConfigError, load_config
from src.tls.cert_manager import CertManager


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def setup_logging(log_level: str) -> None:
    """
    Configure the root logger with structured output.

    Format: [YYYY-MM-DD HH:MM:SS] [LEVEL   ] [module] message

    The log_level string (e.g. "DEBUG", "INFO") is mapped to Python
    logging constants. All sub-module loggers (proxy, converter, etc.)
    inherit from the root logger configured here.
    """
    level_map = {
        "DEBUG":   logging.DEBUG,
        "INFO":    logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR":   logging.ERROR,
    }
    level = level_map.get(log_level.upper(), logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    # Avoid duplicate handlers if setup_logging() is called more than once
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


# ---------------------------------------------------------------------------
# Config Summary Logger
# ---------------------------------------------------------------------------

def _mask_api_key(key: str) -> str:
    """
    Mask an API key for safe console display.
    Shows first 4 and last 4 characters. Fully masks short keys.

    Examples:
        "zen_abc123xyz789"  ->  "zen_...789"
        "short"             ->  "****"
    """
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


def log_config_summary(config: AppConfig) -> None:
    """
    Log a human-readable summary of the loaded configuration.
    API keys are masked — never logged in plaintext.
    """
    log = logging.getLogger("main")

    log.debug(f"Proxy server : {config.proxy.host}:{config.proxy.port}")
    log.debug(f"Log level    : {config.proxy.log_level}")
    log.debug(f"Incl. thoughts: {config.proxy.include_thoughts}")
    log.debug(f"Upstream hosts: {len(config.upstream.hosts)} configured")

    for host in config.upstream.hosts:
        log.debug(f"  -> {host}")

    log.debug(
        f"TLS cert dir : {config.tls.cert_dir}"
    )
    log.debug(f"  cert: {config.tls.cert_file}")
    log.debug(f"  key : {config.tls.key_file}")

    enabled_providers = [p for p in config.providers if p.enabled]
    disabled_providers = [p for p in config.providers if not p.enabled]

    log.info(
        f"Providers: {len(config.providers)} total — "
        f"{len(enabled_providers)} enabled, {len(disabled_providers)} disabled"
    )

    for p in config.providers:
        status = "ENABLED" if p.enabled else "DISABLED"
        masked_key = _mask_api_key(p.api_key)
        log.info(
            f"  [{status}] {p.name} | protocol={p.protocol} | "
            f"key={masked_key} | {len(p.model_map)} model mapping(s)"
        )
        for ide_model, ext_model in p.model_map.items():
            log.debug(f"    {ide_model}  ->  {ext_model}")

    log.debug(f"Patcher target URL: {config.patcher.target_url}")
    if config.patcher.ide_path:
        log.debug(f"Patcher IDE path override: {config.patcher.ide_path}")

    # Pool summary
    if config.pool_settings is not None:
        ps = config.pool_settings
        log.info(
            f"Model pool: mapped_model={ps.mapped_model!r} | "
            f"fallback={ps.all_cooled_fallback!r} | "
            f"cooldowns_file={ps.cooldowns_file!r}"
        )
    else:
        log.info("Model pool: NOT configured (pool routing disabled)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log = logging.getLogger("main")

    # --- Bootstrap: minimal logging before config is loaded ---
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log.info("=" * 60)
    log.info("  Antigravity Model API Extension")
    log.info("=" * 60)
    log.info("Loading configuration...")

    # --- Load config ---
    # load_config() will sys.exit(1) if file not found (template is generated)
    # ConfigError is raised for structural/validation issues
    try:
        config = load_config("config.json")
    except ConfigError as e:
        log.error(f"Configuration error: {e}")
        log.error("Fix the issue in config.json and restart.")
        sys.exit(1)

    # --- Re-configure logging with the level from config ---
    setup_logging(config.proxy.log_level)

    # Re-get logger after reconfiguration
    log = logging.getLogger("main")

    log.info("Config loaded successfully.")
    log_config_summary(config)

    # --- Phase 2: TLS Certificate Manager ---
    log.info("-" * 60)
    log.info("Initializing TLS certificate manager...")
    try:
        cert_manager = CertManager(config.tls)
        cert_path, key_path = cert_manager.ensure_certs()
        log.info(f"TLS certificate ready: {cert_path}")

        # Verify SSL context can be created — catches corrupt cert/key early
        ssl_context = cert_manager.get_ssl_context()
        log.info("SSL context created successfully")
    except (RuntimeError, OSError) as e:
        log.error(f"TLS initialization failed: {e}")
        log.error("Delete the ag_proxy/ folder and restart to regenerate certificates.")
        sys.exit(1)

    # --- Phase 5: Provider Registry ---
    from src.provider.registry import ProviderRegistry

    log.info("-" * 60)
    log.info("Initializing provider registry...")
    registry = ProviderRegistry()
    registry.load_from_config(config.providers)

    # --- Phase 5b: Configure router feature flags ---
    from src.proxy import router as _router_module
    _router_module.configure(
        dump_model_responses=config.proxy.dump_model_responses,
    )

    # --- Pool: Initialize PoolPicker ---
    if config.pool_settings is not None and config.raw_model_pool is not None:
        from src.pool.config_parser import parse_pool_entries
        from src.pool.picker import init_picker

        log.info("-" * 60)
        log.info("Initializing model key pool...")
        try:
            pool_entries, _ = parse_pool_entries(
                config.raw_model_pool,
                cooldowns_path=config.pool_settings.cooldowns_file,
            )

            picker = init_picker(
                entries=pool_entries,
                pool_settings=config.pool_settings,
            )
            log.info(
                f"Pool ready: {len(pool_entries)} entries | "
                f"fallback={config.pool_settings.all_cooled_fallback!r}"
            )
            # Log each entry's initial state at DEBUG
            for s in picker.status_summary():
                cooled = f" [COOLED until {s['cooldown_until']}]" if s["is_cooled"] else ""
                log.debug(
                    f"  [{s['id']}] {s['label']}{cooled} | "
                    f"usable={s['usable_tokens']} | "
                    f"thinking_field={s['response_thinking_field']!r}"
                )
        except (ValueError, KeyError) as exc:
            log.error(f"Pool initialization failed: {exc}")
            log.error("Check model_pool.providers in config.json.")
            sys.exit(1)

        # --- Initialize on-demand refresh flag file ---
        from src.pool.trigger import init_flag_file
        init_flag_file(config.pool_settings.flag_file)

    else:
        log.info("Model pool not configured — pool routing disabled.")

    # --- Phase 6: Create and start proxy server ---
    from src.proxy.server import create_app

    log.info("-" * 60)
    log.info("Starting proxy server...")
    log.info(f"Listening on https://{config.proxy.host}:{config.proxy.port}")
    log.info("Press Ctrl+C to stop.")
    log.info("-" * 60)

    app = create_app(
        registry=registry,
        upstream_hosts=config.upstream.hosts,
        include_thoughts=config.proxy.include_thoughts,
    )

    import uvicorn
    try:
        uvicorn.run(
            app,
            host=config.proxy.host,
            port=config.proxy.port,
            ssl_certfile=str(cert_path),
            ssl_keyfile=str(key_path),
            log_level="warning",
        )
    except KeyboardInterrupt:
        pass

    log.info("Proxy stopped.")


if __name__ == "__main__":
    main()
