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

    # --- Phase 4 Step 0: Enhanced Traffic-Capture Stub Server ---
    # This stub categorises every IDE request, mocks telemetry with empty JSON,
    # and dumps full headers + decoded body for any non-telemetry requests so we
    # can discover the exact AI endpoint path and internal model name strings.
    # Phase 6 replaces this with the real router.
    log.info("-" * 60)
    log.info("Starting enhanced stub HTTPS server (Phase 4 traffic capture)...")
    log.info(f"Listening on https://{config.proxy.host}:{config.proxy.port}")
    log.info("Press Ctrl+C to stop.")
    log.info("-" * 60)

    try:
        import json as _json
        import uvicorn
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse

        app = FastAPI()

        # ---------------------------------------------------------------
        # Telemetry endpoints — mock with empty JSON (no upstream needed).
        # Identical behaviour to reference project proxy.rs L1541-L1566.
        # ---------------------------------------------------------------
        _TELEMETRY_PATHS = (
            "cascadeNuxes",
            "recordCodeAssistMetrics",
            "recordTrajectoryAnalytics",
            "fetchAdminControls",
            "/log",
        )

        # Auxiliary init endpoints — mock so the IDE doesn't stall.
        _AUX_PATHS = (
            "loadCodeAssist",
            "fetchUserInfo",
            "fetchAvailableModels",
        )

        @app.api_route(
            "/{path:path}",
            methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
        )
        async def catch_all(request: Request, path: str):
            body_bytes = await request.body()
            content_type = request.headers.get("content-type", "")
            stub_log = logging.getLogger("proxy.stub")

            full_path = f"/{path}"
            if request.url.query:
                full_path += f"?{request.url.query}"

            # --- Classify ---
            is_telemetry = any(ep in full_path for ep in _TELEMETRY_PATHS)
            is_aux = any(ep in full_path for ep in _AUX_PATHS)

            tag = "[TELEM]" if is_telemetry else "[AUX  ]" if is_aux else "[AI?  ]"

            # Always log the one-liner
            stub_log.info(
                f"{tag} {request.method} {full_path} | {len(body_bytes)}B | {content_type}"
            )

            # For non-telemetry: dump headers + body for analysis
            if not is_telemetry:
                stub_log.debug(f"  Headers : {dict(request.headers)}")

                if body_bytes:
                    # Try JSON parse first (readable output)
                    try:
                        body_obj = _json.loads(body_bytes)
                        pretty = _json.dumps(body_obj, indent=2, ensure_ascii=False)
                        # Cap at 3000 chars to avoid log flood on large requests
                        stub_log.info(
                            f"  Body (JSON):\n{pretty[:3000]}"
                            + (" ...[truncated]" if len(pretty) > 3000 else "")
                        )
                    except (_json.JSONDecodeError, UnicodeDecodeError):
                        # Binary / protobuf — hex dump first 200 bytes
                        stub_log.info(
                            f"  Body (binary/proto, first 200B hex):\n  {body_bytes[:200].hex()}"
                        )

            # --- Mock responses ---
            if is_telemetry:
                # Swallow /log with 204; everything else gets empty 200 JSON
                if full_path.startswith("/log"):
                    from starlette.responses import Response as _Resp
                    return _Resp(status_code=204)
                return JSONResponse(content={}, status_code=200)

            if is_aux:
                return JSONResponse(content={}, status_code=200)

            # Unknown / AI endpoint — return empty stub JSON (IDE will fail gracefully)
            stub_log.info(
                f"  [AI?  ] Returning empty stub JSON. "
                f"Implement real handler in Phase 6."
            )
            return JSONResponse(
                content={"candidates": [], "modelVersion": "stub"},
                status_code=200,
            )

        uvicorn.run(
            app,
            host=config.proxy.host,
            port=config.proxy.port,
            ssl_certfile=cert_path,
            ssl_keyfile=key_path,
            log_level="warning",  # suppress uvicorn's own access logs
        )
    except KeyboardInterrupt:
        log.info("Proxy stopped by user (Ctrl+C).")


if __name__ == "__main__":
    main()
