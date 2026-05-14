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

    # --- Phase 5: Provider Registry ---
    from src.provider.registry import ProviderRegistry

    log.info("-" * 60)
    log.info("Initializing provider registry...")
    registry = ProviderRegistry()
    registry.load_from_config(config.providers)

    # --- Phase 5: Proxy Server (provider routing enabled) ---
    log.info("-" * 60)
    log.info("Starting proxy server (Phase 5 — provider routing enabled)...")
    log.info(f"Listening on https://{config.proxy.host}:{config.proxy.port}")
    log.info("Press Ctrl+C to stop.")
    log.info("-" * 60)

    try:
        import json as _json
        import httpx
        import uvicorn
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, Response as _FastResp

        app = FastAPI()

        # ---------------------------------------------------------------
        # Pure telemetry — fire-and-forget analytics. IDE ignores response.
        # Safe to swallow. Identical to reference proxy.rs L1541-L1566.
        # ---------------------------------------------------------------
        _TELEMETRY_PATHS = (
            "cascadeNuxes",
            "recordCodeAssistMetrics",
            "recordTrajectoryAnalytics",
            "fetchAdminControls",
            "/log",
        )

        # Google upstream for pass-through forwarding.
        _UPSTREAM_HOST = (
            config.upstream.hosts[0]
            if config.upstream.hosts
            else "cloudcode-pa.googleapis.com"
        )
        _UPSTREAM_BASE = f"https://{_UPSTREAM_HOST}"

        # Shared async HTTP client — real SSL to Google (not our proxy CA).
        # Suppress httpcore debug noise (connect/TLS handshake spam)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        _http = httpx.AsyncClient(verify=True, timeout=30.0)

        async def _forward(request: Request, full_path: str) -> _FastResp:
            """Forward request verbatim to Google, log and return the raw response."""
            stub_log = logging.getLogger("proxy.stub")
            target_url = f"{_UPSTREAM_BASE}{full_path}"

            # Strip hop-by-hop headers before forwarding
            skip = {"host", "content-length", "connection", "transfer-encoding",
                    "te", "trailers", "upgrade", "proxy-authorization"}
            fwd_headers = {k: v for k, v in request.headers.items()
                           if k.lower() not in skip}

            body_bytes = await request.body()
            try:
                resp = await _http.request(
                    method=request.method,
                    url=target_url,
                    headers=fwd_headers,
                    content=body_bytes,
                )
                stub_log.info(
                    f"  [FWD ] -> {target_url} | "
                    f"status={resp.status_code} | {len(resp.content)}B"
                )
                # Log response for traffic analysis
                if resp.content:
                    try:
                        rj = _json.loads(resp.content)
                        stub_log.info(
                            f"  [FWD ] Response (JSON):\n"
                            f"{_json.dumps(rj, indent=2)[:8000]}"
                            + (" ...[TRUNCATED - increase cap to see more]"
                               if len(_json.dumps(rj)) > 8000 else "")
                        )
                    except Exception:
                        stub_log.info(
                            f"  [FWD ] Response (binary, first 500B hex):\n"
                            f"  {resp.content[:500].hex()}"
                        )

                # Pass response back to IDE
                passthrough_headers = {
                    k: v for k, v in resp.headers.items()
                    if k.lower() not in ("content-encoding", "transfer-encoding",
                                         "connection")
                }
                return _FastResp(
                    content=resp.content,
                    status_code=resp.status_code,
                    headers=passthrough_headers,
                )
            except Exception as exc:
                stub_log.error(f"  [FWD ] Forward failed for {target_url}: {exc}")
                return JSONResponse(content={"error": str(exc)}, status_code=502)

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
            tag = "[TELEM]" if is_telemetry else "[FWD  ]"

            stub_log.info(
                f"{tag} {request.method} {full_path} | {len(body_bytes)}B | {content_type}"
            )

            # Log request body for all non-telemetry requests
            if not is_telemetry and body_bytes:
                try:
                    body_obj = _json.loads(body_bytes)
                    pretty = _json.dumps(body_obj, indent=2, ensure_ascii=False)
                    stub_log.info(
                        f"  Req Body (JSON):\n{pretty[:2000]}"
                        + (" ...[truncated]" if len(pretty) > 2000 else "")
                    )
                except (_json.JSONDecodeError, UnicodeDecodeError):
                    stub_log.info(
                        f"  Req Body (binary, first 200B hex):\n  {body_bytes[:200].hex()}"
                    )

            # --- Pure telemetry: swallow entirely ---
            if is_telemetry:
                if full_path.startswith("/log"):
                    return _FastResp(status_code=204)
                return JSONResponse(content={}, status_code=200)

            # --- Phase 5: Provider routing ---
            # Try to extract model name from body first, then from URL path.
            # If a mapping exists → run full provider pipeline.
            # If no mapping → fall through to Google pass-through.
            from src.converter.model_extractor import (
                extract_model_from_body,
                extract_model_from_path,
            )
            from src.provider.openai_compat import OpenAICompatProvider

            model_name = (
                extract_model_from_body(body_bytes)
                or extract_model_from_path(full_path)
            )

            if model_name:
                match = registry.find_provider_for_model(model_name)
                if match:
                    provider, target_model = match
                    stub_log.info(
                        f"  [ROUTE] [{model_name}] → "
                        f"provider [{provider.name}] → [{target_model}]"
                    )
                    compat = OpenAICompatProvider(provider)
                    status, resp_headers, sse_body = await compat.forward_request(
                        body_bytes,
                        target_model,
                        config.proxy.include_thoughts,
                    )
                    return _FastResp(
                        content=sse_body.encode("utf-8"),
                        status_code=status,
                        headers=resp_headers,
                    )

            # --- No match: forward to Google verbatim ---
            # Covers: loadCodeAssist, fetchAvailableModels, onboardUser,
            # fetchUserInfo, unmapped AI models, and all other endpoints.
            return await _forward(request, full_path)

        uvicorn.run(
            app,
            host=config.proxy.host,
            port=config.proxy.port,
            ssl_certfile=cert_path,
            ssl_keyfile=key_path,
            log_level="warning",
        )
    except KeyboardInterrupt:
        log.info("Proxy stopped by user (Ctrl+C).")


if __name__ == "__main__":
    main()

