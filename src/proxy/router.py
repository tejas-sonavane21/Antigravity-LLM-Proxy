"""
src/proxy/router.py — Request Router
======================================
Classifies every incoming IDE request and routes it to the correct handler:

  1. TELEMETRY  → mock response (200 {} or 204)
  2. PROVIDER   → model is mapped → full provider pipeline (streaming)
  3. PASSTHROUGH→ everything else → forward to Google verbatim

Classification is deliberately conservative: we only intercept AI generation
requests that have a model mapping in the provider registry. Everything else
(init, auth, model discovery, unmapped models) goes straight to Google.

Reference:
  - proxy.rs L1452-L1566 — model extraction + provider routing + telemetry
  - proxy.rs L1541-L1566 — telemetry identification and mocking
"""

import logging
from collections.abc import AsyncIterator
from enum import Enum

from src.converter.model_extractor import (
    extract_model_from_body,
    extract_model_from_path,
)
from src.provider.openai_compat import OpenAICompatProvider
from src.provider.registry import ProviderRegistry
from src.proxy.forwarder import forward_to_google

log = logging.getLogger("proxy.router")


# ---------------------------------------------------------------------------
# Request categories
# ---------------------------------------------------------------------------

class RequestCategory(Enum):
    """Classification result for an incoming request."""
    TELEMETRY = "telemetry"       # Mock and swallow
    PROVIDER = "provider"         # Route through provider pipeline
    PASSTHROUGH = "passthrough"   # Forward to Google verbatim


# ---------------------------------------------------------------------------
# Telemetry detection
# ---------------------------------------------------------------------------

# Path markers that identify telemetry/analytics endpoints.
# These are fire-and-forget — the IDE ignores the response body.
# Reference: proxy.rs L1541-L1566
_TELEMETRY_MARKERS = (
    "cascadeNuxes",
    "recordCodeAssistMetrics",
    "recordTrajectoryAnalytics",
    "fetchAdminControls",
)


def _is_telemetry(path: str) -> bool:
    """Check if the path is a telemetry/analytics endpoint."""
    if path == "/log" or path.startswith("/log?"):
        return True
    return any(marker in path for marker in _TELEMETRY_MARKERS)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_request(
    path: str,
    body: bytes,
    registry: ProviderRegistry,
) -> tuple[RequestCategory, str | None, str | None]:
    """
    Classify an incoming request.

    Args:
        path:     Full path with query string.
        body:     Raw request body bytes.
        registry: Provider registry for model lookup.

    Returns:
        ``(category, model_name_or_None, target_model_or_None)``
    """
    # 1. Telemetry — check first (cheapest)
    if _is_telemetry(path):
        return (RequestCategory.TELEMETRY, None, None)

    # 2. Extract model name from body, then from URL path
    model_name = extract_model_from_body(body) or extract_model_from_path(path)

    # 3. If model found, check provider registry
    if model_name:
        match = registry.find_provider_for_model(model_name)
        if match:
            provider, target_model = match
            return (RequestCategory.PROVIDER, model_name, target_model)

    # 4. Everything else → pass through to Google
    return (RequestCategory.PASSTHROUGH, model_name, None)


# ---------------------------------------------------------------------------
# Main routing function
# ---------------------------------------------------------------------------

async def route_request(
    method: str,
    path: str,
    headers: dict[str, str],
    body: bytes,
    registry: ProviderRegistry,
    upstream_hosts: list[str],
    include_thoughts: bool,
) -> tuple[int, dict[str, str], bytes | AsyncIterator[bytes]]:
    """
    Main routing entry point — called by the server for every request.

    Returns:
        For TELEMETRY / PASSTHROUGH:
            ``(status_code, response_headers, bytes_body)``
        For PROVIDER:
            ``(200, sse_headers, AsyncIterator[bytes])``
            The server must use StreamingResponse for the async iterator case.
    """
    category, model_name, target_model = classify_request(path, body, registry)

    # ------------------------------------------------------------------
    # 1. TELEMETRY — mock and swallow
    # ------------------------------------------------------------------
    if category is RequestCategory.TELEMETRY:
        log.debug(f"[TELEM] {method} {path}")
        if path == "/log" or path.startswith("/log?"):
            return (204, {}, b"")
        return (200, {"Content-Type": "application/json"}, b"{}")

    # ------------------------------------------------------------------
    # 2. PROVIDER — mapped model → streaming provider pipeline
    # ------------------------------------------------------------------
    if category is RequestCategory.PROVIDER:
        assert model_name is not None
        assert target_model is not None

        match = registry.find_provider_for_model(model_name)
        assert match is not None
        provider, _ = match

        log.info(
            f"[ROUTE] {method} {path} | "
            f"{model_name} → [{provider.name}] → {target_model}"
        )

        compat = OpenAICompatProvider(provider)
        sse_headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx/proxy buffering
        }
        # Return the async generator — server.py routes this to StreamingResponse
        return (
            200,
            sse_headers,
            compat.stream_request(body, target_model, include_thoughts),
        )

    # ------------------------------------------------------------------
    # 3. PASSTHROUGH — forward to Google verbatim
    # ------------------------------------------------------------------
    model_tag = f" | model={model_name}" if model_name else ""
    log.info(f"[PASS] {method} {path}{model_tag}")

    status, resp_headers, resp_body = await forward_to_google(
        method=method,
        path=path,
        headers=headers,
        body=body,
        upstream_hosts=upstream_hosts,
    )

    return (status, resp_headers, resp_body)
