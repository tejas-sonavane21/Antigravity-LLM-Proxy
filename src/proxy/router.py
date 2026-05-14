"""
src/proxy/router.py — Request Router
======================================
Classifies every incoming IDE request and routes it to the correct handler:

  1. TELEMETRY  → mock response (200 {} or 204)
  2. PROVIDER   → model is mapped → full provider pipeline (Phase 5)
  3. PASSTHROUGH→ everything else → forward to Google verbatim

Classification is deliberately conservative: we only intercept AI generation
requests that have a model mapping in the provider registry. Everything else
(init, auth, model discovery, unmapped models) goes straight to Google.

Reference:
  - proxy.rs L1452-L1566 — model extraction + provider routing + telemetry
  - proxy.rs L1541-L1566 — telemetry identification and mocking
"""

import json
import logging
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
    # /log and /log?... are pure telemetry uploads
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

        - TELEMETRY: ``("telemetry", None, None)``
        - PROVIDER:  ``("provider", "gpt-oss-120b-medium", "minimax-m2.5-free")``
        - PASSTHROUGH: ``("passthrough", "some-model" or None, None)``
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
) -> tuple[int, dict[str, str], bytes]:
    """
    Main routing entry point — called by the server for every request.

    Decision flow:
      1. Telemetry? → mock response
      2. Extract model from body/path
      3. Model mapped in registry? → provider pipeline
      4. Otherwise → forward to Google

    Args:
        method:           HTTP method (GET, POST, etc.).
        path:             Full path with query string.
        headers:          Raw incoming headers dict.
        body:             Raw request body bytes.
        registry:         Provider registry (Phase 5).
        upstream_hosts:   Ordered list of Google hostnames.
        include_thoughts: Whether to include thought parts in conversion.

    Returns:
        ``(status_code, response_headers, response_body_bytes)``
    """
    category, model_name, target_model = classify_request(path, body, registry)

    # ------------------------------------------------------------------
    # 1. TELEMETRY — mock and swallow
    # ------------------------------------------------------------------
    if category is RequestCategory.TELEMETRY:
        log.debug(f"[TELEM] {method} {path}")

        # /log endpoint returns 204 No Content (proxy.rs L1554-L1559)
        if path == "/log" or path.startswith("/log?"):
            return (204, {}, b"")

        # All other telemetry returns 200 {} (proxy.rs L1561-L1565)
        return (200, {"Content-Type": "application/json"}, b"{}")

    # ------------------------------------------------------------------
    # 2. PROVIDER — mapped model → full provider pipeline
    # ------------------------------------------------------------------
    if category is RequestCategory.PROVIDER:
        # At this point model_name and target_model are guaranteed non-None
        # (set by classify_request when PROVIDER is returned)
        assert model_name is not None
        assert target_model is not None

        # Find the provider again to get the full Provider object
        match = registry.find_provider_for_model(model_name)
        assert match is not None  # classify_request already verified this
        provider, _ = match

        log.info(
            f"[ROUTE] {method} {path} | "
            f"{model_name} → [{provider.name}] → {target_model}"
        )

        compat = OpenAICompatProvider(provider)
        status, resp_headers, sse_body = await compat.forward_request(
            body, target_model, include_thoughts
        )

        return (status, resp_headers, sse_body.encode("utf-8"))

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
