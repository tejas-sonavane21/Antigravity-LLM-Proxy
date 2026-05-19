"""
src/proxy/forwarder.py — Google Upstream Forwarder
===================================================
Forwards requests to Google's cloudcode-pa API with host fallback.

Design:
  - Tries each upstream host in config order (sandbox → daily → prod)
  - On connection/timeout error → next host
  - On HTTP error (4xx/5xx) → return immediately (real error, not connectivity)
  - Preserves all original headers except hop-by-hop
  - Authorization header is passed through untouched — the IDE's own
    Google auth is the only auth needed for pass-through requests

Reference:
  - proxy.rs build_legacy_forward_targets() — 3-host fallback chain
  - proxy.rs should_skip_forward_header() L1087-L1094 — header stripping
  - constants.rs L22-L24 — TARGET_HOST_1/2/3
"""

import logging

import httpx

log = logging.getLogger("proxy.forwarder")


# ---------------------------------------------------------------------------
# Headers to strip
# ---------------------------------------------------------------------------

# Headers to STRIP from the incoming IDE request before forwarding to Google.
# These are hop-by-hop headers that must not be forwarded by proxies.
# NOTE: "authorization" is NOT stripped — we pass the IDE's original
# Google OAuth token through to Google untouched.
# Reference: proxy.rs should_skip_forward_header() L1087-L1094
_SKIP_REQUEST_HEADERS = frozenset({
    "host",                # We set this to the upstream host
    "content-length",      # httpx recalculates from body
    "connection",          # hop-by-hop
    "transfer-encoding",   # hop-by-hop
    "te",                  # hop-by-hop
    "trailers",            # hop-by-hop
    "upgrade",             # hop-by-hop
    "proxy-authorization", # proxy-specific, not for upstream
})

# Headers to STRIP from Google's response before returning to the IDE.
# httpx already decodes content-encoding; FastAPI handles chunked transfer.
_SKIP_RESPONSE_HEADERS = frozenset({
    "content-encoding",    # httpx decoded gzip/br already
    "transfer-encoding",   # chunked is handled by FastAPI/uvicorn
    "connection",          # hop-by-hop
})


# ---------------------------------------------------------------------------
# Shared HTTP client
# ---------------------------------------------------------------------------

# Single shared AsyncClient for all upstream Google calls.
# verify=True — real SSL to Google (not our proxy CA).
#
# Timeout breakdown (not a single flat value):
#   connect= 10s  — TLS + TCP handshake to Google
#   read   = 55s  — longest init endpoint (loadCodeAssist, listExperiments)
#   write  = 10s  — sending the request body
#   pool   =  5s  — time to acquire a connection from the pool
#
# keepalive_expiry = 20s — proactively close idle pooled connections.
#   Google closes idle keepalive connections at ~60-90s server-side.
#   Without this, the pool holds stale TCP sockets; the next request hangs
#   waiting for a response on a dead socket until asyncio detects the RST.
#   Setting 20s ensures we close connections before Google does.
#
# max_keepalive_connections = 10 — cap idle pool size (3 hosts × headroom)
# Timeout for non-SSE buffered requests (loadCodeAssist, listExperiments, etc.)
_TIMEOUT = httpx.Timeout(connect=10.0, read=55.0, write=10.0, pool=5.0)
# Timeout for SSE streaming pass-through (claude-sonnet, gemini, etc.).
# read= applies to each individual chunk read, NOT the total stream duration.
# We set it to 120s so a model that pauses mid-generation doesn't time out,
# while still catching truly dead connections.
_TIMEOUT_SSE = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=5.0)
_LIMITS = httpx.Limits(
    max_keepalive_connections=10,
    max_connections=20,
    keepalive_expiry=20.0,   # seconds — close idle connections proactively
)
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Lazy-init the shared httpx client with proper pool and timeout config."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            verify=True,
            timeout=_TIMEOUT,
            limits=_LIMITS,
        )
    return _client


# ---------------------------------------------------------------------------
# Header helpers
# ---------------------------------------------------------------------------

def _build_forward_headers(incoming: dict[str, str]) -> dict[str, str]:
    """
    Build headers for the upstream Google request.

    Strips hop-by-hop headers while preserving everything else
    (including Authorization, Content-Type, Accept, etc.).
    """
    return {
        k: v
        for k, v in incoming.items()
        if k.lower() not in _SKIP_REQUEST_HEADERS
    }


def _build_response_headers(upstream: httpx.Headers) -> dict[str, str]:
    """
    Build headers for the response back to the IDE.

    Strips hop-by-hop and content-encoding (already decoded by httpx).
    """
    return {
        k: v
        for k, v in upstream.items()
        if k.lower() not in _SKIP_RESPONSE_HEADERS
    }


# ---------------------------------------------------------------------------
# Main forwarding function
# ---------------------------------------------------------------------------

async def forward_to_google(
    method: str,
    path: str,
    headers: dict[str, str],
    body: bytes,
    upstream_hosts: list[str],
) -> tuple[int, dict[str, str], bytes]:
    """
    Forward a request to Google with host fallback chain.

    Tries each host in ``upstream_hosts`` order:
      - Connection error / timeout → log warning, try next host
      - HTTP error (4xx/5xx) → return immediately (it's a real API error)
      - HTTP success (2xx/3xx) → return immediately

    Args:
        method:         HTTP method (``GET``, ``POST``, etc.).
        path:           Full path including query string,
                        e.g. ``/v1internal:loadCodeAssist?key=...``.
        headers:        Raw incoming headers dict from the IDE request.
        body:           Raw request body bytes.
        upstream_hosts: Ordered list of Google hostnames from config
                        (e.g. ``["daily-cloudcode-pa.sandbox.googleapis.com", ...]``).

    Returns:
        ``(status_code, response_headers_dict, response_body_bytes)``

    Raises:
        Nothing — connection failures across all hosts return a 502 error tuple.
    """
    fwd_headers = _build_forward_headers(headers)
    client = _get_client()
    last_error: str = ""

    for host in upstream_hosts:
        target_url = f"https://{host}{path}"

        try:
            resp = await client.request(
                method=method,
                url=target_url,
                headers=fwd_headers,
                content=body,
            )
        except (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.RequestError,
            httpx.PoolTimeout,
        ) as exc:
            last_error = f"{host}: {exc}"
            log.warning(f"  → {host} | FAILED ({type(exc).__name__}: {exc})")
            continue  # Try next host

        resp_headers = _build_response_headers(resp.headers)

        log.info(
            f"  → {host} | {resp.status_code} | {len(resp.content)}B"
        )

        # Return immediately — whether success or HTTP error.
        # HTTP errors (401, 403, etc.) are real API errors, not
        # connectivity issues. Retrying a different host won't help.
        return (resp.status_code, resp_headers, resp.content)

    # All hosts exhausted — return 502
    log.error(f"  All upstream hosts failed. Last: {last_error}")
    return (
        502,
        {"Content-Type": "application/json"},
        f'{{"error":{{"message":"All upstream hosts failed: {last_error}","code":502}}}}'.encode(),
    )
# ---------------------------------------------------------------------------
# Streaming forward (SSE pass-through)
# ---------------------------------------------------------------------------

async def forward_to_google_stream(
    method: str,
    path: str,
    headers: dict[str, str],
    body: bytes,
    upstream_hosts: list[str],
):
    """
    Stream a pass-through SSE request to Google, yielding raw bytes as they
    arrive without buffering the complete body first.

    Used for `?alt=sse` pass-through requests (e.g. claude-sonnet-4-6,
    gemini-3-flash when not pool-routed) so the IDE receives tokens in
    real-time instead of waiting for the entire generation to complete.

    Yields:
        Raw bytes chunks from the Google SSE stream.

    On connection failure: yields a Gemini-format SSE error event so the IDE
    receives a well-formed response instead of a timeout.
    """
    from collections.abc import AsyncIterator
    import json as _json

    fwd_headers = _build_forward_headers(headers)
    client = _get_client()

    for host in upstream_hosts:
        target_url = f"https://{host}{path}"
        try:
            async with client.stream(
                method,
                target_url,
                headers=fwd_headers,
                content=body,
                timeout=_TIMEOUT_SSE,
            ) as resp:
                log.info(
                    f"  → {host} | {resp.status_code} | SSE stream open"
                )
                if resp.status_code >= 400:
                    # HTTP error — read body and return an SSE error event
                    err_body = await resp.aread()
                    log.warning(
                        f"  → {host} | {resp.status_code} | "
                        f"SSE pass-through error: {err_body[:200]}"
                    )
                    err_event = _json.dumps({
                        "response": {
                            "candidates": [{
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": (
                                        f"[Proxy] Upstream error {resp.status_code}: "
                                        f"{err_body[:200].decode('utf-8', errors='replace')}"
                                    )}]
                                },
                                "finishReason": "STOP",
                            }],
                        }
                    })
                    yield f"data: {err_event}\n\n".encode("utf-8")
                    return

                # Stream bytes as they arrive
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk
                return  # done — don't try next host

        except (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.RequestError,
            httpx.PoolTimeout,
        ) as exc:
            log.warning(f"  → {host} | FAILED ({type(exc).__name__}: {exc})")
            continue  # try next host

    # All hosts exhausted
    log.error("  All upstream hosts failed for SSE stream.")
    err_event = _json.dumps({
        "response": {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "[Proxy] All upstream hosts failed."}]
                },
                "finishReason": "STOP",
            }],
        }
    })
    yield f"data: {err_event}\n\n".encode("utf-8")
