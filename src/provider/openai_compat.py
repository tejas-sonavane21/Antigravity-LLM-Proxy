"""
src/provider/openai_compat.py — OpenAI-Compatible Provider
===========================================================
Implements the full request-forwarding pipeline for providers that
speak the OpenAI chat completions API:

  Gemini JSON (IDE)
      │
      ▼  gemini_to_openai.convert_request()
  OpenAI JSON
      │
      ▼  POST {base_url}/chat/completions  (httpx, stream=True)
  OpenAI response body  ──►  collected then converted
      │
      ▼  openai_to_gemini.convert_response()
  Gemini SSE string  ──►  yielded immediately to IDE

Port of: provider.rs forward_to_provider() L499-L700

Streaming design:
  We use httpx.stream() so the HTTP connection to the provider stays
  alive and delivers its response headers immediately without blocking
  the asyncio event loop on the full body read. This eliminates the
  silent-freeze bug where the IDE chat showed no activity for 30+
  seconds and required Ctrl+C to unblock.

  We still collect the full body before converting (the provider is
  called with stream=False in the JSON payload). The key improvement is
  that the *connection* itself is non-blocking — keepalive management
  and header receipt happen without stalling the event loop.

Retry logic (proxy.rs L1467-L1537):
  - MAX_RETRIES = 3
  - Backoff: 0.5 × attempt seconds
  - Retry only on transient errors: 429, 500, 502, 503, 504
  - Permanent errors (4xx except 429): no retry, return error SSE

Error format:
  Errors are wrapped in Gemini SSE so the IDE shows them as chat text.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator

import httpx

from .base import Provider
from src.converter.gemini_to_openai import convert_request
from src.converter.openai_to_gemini import convert_response


# ---------------------------------------------------------------------------
# Shared HTTP client for provider requests
# ---------------------------------------------------------------------------
#
# Per-phase timeout breakdown:
#   connect =  15s  — TLS + TCP handshake to external provider
#   read    = 300s  — AI responses can take minutes; keep generous
#   write   =  15s  — sending the (potentially large) request body
#   pool    =   5s  — time to acquire a free connection from the pool
#
# keepalive_expiry = 20s — proactively close idle connections.
#   External providers close idle HTTP/1.1 keepalive connections at
#   varying intervals. Without this, pooled sockets go stale and the
#   next request hangs silently until asyncio detects the RST.
#
_PROVIDER_TIMEOUT = httpx.Timeout(connect=15.0, read=300.0, write=15.0, pool=5.0)
_PROVIDER_LIMITS  = httpx.Limits(
    max_keepalive_connections=5,
    max_connections=10,
    keepalive_expiry=20.0,
)
_provider_client: httpx.AsyncClient | None = None


def _get_provider_client() -> httpx.AsyncClient:
    """Lazy-init the shared provider httpx client."""
    global _provider_client
    if _provider_client is None or _provider_client.is_closed:
        _provider_client = httpx.AsyncClient(
            timeout=_PROVIDER_TIMEOUT,
            limits=_PROVIDER_LIMITS,
        )
    return _provider_client


# Transient HTTP status codes that warrant a retry.
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}

# Maximum number of attempts (1 original + 2 retries = 3 total).
_MAX_RETRIES = 3

# Backoff: attempt=2 → 0.5s, attempt=3 → 1.0s
_BACKOFF_MULTIPLIER = 0.5


class OpenAICompatProvider:
    """
    Handles the complete forward pipeline for OpenAI-compatible providers.

    Usage::

        compat = OpenAICompatProvider(provider)
        async for chunk in compat.stream_request(body_bytes, target_model):
            # chunk is bytes — forward directly to IDE
            ...

    Port of: provider.rs forward_to_provider() L499-L700
    """

    def __init__(self, provider: Provider) -> None:
        self._provider = provider
        self._log = logging.getLogger(f"provider.{provider.name}")

    # ------------------------------------------------------------------
    # Public API — async streaming generator
    # ------------------------------------------------------------------

    async def stream_request(
        self,
        body_bytes: bytes,
        target_model: str,
        include_thoughts: bool = False,
    ) -> AsyncIterator[bytes]:
        """
        Full forwarding pipeline as an async generator of bytes chunks.

        Yields the Gemini SSE response as bytes the moment it is ready.
        The IDE receives data immediately — no silent freeze.

        Yields:
            bytes: UTF-8 encoded SSE chunks (``b"data: {json}\\n\\n"``)

        On error: yields a single error SSE chunk and stops.
        """
        # Step 1: Convert Gemini → OpenAI
        try:
            openai_req = convert_request(body_bytes, target_model, include_thoughts)
        except ValueError as exc:
            self._log.error(f"Request conversion failed: {exc}")
            yield self._error_sse_bytes(400, f"Request conversion error: {exc}")
            return

        # Force non-streaming payload (we handle the connection as streaming)
        openai_req["stream"] = False

        # Step 2: Build target URL
        base = self._provider.base_url.rstrip("/")
        protocol = self._provider.protocol.lower()
        if protocol == "openai":
            target_url = f"{base}/chat/completions"
        elif protocol == "gemini":
            target_url = f"{base}/v1beta/models/{target_model}:streamGenerateContent?alt=sse"
        elif protocol == "claude":
            target_url = f"{base}/v1/messages"
        else:
            target_url = f"{base}/chat/completions"

        self._log.info(
            f"Forwarding [{self._provider.name}] "
            f"{target_model!r} → {target_url}"
        )

        # Step 3: Build request headers
        req_headers: dict[str, str] = {"Content-Type": "application/json"}
        if protocol == "claude":
            req_headers["x-api-key"] = self._provider.api_key
            req_headers["anthropic-version"] = "2023-06-01"
        else:
            req_headers["Authorization"] = f"Bearer {self._provider.api_key}"

        # Step 4: Send with retry using httpx streaming transport.
        # client.stream() opens the connection and receives headers immediately
        # without blocking on the full body — this is the key fix.
        last_error: str = ""
        client = _get_provider_client()
        resp_bytes: bytes | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            if attempt > 1:
                backoff = _BACKOFF_MULTIPLIER * attempt
                self._log.warning(
                    f"  Retry {attempt}/{_MAX_RETRIES} "
                    f"[{self._provider.name}], backoff={backoff:.1f}s"
                )
                await asyncio.sleep(backoff)

            try:
                async with client.stream(
                    "POST",
                    target_url,
                    json=openai_req,
                    headers=req_headers,
                ) as resp:

                    if resp.status_code >= 400:
                        err_body = await resp.aread()
                        err_preview = err_body.decode("utf-8", errors="replace")[:500]
                        last_error = err_preview

                        if resp.status_code in _RETRYABLE_STATUSES:
                            self._log.warning(
                                f"  [{self._provider.name}] attempt {attempt}/{_MAX_RETRIES}: "
                                f"transient {resp.status_code}: {err_preview[:200]}"
                            )
                            continue  # retry
                        else:
                            self._log.error(
                                f"  [{self._provider.name}] permanent error "
                                f"{resp.status_code}: {err_preview[:200]}"
                            )
                            yield self._error_sse_bytes(
                                resp.status_code,
                                f"Provider error {resp.status_code}: {err_preview}",
                            )
                            return

                    # Success — read full body inside the stream context
                    resp_bytes = await resp.aread()
                    self._log.info(
                        f"  [{self._provider.name}] Response: "
                        f"status={resp.status_code} | {len(resp_bytes)}B"
                    )
                    break  # exit retry loop

            except httpx.TimeoutException as exc:
                last_error = f"Request timed out: {exc}"
                self._log.error(
                    f"  [{self._provider.name}] attempt {attempt}/{_MAX_RETRIES}: timeout"
                )
                continue

            except httpx.RequestError as exc:
                last_error = f"Connection error: {exc}"
                self._log.error(
                    f"  [{self._provider.name}] attempt {attempt}/{_MAX_RETRIES}: "
                    f"connection error: {exc}"
                )
                continue

        else:
            # All retries exhausted
            self._log.error(
                f"  [{self._provider.name}] all {_MAX_RETRIES} attempts failed. "
                f"Last: {last_error[:300]}"
            )
            yield self._error_sse_bytes(
                502,
                f"Provider [{self._provider.name}] failed after "
                f"{_MAX_RETRIES} retries: {last_error[:300]}",
            )
            return

        if resp_bytes is None:
            yield self._error_sse_bytes(502, "No response received from provider")
            return

        # Step 5: Parse JSON response
        try:
            openai_resp = json.loads(resp_bytes)
        except Exception as exc:
            raw_preview = resp_bytes.decode("utf-8", errors="replace")[:500]
            self._log.error(
                f"  [{self._provider.name}] Failed to parse response JSON: {exc}\n"
                f"  Raw: {raw_preview}"
            )
            yield self._error_sse_bytes(502, f"Provider response is not valid JSON: {exc}")
            return

        self._log.debug(
            f"  [{self._provider.name}] OpenAI response (first 1000 chars):\n"
            f"  {json.dumps(openai_resp)[:1000]}"
        )

        # Step 6: Convert OpenAI → Gemini SSE
        try:
            sse_body = convert_response(openai_resp, target_model)
        except Exception as exc:
            self._log.error(
                f"  [{self._provider.name}] Response conversion failed: {exc}"
            )
            yield self._error_sse_bytes(502, f"Response conversion error: {exc}")
            return

        self._log.info(
            f"  [{self._provider.name}] SSE ready: {len(sse_body)} chars → IDE"
        )

        # Yield the complete SSE event — IDE receives it the moment we yield
        yield sse_body.encode("utf-8")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _error_sse_bytes(self, status: int, message: str) -> bytes:
        """
        Wrap a provider error in Gemini SSE format (as bytes).
        The IDE displays this as a chat error message.
        """
        self._log.error(
            f"  [{self._provider.name}] Returning error SSE "
            f"[{status}]: {message[:200]}"
        )
        gemini_error: dict = {
            "response": {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": f"[Provider Error {status}]: {message[:500]}"}
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "modelVersion": "error",
                "responseId": "",
            }
        }
        sse = f"data: {json.dumps(gemini_error, ensure_ascii=False)}\n\n"
        return sse.encode("utf-8")
