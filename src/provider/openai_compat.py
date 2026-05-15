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
      ▼  POST {base_url}/chat/completions  (httpx, Bearer auth)
  OpenAI JSON response
      │
      ▼  openai_to_gemini.convert_response()
  Gemini SSE string
      │
      ▼  returned to proxy / IDE

Port of: provider.rs forward_to_provider() L499-L700

Retry logic (proxy.rs L1467-L1537):
  - MAX_RETRIES = 3
  - Backoff: 0.5 × attempt seconds (attempt 1 → 0.5s, attempt 2 → 1.0s)
  - Retry only on transient errors: 429, 500, 502, 503, 504
  - Permanent errors (4xx except 429): no retry, return error SSE immediately

Error format:
  Provider errors are wrapped in Gemini SSE format so the IDE can display
  them as a chat message rather than a cryptic HTTP failure:
    data: {"response":{"candidates":[{"content":{"role":"model",
           "parts":[{"text":"[Provider Error 401]: Unauthorized"}]},
           "finishReason":"STOP"}],"modelVersion":"error"}}\n\n
"""

import asyncio
import json
import logging

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
#   External providers close idle HTTP/1.1 keepalive connections at varying
#   intervals. Without this, pooled sockets go stale and the next request
#   hangs silently until asyncio detects the RST (the 'silent freeze' bug).
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


# External API timeout in seconds (kept for reference; actual timeout is above).
_REQUEST_TIMEOUT = 300.0

# Transient HTTP status codes that warrant a retry attempt.
# Permanent errors (400, 401, 403, 404, etc.) are not retried.
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}

# Maximum number of attempts (1 original + 2 retries = 3 total).
# Reference: proxy.rs L1467
_MAX_RETRIES = 3

# Backoff multiplier in seconds per attempt.
# attempt=2 → 0.5s, attempt=3 → 1.0s  (proxy.rs L1471: 500ms × attempt)
_BACKOFF_MULTIPLIER = 0.5


class OpenAICompatProvider:
    """
    Handles the complete forward pipeline for OpenAI-compatible providers.

    Usage::

        compat = OpenAICompatProvider(provider)
        status, headers, body = await compat.forward_request(
            body_bytes, target_model, include_thoughts=False
        )

    Port of: provider.rs forward_to_provider() L499-L700
    """

    def __init__(self, provider: Provider) -> None:
        self._provider = provider
        self._log = logging.getLogger(f"provider.{provider.name}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def forward_request(
        self,
        body_bytes: bytes,
        target_model: str,
        include_thoughts: bool = False,
    ) -> tuple[int, dict, str]:
        """
        Full forwarding pipeline: Gemini request → OpenAI → Gemini SSE.

        Args:
            body_bytes:       Raw bytes of the IDE's Gemini-format request body.
            target_model:     External model name (value from model_map),
                              e.g. "minimax-m2.5-free".
            include_thoughts: If True, thought-tagged parts are included in the
                              converted request (usually False to save tokens).

        Returns:
            A tuple ``(status_code, response_headers, body_string)`` where:

            - ``status_code`` is always 200 (errors are wrapped in SSE).
            - ``response_headers`` contains ``Content-Type: text/event-stream``.
            - ``body_string`` is the Gemini SSE string:
              ``"data: {json}\\n\\n"``

        Port of: provider.rs forward_to_provider() L499-L700
        """
        # --- Step 1: Convert request Gemini → OpenAI ---
        try:
            openai_req = convert_request(body_bytes, target_model, include_thoughts)
        except ValueError as exc:
            self._log.error(f"Request conversion failed: {exc}")
            return self._error_sse(400, f"Request conversion error: {exc}")

        # Ensure non-streaming (stream: false) — reference project L194
        openai_req["stream"] = False

        # --- Step 2: Build target URL ---
        base = self._provider.base_url.rstrip("/")
        # Protocol dispatch (provider.rs L508-L516) — v1 only supports "openai"
        protocol = self._provider.protocol.lower()
        if protocol == "openai":
            target_url = f"{base}/chat/completions"
        elif protocol == "gemini":
            target_url = (
                f"{base}/v1beta/models/{target_model}:streamGenerateContent?alt=sse"
            )
        elif protocol == "claude":
            target_url = f"{base}/v1/messages"
        else:
            # Unknown protocol → default to OpenAI-style
            target_url = f"{base}/chat/completions"

        self._log.info(
            f"Forwarding [{self._provider.name}] "
            f"{target_model!r} → {target_url}"
        )

        # --- Step 3: Build request headers ---
        headers = {"Content-Type": "application/json"}
        if protocol == "claude":
            # Claude uses x-api-key + anthropic-version (provider.rs L538-L542)
            headers["x-api-key"] = self._provider.api_key
            headers["anthropic-version"] = "2023-06-01"
        else:
            # OpenAI and others use Bearer token (provider.rs L544-L545)
            headers["Authorization"] = f"Bearer {self._provider.api_key}"

        # --- Step 4: Send with retry ---
        last_error: str = ""
        last_status: int = 0
        client = _get_provider_client()

        for attempt in range(1, _MAX_RETRIES + 1):
            # Backoff before retry attempts (not before the first attempt)
            if attempt > 1:
                backoff = _BACKOFF_MULTIPLIER * attempt
                self._log.warning(
                    f"  Retry {attempt}/{_MAX_RETRIES} "
                    f"[{self._provider.name}], backoff={backoff:.1f}s"
                )
                await asyncio.sleep(backoff)

            try:
                resp = await client.post(
                    target_url,
                    json=openai_req,
                    headers=headers,
                )
            except httpx.TimeoutException as exc:
                last_error = f"Request timed out: {exc}"
                last_status = 504
                self._log.error(
                    f"  [{self._provider.name}] attempt {attempt}/{_MAX_RETRIES}: "
                    f"timeout"
                )
                continue  # timeout is always retryable

            except httpx.RequestError as exc:
                last_error = f"Connection error: {exc}"
                last_status = 502
                self._log.error(
                    f"  [{self._provider.name}] attempt {attempt}/{_MAX_RETRIES}: "
                    f"connection error: {exc}"
                )
                continue  # connection errors are retryable

            last_status = resp.status_code

            if resp.status_code < 400:
                # --- Success path ---
                if attempt > 1:
                    self._log.info(
                        f"  [{self._provider.name}] retry succeeded "
                        f"on attempt {attempt}"
                    )
                break  # exit retry loop

            # --- Error path ---
            try:
                err_preview = resp.text[:500]
            except Exception:
                err_preview = f"<status {resp.status_code}>"

            last_error = err_preview

            if resp.status_code in _RETRYABLE_STATUSES:
                self._log.warning(
                    f"  [{self._provider.name}] attempt {attempt}/{_MAX_RETRIES}: "
                    f"transient error {resp.status_code}: "
                    f"{err_preview[:200]}"
                )
                # continue to next attempt
            else:
                # Permanent error — do not retry
                self._log.error(
                    f"  [{self._provider.name}] permanent error "
                    f"{resp.status_code}: {err_preview[:200]}"
                )
                return self._error_sse(
                    resp.status_code,
                    f"Provider error {resp.status_code}: {err_preview}",
                )

        else:
            # All retries exhausted
            self._log.error(
                f"  [{self._provider.name}] all {_MAX_RETRIES} attempts failed. "
                f"Last error: {last_error[:300]}"
            )
            return self._error_sse(
                502,
                f"Provider [{self._provider.name}] failed after "
                f"{_MAX_RETRIES} retries: {last_error[:300]}",
            )

        # --- Step 5: Parse and convert response ---
        self._log.info(
            f"  [{self._provider.name}] Response: "
            f"status={resp.status_code} | {len(resp.content)}B"
        )

        try:
            openai_resp = resp.json()
        except Exception as exc:
            self._log.error(
                f"  [{self._provider.name}] Failed to parse response JSON: {exc}"
            )
            # Log raw body (first 500 chars) for debugging
            try:
                raw_preview = resp.text[:500]
            except Exception:
                raw_preview = "<unreadable>"
            self._log.error(f"  Raw response: {raw_preview}")
            return self._error_sse(502, f"Provider response is not valid JSON: {exc}")

        # Log the OpenAI response at DEBUG level for analysis
        self._log.debug(
            f"  [{self._provider.name}] OpenAI response (first 1000 chars):\n"
            f"  {json.dumps(openai_resp)[:1000]}"
        )

        # --- Step 6: Convert OpenAI response → Gemini SSE ---
        try:
            sse_body = convert_response(openai_resp, target_model)
        except Exception as exc:
            self._log.error(
                f"  [{self._provider.name}] Response conversion failed: {exc}"
            )
            return self._error_sse(502, f"Response conversion error: {exc}")

        self._log.info(
            f"  [{self._provider.name}] SSE ready: {len(sse_body)} chars → IDE"
        )

        return (
            200,
            {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
            },
            sse_body,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _error_sse(self, status: int, message: str) -> tuple[int, dict, str]:
        """
        Wrap a provider error in Gemini SSE format so the IDE displays it
        as a chat message instead of a silent failure or cryptic HTTP error.

        Always returns HTTP 200 with SSE body — the IDE expects this format
        for all AI responses regardless of upstream status.

        Args:
            status:  The upstream HTTP status code (included in the message).
            message: Human-readable error description.

        Returns:
            ``(200, sse_headers, sse_body_string)``
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

        sse_body = f"data: {json.dumps(gemini_error, ensure_ascii=False)}\n\n"

        return (
            200,
            {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
            },
            sse_body,
        )
