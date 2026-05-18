"""
src/pool/handler.py — Pool Request Handler
============================================
Handles POOL-classified requests: picks an entry, converts the request,
injects thinking parameters, streams the response, and releases the entry.

Entry point:
    handle_pool_request(method, path, headers, body, upstream_hosts, include_thoughts)
    -> AsyncIterator[bytes]  (Gemini SSE format for the IDE)

Sub-handlers:
    _inject_thinking(openai_body, entry)  -> dict
    _handle_all_cooled(method, path, headers, body, upstream_hosts, pool_settings)
        -> AsyncIterator[bytes]

Reference: pool_implementation_plan.md Phase 3 Steps 3.2, 3.3, 3.4, 3.6
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from src.pool.entry import PoolEntry
from src.pool.picker import PoolPicker
from src.pool.trigger import trigger_model_refresh
from src.config import PoolSettings
from src.converter.gemini_to_openai import convert_request
from src.provider.base import Provider
from src.provider.openai_compat import OpenAICompatProvider

_log = logging.getLogger("pool.handler")


# ---------------------------------------------------------------------------
# Thinking parameter injection
# ---------------------------------------------------------------------------

def _inject_thinking(openai_body: dict, entry: PoolEntry) -> dict:
    """
    Add provider-specific thinking/chain-of-thought parameters to the OpenAI
    request body based on the pool entry's ThinkingConfig.

    Handles two provider conventions:
      - OpenCode (OpenAI-compat): thinking={"type":"enabled","budget_tokens":N}
        but since we call the OpenAI-format API, the field is just `thinking`
        with a boolean + budget nested under the configured param name.
      - SiliconFlow: enable_thinking=True/False, thinking_budget=N

    The exact field names are driven by entry.thinking.enable_param and
    entry.thinking.budget_param, so adding a new provider is just config.

    Args:
        openai_body: The converted OpenAI request dict (mutated in-place copy).
        entry:       The selected pool entry with its ThinkingConfig.

    Returns:
        The updated openai_body dict (same object, returned for chaining).
    """
    t = entry.thinking
    if not t.enabled:
        # Explicit disable — required for models that break on thinking + tool calls
        # (e.g. DeepSeek-V3.1; not our current models but handled for future entries)
        openai_body[t.enable_param] = False
        _log.debug(
            f"  [thinking] DISABLED for [{entry.id}] "
            f"({t.enable_param}=False)"
        )
        return openai_body

    # Enable thinking
    openai_body[t.enable_param] = True

    # Set budget if configured
    if t.budget is not None:
        openai_body[t.budget_param] = t.budget
        _log.debug(
            f"  [thinking] ENABLED for [{entry.id}] "
            f"({t.enable_param}=True, {t.budget_param}={t.budget})"
        )
    else:
        _log.debug(
            f"  [thinking] ENABLED for [{entry.id}] "
            f"({t.enable_param}=True, no budget set)"
        )

    return openai_body


# ---------------------------------------------------------------------------
# All-cooled fallback handler
# ---------------------------------------------------------------------------

async def _handle_all_cooled(
    method: str,
    path: str,
    headers: dict[str, str],
    body: bytes,
    upstream_hosts: list[str],
    pool_settings: PoolSettings,
) -> AsyncIterator[bytes]:
    """
    Fallback path when ALL pool entries are cooled or disabled.

    Behaviour is controlled by pool_settings.all_cooled_fallback:
      "passthrough" -> forward to Google with the original model name (no change)
      "<model-id>"  -> substitute the model name in the request body,
                       then forward to Google in Gemini format as-is

    In both cases NO protocol conversion is done — the request stays in
    Gemini format and goes directly to Google.

    This is intentionally simple: we're just letting Google handle it.
    The IDE will see a normal response and continue operating.
    """
    from src.proxy.forwarder import forward_to_google

    fallback = pool_settings.all_cooled_fallback
    _log.warning(
        f"[POOL] All entries cooled/disabled. "
        f"Fallback: {fallback!r}"
    )

    if fallback != "passthrough":
        # Substitute the model name in the Gemini request body
        try:
            body_dict = json.loads(body)
            # Model name sits at body["request"]["model"] or body["model"]
            if "request" in body_dict and isinstance(body_dict["request"], dict):
                body_dict["request"]["model"] = fallback
            else:
                body_dict["model"] = fallback
            body = json.dumps(body_dict).encode("utf-8")
            _log.info(f"[POOL] Fallback model substituted: {fallback!r}")
        except (json.JSONDecodeError, KeyError) as exc:
            _log.warning(
                f"[POOL] Could not substitute fallback model in body: {exc}. "
                f"Forwarding with original model."
            )

    # Forward to Google verbatim in Gemini format
    status, resp_headers, resp_body = await forward_to_google(
        method=method,
        path=path,
        headers=headers,
        body=body,
        upstream_hosts=upstream_hosts,
    )

    # Wrap the response as a single SSE event so the server can stream it
    # (the forwarder returns bytes, but the POOL handler must return AsyncIterator)
    yield resp_body


# ---------------------------------------------------------------------------
# Main pool request handler
# ---------------------------------------------------------------------------

async def handle_pool_request(
    method: str,
    path: str,
    headers: dict[str, str],
    body: bytes,
    picker: PoolPicker,
    upstream_hosts: list[str],
    include_thoughts: bool,
) -> AsyncIterator[bytes]:
    """
    Full pipeline for a POOL-classified generation request.

    Flow:
      1. pick() — select the least-used available entry
      2. If None (all cooled) — delegate to _handle_all_cooled()
      3. Fire-and-forget trigger: asyncio.create_task(trigger_model_refresh())
         so the IDE refreshes its model metadata AFTER this stream completes.
         mark picker.pending_advance = True so the interceptor knows to serve
         the patched response.
      4. Convert Gemini body -> OpenAI format (using entry's model name)
      5. Inject thinking parameters (per-entry ThinkingConfig)
      6. Inject max_tokens = entry.usable_tokens (context window management)
      7. Stream via OpenAICompatProvider._iter_stream_events with correct
         thinking_field from entry.response_thinking_field
      8. release() — decrement in_flight_count; apply cooldown on error

    Args:
        method:          HTTP method ("POST").
        path:            Full request path.
        headers:         Request headers dict (lowercased keys).
        body:            Raw Gemini-format request body bytes.
        picker:          The PoolPicker singleton.
        upstream_hosts:  Google upstream hosts (for fallback path).
        include_thoughts: Whether to pass through thought content.

    Yields:
        Gemini SSE bytes suitable for StreamingResponse.
    """
    # ── Step 1: pick ──────────────────────────────────────────────────────
    entry = picker.pick()

    if entry is None:
        _log.warning("[POOL] pick() returned None -> all-cooled fallback")
        async for chunk in _handle_all_cooled(
            method, path, headers, body, upstream_hosts, picker.pool_settings
        ):
            yield chunk
        return

    _log.info(
        f"[POOL] Picked [{entry.id}] | {entry.model} | "
        f"ctx={entry.context_window} usable={entry.usable_tokens} | "
        f"thinking_field={entry.response_thinking_field!r}"
    )

    # ── Step 2: fire-and-forget model refresh trigger ─────────────────────
    # Schedule the trigger BEFORE we start streaming so it fires as early as
    # possible. pending_advance tells the fetchAvailableModels interceptor
    # to serve the NEXT entry's context window in its patched response.
    asyncio.create_task(trigger_model_refresh())
    picker.pending_advance = True

    # ── Step 3: convert Gemini body -> OpenAI format ─────────────────────
    try:
        openai_body = convert_request(body, entry.model, include_thoughts)
    except ValueError as exc:
        _log.error(f"[POOL] Request conversion failed: {exc}")
        picker.release(entry.id, 400, None)
        from src.provider.openai_compat import OpenAICompatProvider as _OAP
        # Yield a minimal error SSE so the IDE gets a response
        err_event = json.dumps({
            "response": {
                "candidates": [{"content": {"role": "model", "parts": [
                    {"text": f"[Proxy Error] Request conversion failed: {exc}"}
                ]}, "finishReason": "STOP"}],
                "modelVersion": entry.model,
            }
        })
        yield f"data: {err_event}\n\n".encode("utf-8")
        return

    openai_body["stream"] = True

    # ── Step 4: inject thinking parameters ───────────────────────────────
    openai_body = _inject_thinking(openai_body, entry)

    # ── Step 5: inject max_tokens (context window management) ─────────────
    # Override whatever max_tokens the IDE requested with our computed
    # usable_tokens value (= context_window - safety_buffer). This ensures
    # the provider never exceeds the model's actual limit.
    openai_body["max_tokens"] = entry.usable_tokens

    # ── Step 6: build a lightweight Provider shim for OpenAICompatProvider ─
    # OpenAICompatProvider expects a Provider dataclass. We construct a minimal
    # shim from the pool entry so we don't need to duplicate streaming logic.
    provider_shim = Provider(
        name=entry.id,
        base_url=entry.base_url,
        api_key=entry.api_key,
        protocol="openai",
        enabled=True,
        streaming=entry.streaming,
        model_map={},
    )

    # ── Step 7: stream with per-entry thinking field ──────────────────────
    # We use the internal _iter_stream_events directly so we can pass the
    # entry-specific thinking_field without duplicating streaming logic.
    import httpx as _httpx
    from src.provider.openai_compat import (
        _get_provider_client,
        _MAX_RETRIES,
        _BACKOFF_MULTIPLIER,
        _RETRYABLE_STATUSES,
        _SSE_KEEPALIVE,
    )

    target_url = f"{entry.base_url.rstrip('/')}/chat/completions"
    req_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {entry.api_key}",
    }

    compat = OpenAICompatProvider(provider_shim)
    client = _get_provider_client()
    last_http_status = 200
    last_error_body: bytes | None = None
    success = False

    for attempt in range(1, _MAX_RETRIES + 1):
        if attempt > 1:
            backoff = _BACKOFF_MULTIPLIER * attempt
            _log.warning(
                f"  [POOL] Retry {attempt}/{_MAX_RETRIES} [{entry.id}], "
                f"backoff={backoff:.1f}s"
            )
            await asyncio.sleep(backoff)

        try:
            async with client.stream(
                "POST", target_url,
                json=openai_body, headers=req_headers,
            ) as resp:
                last_http_status = resp.status_code

                if resp.status_code >= 400:
                    last_error_body = await resp.aread()
                    err_preview = last_error_body.decode("utf-8", errors="replace")[:300]
                    if resp.status_code in _RETRYABLE_STATUSES:
                        _log.warning(
                            f"  [POOL] [{entry.id}] attempt {attempt}: "
                            f"transient {resp.status_code}"
                        )
                        continue
                    _log.error(
                        f"  [POOL] [{entry.id}] permanent {resp.status_code}: "
                        f"{err_preview[:200]}"
                    )
                    # Non-retryable error — release with status and break
                    picker.release(entry.id, resp.status_code, last_error_body)
                    # Fall through to all-cooled fallback
                    async for chunk in _handle_all_cooled(
                        method, path, headers, body,
                        upstream_hosts, picker.pool_settings
                    ):
                        yield chunk
                    return

                # Successful connection — stream with entry-specific thinking field
                event_count = 0
                async for sse_bytes in compat._iter_stream_events(
                    resp,
                    entry.model,
                    thinking_field=entry.response_thinking_field,
                ):
                    event_count += 1
                    yield sse_bytes

                _log.info(
                    f"  [POOL] [{entry.id}] stream done: "
                    f"{event_count} Gemini events -> IDE"
                )
                success = True
                break  # exit retry loop

        except _httpx.TimeoutException as exc:
            _log.error(
                f"  [POOL] [{entry.id}] attempt {attempt}: timeout: {exc}"
            )
            last_http_status = 504
            if attempt == _MAX_RETRIES:
                last_error_body = str(exc).encode()
            continue

        except _httpx.RequestError as exc:
            _log.error(
                f"  [POOL] [{entry.id}] attempt {attempt}: "
                f"connection error: {exc}"
            )
            last_http_status = 502
            if attempt == _MAX_RETRIES:
                last_error_body = str(exc).encode()
            continue

    # ── Step 8: release ───────────────────────────────────────────────────
    if success:
        picker.release(entry.id, 200, None)
    else:
        _log.error(
            f"[POOL] [{entry.id}] failed after {_MAX_RETRIES} retries "
            f"(status={last_http_status})"
        )
        picker.release(entry.id, last_http_status, last_error_body)
        # Attempt fallback on exhausted retries
        async for chunk in _handle_all_cooled(
            method, path, headers, body,
            upstream_hosts, picker.pool_settings
        ):
            yield chunk
