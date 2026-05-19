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
    _try_next_entry(entry_id, status, body, picker, method, path, headers,
                    orig_body, upstream_hosts)
        -> AsyncIterator[bytes]

Error handling strategy:
    Retryable (429, 500, 502, 503, 504):
        Retry the SAME entry up to _MAX_RETRIES times with exponential backoff.
    Non-retryable (401, 400, 403, 404 ...):
        Cool the current entry and try the NEXT available pool entry immediately
        (pool-level retry). Only fall back to Google if pick() returns None,
        meaning ALL entries are now cooled/disabled.
    All entries exhausted (pick() returns None on first call):
        _handle_all_cooled() — substitute fallback model and forward to Google.

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

# Maximum number of DIFFERENT pool entries to try before falling back to Google.
# This is separate from the per-entry retry count (_MAX_RETRIES for transient errors).
_MAX_POOL_ATTEMPTS = 6


# ---------------------------------------------------------------------------
# Thinking parameter injection
# ---------------------------------------------------------------------------

def _inject_thinking(openai_body: dict, entry: PoolEntry) -> dict:
    """
    Add provider-specific thinking/chain-of-thought parameters to the OpenAI
    request body based on the pool entry's ThinkingConfig.

    Handles two provider conventions:
      - OpenCode (OpenAI-compat): thinking={...} / thinking_budget=N
      - SiliconFlow: enable_thinking=True/False, thinking_budget=N

    The exact field names are driven by entry.thinking.enable_param and
    entry.thinking.budget_param, so adding a new provider is just config.
    """
    t = entry.thinking
    if not t.enabled:
        openai_body[t.enable_param] = False
        _log.debug(
            f"  [thinking] DISABLED for [{entry.id}] "
            f"({t.enable_param}=False)"
        )
        return openai_body

    openai_body[t.enable_param] = True

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

    On error from Google (4xx/5xx), yields a proper Gemini SSE error event
    so the IDE receives a well-formed response rather than raw error bytes.
    Raw error bytes on a streaming endpoint cause the IDE to misparse the
    stream and show "agent execution terminated" without any useful context.
    """
    from src.proxy.forwarder import forward_to_google
    from src.pool.picker import get_picker

    fallback = pool_settings.all_cooled_fallback

    # ------------------------------------------------------------------
    # "keep-alive" mode — wait for any pool entry to become available
    # ------------------------------------------------------------------
    if fallback == "keep-alive":
        timeout_s = pool_settings.keep_alive_timeout_minutes * 60
        poll_interval_s = 5.0
        _SSE_KA = b": keepalive\n\n"

        picker = get_picker()
        eta = picker.earliest_cooldown_seconds() if picker else 0.0
        eta_display = f"{eta:.0f}s" if eta > 0 else "unknown"

        _log.warning(
            f"[POOL] keep-alive: all entries cooled. "
            f"Earliest key available in ~{eta_display}. "
            f"Waiting up to {pool_settings.keep_alive_timeout_minutes}m "
            f"(poll every {poll_interval_s:.0f}s)..."
        )

        waited_s: float = 0.0
        while waited_s < timeout_s:
            await asyncio.sleep(poll_interval_s)
            waited_s += poll_interval_s

            # Yield keepalive to keep IDE connection open
            yield _SSE_KA

            # Check if any entry is now available
            if picker is not None:
                entry = picker.pick()
                if entry is not None:
                    _log.info(
                        f"[POOL] keep-alive: [{entry.id}] cooldown expired after "
                        f"~{waited_s:.0f}s — resuming pool stream"
                    )
                    # Hand the picked entry back so handle_pool_request can
                    # use it properly with full thinking injection and release().
                    # We release it immediately (no error) so it goes back into
                    # the pool, then return a special sentinel that handle_pool_request
                    # detects to re-enter its pick() loop.
                    #
                    # Simpler approach: just stream through this entry directly here.
                    picker.release(entry.id, 200, None)
                    # Signal caller to retry: we yield nothing more and return.
                    # The caller (handle_pool_request) will loop back and pick().
                    return

        # Timed out — fall through to passthrough
        _log.warning(
            f"[POOL] keep-alive: timed out after "
            f"{pool_settings.keep_alive_timeout_minutes}m — "
            f"falling through to passthrough"
        )
        fallback = "passthrough"

    # ------------------------------------------------------------------
    # Original fallback logic (passthrough or model substitution)
    # ------------------------------------------------------------------
    _log.warning(
        f"[POOL] All pool entries cooled/disabled — "
        f"forwarding to Google with fallback model {fallback!r}"
    )

    if fallback != "passthrough":
        try:
            body_dict = json.loads(body)
            limits = pool_settings.fallback_limits  # FallbackModelLimits from config

            # Locate the Gemini request sub-dict (body has a "request" wrapper from IDE)
            req = body_dict.get("request") if isinstance(body_dict.get("request"), dict) else body_dict

            # 1. Update model name
            req["model"] = fallback

            # 2. Patch generationConfig fields to match fallback model's real limits.
            #    The IDE populates these from FAMS metadata (which reports pool entry
            #    usable_tokens, e.g. 188808). The fallback model has lower limits.
            #    We DO NOT strip the fields — Google expects them to be present.
            #    We only update the values when we have a configured limit AND the
            #    field exists in the request (meaning the IDE already set it).
            gen_cfg = req.get("generationConfig")
            if isinstance(gen_cfg, dict):
                if limits.max_output_tokens is not None and "maxOutputTokens" in gen_cfg:
                    old = gen_cfg["maxOutputTokens"]
                    gen_cfg["maxOutputTokens"] = limits.max_output_tokens
                    _log.debug(
                        f"[POOL] Fallback: generationConfig.maxOutputTokens "
                        f"{old} -> {limits.max_output_tokens}"
                    )

                # thinkingConfig.thinkingBudget — nested one level deeper
                thinking_cfg = gen_cfg.get("thinkingConfig")
                if isinstance(thinking_cfg, dict) and limits.thinking_budget is not None:
                    if "thinkingBudget" in thinking_cfg:
                        old_budget = thinking_cfg["thinkingBudget"]
                        thinking_cfg["thinkingBudget"] = limits.thinking_budget
                        _log.debug(
                            f"[POOL] Fallback: thinkingConfig.thinkingBudget "
                            f"{old_budget} -> {limits.thinking_budget}"
                        )

            body = json.dumps(body_dict).encode("utf-8")
            _log.info(
                f"[POOL] Fallback model substituted: {fallback!r} "
                f"(maxOutputTokens={limits.max_output_tokens}, "
                f"thinkingBudget={limits.thinking_budget})"
            )
        except (json.JSONDecodeError, KeyError) as exc:
            _log.warning(
                f"[POOL] Could not substitute fallback model in body: {exc}. "
                f"Forwarding with original model."
            )

    # Stream the fallback response to the IDE without buffering.
    # forward_to_google() uses client.request() which buffers the ENTIRE body
    # before returning. For a streaming SSE endpoint (?alt=sse) this means
    # waiting for the full model response (~2MB, 35+ seconds) before the IDE
    # receives a single byte — causing the "generating..." freeze.
    #
    # Instead we use client.stream() so chunks flow to the IDE immediately
    # as Google sends them, giving the same live streaming feel as the pool path.
    import httpx as _httpx
    from src.proxy.forwarder import _get_client, _build_forward_headers

    fwd_headers = _build_forward_headers(headers)
    client = _get_client()
    last_error = ""

    from src.proxy.forwarder import _SKIP_REQUEST_HEADERS  # already imported above

    for host in upstream_hosts:
        target_url = f"https://{host}{path}"
        try:
            async with client.stream(
                method=method,
                url=target_url,
                headers=fwd_headers,
                content=body,
            ) as resp:
                _log.info(
                    f"[POOL] Fallback streaming from Google: {host} | {resp.status_code}"
                )
                if resp.status_code >= 400:
                    err_body = await resp.aread()
                    err_text = err_body.decode("utf-8", errors="replace")[:300]
                    _log.warning(
                        f"[POOL] Fallback Google returned {resp.status_code}: {err_text[:120]}"
                    )
                    error_event = json.dumps({
                        "response": {
                            "candidates": [{
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": f"[Proxy] All pool entries unavailable. "
                                                       f"Fallback error {resp.status_code}: {err_text}"}]
                                },
                                "finishReason": "STOP",
                            }],
                            "modelVersion": fallback,
                        }
                    })
                    yield f"data: {error_event}\n\n".encode("utf-8")
                    await asyncio.sleep(0.05)
                    return
                # Stream success — yield chunks as they arrive
                async for chunk in resp.aiter_bytes(chunk_size=4096):
                    if chunk:
                        yield chunk
                return  # done
        except (_httpx.TimeoutException, _httpx.ConnectError, _httpx.RequestError) as exc:
            last_error = f"{host}: {exc}"
            _log.warning(f"[POOL] Fallback: {host} failed ({exc}), trying next host")
            continue

    # All hosts failed
    _log.error(f"[POOL] Fallback: all upstream hosts failed. Last: {last_error}")
    error_event = json.dumps({
        "response": {
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": f"[Proxy] All pool entries unavailable and all upstream hosts failed."}]},
                "finishReason": "STOP",
            }],
            "modelVersion": fallback,
        }
    })
    yield f"data: {error_event}\n\n".encode("utf-8")
    await asyncio.sleep(0.05)



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
      4. Convert Gemini body -> OpenAI format
      5. Inject thinking parameters
      6. Inject max_tokens = entry.usable_tokens
      7. Stream via OpenAICompatProvider._iter_stream_events
      8. release() — decrement in_flight_count; apply cooldown on error
      9. On non-retryable error: cool entry and try NEXT pool entry (pool-level
         retry). Only fall back to Google once ALL entries are exhausted.

    Pool-level retry logic:
      - Transient errors (429/5xx): retry SAME entry up to _MAX_RETRIES times
      - Permanent errors (401/403/400): skip to NEXT available pool entry
      - All entries exhausted: _handle_all_cooled() -> Google fallback
    """
    import httpx as _httpx
    from src.provider.openai_compat import (
        _get_provider_client,
        _MAX_RETRIES,
        _BACKOFF_MULTIPLIER,
        _RETRYABLE_STATUSES,
    )

    attempted_ids: set[str] = set()
    pool_attempt = 0

    while True:

        # ── pick() ────────────────────────────────────────────────────────
        entry = picker.pick()
        pool_attempt += 1

        if entry is None:
            # All entries are cooled/disabled
            _log.warning(
                f"[POOL] pick() returned None after {pool_attempt - 1} "
                f"pool attempt(s) -> all-cooled fallback"
            )
            _got_data = False
            async for chunk in _handle_all_cooled(
                method, path, headers, body, upstream_hosts, picker.pool_settings
            ):
                _got_data = True
                yield chunk
            if _got_data:
                return   # normal fallback finished streaming — done
            # _got_data=False: keep-alive freed a key — retry pick() from top.
            # Reset attempt counter and tried-set so we treat this as a fresh start.
            attempted_ids.clear()
            pool_attempt = 0
            continue

        # Guard: don't re-attempt an entry we already tried this request
        if entry.id in attempted_ids:
            _log.warning(
                f"[POOL] pick() returned already-tried entry [{entry.id}] "
                f"— no more distinct entries available, using fallback"
            )
            _got_data = False
            async for chunk in _handle_all_cooled(
                method, path, headers, body, upstream_hosts, picker.pool_settings
            ):
                _got_data = True
                yield chunk
            if _got_data:
                return
            attempted_ids.clear()
            pool_attempt = 0
            continue

        # Guard: max distinct pool entries tried in one request
        if pool_attempt > _MAX_POOL_ATTEMPTS:
            break

        attempted_ids.add(entry.id)

        _log.info(
            f"[POOL] Picked [{entry.id}] | {entry.model} | "
            f"ctx={entry.context_window} usable={entry.usable_tokens} | "
            f"thinking_field={entry.response_thinking_field!r}"
            + (f" (pool attempt {pool_attempt})" if pool_attempt > 1 else "")
        )

        # ── Fire-and-forget trigger (only on first attempt) ───────────────
        if pool_attempt == 1:
            asyncio.create_task(trigger_model_refresh(picker.pool_settings.flag_file))
            picker.pending_advance = True

        # ── Convert Gemini -> OpenAI ──────────────────────────────────────
        try:
            openai_body = convert_request(body, entry.model, include_thoughts)
        except ValueError as exc:
            _log.error(f"[POOL] Request conversion failed: {exc}")
            picker.release(entry.id, 400, None)
            err_event = json.dumps({
                "response": {
                    "candidates": [{
                        "content": {"role": "model", "parts": [
                            {"text": f"[Proxy Error] Request conversion failed: {exc}"}
                        ]},
                        "finishReason": "STOP",
                    }],
                    "modelVersion": entry.model,
                }
            })
            yield f"data: {err_event}\n\n".encode("utf-8")
            return

        openai_body["stream"] = True
        openai_body = _inject_thinking(openai_body, entry)
        # max_tokens is NOT overridden here. usable_tokens is the context window
        # we announce to the IDE via FAMS — not the per-request output limit.
        # The Gemini->OpenAI converter maps generationConfig.maxOutputTokens from
        # the IDE request. Overriding with usable_tokens (e.g. 188808) causes
        # input + output to exceed the provider's total context limit (HTTP 400).

        # ── Build provider shim ───────────────────────────────────────────
        provider_shim = Provider(
            name=entry.id,
            base_url=entry.base_url,
            api_key=entry.api_key,
            protocol="openai",
            enabled=True,
            streaming=entry.streaming,
            model_map={},
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
        permanent_error = False

        # ── Per-entry retry loop (transient errors only) ──────────────────
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
                        err_preview = last_error_body.decode(
                            "utf-8", errors="replace"
                        )[:300]

                        if resp.status_code in _RETRYABLE_STATUSES:
                            # Transient — retry same entry
                            _log.warning(
                                f"  [POOL] [{entry.id}] attempt {attempt}: "
                                f"transient {resp.status_code}"
                            )
                            continue

                        # Permanent error — cool this entry and skip to next
                        _log.error(
                            f"  [POOL] [{entry.id}] permanent {resp.status_code}: "
                            f"{err_preview[:200]}"
                        )
                        permanent_error = True
                        break  # exit per-entry retry; outer loop picks next entry

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
                    break

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

        # ── release entry ─────────────────────────────────────────────────
        if success:
            picker.release(entry.id, 200, None)
            return  # done

        # Entry failed — release with appropriate status
        if permanent_error:
            _log.warning(
                f"[POOL] [{entry.id}] permanent error {last_http_status} — "
                f"cooling and trying next pool entry "
                f"(attempt {pool_attempt}/{_MAX_POOL_ATTEMPTS})"
            )
        else:
            _log.error(
                f"[POOL] [{entry.id}] failed after {_MAX_RETRIES} retries "
                f"(status={last_http_status}) — "
                f"cooling and trying next pool entry "
                f"(attempt {pool_attempt}/{_MAX_POOL_ATTEMPTS})"
            )

        picker.release(entry.id, last_http_status, last_error_body)
        # Loop continues — pick() will select the next available entry

    # All pool attempts exhausted without success
    _log.error(
        f"[POOL] All {_MAX_POOL_ATTEMPTS} pool attempt(s) exhausted — "
        f"falling back to Google"
    )
    async for chunk in _handle_all_cooled(
        method, path, headers, body, upstream_hosts, picker.pool_settings
    ):
        yield chunk
