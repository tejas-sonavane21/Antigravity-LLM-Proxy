"""
src/proxy/router.py — Request Router
======================================
Classifies every incoming IDE request and routes it to the correct handler:

  1. TELEMETRY  -> mock response (200 {} or 204)
  2. POOL       -> model matches pool mapped_model -> pool handler pipeline
  3. PROVIDER   -> model has a mapping in ProviderRegistry -> legacy provider pipeline
  4. PASSTHROUGH-> everything else -> forward to Google verbatim

Classification priority: POOL takes precedence over PROVIDER for the
primary mapped model (gpt-oss-120b-medium). This means when the pool is
configured, the ProviderRegistry path for that model is bypassed entirely.

Reference:
  - proxy.rs L1452-L1566 -- model extraction + provider routing + telemetry
  - proxy.rs L1541-L1566 -- telemetry identification and mocking
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
# Feature flags (set at startup via configure())
# ---------------------------------------------------------------------------

# When True: dump the full model list on every fetchAvailableModels response.
# Useful for investigating Antigravity updates (new models, changed metadata).
# Set via config.json → proxy.dump_model_responses
_dump_model_responses: bool = False

# When True: print every request intercepted from the IDE to the terminal.
# Shows method, path, headers, and the FULL decoded body (system prompt, messages, etc.).
# Set via config.json → proxy.dump_requests
_dump_requests: bool = False


def configure(*, dump_model_responses: bool = False, dump_requests: bool = False) -> None:
    """Apply runtime configuration to this module. Call once at startup."""
    global _dump_model_responses, _dump_requests
    _dump_model_responses = dump_model_responses
    _dump_requests = dump_requests
    if dump_model_responses:
        log.info("[router] Model response dumping ENABLED (dump_model_responses=true)")
    if dump_requests:
        log.info("[router] Request dumping ENABLED (dump_requests=true) — all IDE requests will be printed in full")


# ---------------------------------------------------------------------------
# Request categories
# ---------------------------------------------------------------------------

class RequestCategory(Enum):
    """Classification result for an incoming request."""
    TELEMETRY   = "telemetry"     # Mock and swallow
    POOL        = "pool"          # Route through pool handler pipeline
    PROVIDER    = "provider"      # Route through legacy provider pipeline
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
        For POOL: target_model is None (entry selection done in handler).
        For PROVIDER: target_model is the external model name.
    """
    # 1. Telemetry -- check first (cheapest)
    if _is_telemetry(path):
        return (RequestCategory.TELEMETRY, None, None)

    # 2. Extract model name from body, then from URL path
    model_name = extract_model_from_body(body) or extract_model_from_path(path)

    # 3. Check pool first (takes priority over legacy ProviderRegistry path)
    if model_name:
        from src.pool.picker import get_picker
        picker = get_picker()
        if picker is not None:
            # Pool is active -- check if this model is the pool's mapped model
            pool_mapped = picker.pool_settings.mapped_model
            if model_name == pool_mapped:
                return (RequestCategory.POOL, model_name, None)

    # 4. Legacy ProviderRegistry path (for any other mapped models)
    if model_name:
        match = registry.find_provider_for_model(model_name)
        if match:
            provider, target_model = match
            return (RequestCategory.PROVIDER, model_name, target_model)

    # 5. Everything else -> pass through to Google
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
    # 0. REQUEST DUMP — print full raw request before routing (opt-in)
    # ------------------------------------------------------------------
    if _dump_requests and category is not RequestCategory.TELEMETRY:
        import json as _json
        _SEP = "=" * 72
        print(_SEP, flush=True)
        print(f"[REQ-DUMP] {method} {path}", flush=True)
        print(f"[REQ-DUMP] Body size: {len(body)} bytes", flush=True)

        # Headers (sanitise Authorization: show only first 20 chars)
        print("[REQ-DUMP] Headers:", flush=True)
        for _hk, _hv in headers.items():
            _hk_lower = _hk.lower()
            if _hk_lower == "authorization" and len(_hv) > 28:
                _hv = _hv[:20] + "...<redacted>"
            print(f"  {_hk}: {_hv}", flush=True)

        # Body
        if body:
            try:
                _parsed = _json.loads(body.decode("utf-8"))
                print("[REQ-DUMP] Body (decoded JSON):", flush=True)

                # ── System prompt ────────────────────────────────────────
                _sys = _parsed.get("systemInstruction") or _parsed.get("system")
                if _sys:
                    print("  [SYSTEM PROMPT]:", flush=True)
                    if isinstance(_sys, dict):
                        # Gemini format: {role, parts: [{text}]}
                        for _part in _sys.get("parts", []):
                            _txt = _part.get("text", "")
                            print(f"    {_txt}", flush=True)
                    else:
                        print(f"    {_sys}", flush=True)
                else:
                    print("  [SYSTEM PROMPT]: <none>", flush=True)

                # ── Generation config ────────────────────────────────────
                _gcfg = _parsed.get("generationConfig") or {}
                if _gcfg:
                    print(f"  [GEN CONFIG]: {_json.dumps(_gcfg)}", flush=True)

                # ── Tools ────────────────────────────────────────────────
                _tools = _parsed.get("tools") or []
                if _tools:
                    print(f"  [TOOLS]: {len(_tools)} tool(s) defined", flush=True)
                    for _t in _tools:
                        _fdecls = _t.get("functionDeclarations") or []
                        for _fd in _fdecls:
                            print(f"    - {_fd.get('name','?')}: {_fd.get('description','')[:80]}", flush=True)

                # ── Conversation turns ───────────────────────────────────
                _contents = _parsed.get("contents") or []
                print(f"  [CONTENTS]: {len(_contents)} turn(s)", flush=True)
                for _i, _turn in enumerate(_contents):
                    _role = _turn.get("role", "?")
                    _parts = _turn.get("parts") or []
                    for _p in _parts:
                        if "text" in _p:
                            _txt = _p["text"]
                            # Print full text, no truncation
                            print(f"  [TURN {_i}][{_role}] text ({len(_txt)} chars):", flush=True)
                            print(f"    {_txt}", flush=True)
                        elif "functionCall" in _p:
                            _fc = _p["functionCall"]
                            print(f"  [TURN {_i}][{_role}] functionCall: {_fc.get('name','?')}", flush=True)
                            print(f"    args: {_json.dumps(_fc.get('args', {}))}", flush=True)
                        elif "functionResponse" in _p:
                            _fr = _p["functionResponse"]
                            _resp_str = _json.dumps(_fr.get('response', {}))
                            print(f"  [TURN {_i}][{_role}] functionResponse: {_fr.get('name','?')}", flush=True)
                            print(f"    response: {_resp_str}", flush=True)
                        else:
                            print(f"  [TURN {_i}][{_role}] part keys: {list(_p.keys())}", flush=True)

                # ── Anything else at top level ───────────────────────────
                _shown = {"systemInstruction", "system", "generationConfig", "tools", "contents"}
                _extra = {k: v for k, v in _parsed.items() if k not in _shown}
                if _extra:
                    print(f"  [OTHER FIELDS]: {_json.dumps(_extra)}", flush=True)

            except (_json.JSONDecodeError, UnicodeDecodeError):
                print(f"[REQ-DUMP] Body (raw, non-JSON):", flush=True)
                print(body.decode("utf-8", errors="replace"), flush=True)
        else:
            print("[REQ-DUMP] Body: <empty>", flush=True)

        print(_SEP, flush=True)

    if category is RequestCategory.TELEMETRY:
        log.debug(f"[TELEM] {method} {path}")
        if path == "/log" or path.startswith("/log?"):
            return (204, {}, b"")
        return (200, {"Content-Type": "application/json"}, b"{}")

    # ------------------------------------------------------------------
    # 2. POOL -- pool-managed model -> full pool handler pipeline
    # ------------------------------------------------------------------
    if category is RequestCategory.POOL:
        from src.pool.picker import get_picker
        from src.pool.handler import handle_pool_request

        picker = get_picker()
        assert picker is not None  # guaranteed by classify_request

        log.info(
            f"[POOL] {method} {path} | "
            f"{model_name} -> pool ({len(picker.entries)} entries)"
        )

        sse_headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        return (
            200,
            sse_headers,
            handle_pool_request(
                method=method,
                path=path,
                headers=headers,
                body=body,
                picker=picker,
                upstream_hosts=upstream_hosts,
                include_thoughts=include_thoughts,
            ),
        )

    # ------------------------------------------------------------------
    # 3. PROVIDER -- legacy ProviderRegistry path (non-pool mapped models)
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
    # 4. PASSTHROUGH -- forward to Google verbatim
    # ------------------------------------------------------------------
    model_tag = f" | model={model_name}" if model_name else ""

    # ── fetchAvailableModels intercept ──────────────────────────────────
    # Check BEFORE forwarding to Google so we can short-circuit on-demand
    # requests (pending_advance=True) without wasting a Google round-trip.
    _is_model_fetch = (
        "fetchAvailableModels" in path or
        "fetchAvailableCodeAssistModels" in path
    )

    if _is_model_fetch:
        from src.pool.picker import get_picker
        from src.pool.metadata import patch_model_metadata

        _picker = get_picker()

        if _picker is not None and _picker.pending_advance:
            # ── ON-DEMAND path ──────────────────────────────────────────
            # This request was triggered by trigger_model_refresh() after a
            # POOL generation completed. The IDE fired fetchAvailableModels
            # to get updated model metadata.
            #
            # We do NOT forward to Google. Instead:
            #   1. Clear pending_advance flag
            #   2. Advance the peek cursor to the NEXT pool entry
            #   3. Serve cached body patched with that entry's usable_tokens
            _picker.pending_advance = False
            next_entry = _picker.advance_peek()

            cached = _picker.get_cached_model_response()
            if cached is not None and next_entry is not None:
                mapped = _picker.pool_settings.mapped_model
                patched_body = patch_model_metadata(cached, mapped, next_entry)
                log.info(
                    f"[FAMS] On-demand intercept: serving cached body "
                    f"patched for [{next_entry.id}] "
                    f"usable_tokens={next_entry.usable_tokens}"
                )
                sse_headers = {"Content-Type": "application/json"}
                return (200, sse_headers, patched_body)
            else:
                # Cache not yet populated or no available entry --
                # fall through to forward to Google normally
                log.info(
                    f"[FAMS] On-demand intercept: cache empty or no entry "
                    f"available -- forwarding to Google"
                )
                _picker.pending_advance = False  # reset anyway

    # ── Forward to Google ────────────────────────────────────────────────
    log.info(f"[PASS] {method} {path}{model_tag}")

    # SSE streaming endpoints must be streamed, not buffered.
    # Buffering a streamGenerateContent?alt=sse response means the IDE
    # gets nothing until the ENTIRE model generation completes — this
    # appears as a multi-minute hang for long responses (claude-sonnet, etc.)
    _is_sse = "alt=sse" in path

    if _is_sse:
        from src.proxy.forwarder import forward_to_google_stream
        sse_headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        return (
            200,
            sse_headers,
            forward_to_google_stream(
                method=method,
                path=path,
                headers=headers,
                body=body,
                upstream_hosts=upstream_hosts,
            ),
        )

    status, resp_headers, resp_body = await forward_to_google(
        method=method,
        path=path,
        headers=headers,
        body=body,
        upstream_hosts=upstream_hosts,
    )

    # ── fetchAvailableModels: natural request post-processing ─────────────
    # Now that we have Google's fresh response, cache it and patch the
    # mapped model's maxTokens to match the current peek entry's usable_tokens.
    if _is_model_fetch and status == 200:
        from src.pool.picker import get_picker
        from src.pool.metadata import patch_model_metadata

        _picker = get_picker()
        if _picker is not None:
            # Cache the full unmodified Google response for on-demand replays
            _picker.set_cached_model_response(resp_body)

            # Patch with the CURRENT peek entry's usable_tokens
            # (this is what the IDE will use for its next generation request)
            current_entry = _picker.peek()
            if current_entry is not None:
                mapped = _picker.pool_settings.mapped_model
                resp_body = patch_model_metadata(resp_body, mapped, current_entry)
                log.info(
                    f"[FAMS] Natural request: cached + patched "
                    f"[{current_entry.id}] usable_tokens={current_entry.usable_tokens}"
                )

    # ── Q3 INVESTIGATOR: dump fetchAvailableModels response ──────────────
    # Runs AFTER patching so the dump shows exactly what the IDE receives.
    # Controlled by config.proxy.dump_model_responses (default: False).
    if _dump_model_responses and _is_model_fetch:
        import json as _json
        log.info("=" * 60)
        log.info("[Q3-DUMP] fetchAvailableModels response intercepted")
        log.info(f"[Q3-DUMP] Status: {status} | Body size: {len(resp_body)}B")
        try:
            _data = _json.loads(resp_body)
            _models_dict = _data.get("models") or {}
            if isinstance(_models_dict, dict) and _models_dict:
                log.info(f"[Q3-DUMP] Found {len(_models_dict)} model(s):")
                for _mid, _m in _models_dict.items():
                    _ctx   = _m.get("maxTokens") or "N/A"
                    _out   = _m.get("maxOutputTokens") or "N/A"
                    _disp  = _m.get("displayName") or ""
                    _think = _m.get("supportsThinking")
                    _tbud  = _m.get("thinkingBudget")
                    _int   = _m.get("isInternal", False)
                    _api   = _m.get("apiProvider") or ""
                    log.info(
                        f"  [{_mid}]  display={_disp!r}  "
                        f"maxTokens={_ctx}  maxOutput={_out}  "
                        f"thinking={_think}  thinkBudget={_tbud}  "
                        f"internal={_int}  api={_api}"
                    )
            else:
                log.info("[Q3-DUMP] Unexpected structure -- raw JSON (first 3000 chars):")
                log.info(_json.dumps(_data, indent=2)[:3000])
        except Exception as _e:
            log.info(f"[Q3-DUMP] Parse error: {_e}")
            log.info(f"[Q3-DUMP] Raw body (first 1000 chars): {resp_body[:1000]}")
        log.info("=" * 60)
    # ── end Q3 INVESTIGATOR ──────────────────────────────────────────────

    return (status, resp_headers, resp_body)
