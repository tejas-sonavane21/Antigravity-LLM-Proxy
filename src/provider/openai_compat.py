"""
src/provider/openai_compat.py — OpenAI-Compatible Provider
===========================================================
Full request-forwarding pipeline for OpenAI-compatible providers.

Supports two provider modes, configured per-provider in config.json:

  streaming: true  (default, recommended)
  ─────────────────────────────────────────
  Calls provider with stream=True.  Processes each SSE chunk as it arrives:
    • text/thought deltas  → yielded immediately → IDE sees tokens as typed
    • tool call deltas     → accumulated then yielded complete on finish
    • finish event         → yields STOP + usage metadata

  streaming: false  (fallback for providers that don't support SSE)
  ──────────────────────────────────────────────────────────────────
  Calls provider with stream=False.  Uses httpx.stream() for non-blocking
  I/O (headers received immediately, body read without stalling the loop),
  then converts and yields one Gemini SSE event.  Works correctly in multi-
  step agentic flows — the IDE waits for the connection to close after each
  turn before sending the next request, so there is no buffering issue.

Multi-step agentic flow (tool calls):
  In a real agentic task the IDE sends many sequential requests:
    Turn 1: user message  → model returns functionCall  → IDE executes tool
    Turn 2: tool result   → model returns more text/calls → IDE executes tool
    ...
    Turn N: final answer
  Each turn is an independent HTTP request.  Our proxy handles each turn
  identically — the agentic complexity lives in the IDE, not the proxy.

  For the streaming path, tool call SSE events are special:
    - Provider sends tool_call deltas (id, name, args fragments) across
      multiple chunks.  We accumulate them fully before yielding.
    - We yield a thought event immediately for each reasoning chunk.
    - When finish_reason=tool_calls arrives, we yield the complete
      functionCall event with thoughtSignature injected (required by IDE).
    - The connection then closes normally → IDE processes the tool.

Retry logic (proxy.rs L1467-L1537):
  MAX_RETRIES = 3  |  Backoff = 0.5s × attempt
  Retry on: 429, 500, 502, 503, 504
  No retry on: 400, 401, 403, 404, etc.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from .base import Provider
from src.converter.gemini_to_openai import convert_request
from src.converter.openai_to_gemini import convert_response


# ---------------------------------------------------------------------------
# Shared HTTP client
# ---------------------------------------------------------------------------
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


_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BACKOFF_MULTIPLIER = 0.5

# OpenAI finish_reason → Gemini finishReason
_FINISH_REASON_MAP = {
    "stop":       "STOP",
    "tool_calls": "STOP",
    "length":     "MAX_TOKENS",
    "max_tokens": "MAX_TOKENS",
}

# ---------------------------------------------------------------------------
# Heartbeat / keepalive
# ---------------------------------------------------------------------------
# SSE spec §9.2: lines beginning with ':' are comments and MUST be ignored.
# We send these when the provider goes silent (long internal reasoning or
# tool-call argument generation) to prevent the IDE's HTTP keep-alive timer
# from closing the connection (manifests as ConnectionResetError 10054 on
# Windows after ~4 minutes of silence).
_HEARTBEAT_INTERVAL: float = 25.0   # seconds between keepalive injections
_SSE_KEEPALIVE: bytes = b": keepalive\n\n"  # SSE comment — ignored by IDE parser


# ---------------------------------------------------------------------------
# Delta accumulator for streaming tool calls
# ---------------------------------------------------------------------------

@dataclass
class _ToolCallAcc:
    """Accumulates one tool call's deltas (id, name, args) across SSE chunks."""
    index:     int
    call_id:   str = ""
    name:      str = ""
    args_json: str = ""   # JSON string, built up fragment by fragment


# ---------------------------------------------------------------------------
# SSE frame builders
# ---------------------------------------------------------------------------

def _make_sse(gemini_dict: dict) -> bytes:
    """Serialise a Gemini response dict to ``data: {json}\\n\\n`` bytes."""
    return f"data: {json.dumps(gemini_dict, ensure_ascii=False)}\n\n".encode("utf-8")


def _partial_text_event(text: str, thought: bool = False) -> bytes:
    """
    Build a partial Gemini SSE event for one text or thought delta.
    finishReason is omitted (empty string) to signal 'more coming'.
    """
    part: dict = {"text": text}
    if thought:
        part["thought"] = True
    return _make_sse({
        "response": {
            "candidates": [{
                "content": {"role": "model", "parts": [part]},
                "finishReason": "",
            }],
            "modelVersion": "",
        }
    })


def _tool_call_event(
    tool_accs: list[_ToolCallAcc],
    finish_reason: str,
    model_version: str,
    usage: dict | None = None,
) -> bytes:
    """
    Build the complete Gemini SSE event for all accumulated tool calls.
    Includes thoughtSignature on every functionCall — required by the IDE;
    without it the IDE silently discards the tool call (no error, nothing).
    """
    parts = []
    for tc in tool_accs:
        try:
            args_parsed = json.loads(tc.args_json) if tc.args_json.strip() else {}
        except json.JSONDecodeError:
            args_parsed = {}

        fc: dict = {
            "name": tc.name,
            "args": args_parsed,
            # CRITICAL: IDE silently drops tool calls without this field
            # Reference: openai_to_gemini.py + brainstorming_session §11.6.1
            "thoughtSignature": "skip_thought_signature_validator",
        }
        if tc.call_id:
            fc["id"] = tc.call_id

        parts.append({"functionCall": fc})

    gemini_finish = _FINISH_REASON_MAP.get(finish_reason.lower(), "STOP")
    resp: dict = {
        "candidates": [{
            "content": {"role": "model", "parts": parts},
            "finishReason": gemini_finish,
        }],
        "modelVersion": model_version,
        "responseId": "",
    }
    if usage:
        resp["usageMetadata"] = {
            "promptTokenCount":     usage.get("prompt_tokens", 0),
            "candidatesTokenCount": usage.get("completion_tokens", 0),
            "totalTokenCount":      usage.get("total_tokens", 0),
        }
    return _make_sse({"response": resp})


def _finish_event(
    model_version: str,
    finish_reason: str,
    usage: dict | None = None,
) -> bytes:
    """
    Build a terminal Gemini SSE event: empty parts, STOP finishReason, usage.
    Sent after all text/thought deltas to signal end-of-turn to the IDE.
    """
    gemini_finish = _FINISH_REASON_MAP.get(finish_reason.lower(), "STOP")
    resp: dict = {
        "candidates": [{
            "content": {"role": "model", "parts": []},
            "finishReason": gemini_finish,
        }],
        "modelVersion": model_version,
        "responseId": "",
    }
    if usage:
        resp["usageMetadata"] = {
            "promptTokenCount":     usage.get("prompt_tokens", 0),
            "candidatesTokenCount": usage.get("completion_tokens", 0),
            "totalTokenCount":      usage.get("total_tokens", 0),
        }
    return _make_sse({"response": resp})


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------

class OpenAICompatProvider:
    """
    Handles the complete Gemini→provider→Gemini pipeline.

    Reads ``provider.streaming`` from the Provider dataclass (set from
    config.json) to choose between true SSE streaming or non-streaming
    fallback mode.

    Usage::

        compat = OpenAICompatProvider(provider)
        async for chunk in compat.stream_request(body_bytes, target_model):
            # chunk: bytes — forwarded to IDE via StreamingResponse
            ...
    """

    def __init__(self, provider: Provider) -> None:
        self._provider = provider
        self._log = logging.getLogger(f"provider.{provider.name}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def stream_request(
        self,
        body_bytes: bytes,
        target_model: str,
        include_thoughts: bool = False,
    ) -> AsyncIterator[bytes]:
        """
        Async generator — yields Gemini SSE bytes for the IDE.

        Branches on ``self._provider.streaming``:
          True  → _stream_from_provider()  (per-token streaming)
          False → _collect_from_provider() (full-body then convert)
        """
        if self._provider.streaming:
            async for chunk in self._stream_from_provider(
                body_bytes, target_model, include_thoughts
            ):
                yield chunk
        else:
            async for chunk in self._collect_from_provider(
                body_bytes, target_model, include_thoughts
            ):
                yield chunk

    # ------------------------------------------------------------------
    # Shared: request preparation
    # ------------------------------------------------------------------

    def _prepare_request(
        self,
        body_bytes: bytes,
        target_model: str,
        include_thoughts: bool,
        streaming: bool,
    ) -> tuple[dict, str, dict[str, str]] | None:
        """
        Convert body and build URL + headers.
        Returns ``(openai_req, target_url, req_headers)`` or None on error
        (error SSE already yielded by caller).
        """
        try:
            openai_req = convert_request(body_bytes, target_model, include_thoughts)
        except ValueError as exc:
            return None  # caller yields error

        openai_req["stream"] = streaming

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

        req_headers: dict[str, str] = {"Content-Type": "application/json"}
        if protocol == "claude":
            req_headers["x-api-key"] = self._provider.api_key
            req_headers["anthropic-version"] = "2023-06-01"
        else:
            req_headers["Authorization"] = f"Bearer {self._provider.api_key}"

        return openai_req, target_url, req_headers

    # ------------------------------------------------------------------
    # Path A: True streaming (provider.streaming = True)
    # ------------------------------------------------------------------

    async def _stream_from_provider(
        self,
        body_bytes: bytes,
        target_model: str,
        include_thoughts: bool,
    ) -> AsyncIterator[bytes]:
        """
        Send stream=True, process each SSE chunk as it arrives.

        Text/thought deltas → yield partial Gemini event immediately.
        Tool call deltas    → accumulate, yield complete event at finish.
        Finish event        → yield STOP + usage.

        This ensures the IDE sees content within seconds of the provider
        starting to generate, rather than waiting for the full response.
        """
        try:
            openai_req = convert_request(body_bytes, target_model, include_thoughts)
        except ValueError as exc:
            self._log.error(f"Request conversion failed: {exc}")
            yield self._error_sse_bytes(400, f"Request conversion error: {exc}")
            return

        openai_req["stream"] = True

        base = self._provider.base_url.rstrip("/")
        protocol = self._provider.protocol.lower()
        target_url = self._build_url(base, protocol, target_model)
        req_headers = self._build_headers(protocol)

        self._log.info(
            f"[STREAM] [{self._provider.name}] {target_model!r} → {target_url}"
        )

        client = _get_provider_client()

        for attempt in range(1, _MAX_RETRIES + 1):
            if attempt > 1:
                backoff = _BACKOFF_MULTIPLIER * attempt
                self._log.warning(
                    f"  Retry {attempt}/{_MAX_RETRIES} [{self._provider.name}], "
                    f"backoff={backoff:.1f}s"
                )
                await asyncio.sleep(backoff)

            try:
                async with client.stream(
                    "POST", target_url,
                    json=openai_req, headers=req_headers,
                ) as resp:

                    if resp.status_code >= 400:
                        err_body = await resp.aread()
                        err_preview = err_body.decode("utf-8", errors="replace")[:500]
                        if resp.status_code in _RETRYABLE_STATUSES:
                            self._log.warning(
                                f"  [{self._provider.name}] attempt {attempt}: "
                                f"transient {resp.status_code}"
                            )
                            continue
                        self._log.error(
                            f"  [{self._provider.name}] permanent {resp.status_code}"
                        )
                        yield self._error_sse_bytes(
                            resp.status_code,
                            f"Provider error {resp.status_code}: {err_preview}",
                        )
                        return

                    # Process the SSE stream line by line
                    event_count = 0
                    async for sse_bytes in self._iter_stream_events(resp, target_model):
                        event_count += 1
                        yield sse_bytes

                    self._log.info(
                        f"  [{self._provider.name}] stream done: "
                        f"{event_count} Gemini events → IDE"
                    )
                    return  # success

            except httpx.TimeoutException as exc:
                self._log.error(
                    f"  [{self._provider.name}] attempt {attempt}: timeout: {exc}"
                )
                if attempt == _MAX_RETRIES:
                    yield self._error_sse_bytes(504, f"Provider timed out: {exc}")
                continue

            except httpx.RequestError as exc:
                self._log.error(
                    f"  [{self._provider.name}] attempt {attempt}: "
                    f"connection error: {exc}"
                )
                if attempt == _MAX_RETRIES:
                    yield self._error_sse_bytes(502, f"Connection error: {exc}")
                continue

        else:
            yield self._error_sse_bytes(
                502,
                f"Provider [{self._provider.name}] failed after {_MAX_RETRIES} retries",
            )

    async def _iter_stream_events(
        self,
        resp: httpx.Response,
        target_model: str,
    ) -> AsyncIterator[bytes]:
        """
        Read the provider's SSE wire format line by line and yield
        converted Gemini SSE bytes.

        Invariants maintained:
          - Text/thought partial events have finishReason=""
          - Tool call event has finishReason="STOP" (set by _tool_call_event)
          - Finish event always terminates the generator with a STOP frame
          - Connection closes as soon as the generator returns
        """
        tool_accs: dict[int, _ToolCallAcc] = {}
        has_tool_calls = False
        finish_reason = "stop"
        last_model = target_model
        last_usage: dict | None = None
        chunk_num = 0
        heartbeat_count = 0

        import time as _time
        # Track wall-clock time so we can inject keepalives even when the
        # provider is ACTIVELY sending tool-call argument chunks.  Without
        # this, large write_to_file generations (file content = hundreds of
        # small chunks over several minutes) never yield anything to the IDE,
        # the IDE's SSE keep-alive fires, it closes the connection, and all
        # accumulated data is lost — the "15-minute freeze" bug.
        _last_keepalive_t = _time.monotonic()

        _aiter = resp.aiter_lines().__aiter__()
        while True:
            # ── Time-based keepalive (covers BOTH silent and busy providers) ──
            # Check BEFORE awaiting next line so we catch long accumulation runs.
            _now = _time.monotonic()
            if _now - _last_keepalive_t >= _HEARTBEAT_INTERVAL:
                heartbeat_count += 1
                self._log.debug(
                    f"  [{self._provider.name}] keepalive #{heartbeat_count} "
                    f"(chunk#{chunk_num}, has_tool_calls={has_tool_calls})"
                )
                yield _SSE_KEEPALIVE
                _last_keepalive_t = _time.monotonic()

            try:
                raw_line = await asyncio.wait_for(
                    _aiter.__anext__(), timeout=_HEARTBEAT_INTERVAL
                )
            except asyncio.TimeoutError:
                # Provider went silent → keepalive
                heartbeat_count += 1
                self._log.debug(
                    f"  [{self._provider.name}] keepalive #{heartbeat_count} "
                    f"(provider silent)"
                )
                yield _SSE_KEEPALIVE
                _last_keepalive_t = _time.monotonic()
                continue
            except StopAsyncIteration:
                break

            line = raw_line.strip()
            if not line or not line.startswith("data: "):
                continue

            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                self._log.debug(f"  Non-JSON SSE line: {data_str[:80]}")
                continue

            chunk_num += 1

            if chunk.get("model"):
                last_model = chunk["model"]
            if chunk.get("usage"):
                last_usage = chunk["usage"]

            choices = chunk.get("choices", [])
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta", {})
            finish = choice.get("finish_reason")
            if finish:
                finish_reason = finish

            # ── Reasoning / thinking ──────────────────────────────────
            reasoning_delta = (
                delta.get("reasoning") or
                delta.get("thinking_content") or
                ""
            )
            if reasoning_delta:
                self._log.debug(
                    f"  chunk#{chunk_num}: thought +{len(reasoning_delta)}ch"
                )
                yield _partial_text_event(reasoning_delta, thought=True)
                _last_keepalive_t = _time.monotonic()  # reset — we just yielded

            # ── Text content ──────────────────────────────────────────
            content_delta = delta.get("content") or ""
            if content_delta:
                self._log.debug(
                    f"  chunk#{chunk_num}: text +{len(content_delta)}ch"
                )
                yield _partial_text_event(content_delta, thought=False)
                _last_keepalive_t = _time.monotonic()  # reset — we just yielded

            # ── Tool call deltas — accumulate (NO yield to IDE here) ──
            # We log every 50 chunks so the server terminal shows activity.
            for tc_delta in delta.get("tool_calls", []):
                idx = tc_delta.get("index", 0)
                has_tool_calls = True
                if idx not in tool_accs:
                    tool_accs[idx] = _ToolCallAcc(index=idx)
                acc = tool_accs[idx]
                if tc_delta.get("id"):
                    acc.call_id = tc_delta["id"]
                fn = tc_delta.get("function", {})
                if fn.get("name"):
                    acc.name += fn["name"]
                if fn.get("arguments"):
                    acc.args_json += fn["arguments"]

            if has_tool_calls and chunk_num % 50 == 0:
                names = [a.name for a in tool_accs.values() if a.name]
                args_len = sum(len(a.args_json) for a in tool_accs.values())
                self._log.debug(
                    f"  chunk#{chunk_num}: accumulating tool_call "
                    f"{names} | args={args_len}ch so far"
                )

        # ── End of stream: yield terminal event ───────────────────────
        if heartbeat_count:
            self._log.info(
                f"  [{self._provider.name}] {heartbeat_count} keepalive(s) "
                f"sent during stream"
            )
        if has_tool_calls and tool_accs:
            tool_list = sorted(tool_accs.values(), key=lambda x: x.index)
            self._log.info(
                f"  [{self._provider.name}] Tool calls complete: "
                f"{[t.name for t in tool_list]} → IDE"
            )
            yield _tool_call_event(tool_list, finish_reason, last_model, last_usage)
            # Small flush delay: lets uvicorn/IOCP deliver the tool_call
            # bytes before the connection close signal is sent.
            await asyncio.sleep(0.15)
        else:
            self._log.info(
                f"  [{self._provider.name}] Text turn complete "
                f"({chunk_num} chunks) → IDE"
            )
            yield _finish_event(last_model, finish_reason, last_usage)
            await asyncio.sleep(0.05)

    # ------------------------------------------------------------------
    # Path B: Non-streaming fallback (provider.streaming = False)
    # ------------------------------------------------------------------

    async def _collect_from_provider(
        self,
        body_bytes: bytes,
        target_model: str,
        include_thoughts: bool,
    ) -> AsyncIterator[bytes]:
        """
        Send stream=False, collect full body, convert, yield one SSE event.

        Uses httpx.stream() for non-blocking I/O — we still get headers
        immediately and the event loop is free while the body arrives.
        This works correctly for multi-step agentic flows: the IDE waits
        for the connection to close before sending the next request, so
        there is no buffering ambiguity.
        """
        try:
            openai_req = convert_request(body_bytes, target_model, include_thoughts)
        except ValueError as exc:
            self._log.error(f"Request conversion failed: {exc}")
            yield self._error_sse_bytes(400, f"Request conversion error: {exc}")
            return

        openai_req["stream"] = False

        base = self._provider.base_url.rstrip("/")
        protocol = self._provider.protocol.lower()
        target_url = self._build_url(base, protocol, target_model)
        req_headers = self._build_headers(protocol)

        self._log.info(
            f"[COLLECT] [{self._provider.name}] {target_model!r} → {target_url}"
        )

        client = _get_provider_client()

        for attempt in range(1, _MAX_RETRIES + 1):
            if attempt > 1:
                backoff = _BACKOFF_MULTIPLIER * attempt
                self._log.warning(
                    f"  Retry {attempt}/{_MAX_RETRIES} [{self._provider.name}], "
                    f"backoff={backoff:.1f}s"
                )
                await asyncio.sleep(backoff)

            try:
                # httpx.stream() = non-blocking I/O even for non-streaming payloads
                async with client.stream(
                    "POST", target_url,
                    json=openai_req, headers=req_headers,
                ) as resp:

                    if resp.status_code >= 400:
                        err_body = await resp.aread()
                        err_preview = err_body.decode("utf-8", errors="replace")[:500]
                        if resp.status_code in _RETRYABLE_STATUSES:
                            self._log.warning(
                                f"  [{self._provider.name}] attempt {attempt}: "
                                f"transient {resp.status_code}"
                            )
                            continue
                        self._log.error(
                            f"  [{self._provider.name}] permanent {resp.status_code}: "
                            f"{err_preview[:200]}"
                        )
                        yield self._error_sse_bytes(
                            resp.status_code,
                            f"Provider error {resp.status_code}: {err_preview}",
                        )
                        return

                    # Keepalive-aware body collection 
                    # For non-streaming providers the ENTIRE response body
                    # arrives as one blob only after the provider finishes
                    # generating (can take 4+ minutes for complex tasks).
                    # Without keepalives the IDE's HTTP connection times out
                    # (ConnectionResetError 10054 on Windows).
                    # We read body in chunks with a per-chunk timeout so we
                    # can inject SSE comments while waiting.
                    _body_chunks: list[bytes] = []
                    _hb = 0
                    _body_aiter = resp.aiter_bytes(chunk_size=65536).__aiter__()
                    while True:
                        try:
                            _chunk = await asyncio.wait_for(
                                _body_aiter.__anext__(),
                                timeout=_HEARTBEAT_INTERVAL,
                            )
                            _body_chunks.append(_chunk)
                        except asyncio.TimeoutError:
                            _hb += 1
                            self._log.debug(
                                f"  [{self._provider.name}] collect keepalive "
                                f"#{_hb} (provider still generating...)"
                            )
                            yield _SSE_KEEPALIVE
                        except StopAsyncIteration:
                            break
                    resp_bytes = b"".join(_body_chunks)
                    if _hb:
                        self._log.info(
                            f"  [{self._provider.name}] {_hb} keepalive(s) sent "
                            f"while collecting {len(resp_bytes)}B response"
                        )

                self._log.info(
                    f"  [{self._provider.name}] Response: "
                    f"status={resp.status_code} | {len(resp_bytes)}B"
                )
                break  # success

            except httpx.TimeoutException as exc:
                self._log.error(
                    f"  [{self._provider.name}] attempt {attempt}: timeout: {exc}"
                )
                if attempt == _MAX_RETRIES:
                    yield self._error_sse_bytes(504, f"Provider timed out: {exc}")
                resp_bytes = None
                continue

            except httpx.RequestError as exc:
                self._log.error(
                    f"  [{self._provider.name}] attempt {attempt}: "
                    f"connection error: {exc}"
                )
                if attempt == _MAX_RETRIES:
                    yield self._error_sse_bytes(502, f"Connection error: {exc}")
                resp_bytes = None
                continue

        else:
            yield self._error_sse_bytes(
                502,
                f"Provider [{self._provider.name}] failed after {_MAX_RETRIES} retries",
            )
            return

        if not resp_bytes:
            yield self._error_sse_bytes(502, "No response received from provider")
            return

        # Parse and convert
        try:
            openai_resp = json.loads(resp_bytes)
        except Exception as exc:
            raw_preview = resp_bytes.decode("utf-8", errors="replace")[:500]
            self._log.error(
                f"  [{self._provider.name}] JSON parse error: {exc} | raw: {raw_preview}"
            )
            yield self._error_sse_bytes(502, f"Provider response not valid JSON: {exc}")
            return

        self._log.debug(
            f"  [{self._provider.name}] OpenAI response: "
            f"{json.dumps(openai_resp)[:500]}"
        )

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
        yield sse_body.encode("utf-8")

    # ------------------------------------------------------------------
    # Shared URL / header builders
    # ------------------------------------------------------------------

    def _build_url(self, base: str, protocol: str, target_model: str) -> str:
        if protocol == "openai":
            return f"{base}/chat/completions"
        if protocol == "gemini":
            return f"{base}/v1beta/models/{target_model}:streamGenerateContent?alt=sse"
        if protocol == "claude":
            return f"{base}/v1/messages"
        return f"{base}/chat/completions"

    def _build_headers(self, protocol: str) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if protocol == "claude":
            h["x-api-key"] = self._provider.api_key
            h["anthropic-version"] = "2023-06-01"
        else:
            h["Authorization"] = f"Bearer {self._provider.api_key}"
        return h

    # ------------------------------------------------------------------
    # Error helper
    # ------------------------------------------------------------------

    def _error_sse_bytes(self, status: int, message: str) -> bytes:
        """Wrap an error in Gemini SSE format so the IDE shows it as chat text."""
        self._log.error(
            f"  [{self._provider.name}] Error SSE [{status}]: {message[:200]}"
        )
        gemini_error: dict = {
            "response": {
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{"text": f"[Provider Error {status}]: {message[:500]}"}],
                    },
                    "finishReason": "STOP",
                }],
                "modelVersion": "error",
                "responseId": "",
            }
        }
        return f"data: {json.dumps(gemini_error, ensure_ascii=False)}\n\n".encode("utf-8")
