"""
src/proxy/server.py — FastAPI Application Factory
===================================================
Thin wrapper that creates the FastAPI app with a single catch-all route.
All routing logic lives in router.py — this file only bridges FastAPI
to the router and handles the two response types:

  - bytes body  -> fastapi.responses.Response  (TELEMETRY / PASSTHROUGH)
  - AsyncIterator[bytes] -> fastapi.responses.StreamingResponse  (POOL/PROVIDER)

Enterprise-grade streaming:
  The IDE requires each SSE chunk to be flushed to the TCP socket immediately.
  Uvicorn's h11 transport batches write() calls into a single kernel send when
  the event loop is idle — meaning all chunks may be held in the OS write buffer
  until the connection closes or Ctrl-C is pressed (the bug observed on Windows).

  Fix: wrap every AsyncIterator in _flush_after_each_chunk(), which inserts
  `await asyncio.sleep(0)` after each yielded chunk. This single event-loop
  yield forces the asyncio selector/IOCP to drain the h11 write buffer before
  processing the next chunk. This is the standard pattern for SSE proxies.

  Why asyncio.sleep(0) works:
    - h11 calls transport.write(data) — this enqueues data into asyncio's
      write buffer, but does NOT send it yet.
    - When we yield to the event loop (via sleep(0)), asyncio's selector runs
      and calls the socket's send() on all pending write buffers.
    - The IDE receives each SSE line within milliseconds of the provider
      emitting it, instead of in a burst when the stream ends.

  The previous `await asyncio.sleep(0.05)` in openai_compat.py only flushed
  after the FINAL event. This fix flushes after EVERY chunk — true incremental
  delivery. The existing per-final-event sleeps in openai_compat.py are kept
  as-is (they serve a different purpose: IOCP drain for the close signal).

  Why this doesn't break the chat UI burst fix:
    The queue-based line reader in openai_compat._iter_stream_events uses
    asyncio.Queue + asyncio.wait_for() for safe timeout/keepalive handling.
    sleep(0) between chunks does not affect that — it only adds one extra
    event-loop tick per yielded Gemini SSE frame.

Usage::

    from src.proxy.server import create_app

    app = create_app(registry, upstream_hosts, include_thoughts)
    uvicorn.run(app, ...)
"""

import asyncio
import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from src.provider.registry import ProviderRegistry
from src.proxy.router import route_request

log = logging.getLogger("proxy.server")


# ---------------------------------------------------------------------------
# Streaming flush wrapper
# ---------------------------------------------------------------------------

async def _flush_after_each_chunk(
    source: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    """
    Wrap an async generator so that after every yielded chunk, control is
    returned to the asyncio event loop for one cycle (asyncio.sleep(0)).

    This forces uvicorn's h11 transport to drain its write buffer to the
    kernel TCP socket immediately, giving the IDE per-token streaming instead
    of burst delivery at stream end.

    Preserves the original generator's exception behaviour:
      - GeneratorExit from StreamingResponse.body_iterator close() propagates
        cleanly via try/finally.
      - Any exception from source propagates to FastAPI's exception handler.
    """
    try:
        async for chunk in source:
            yield chunk
            # One event-loop tick: force asyncio to flush the h11 write buffer.
            # Cost: ~0.001ms per chunk. Benefit: true incremental SSE delivery.
            await asyncio.sleep(0)
    except GeneratorExit:
        pass  # Client disconnected — clean shutdown


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    registry: ProviderRegistry,
    upstream_hosts: list[str],
    include_thoughts: bool,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        registry:         Initialized ProviderRegistry (Phase 5).
        upstream_hosts:   Ordered list of Google upstream hostnames.
        include_thoughts: Whether to include thought parts in conversion.

    Returns:
        Configured FastAPI application ready for ``uvicorn.run()``.
    """
    # Suppress noisy library loggers
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    app = FastAPI(
        title="Antigravity Model API Extension",
        docs_url=None,    # No Swagger UI (security)
        redoc_url=None,   # No ReDoc (security)
        openapi_url=None, # No OpenAPI schema endpoint
    )

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    async def catch_all(request: Request, path: str) -> Response:
        """
        Catch-all route — every IDE request lands here.

        Reads the raw body and headers, reconstructs the full path
        with query string, then delegates to the router.

        For POOL/PROVIDER requests the router returns an AsyncIterator[bytes],
        which we wrap in _flush_after_each_chunk() and then StreamingResponse
        so the IDE receives each SSE chunk immediately rather than waiting
        for the full buffered body.
        """
        body = await request.body()

        # Reconstruct full path with query string
        full_path = f"/{path}"
        if request.url.query:
            full_path += f"?{request.url.query}"

        # Extract headers as a plain dict
        req_headers = dict(request.headers)

        # Route the request
        status, resp_headers, resp_body = await route_request(
            method=request.method,
            path=full_path,
            headers=req_headers,
            body=body,
            registry=registry,
            upstream_hosts=upstream_hosts,
            include_thoughts=include_thoughts,
        )

        # POOL/PROVIDER route returns AsyncIterator[bytes] — wrap in the flush
        # adapter so uvicorn drains the write buffer after every SSE chunk.
        if isinstance(resp_body, AsyncIterator):
            return StreamingResponse(
                content=_flush_after_each_chunk(resp_body),
                status_code=status,
                headers=resp_headers,
                media_type="text/event-stream",
            )

        # TELEMETRY / PASSTHROUGH return plain bytes — use regular Response
        return Response(
            content=resp_body,
            status_code=status,
            headers=resp_headers,
        )

    return app
