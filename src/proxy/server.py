"""
src/proxy/server.py — FastAPI Application Factory
===================================================
Thin wrapper that creates the FastAPI app with a single catch-all route.
All routing logic lives in router.py — this file only bridges FastAPI
to the router and handles the two response types:

  - bytes body  -> fastapi.responses.Response  (TELEMETRY / PASSTHROUGH)
  - AsyncIterator[bytes] -> fastapi.responses.StreamingResponse  (POOL/PROVIDER)

Streaming behaviour (Windows IOCP note):
  The IDE requires each SSE chunk to be flushed to the TCP socket immediately.
  On Linux, asyncio.sleep(0) reliably drains the selector write buffer.
  On Windows with IocpProactor, sleep(0) is NOT sufficient — the IOCP
  completion port processes with timeout=0 finds no pending I/O and returns
  immediately, leaving data sitting in the kernel write buffer until the
  connection closes (manifests as the Ctrl+C burst-flush symptom).

  The correct fix for Windows IOCP is the queue-based approach already
  implemented in openai_compat._iter_stream_events:
    - A background asyncio.Task reads SSE lines into an asyncio.Queue
    - The generator does asyncio.wait_for(queue.get(), timeout=25)
    - wait_for() runs the event loop with a REAL timeout, giving IOCP
      enough time to process all pending overlapped write completions
    - Each yield therefore arrives at the IDE within milliseconds

  This server.py file intentionally does NOT add any extra wrapping or
  sleep() calls. The streaming generators from handler.py and openai_compat.py
  already contain the correct IOCP-compatible flush logic.

  Background: the previous _flush_after_each_chunk() wrapper using sleep(0)
  was reverted because it re-introduced the exact buffering bug it was meant
  to fix — on Windows IOCP, sleep(0) has no write-flushing effect.

Usage::

    from src.proxy.server import create_app

    app = create_app(registry, upstream_hosts, include_thoughts)
    uvicorn.run(app, ...)
"""

import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from src.provider.registry import ProviderRegistry
from src.proxy.router import route_request

log = logging.getLogger("proxy.server")


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
        which we wrap in StreamingResponse so the IDE receives SSE data
        as it arrives. The generators from handler.py/openai_compat.py
        already implement Windows IOCP-compatible write flushing internally
        via their queue + wait_for pattern (no extra wrapping needed here).
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

        # POOL/PROVIDER route returns AsyncIterator[bytes] — use StreamingResponse
        # so uvicorn delivers each SSE chunk to the IDE as it is yielded.
        # No additional wrapping: the generator's internal queue+wait_for loop
        # already ensures IOCP write buffers are drained between chunks.
        if isinstance(resp_body, AsyncIterator):
            return StreamingResponse(
                content=resp_body,
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
