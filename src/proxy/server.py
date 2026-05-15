"""
src/proxy/server.py — FastAPI Application Factory
===================================================
Thin wrapper that creates the FastAPI app with a single catch-all route.
All routing logic lives in router.py — this file only bridges FastAPI
to the router and handles the two response types:

  - bytes body  → fastapi.responses.Response  (TELEMETRY / PASSTHROUGH)
  - AsyncIterator[bytes] → fastapi.responses.StreamingResponse  (PROVIDER)

The StreamingResponse for PROVIDER requests is critical — it delivers
SSE data to the IDE the moment each chunk is yielded, without buffering
the entire body in memory first.

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

        For PROVIDER requests the router returns an AsyncIterator[bytes],
        which we wrap in StreamingResponse so the IDE receives SSE data
        as it arrives rather than waiting for the full buffered body.
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

        # PROVIDER route returns AsyncIterator[bytes] — use StreamingResponse
        # so uvicorn flushes each SSE chunk to the IDE immediately.
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
