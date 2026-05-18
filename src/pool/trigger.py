"""
src/pool/trigger.py — On-Demand fetchAvailableModels Refresh Trigger
=====================================================================
Fire-and-forget async function that calls GET http://127.0.0.1:9528/refresh-models
to signal the patched Antigravity IDE to re-issue fetchAvailableModels.

This is used by the pool request handler (Phase 3) to refresh the IDE's
model metadata after every POOL-routed generation request, so that the
IDE always has the correct maxTokens for the next pool entry's context window.

The trigger is fire-and-forget via asyncio.create_task() — it NEVER blocks
the streaming response to the IDE. If the IDE is not patched or port 9528
is not listening, the error is silently swallowed after the timeout.

Reference: pool_implementation_plan.md Phase 3 Step 3.6
           fetchAvailableModels_investigation.md Section 5
           patcher.py: do_pool_trigger_patch() injects the HTTP server on port 9528
"""

import asyncio
import logging

_log = logging.getLogger("pool.trigger")

# The port injected into main.js via patcher.py do_pool_trigger_patch()
_TRIGGER_PORT = 9528
_TRIGGER_URL  = f"http://127.0.0.1:{_TRIGGER_PORT}/refresh-models"

# Short timeout — if the IDE is not patched or port is not up, fail fast
_TRIGGER_TIMEOUT_S = 3.0


async def trigger_model_refresh() -> None:
    """
    Fire GET http://127.0.0.1:9528/refresh-models to the patched IDE.

    This causes the IDE to call refreshUserStatus() which then issues a
    fetchAvailableModels request. Our proxy's PASSTHROUGH handler intercepts
    that request and serves the patched response with the next pool entry's
    context window as maxTokens.

    All errors are caught and logged at DEBUG level — this is best-effort.
    The proxy continues serving the stream regardless of trigger success/failure.
    """
    try:
        # Use asyncio streams for a lightweight single-shot GET with no
        # external dependency (no httpx — we don't want to share the provider
        # client pool for this internal loopback call).
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", _TRIGGER_PORT),
            timeout=_TRIGGER_TIMEOUT_S,
        )
        try:
            # Minimal HTTP/1.1 GET — the injected Node.js server accepts this
            request_line = (
                f"GET /refresh-models HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{_TRIGGER_PORT}\r\n"
                f"Connection: close\r\n"
                f"\r\n"
            )
            writer.write(request_line.encode("ascii"))
            await writer.drain()

            # Read the response (we don't care about the body, just that it arrived)
            response = await asyncio.wait_for(
                reader.read(512), timeout=_TRIGGER_TIMEOUT_S
            )
            status_line = response.split(b"\r\n", 1)[0].decode("ascii", errors="replace")
            if "200" in status_line:
                _log.debug(f"trigger_model_refresh: IDE acknowledged -> {status_line}")
            else:
                _log.debug(f"trigger_model_refresh: IDE returned non-200 -> {status_line}")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    except asyncio.TimeoutError:
        _log.debug(
            f"trigger_model_refresh: timeout after {_TRIGGER_TIMEOUT_S}s "
            f"(IDE not patched or port {_TRIGGER_PORT} not listening)"
        )
    except ConnectionRefusedError:
        _log.debug(
            f"trigger_model_refresh: connection refused on port {_TRIGGER_PORT} "
            f"(IDE not running or not patched yet)"
        )
    except Exception as exc:
        _log.debug(f"trigger_model_refresh: unexpected error: {exc}")
