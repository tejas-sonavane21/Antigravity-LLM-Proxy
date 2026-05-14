"""
src/proxy/__init__.py — Proxy System Package
==============================================
Public API for the proxy server system.

Phase 6 implementation:
  - server    : FastAPI app factory (create_app)
  - router    : Request classification and routing
  - forwarder : Google upstream forwarding with host fallback
"""

from .server import create_app
from .forwarder import forward_to_google

__all__ = [
    "create_app",
    "forward_to_google",
]
