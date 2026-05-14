"""
src/provider/__init__.py — Provider System Package
====================================================
Public API for the provider system.

Phase 5 implementation:
  - base         : Provider base dataclass
  - registry     : ProviderRegistry — loads from config, finds by model
  - openai_compat: OpenAICompatProvider — full forward pipeline
"""

from .base import Provider
from .registry import ProviderRegistry
from .openai_compat import OpenAICompatProvider

__all__ = [
    "Provider",
    "ProviderRegistry",
    "OpenAICompatProvider",
]
