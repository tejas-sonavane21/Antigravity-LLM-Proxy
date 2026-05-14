"""
src/converter/__init__.py — Protocol Converter Package
=======================================================
Public API for all Gemini ↔ OpenAI conversion functions.

Phase 4 implementation:
  - schema_cleaner   : Recursive schema cleaning for OpenAI compatibility
  - model_extractor  : Extract + normalize model names from request body/path
  - gemini_to_openai : Convert IDE Gemini-format request → OpenAI request
  - openai_to_gemini : Convert OpenAI response → Gemini SSE response
"""

from .schema_cleaner import clean_schema_for_openai
from .model_extractor import (
    extract_model_from_body,
    extract_model_from_path,
    normalize_model_name,
)
from .gemini_to_openai import convert_request
from .openai_to_gemini import convert_response

__all__ = [
    "clean_schema_for_openai",
    "extract_model_from_body",
    "extract_model_from_path",
    "normalize_model_name",
    "convert_request",
    "convert_response",
]
