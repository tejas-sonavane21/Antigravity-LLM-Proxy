"""
src/pool/metadata.py — fetchAvailableModels Response Patcher
=============================================================
Patches the Google fetchAvailableModels response body to replace the
mapped model's maxTokens (and related fields) with the pool entry's
computed usable_tokens value.

Design rules (from model_list_analysis.md Section 5):
  - ALWAYS cache and return the FULL Google response body (all 18 models).
  - ONLY modify the fields of the mapped model (gpt-oss-120b-medium).
  - Never modify any other model's metadata.
  - If the mapped model is missing from the response (upstream rename/removal),
    log a warning and return the body unmodified — never crash the IDE.

The patching is deliberately minimal:
  - maxTokens        -> pool_entry.usable_tokens  (context window visible to IDE)
  - supportsThinking -> True                       (all our pool models support thinking)
  - thinkingBudget   -> entry.thinking.budget ONLY when numeric; string budgets
                        (reasoning_effort levels) are skipped to keep the FAMS
                        handshake well-formed (IDE expects an integer here)

We do NOT patch:
  - maxOutputTokens  (we let the provider cap its own output naturally)
  - displayName      (cosmetic, not functional)
  - apiProvider      (Google-internal routing label, IDE reads it but does not act on it)
  - Any other model's fields

Reference: pool_implementation_plan.md Phase 4 Step 4.2
           model_list_analysis.md Section 5 (Finalized Caching Strategy)
"""

import json
import logging
from typing import Optional

from src.pool.entry import PoolEntry

_log = logging.getLogger("pool.metadata")


def patch_model_metadata(
    response_body: bytes,
    mapped_model: str,
    pool_entry: Optional[PoolEntry],
) -> bytes:
    """
    Patch the mapped model's maxTokens in a full fetchAvailableModels response.

    Args:
        response_body: Raw bytes of Google's fetchAvailableModels response.
        mapped_model:  The model ID to patch (e.g. "gpt-oss-120b-medium").
                       This must match the key in the response["models"] dict.
        pool_entry:    The pool entry whose usable_tokens to advertise.
                       If None, the body is returned completely unmodified.

    Returns:
        The patched response body bytes (all models present, only mapped model
        fields changed). Returns original bytes on any parse error.

    Safety:
        - JSON parse errors: returns original bytes, logs warning
        - Mapped model missing: returns original bytes, logs warning
        - pool_entry is None: returns original bytes (no picker available)
    """
    if pool_entry is None:
        return response_body

    # Parse
    try:
        data = json.loads(response_body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        _log.warning(
            f"[PATCH] Failed to parse fetchAvailableModels body: {exc}. "
            f"Returning original unmodified."
        )
        return response_body

    models = data.get("models")
    if not isinstance(models, dict):
        _log.warning(
            f"[PATCH] Unexpected models structure (type={type(models).__name__}). "
            f"Returning original unmodified."
        )
        return response_body

    if mapped_model not in models:
        _log.warning(
            f"[PATCH] Mapped model {mapped_model!r} not found in "
            f"fetchAvailableModels response ({len(models)} models present). "
            f"Google may have renamed or removed it. "
            f"Returning full original response unmodified."
        )
        return response_body

    # Patch only the mapped model's fields
    target = models[mapped_model]
    old_tokens = target.get("maxTokens", "?")

    # maxTokens: advertise the pool entry's usable_tokens (context window for IDE)
    target["maxTokens"] = pool_entry.usable_tokens

    # supportsThinking: reflect the ACTUAL entry's thinking support.
    # Previously hardcoded True — this broke non-thinking models (e.g. OpenRouter
    # owl-alpha) because the IDE would inject thinkingConfig into every request,
    # and the model would reject it with a tool-call failure.
    target["supportsThinking"] = pool_entry.thinking.enabled

    # thinkingBudget: the IDE's fetchAvailableModels (FAMS) handshake expects an
    # INTEGER token count here. Only patch it when the configured budget is
    # numeric. A STRING budget (e.g. "medium" for a reasoning_effort provider)
    # must NEVER be written here — a string value makes the IDE reject the model
    # metadata with "There was an error with your authentication". For string
    # budgets we leave Google's default thinkingBudget untouched (it is
    # irrelevant to reasoning_effort providers, which drive thinking via
    # thinkingLevel / reasoning_effort rather than a numeric budget).
    if pool_entry.thinking.enabled and isinstance(
        pool_entry.thinking.budget, (int, float)
    ):
        target["thinkingBudget"] = int(pool_entry.thinking.budget)

    # maxOutputTokens: patch only when the entry has an explicit value configured.
    # When None (most entries), leave Google's value untouched — the provider
    # enforces its own output cap naturally.
    old_output = target.get("maxOutputTokens", "?")
    if pool_entry.max_output_tokens is not None:
        target["maxOutputTokens"] = pool_entry.max_output_tokens
        output_patch_msg = f" maxOutputTokens {old_output} -> {pool_entry.max_output_tokens}"
    else:
        output_patch_msg = ""

    _log.info(
        f"[PATCH] fetchAvailableModels: {mapped_model!r} "
        f"maxTokens {old_tokens} -> {pool_entry.usable_tokens} "
        f"(entry={pool_entry.id}, ctx={pool_entry.context_window}, "
        f"buffer={pool_entry.safety_buffer_tokens})"
        f"{output_patch_msg}"
    )

    try:
        return json.dumps(data, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _log.warning(
            f"[PATCH] Failed to re-serialise patched response: {exc}. "
            f"Returning original unmodified."
        )
        return response_body
