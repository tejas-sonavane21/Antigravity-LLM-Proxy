"""
src/converter/openai_to_gemini.py — Response Converter: OpenAI → Gemini SSE
=============================================================================
Converts an OpenAI chat completion response dict into a Gemini-format
Server-Sent Event (SSE) string that the Antigravity IDE can consume.

Port of reference project: provider.rs forward_to_provider() L586-L700

Critical behaviours:

  1. thoughtSignature injection (provider.rs L614-L615, CRITICAL)
     ─────────────────────────────────────────────────────────────
     Every functionCall in the Gemini response MUST include:
       "thoughtSignature": "skip_thought_signature_validator"
     Without this the IDE silently discards the tool call — no error,
     just nothing happens. This was discovered in the reference project
     (brainstorming_session.md §11.6.1).

  2. SSE format (provider.rs L677)
     ─────────────────────────────
     Response must be:  data: <json>\n\n
     Exactly two trailing newlines. The IDE's SSE parser requires this.

  3. finishReason mapping (provider.rs L631-L635)
     ─────────────────────────────────────────────
     OpenAI "stop"/"tool_calls" -> Gemini "STOP"
     OpenAI "length"/"max_tokens" -> Gemini "MAX_TOKENS"

  4. arguments parsing (provider.rs L602-L609)
     ──────────────────────────────────────────
     OpenAI returns tool arguments as a JSON string.
     Gemini expects "args" as a parsed object. We parse the string.
"""

import json
import logging

log = logging.getLogger("converter.openai_to_gemini")

# SSE format constant — MUST be exactly "data: {json}\n\n"
_SSE_PREFIX = "data: "
_SSE_SUFFIX = "\n\n"

# finishReason mapping (provider.rs L631-L635)
_FINISH_REASON_MAP = {
    "stop": "STOP",
    "tool_calls": "STOP",
    "length": "MAX_TOKENS",
    "max_tokens": "MAX_TOKENS",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_response(openai_resp: dict, target_model: str) -> str:
    """
    Convert an OpenAI chat completion response to a Gemini SSE string.

    Args:
        openai_resp:  Parsed OpenAI response dict (from json.loads).
        target_model: The internal model name to use as modelVersion fallback
                      if the response doesn't include a "model" field.

    Returns:
        A complete SSE string:  "data: {json}\n\n"

    Port of provider.rs forward_to_provider() response handling L586-L700.
    """
    parts: list[dict] = []
    finish_reason_raw = "stop"

    choices = openai_resp.get("choices", [])

    for choice in choices:
        if not isinstance(choice, dict):
            continue

        msg = choice.get("message", {})
        if not isinstance(msg, dict):
            continue

        # Capture finish_reason from the first choice (provider.rs L624-L629)
        if not finish_reason_raw or finish_reason_raw == "stop":
            finish_reason_raw = choice.get("finish_reason") or "stop"

        # --- Text content -> {"text": "..."} (provider.rs L592-L594) ---
        content = msg.get("content")
        if isinstance(content, str) and content:
            parts.append({"text": content})

        # --- tool_calls -> functionCall parts (provider.rs L595-L620) ---
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue

                func = tc.get("function", {})
                if not isinstance(func, dict):
                    continue

                fc: dict = {}

                # Function name (provider.rs L599-L601)
                name = func.get("name")
                if name:
                    fc["name"] = name

                # Parse arguments from JSON string to object (provider.rs L602-L609)
                args_raw = func.get("arguments", "{}")
                if isinstance(args_raw, str):
                    try:
                        fc["args"] = json.loads(args_raw)
                    except json.JSONDecodeError:
                        log.warning(
                            f"Could not parse tool arguments for {name!r}: {args_raw!r}"
                        )
                        fc["args"] = {}
                elif isinstance(args_raw, dict):
                    fc["args"] = args_raw
                else:
                    fc["args"] = {}

                # Preserve tool call ID (provider.rs L611-L613)
                call_id = tc.get("id")
                if call_id:
                    fc["id"] = call_id

                # CRITICAL: Inject thoughtSignature (provider.rs L614-L615)
                # Without this the IDE silently drops the function call.
                fc["thoughtSignature"] = "skip_thought_signature_validator"

                parts.append({"functionCall": fc})

    # --- finishReason mapping (provider.rs L631-L635) ---
    gemini_finish = _FINISH_REASON_MAP.get(
        (finish_reason_raw or "stop").lower(), "STOP"
    )

    # --- modelVersion (provider.rs L637-L640) ---
    model_version = openai_resp.get("model") or target_model

    # --- Build Gemini response envelope (provider.rs L642-L654) ---
    gemini: dict = {
        "response": {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": parts,
                    },
                    "finishReason": gemini_finish,
                }
            ],
            "modelVersion": model_version,
            "responseId": openai_resp.get("id", ""),
        }
    }

    # --- Usage metadata (provider.rs L656-L674) ---
    usage = openai_resp.get("usage")
    if isinstance(usage, dict):
        gemini["response"]["usageMetadata"] = {
            "promptTokenCount": usage.get("prompt_tokens", 0),
            "candidatesTokenCount": usage.get("completion_tokens", 0),
            "totalTokenCount": usage.get("total_tokens", 0),
        }

    # --- Serialise to SSE (provider.rs L676-L677) ---
    json_str = json.dumps(gemini, ensure_ascii=False)
    return _SSE_PREFIX + json_str + _SSE_SUFFIX
