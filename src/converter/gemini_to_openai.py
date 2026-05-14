"""
src/converter/gemini_to_openai.py — Request Converter: Gemini → OpenAI
========================================================================
Converts the Antigravity IDE's Gemini-format request body into an
OpenAI-compatible chat completion request dict.

Port of reference project: provider.rs convert_antigravity_to_openai() L188-L497

This is the most complex converter. Key behaviours:

  1. generationConfig  -> top-level OpenAI params (temperature, max_tokens, etc.)
  2. systemInstruction -> {"role": "system", "content": "..."}
  3. TWO-PASS content processing:
       Pass 1: Scan all functionCall parts to build ID registry by function name.
               This is needed so functionResponse parts can match IDs even when
               the response doesn't carry an explicit ID.
       Pass 2: Build OpenAI messages, mapping Gemini roles/parts to OpenAI.
  4. Thought filtering: parts with "thought": true are skipped (or included
     with a prefix) based on the include_thoughts config flag.
  5. Tool validation: every assistant message that has tool_calls must have a
     matching tool message for each call ID. Missing ones are injected as {}.
  6. Function declaration schemas cleaned via schema_cleaner.
"""

import json
import logging

from .schema_cleaner import clean_schema_for_openai

log = logging.getLogger("converter.gemini_to_openai")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_response_content(response: object) -> str:
    """
    Extract a string from a functionResponse.response value.

    Mirrors the content_str extraction in provider.rs L344-L358.
    """
    if response is None:
        return "{}"
    if isinstance(response, dict):
        result = response.get("result")
        if result is not None:
            return result if isinstance(result, str) else json.dumps(result)
        return json.dumps(response)
    if isinstance(response, str):
        return response
    return json.dumps(response)


def _validate_tool_responses(messages: list[dict]) -> list[dict]:
    """
    Ensure every assistant message that has tool_calls has a matching
    {role: tool} message for each call ID.

    If a tool response is missing, inject an empty placeholder:
      {"role": "tool", "tool_call_id": "<id>", "content": "{}"}

    Port of provider.rs L394-L463 (the validated_messages loop).
    """
    validated: list[dict] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        validated.append(msg)

        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            required_ids = [
                tc["id"]
                for tc in msg["tool_calls"]
                if isinstance(tc, dict) and "id" in tc
            ]

            # Collect tool messages immediately following this assistant message
            found_ids: set[str] = set()
            j = i + 1
            while j < len(messages):
                next_msg = messages[j]
                if next_msg.get("role") == "tool":
                    tid = next_msg.get("tool_call_id")
                    if tid:
                        found_ids.add(tid)
                else:
                    break
                j += 1

            # Append the tool messages we found
            i += 1
            while i < len(messages) and messages[i].get("role") == "tool":
                validated.append(messages[i])
                tid = messages[i].get("tool_call_id")
                if tid:
                    found_ids.add(tid)
                i += 1

            # Inject placeholders for any missing tool responses (provider.rs L450-L458)
            for req_id in required_ids:
                if req_id not in found_ids:
                    log.debug(
                        f"Injecting missing tool response placeholder for id={req_id!r}"
                    )
                    validated.append({
                        "role": "tool",
                        "tool_call_id": req_id,
                        "content": "{}",
                    })

            continue  # i already advanced inside the inner loop

        i += 1

    return validated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_request(
    body: bytes,
    target_model: str,
    include_thoughts: bool = False,
) -> dict:
    """
    Convert an Antigravity IDE Gemini-format request body to an OpenAI
    chat completion request dict.

    Args:
        body:             Raw bytes of the IDE's request body (JSON).
        target_model:     The external provider model name to put in "model".
        include_thoughts: If True, parts with "thought": true are included as
                          "[Previous Reasoning]: <text>". If False, skipped.

    Returns:
        A dict ready to be serialised and sent to an OpenAI-compatible endpoint.

    Raises:
        ValueError: If body is not valid JSON.

    Port of provider.rs convert_antigravity_to_openai() L188-L497.
    """
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Request body is not valid JSON: {exc}") from exc

    # The IDE wraps the actual request in a "request" key (provider.rs L191)
    request = data.get("request", data)
    if not isinstance(request, dict):
        request = data

    # --- Base OpenAI request (provider.rs L192-L195) ---
    openai_req: dict = {
        "model": target_model,
        "stream": False,
    }

    # --- generationConfig -> top-level params (provider.rs L197-L212) ---
    gen_config = request.get("generationConfig")
    if isinstance(gen_config, dict):
        temperature = gen_config.get("temperature")
        if isinstance(temperature, (int, float)):
            openai_req["temperature"] = float(temperature)

        top_p = gen_config.get("topP")
        if isinstance(top_p, (int, float)):
            openai_req["top_p"] = float(top_p)

        max_tokens = gen_config.get("maxOutputTokens")
        if isinstance(max_tokens, int):
            openai_req["max_tokens"] = max_tokens

        thinking_config = gen_config.get("thinkingConfig")
        if isinstance(thinking_config, dict):
            thinking_level = thinking_config.get("thinkingLevel")
            if isinstance(thinking_level, str) and thinking_level:
                openai_req["reasoning_effort"] = thinking_level

    # --- systemInstruction -> system message (provider.rs L215-L232) ---
    messages: list[dict] = []

    system_instruction = request.get("systemInstruction")
    if isinstance(system_instruction, dict):
        parts = system_instruction.get("parts", [])
        if isinstance(parts, list):
            text_chunks = [
                p["text"]
                for p in parts
                if isinstance(p, dict) and isinstance(p.get("text"), str)
            ]
            if text_chunks:
                messages.append({
                    "role": "system",
                    "content": "\n".join(text_chunks),
                })

    # -------------------------------------------------------------------
    # PASS 1: Build function-call ID registry (provider.rs L234-L260)
    # Scan all contents upfront to record which IDs belong to which
    # function name. This allows functionResponse to match by name when
    # no explicit ID is provided.
    # -------------------------------------------------------------------
    fc_name_to_ids: dict[str, list[str]] = {}  # {"func_name": ["call_1", ...]}
    fc_id_counter = 0

    contents = request.get("contents", [])
    if isinstance(contents, list):
        for content in contents:
            if not isinstance(content, dict):
                continue
            for part in content.get("parts", []):
                if not isinstance(part, dict):
                    continue
                fc = part.get("functionCall")
                if isinstance(fc, dict):
                    name = fc.get("name") or "unknown"
                    call_id = fc.get("id")
                    if not call_id:
                        fc_id_counter += 1
                        call_id = f"call_{fc_id_counter}"
                    fc_name_to_ids.setdefault(name, []).append(call_id)

    # -------------------------------------------------------------------
    # PASS 2: Convert contents to OpenAI messages (provider.rs L265-L392)
    # -------------------------------------------------------------------
    fc_name_consume_idx: dict[str, int] = {}  # tracks next ID to use per name
    fc_id_counter2 = 0  # separate counter for pass 2 ID generation

    if isinstance(contents, list):
        for content in contents:
            if not isinstance(content, dict):
                continue

            role = content.get("role", "user")
            # Gemini "model" role maps to OpenAI "assistant" (provider.rs L272)
            openai_role = "assistant" if role == "model" else role

            parts_list = content.get("parts", [])
            if not isinstance(parts_list, list):
                continue

            text_parts: list[str] = []
            tool_calls: list[dict] = []
            tool_messages: list[dict] = []

            for part in parts_list:
                if not isinstance(part, dict):
                    continue

                # --- Text parts (provider.rs L282-L290) ---
                text = part.get("text")
                if isinstance(text, str):
                    is_thought = part.get("thought") is True
                    if is_thought:
                        if include_thoughts:
                            # Include with prefix so external model understands context
                            text_parts.append(f"[Previous Reasoning]: {text}")
                        # else: skip (default — saves tokens, cleaner context)
                    else:
                        text_parts.append(text)

                # --- functionCall -> tool_calls (provider.rs L292-L315) ---
                fc = part.get("functionCall")
                if isinstance(fc, dict):
                    fc_name = fc.get("name") or "unknown"
                    call_id = fc.get("id")
                    if not call_id:
                        fc_id_counter2 += 1
                        call_id = f"call_{fc_id_counter2}"

                    args = fc.get("args", {})
                    args_str = (
                        json.dumps(args) if not isinstance(args, str) else args
                    )

                    tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": fc_name,
                            "arguments": args_str,
                        },
                    })

                # --- functionResponse -> tool message (provider.rs L317-L365) ---
                fr = part.get("functionResponse")
                if isinstance(fr, dict):
                    fr_name = fr.get("name") or ""
                    call_id = fr.get("id")

                    # If no explicit ID, match by function name (provider.rs L328-L341)
                    if not call_id and fr_name:
                        ids_for_name = fc_name_to_ids.get(fr_name, [])
                        idx = fc_name_consume_idx.get(fr_name, 0)
                        if idx < len(ids_for_name):
                            call_id = ids_for_name[idx]
                            fc_name_consume_idx[fr_name] = idx + 1

                    if not call_id:
                        call_id = f"call_unknown_{len(tool_messages)}"

                    content_str = _extract_response_content(fr.get("response"))

                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": content_str,
                    })

            # --- Build the OpenAI message(s) (provider.rs L368-L391) ---
            if openai_role == "assistant":
                msg: dict = {"role": "assistant"}
                if text_parts:
                    msg["content"] = "".join(text_parts)
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                # Only append if there is something meaningful
                if "content" in msg or "tool_calls" in msg:
                    messages.append(msg)
            else:
                # user / tool role: tool messages first, then text
                for tm in tool_messages:
                    messages.append(tm)
                if text_parts:
                    messages.append({
                        "role": openai_role,
                        "content": "".join(text_parts),
                    })

    # --- Validate tool-response completeness (provider.rs L394-L463) ---
    validated = _validate_tool_responses(messages)
    openai_req["messages"] = validated

    # --- Convert tools / functionDeclarations (provider.rs L467-L494) ---
    tools_raw = request.get("tools")
    if isinstance(tools_raw, list):
        openai_tools: list[dict] = []
        for tool in tools_raw:
            if not isinstance(tool, dict):
                continue
            fds = tool.get("functionDeclarations", [])
            if not isinstance(fds, list):
                continue
            for fd in fds:
                if not isinstance(fd, dict):
                    continue
                name = fd.get("name") or "unknown"
                desc = fd.get("description") or ""
                # Prefer parametersJsonSchema, fall back to parameters (provider.rs L475-L478)
                params = fd.get("parametersJsonSchema") or fd.get("parameters")
                if params is None:
                    params = {"type": "object", "properties": {}}
                params = clean_schema_for_openai(params)
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": desc,
                        "parameters": params,
                    },
                })
        if openai_tools:
            openai_req["tools"] = openai_tools

    return openai_req
