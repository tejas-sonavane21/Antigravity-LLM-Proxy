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

# Recognised tool-output keys, in priority order. The Antigravity IDE wraps a
# tool's textual output inside functionResponse.response under ONE of these
# keys, but the exact key varies by tool / IDE version ("output" for view_file,
# "result" for some others, etc.). We check these first, then fall back to a
# fully key-agnostic extraction so a future / unknown key name can never again
# silently double-wrap the payload back into the model's context.
_RESPONSE_CONTENT_KEYS = ("output", "result", "content", "response", "text", "stdout")


def _stringify(value: object) -> str:
    """Return a model-ready string for a single tool-response value."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _extract_response_content(response: object) -> str:
    """
    Extract the tool-output string from a functionResponse.response value.

    The Antigravity IDE delivers tool output as a single-key dict, e.g.
        {"output": "...file contents..."}
    Older code only read the "result" key, so when the IDE used "output"
    (view_file, etc.) the WHOLE dict was re-serialised via json.dumps and the
    model received a double-wrapped {"output": "..."} blob instead of the clean
    text. That is the root cause of the model mis-reading tool results (and
    "seeing an empty object {}"). This extraction is now key-agnostic so it
    survives the IDE renaming the wrapper key:

      1. None            -> ""  (never the misleading literal "{}").
      2. str             -> returned as-is.
      3. dict            -> first matching known content key (output/result/...);
                            else, if the dict has exactly ONE key, return that
                            value REGARDLESS of the key's name (the IDE always
                            wraps output in a single-key dict);
                            else (multiple unknown keys) -> json.dumps the dict.
      4. anything else   -> json.dumps.
    """
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        # 1) Preferred, explicitly-recognised content keys.
        for key in _RESPONSE_CONTENT_KEYS:
            if key in response and response[key] is not None:
                return _stringify(response[key])
        # 2) Key-agnostic fallback: a single-key wrapper dict — return its
        #    value no matter what the key is named.
        if len(response) == 1:
            return _stringify(next(iter(response.values())))
        # 3) Genuinely structured, multi-key payload: serialise as-is.
        return json.dumps(response, ensure_ascii=False)
    return json.dumps(response, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Strict tool-call contract
# ---------------------------------------------------------------------------
# Injected into the system prompt for the MAPPED (pool) model ONLY, appended
# AFTER the Antigravity IDE's own system instructions so the editor's agentic
# workflow stays authoritative and these rules merely refine the model's
# tool-call discipline. NEVER injected for passthrough requests (those never
# reach this converter with inject_strict_instructions=True).
#
# Purpose (from the full streaming/non-streaming log analysis): gpt-oss is
# *trained* to interleave the harmony channels analysis -> commentary (tool
# call) -> analysis -> commentary -> final. We do NOT fight that structure.
# What we DO suppress is the model's BEHAVIOURAL failure mode: fabricating /
# predicting a tool's result instead of emitting the call and waiting, and
# leaking channel/role markers into the visible answer.
# Injected (as plain text) at the END of the system prompt for the mapped pool
# model only. Written as a normal triple-quoted string so the line breaks below
# are REAL newlines in the delivered prompt — there are no literal \n escape
# sequences in the text. Wrapped in a <tool_calling_contract> block so it slots
# in beside the IDE's own <identity>/<guidelines>/<communication_style> sections
# and the model reads it as just another native section of its system prompt.
_STRICT_TOOL_INSTRUCTIONS = """
<tool_calling_contract>
TOOL-CALL EXECUTION CONTRACT (MANDATORY - proxy-enforced)
The rules below are mandatory and take precedence over any conflicting habit. They make your tool use reliable inside this agentic code editor. Follow them exactly while still completing the user's task normally and obeying all of the editor's instructions above.

1. NEVER assume, fabricate, guess, or predict the result or output of a tool/function call. You do NOT know what a tool returns until the editor actually sends you its result in a later turn.
2. When you decide to use a tool, emit ONLY the tool call(s) through the normal function-calling mechanism, then STOP and end your turn. Do not write, narrate, or invent the tool's result yourself.
3. After issuing tool call(s), WAIT for the real tool result before doing any further reasoning, planning, or output that depends on it. Never continue as if a result you have not received already exists.
4. Take ONE logical step per turn: either (a) issue the tool call(s) you can make right now, or (b) give your final answer. Never invent a tool response and then chain another call or a conclusion on top of it.
5. Only batch multiple tool calls in one turn when they are independent and none of them needs another's result.
6. Do NOT emit internal channel or role markers as visible text. Never write tokens such as 'assistant', 'analysis', 'commentary', 'final', or 'to=functions.<name>' into your answer. Your reply must contain only the natural-language response plus properly-structured tool calls.
7. Do NOT transcribe tool arguments or any fabricated tool output as plain text in your reply.
8. If you lack the information needed to proceed, call the appropriate tool to obtain it rather than guessing.
</tool_calling_contract>"""
_STRICT_TOOL_INSTRUCTIONS = """"""

# Placeholder injected for a tool_call that has NO matching tool response in
# the conversation (e.g. duplicate / interleaved streamed calls the IDE never
# answered). The OpenAI schema REQUIRES a tool message per tool_call_id, so we
# must inject something — but the literal "{}" reads to the model like a real
# (empty) tool result, which gpt-oss then hallucinates around ("view_file
# returned an empty object {} ..."). A short explicit notice is unambiguous and
# is never mistaken for an actual tool payload.
_MISSING_TOOL_RESPONSE_PLACEHOLDER = "[no output returned for this tool call]"


def _validate_tool_responses(messages: list[dict]) -> list[dict]:
    """
    Ensure every assistant message that has tool_calls has a matching
    {role: tool} message for each call ID.

    If a tool response is missing, inject an EXPLICIT placeholder:
      {"role": "tool", "tool_call_id": "<id>",
       "content": _MISSING_TOOL_RESPONSE_PLACEHOLDER}
    The OpenAI schema requires a tool message per tool_call_id, so we must
    inject something. We deliberately avoid the literal "{}" because the model
    reads it as "the tool returned an empty object" and hallucinates around it.

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
                        "content": _MISSING_TOOL_RESPONSE_PLACEHOLDER,
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
    inject_strict_instructions: bool = False,
) -> dict:
    """
    Convert an Antigravity IDE Gemini-format request body to an OpenAI
    chat completion request dict.

    Args:
        body:             Raw bytes of the IDE's request body (JSON).
        target_model:     The external provider model name to put in "model".
        include_thoughts: If True, parts with "thought": true are included as
                          "[Previous Reasoning]: <text>". If False, skipped.
        inject_strict_instructions: If True, append the strict tool-call
                          contract to the system prompt. Enabled ONLY for the
                          mapped (pool) model; never for passthrough requests.

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

    system_text_chunks: list[str] = []
    system_instruction = request.get("systemInstruction")
    if isinstance(system_instruction, dict):
        parts = system_instruction.get("parts", [])
        if isinstance(parts, list):
            system_text_chunks = [
                p["text"]
                for p in parts
                if isinstance(p, dict) and isinstance(p.get("text"), str)
            ]

    # Strict tool-call contract — appended at the END of the system prompt so
    # the IDE's own agentic instructions stay authoritative. Enabled ONLY for
    # the mapped (pool) model via inject_strict_instructions; passthrough
    # requests never reach this converter with the flag set. If the IDE sent no
    # system prompt we still create one so the contract is always delivered.
    if inject_strict_instructions:
        system_text_chunks.append(_STRICT_TOOL_INSTRUCTIONS)

    if system_text_chunks:
        messages.append({
            "role": "system",
            "content": "\n".join(system_text_chunks),
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
                # CRITICAL: the Antigravity IDE delivers functionResponse parts
                # INSIDE a role:"model" content (which maps to assistant here).
                # These tool results must still be emitted as separate
                # role:"tool" messages. Previously they were built above but
                # only appended in the else-branch, so for the IDE's layout
                # they were silently DROPPED — and _validate_tool_responses
                # then injected a "{}" placeholder for every unanswered call.
                # That is the true root cause of the model receiving empty
                # tool output ("view_file returned an empty object {}").
                # A tool message must follow the assistant turn that holds the
                # matching tool_calls, so appending here preserves ordering.
                for tm in tool_messages:
                    messages.append(tm)
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
