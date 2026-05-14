"""
src/converter/model_extractor.py — Model Name Extraction & Normalization
=========================================================================
Extracts model name from IDE request body or URL path, and normalizes
the name for consistent config lookup.

Port of reference project:
  extract_model_from_body()  provider.rs L66-L120
  extract_model_from_path()  provider.rs L122-L134
  normalize_model_name()     provider.rs L136-L151

The IDE can embed the model name in two places:
  1. JSON body: fields named "model", "modelName", "model_name", "modelId",
     "model_id" at any nesting level, plus "request.model" shortcut.
  2. URL path: /v1beta/models/MODEL_NAME:streamGenerateContent?alt=sse
"""

import json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_model_fields(value: object, out: list[str]) -> None:
    """
    Deep-search a parsed JSON value for any key named "model", "modelname",
    "model_name", "modelid", or "model_id" whose value is a non-empty string.

    Results are appended to `out`. Port of provider.rs collect_model_fields()
    (nested function inside extract_model_from_body) L67-L96.
    """
    if isinstance(value, dict):
        for k, v in value.items():
            if k.lower() in ("model", "modelname", "model_name", "modelid", "model_id"):
                if isinstance(v, str):
                    trimmed = v.strip()
                    if trimmed:
                        out.append(trimmed)
            # Always recurse into values
            _collect_model_fields(v, out)
    elif isinstance(value, list):
        for item in value:
            _collect_model_fields(item, out)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_model_from_body(body: bytes) -> str | None:
    """
    Extract the model name from a JSON request body.

    Priority order (provider.rs L101-L119):
      1. Top-level "model" key
      2. Nested "request.model" key
      3. Deep scan for any model-like key

    Returns None if body is not valid JSON, or no model field found.
    Binary / protobuf bodies return None (caller falls back to path extraction).

    Port of provider.rs extract_model_from_body() L66-L120.
    """
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None

    if not isinstance(data, dict):
        return None

    # Priority 1: top-level "model" (provider.rs L101-L106)
    top_model = data.get("model")
    if isinstance(top_model, str):
        trimmed = top_model.strip()
        if trimmed:
            return trimmed

    # Priority 2: request.model (provider.rs L107-L116)
    req = data.get("request")
    if isinstance(req, dict):
        req_model = req.get("model")
        if isinstance(req_model, str):
            trimmed = req_model.strip()
            if trimmed:
                return trimmed

    # Priority 3: deep search (provider.rs L118)
    candidates: list[str] = []
    _collect_model_fields(data, candidates)
    return candidates[0] if candidates else None


def extract_model_from_path(path: str) -> str | None:
    """
    Extract the model name from a URL path.

    Handles paths like:
      /v1beta/models/gemini-2.5-pro:streamGenerateContent?alt=sse
      /v1beta/models/gemini-2.5-pro-preview-05-06:streamGenerateContent
      /models/gemini-v3-byom

    Returns None if "/models/" is not present in the path.

    Port of provider.rs extract_model_from_path() L122-L134.
    """
    if "/models/" not in path:
        return None

    # Take everything after the last "/models/"
    after_models = path.split("/models/", 1)[1]

    # Strip method suffix (":streamGenerateContent") and query ("?alt=sse")
    model_raw = after_models.split(":")[0].split("?")[0].strip("/").strip()

    return model_raw if model_raw else None


def normalize_model_name(raw: str) -> str:
    """
    Normalize a model name for case-insensitive comparison and config lookup.

    Operations (provider.rs normalize_model_name() L136-L151):
      1. Strip surrounding whitespace and quotes
      2. Lowercase
      3. Strip query params (everything after "?")
      4. Strip method suffix (everything after ":")
      5. Strip "/models/" path prefix or "models/" prefix
      6. Strip leading/trailing slashes

    Examples:
      "models/Gemini-2.5-Pro:streamGenerateContent?alt=sse"
        -> "gemini-2.5-pro"
      "  gemini-2.5-flash-preview  "
        -> "gemini-2.5-flash-preview"
      "https://example.com/v1beta/models/my-model"
        -> "my-model"
    """
    s = raw.strip().strip('"').lower()

    # Remove query string
    if "?" in s:
        s = s.split("?", 1)[0]

    # Remove method suffix
    if ":" in s:
        s = s.split(":", 1)[0]

    # Strip path prefix ending with /models/
    if "/models/" in s:
        s = s.rsplit("/models/", 1)[1]

    # Strip bare models/ prefix
    if s.startswith("models/"):
        s = s[len("models/"):]

    return s.strip("/").strip()
