"""
src/converter/schema_cleaner.py — JSON Schema Cleaner
======================================================
Recursively cleans Gemini function declaration schemas for OpenAI compatibility.

Port of reference project: provider.rs clean_schema_for_openai() L37-L63

Changes applied:
  - Remove "format" key from all schema objects
  - Lowercase all "type" values  ("STRING" -> "string")
  - Recurse into "properties", "items", "anyOf", "oneOf", "allOf"

Gemini uses uppercase types ("STRING", "INTEGER", "OBJECT", "ARRAY") and
includes "format" hints that OpenAI rejects. Without this cleaning, tool
schemas sent to OpenAI providers cause 400 validation errors.
"""


def clean_schema_for_openai(schema: object) -> object:
    """
    Recursively clean a JSON schema object for OpenAI compatibility.

    Args:
        schema: A dict, list, or primitive value (str/int/bool/None).
                Dicts are cleaned in-place conceptually (a NEW dict is returned).
                Lists have each element cleaned recursively.
                Primitives are returned unchanged.

    Returns:
        A cleaned copy. The input is NOT mutated.

    Port of provider.rs clean_schema_for_openai() L37-L63.
    """
    if isinstance(schema, dict):
        result: dict = {}
        for key, value in schema.items():

            # Drop "format" entirely — OpenAI rejects these (provider.rs L42)
            if key == "format":
                continue

            # Lowercase "type" values (provider.rs L39-L41)
            if key == "type" and isinstance(value, str):
                result[key] = value.lower()
                continue

            # Recurse into "properties" dict (provider.rs L44-L48)
            if key == "properties" and isinstance(value, dict):
                result[key] = {
                    k: clean_schema_for_openai(v) for k, v in value.items()
                }
                continue

            # Recurse into "items" schema (provider.rs L49-L51)
            if key == "items":
                result[key] = clean_schema_for_openai(value)
                continue

            # Recurse into anyOf / oneOf / allOf arrays (provider.rs L52-L58)
            if key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
                result[key] = [clean_schema_for_openai(item) for item in value]
                continue

            # Everything else — pass through unchanged
            result[key] = value

        return result

    if isinstance(schema, list):
        # provider.rs L59-L62: handle array at top level
        return [clean_schema_for_openai(item) for item in schema]

    # Primitive (str, int, float, bool, None) — return as-is
    return schema
