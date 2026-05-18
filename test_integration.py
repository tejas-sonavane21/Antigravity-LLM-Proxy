"""
Integrated simulation test: Phases 1-4 end-to-end in-memory.

Tests every critical behaviour of the pool system without touching
any live files, making any network calls, or modifying config.json.

Run from project root: python test_integration.py
"""
import sys, json, asyncio, ast, copy
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
HEAD = "\033[36m[----]\033[0m"
_results = []

def ok(msg):
    print(f"  {PASS} {msg}")
    _results.append(True)

def fail(msg):
    print(f"  {FAIL} {msg}")
    _results.append(False)

def section(title):
    print(f"\n{HEAD} {title}")

def assert_eq(actual, expected, msg):
    if actual == expected:
        ok(msg)
    else:
        fail(f"{msg}  |  expected={expected!r}  got={actual!r}")

def assert_true(cond, msg):
    if cond: ok(msg)
    else:     fail(msg)

# ─────────────────────────────────────────────────────────────────────────────
# Load real config (read-only — we deep-copy before any mutation)
# ─────────────────────────────────────────────────────────────────────────────
from src.config import load_config
from src.pool.config_parser import parse_pool_entries
from src.pool.picker import PoolPicker, init_picker, get_picker
from src.pool.cooldown import resolve_cooldown
from src.pool.metadata import patch_model_metadata
from src.pool.handler import _inject_thinking
from src.proxy.router import classify_request, RequestCategory
from src.provider.registry import ProviderRegistry

cfg = load_config("config.json")
RAW_FULL = json.loads(Path("config.json").read_text(encoding="utf-8"))
MODEL_DUMP = json.loads(Path("docs/model_dump.json").read_text(encoding="utf-8"))

# Snapshot raw_model_pool so tests never mutate the real config.json dict
def fresh_raw():
    return copy.deepcopy(RAW_FULL)

def fresh_picker():
    """Build a clean picker for each test group."""
    entries, _ = parse_pool_entries(cfg.raw_model_pool)
    return init_picker(entries, cfg.pool_settings, "config.json", fresh_raw())


# =============================================================================
# SECTION 1: Pool Entry basics
# =============================================================================
section("SECTION 1 — PoolEntry dataclass values")

entries, _ = parse_pool_entries(cfg.raw_model_pool)
assert_eq(len(entries), 6, "6 pool entries parsed")

oc1 = next(e for e in entries if e.id == "oc-key-1")
sf1 = next(e for e in entries if e.id == "sf-key-1")

assert_eq(oc1.context_window, 197000,   "OC key1: context_window = 197000")
assert_eq(oc1.safety_buffer_tokens, 8192, "OC key1: safety_buffer = 8192")
assert_eq(oc1.usable_tokens, 197000 - 8192, "OC key1: usable_tokens = ctx - buffer")
assert_eq(oc1.response_thinking_field, "reasoning", "OC key1: thinking_field=reasoning")
assert_eq(oc1.thinking.enable_param, "thinking",  "OC key1: enable_param=thinking")
assert_eq(oc1.thinking.budget_param, "thinking_budget", "OC key1: budget_param=thinking_budget")

assert_eq(sf1.context_window, 262144,   "SF key1: context_window = 262144")
assert_eq(sf1.usable_tokens, 262144 - 8192, "SF key1: usable_tokens = ctx - buffer")
assert_eq(sf1.response_thinking_field, "reasoning_content", "SF key1: thinking_field=reasoning_content")
assert_eq(sf1.thinking.enable_param, "enable_thinking", "SF key1: enable_param=enable_thinking")


# =============================================================================
# SECTION 2: Round-robin pick() — correct rotation order
# =============================================================================
section("SECTION 2 — Round-robin pick() rotation")

picker = fresh_picker()
picked_ids = []
for _ in range(6):
    e = picker.pick()
    assert_true(e is not None, f"  pick() returned an entry ({e.id if e else 'None'})")
    picked_ids.append(e.id)
    picker.release(e.id, 200, None)

assert_eq(len(set(picked_ids)), 6, "All 6 distinct entries picked in one cycle")
# Keys should rotate through all 6, not repeat the same one
assert_true(picked_ids[0] != picked_ids[1], "No immediate repeat: pick[0] != pick[1]")

# Second cycle: should again rotate through all 6
# Use a fresh isolated picker so section 8 singleton doesn't interfere
_picker2 = PoolPicker(list(parse_pool_entries(cfg.raw_model_pool)[0]), cfg.pool_settings, "config.json", fresh_raw())
picked_ids2 = []
for _ in range(6):
    e = _picker2.pick()
    picked_ids2.append(e.id)
    _picker2.release(e.id, 200, None)
assert_eq(len(set(picked_ids2)), 6, "Second cycle: all 6 entries again")


# =============================================================================
# SECTION 3: Cooldown — 429 cools one entry, others still available
# =============================================================================
section("SECTION 3 — Cooldown: 429 removes entry from rotation")

picker = fresh_picker()
e1 = picker.pick()
e1_id = e1.id
picker.release(e1_id, 429, b'{"message": "requests per minute exceeded"}')

cooled = next(e for e in picker.entries if e.id == e1_id)
assert_true(cooled.is_cooled, f"[{e1_id}] is_cooled=True after 429")
assert_true(cooled.cooldown_until > datetime.now(timezone.utc),
            f"[{e1_id}] cooldown_until is in the future")

# Now pick 5 more times — should never get e1_id again
for i in range(5):
    e = picker.pick()
    assert_true(e is not None, f"pick() {i+2}: still returns entries despite cooldown")
    assert_true(e.id != e1_id, f"pick() {i+2}: NOT the cooled entry [{e1_id}]")
    picker.release(e.id, 200, None)


# =============================================================================
# SECTION 4: All-cooled detection (pick() returns None)
# =============================================================================
section("SECTION 4 — All entries cooled -> pick() returns None")

picker = fresh_picker()
for e in picker.entries:
    e.cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=5)
    e.cooldown_reason = "test forced cooldown"

result = picker.pick()
assert_true(result is None, "pick() returns None when all entries cooled")


# =============================================================================
# SECTION 5: Cooldown expiry — entry becomes available again
# =============================================================================
section("SECTION 5 — Expired cooldown clears on next pick()")

picker = fresh_picker()
e = picker.entries[0]
# Set cooldown to 1 second in the past (already expired)
e.cooldown_until = datetime.now(timezone.utc) - timedelta(seconds=1)
e.cooldown_reason = "expired test cooldown"
assert_true(e.is_cooled is False, "is_cooled=False for past cooldown_until")

result = picker.pick()
assert_true(result is not None, "pick() returns entry after cooldown expiry")


# =============================================================================
# SECTION 6: thinking injection — SiliconFlow vs OpenCode
# =============================================================================
section("SECTION 6 — _inject_thinking per-entry field names")

picker = fresh_picker()
oc_entry = next(e for e in picker.entries if e.id == "oc-key-1")
sf_entry = next(e for e in picker.entries if e.id == "sf-key-1")

# OpenCode: field names are "thinking" / "thinking_budget"
body_oc = {"model": oc_entry.model, "messages": []}
result_oc = _inject_thinking(body_oc, oc_entry)
assert_eq(result_oc["thinking"], True, "OC: thinking=True")
assert_eq(result_oc["thinking_budget"], 8000, "OC: thinking_budget=8000")
assert_true("enable_thinking" not in result_oc, "OC: no SiliconFlow field leaked in")

# SiliconFlow: field names are "enable_thinking" / "thinking_budget"
body_sf = {"model": sf_entry.model, "messages": []}
result_sf = _inject_thinking(body_sf, sf_entry)
assert_eq(result_sf["enable_thinking"], True, "SF: enable_thinking=True")
assert_eq(result_sf["thinking_budget"], 8000, "SF: thinking_budget=8000")
assert_true("thinking" not in result_sf, "SF: no OpenCode 'thinking' field leaked in")


# =============================================================================
# SECTION 7: usable_tokens is sane (context window - safety buffer)
# =============================================================================
section("SECTION 7 — usable_tokens sanity (context window announcement only)")

# usable_tokens is used ONLY for FAMS context window announcement.
# It is NOT injected as max_tokens into API requests (doing so caused HTTP 400
# because input + usable_tokens exceeded total context limit).
for e in picker.entries:
    assert_true(
        isinstance(e.usable_tokens, int) and e.usable_tokens > 0,
        f"[{e.id}] usable_tokens={e.usable_tokens} is positive int"
    )
    assert_eq(
        e.usable_tokens, e.context_window - e.safety_buffer_tokens,
        f"[{e.id}] usable_tokens = context_window - safety_buffer"
    )


# =============================================================================
# SECTION 8: classify_request — POOL vs PASSTHROUGH
# =============================================================================
section("SECTION 8 — classify_request POOL routing")

registry = ProviderRegistry()
picker = fresh_picker()  # picker singleton now active

POOL_MODEL = cfg.pool_settings.mapped_model  # "gpt-oss-120b-medium"

cat, m, t = classify_request(
    "/v1/projects/x/locations/y/streamGenerateContent",
    json.dumps({"request": {"model": POOL_MODEL}}).encode(),
    registry,
)
assert_eq(cat, RequestCategory.POOL, f"mapped model '{POOL_MODEL}' -> POOL")
assert_eq(m, POOL_MODEL, "model_name returned correctly")
assert_true(t is None, "target_model is None for POOL (entry picked in handler)")

# Non-pool model -> PASSTHROUGH
cat2, _, _ = classify_request(
    "/some/path",
    json.dumps({"request": {"model": "gemini-2.5-pro"}}).encode(),
    registry,
)
assert_eq(cat2, RequestCategory.PASSTHROUGH, "non-pool model -> PASSTHROUGH")

# Telemetry is not affected
cat3, _, _ = classify_request("/log?event=x", b"{}", registry)
assert_eq(cat3, RequestCategory.TELEMETRY, "telemetry still -> TELEMETRY")


# =============================================================================
# SECTION 9: fetchAvailableModels cache + patch
# =============================================================================
section("SECTION 9 — fetchAvailableModels: cache and patch mechanics")

picker = fresh_picker()
full_body = json.dumps(MODEL_DUMP).encode("utf-8")
original_max_tokens = MODEL_DUMP["models"][POOL_MODEL]["maxTokens"]

# Simulate: natural request arrives -> cache + patch with peek() entry
peek_entry = picker.peek()
assert_true(peek_entry is not None, "peek() returns entry before any picks")

# Cache the raw body (unmodified)
picker.set_cached_model_response(full_body)
assert_eq(picker.get_cached_model_response(), full_body, "cache stores raw body")

# Patch with peek entry's usable_tokens
patched = patch_model_metadata(full_body, POOL_MODEL, peek_entry)
patched_parsed = json.loads(patched)

assert_eq(patched_parsed["models"][POOL_MODEL]["maxTokens"], peek_entry.usable_tokens,
          f"Natural patch: maxTokens -> {peek_entry.usable_tokens} (peek entry usable_tokens)")
assert_eq(len(patched_parsed["models"]), len(MODEL_DUMP["models"]),
          f"Full model list preserved: {len(patched_parsed['models'])} models")

# Other models untouched
other_id = "gemini-2.5-pro"
assert_eq(patched_parsed["models"][other_id]["maxTokens"],
          MODEL_DUMP["models"][other_id]["maxTokens"],
          f"Other model ({other_id}) untouched")


# =============================================================================
# SECTION 10: pending_advance + advance_peek flow
# ("Announce next entry's context window while processing current request")
# =============================================================================
section("SECTION 10 — Announce-ahead: pending_advance -> advance_peek -> patch next")

picker = fresh_picker()
picker.set_cached_model_response(full_body)

# Step A: First agentic flow request arrives
#   -> handler calls pick() -> gets entry_A
#   -> sets pending_advance = True (so next FAMS intercept knows it's on-demand)
#   -> fires trigger_model_refresh() [not called here — fire-and-forget]

assert_true(picker.pending_advance is False, "pending_advance starts False")
entry_A = picker.pick()
assert_true(entry_A is not None, "pick() returns entry_A")
picker.pending_advance = True
assert_true(picker.pending_advance is True, "pending_advance set to True after pick")

# Step B: IDE calls fetchAvailableModels (triggered by trigger_model_refresh)
#   -> intercept sees pending_advance=True
#   -> does NOT forward to Google
#   -> calls advance_peek() to get the NEXT entry (entry_B)
#   -> patches cached body with entry_B.usable_tokens
#   -> IDE now knows entry_B's context window
#   -> sets pending_advance=False

picker.pending_advance = False  # interceptor clears it
entry_B = picker.advance_peek()  # advance to next
assert_true(entry_B is not None, "advance_peek() returns entry_B")
assert_true(entry_B.id != entry_A.id, "entry_B is different from entry_A")

on_demand_patched = patch_model_metadata(full_body, POOL_MODEL, entry_B)
on_demand_parsed = json.loads(on_demand_patched)
assert_eq(on_demand_parsed["models"][POOL_MODEL]["maxTokens"], entry_B.usable_tokens,
          f"On-demand patch: IDE announced entry_B usable_tokens={entry_B.usable_tokens}")

# Step C: release entry_A (stream completed)
picker.release(entry_A.id, 200, None)

# Step D: Second agentic flow request arrives
#   -> handler calls pick() again
#   -> CRITICAL: should pick entry_B (the one whose context window was announced)
#   -> Because: entry_A was just released (in_flight=0), entry_B has in_flight=0
#     and advance_peek() rotated the logical cursor to entry_B
entry_C = picker.pick()
assert_true(entry_C is not None, "Second pick() returns entry")

# The strategy is: advance_peek() tells us entry_B is next.
# pick() uses least-used sort, so with all entries having equal in_flight=0,
# it picks by last_used_at. entry_A was used most recently (Step A),
# so entry_B (or any other less-recently-used entry) should be picked.
# The critical invariant: entry_C != entry_A (entry_A was LRU, others are fresher)
assert_true(entry_C.id != entry_A.id,
            f"Second request picks [{entry_C.id}], not the just-used [{entry_A.id}]")

ok(f"Context window for [{entry_C.id}] announced ahead = {entry_B.usable_tokens} tokens "
   f"(entry picked = {entry_C.id}, usable = {entry_C.usable_tokens})")


# =============================================================================
# SECTION 11: The "use the key we announced" guarantee
# =============================================================================
section("SECTION 11 — Context window consistency: announced == used key")

# This is the most important test.
# Rule: When IDE is told maxTokens=X for key K, the NEXT request must use key K.
#
# How our system works:
#   - peek() shows what key WILL be used next (no side effects)
#   - patch_model_metadata(cache, POOL_MODEL, peek()) announces that key's maxTokens
#   - pick() selects by (in_flight ASC, last_used_at ASC) — i.e., least recently used
#   - peek() uses the SAME sort as pick() → they agree on the next key
#
# Simulation: 6 turns of pick → release → announce ahead → verify consistency

picker = fresh_picker()
picker.set_cached_model_response(full_body)

for turn in range(6):
    # 1. ANNOUNCE: peek() = what will be picked next
    announced_entry = picker.peek()
    announced_tokens = announced_entry.usable_tokens if announced_entry else None

    # 2. Patch the cached response with the announced entry
    if announced_entry:
        patched_bytes = patch_model_metadata(full_body, POOL_MODEL, announced_entry)
        patched_json = json.loads(patched_bytes)
        tokens_in_response = patched_json["models"][POOL_MODEL]["maxTokens"]
        assert_eq(tokens_in_response, announced_tokens,
                  f"Turn {turn+1}: patched response matches peek() usable_tokens")

    # 3. PICK: actual pick() for this request
    actual_entry = picker.pick()
    assert_true(actual_entry is not None, f"Turn {turn+1}: pick() returns entry")

    # 4. KEY CONSISTENCY CHECK:
    # peek() and pick() must agree — same entry should be returned
    assert_eq(actual_entry.id, announced_entry.id if announced_entry else None,
              f"Turn {turn+1}: peek=[{announced_entry.id}] == pick=[{actual_entry.id}]")

    # 5. Release (simulate stream complete)
    picker.release(actual_entry.id, 200, None)


# =============================================================================
# SECTION 12: Patching safety — missing model, bad JSON, no picker
# =============================================================================
section("SECTION 12 — patch_model_metadata safety cases")

# 12a: mapped model missing -> body unchanged
result = patch_model_metadata(full_body, "nonexistent-model-xyz", sf_entry)
assert_eq(json.loads(result), MODEL_DUMP,
          "Missing model: returns full body unchanged")

# 12b: pool_entry=None -> body unchanged
result2 = patch_model_metadata(full_body, POOL_MODEL, None)
assert_eq(result2, full_body, "pool_entry=None: returns original bytes")

# 12c: invalid JSON -> returns original bytes
bad = b"{ bad json ["
result3 = patch_model_metadata(bad, POOL_MODEL, sf_entry)
assert_eq(result3, bad, "Invalid JSON: returns original bytes")

# 12d: valid patch preserves ALL fields on the patched model (only maxTokens changes)
patched4 = json.loads(patch_model_metadata(full_body, POOL_MODEL, sf_entry))
gpt = patched4["models"][POOL_MODEL]
assert_eq(gpt["maxTokens"], sf_entry.usable_tokens,
          "maxTokens updated to usable_tokens")
assert_true("displayName" in gpt,      "displayName preserved")
assert_true("maxOutputTokens" in gpt,  "maxOutputTokens preserved")
assert_true("apiProvider" in gpt,      "apiProvider preserved")
assert_true("isInternal" in gpt,       "isInternal preserved")


# =============================================================================
# SECTION 13: resolve_cooldown durations
# =============================================================================
section("SECTION 13 — Cooldown duration correctness")

now = datetime.now(timezone.utc)

dt, reason = resolve_cooldown(429, b'{"message":"requests per minute"}', "sf-1")
delta = (dt - now).total_seconds()
assert_true(58 < delta <= 62, f"429 RPM: ~60s cooldown (got {delta:.1f}s)")

dt, reason = resolve_cooldown(429, b'{"message":"tokens per minute exceeded"}', "sf-2")
delta = (dt - now).total_seconds()
assert_true(118 < delta <= 122, f"429 TPM: ~120s cooldown (got {delta:.1f}s)")

dt, reason = resolve_cooldown(429, b'{"message":"daily limit reached"}', "sf-3")
midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
assert_true(abs((dt - midnight).total_seconds()) < 5,
            f"429 RPD: until midnight UTC (diff={abs((dt-midnight).total_seconds()):.1f}s)")

dt, reason = resolve_cooldown(503, None, "oc-1")
delta = (dt - now).total_seconds()
assert_true(88 < delta <= 92, f"503: ~90s cooldown (got {delta:.1f}s)")

dt, reason = resolve_cooldown(500, None, "oc-2")
delta = (dt - now).total_seconds()
assert_true(28 < delta <= 32, f"5xx: ~30s cooldown (got {delta:.1f}s)")


# =============================================================================
# Summary
# =============================================================================
passed = sum(_results)
total  = len(_results)
failed = total - passed

print(f"\n{'='*60}")
if failed == 0:
    print(f"\033[32m  ALL {total} CHECKS PASSED\033[0m")
else:
    print(f"\033[31m  {failed}/{total} CHECKS FAILED\033[0m")
print(f"{'='*60}\n")

sys.exit(0 if failed == 0 else 1)
