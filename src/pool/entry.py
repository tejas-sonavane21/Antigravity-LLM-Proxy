"""
src/pool/entry.py — Pool Entry Dataclass
=========================================
Typed representation of a single pool entry (one api_key + provider + model triple).
Pure data — no network, no file I/O.

Each PoolEntry corresponds to one JSON object inside config.json `model_pool.entries`.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class LimitsConfig:
    """
    Rate limit ceilings for this pool entry.
    All fields are optional (None = unconstrained / unpublished).
    OpenCode does not publish limits — all three are None.
    SiliconFlow publishes rpm=500, tpm=2_000_000, rpd=None.
    """
    rpm: Optional[int]   # requests per minute
    tpm: Optional[int]   # tokens per minute
    rpd: Optional[int]   # requests per day


@dataclass
class ThinkingConfig:
    """
    Per-provider thinking / chain-of-thought configuration.

    Field naming differs by provider:
      - OpenCode (OpenAI-compat): enable_param="thinking", budget_param="thinking_budget"
      - SiliconFlow:              enable_param="enable_thinking", budget_param="thinking_budget"

    Setting enabled=False sends the explicit disable signal using enable_param.
    Required for any model that breaks on thinking+tool_use (e.g. DeepSeek-V3.1).
    """
    enabled: bool
    budget: Optional[int]      # token budget for thinking; None = use provider default
    enable_param: str           # request field name to enable/disable thinking
    budget_param: str           # request field name to set thinking budget


# ---------------------------------------------------------------------------
# Main PoolEntry
# ---------------------------------------------------------------------------

@dataclass
class PoolEntry:
    """
    A single (api_key, provider, model) triple in the round-robin pool.

    Runtime state fields (in_flight_count, last_used_at) are NOT persisted to disk.
    They live only in memory and reset to zero on proxy restart — which is correct
    because in-flight requests from before a crash are gone anyway.

    Cooldown state (cooldown_until, cooldown_reason) IS persisted to config.json
    so that crash/restart recovers correctly without re-hitting rate-limited keys.
    """
    # Identity
    id: str
    label: str

    # Provider connection
    base_url: str
    api_key: str
    model: str
    streaming: bool

    # Selection weight (integer; weight=2 → picked 2× as often as weight=1)
    weight: int

    # Admin flag — if False, this entry is skipped entirely by the picker
    enabled: bool

    # Rate limit ceilings (all optional)
    limits: LimitsConfig

    # Context capacity
    context_window: int          # model's hard per-request token ceiling
    safety_buffer_tokens: int    # resolved from per-entry OR global pool_settings

    # Provider-specific thinking configuration
    thinking: ThinkingConfig

    # SSE delta field that carries thinking/reasoning tokens in streaming responses.
    # SiliconFlow (all reasoning models): "reasoning_content"
    # OpenCode / standard OpenAI:         "reasoning"
    # None = this entry does not support thinking / field unknown
    response_thinking_field: Optional[str]

    # Optional: advertise this entry's max output tokens to the IDE via FAMS.
    # When set, patch_model_metadata() will update the mapped model's
    # maxOutputTokens field. When None, the field is left as Google returns it.
    max_output_tokens: Optional[int] = None

    # Crash-safe cooldown state — persisted to config.json
    cooldown_until: Optional[datetime]    # UTC datetime when cooldown expires; None = not cooled
    cooldown_reason: Optional[str]        # human-readable reason, e.g. "429: TPM limit reached"

    # Runtime-only state (not persisted)
    in_flight_count: int = 0
    last_used_at: Optional[datetime] = None

    # ---------------------------------------------------------------------------
    # Computed properties
    # ---------------------------------------------------------------------------

    @property
    def usable_tokens(self) -> int:
        """
        Effective per-request token budget to advertise to Antigravity and to
        enforce via max_tokens injection.

        Formula:  effective = min(context_window, tpm_if_set)
                  usable    = effective - safety_buffer_tokens

        TPM is the binding constraint only when TPM < context_window.
        For our current SiliconFlow entries (2M TPM, 262K context_window),
        context_window is always the binding constraint.
        """
        effective = self.context_window
        if self.limits.tpm is not None:
            effective = min(effective, self.limits.tpm)
        return max(0, effective - self.safety_buffer_tokens)

    @property
    def is_cooled(self) -> bool:
        """True if this entry is currently in cooldown and should be skipped."""
        if self.cooldown_until is None:
            return False
        return datetime.now(timezone.utc) < self.cooldown_until

    def clear_cooldown_if_expired(self) -> bool:
        """
        Check whether an active cooldown has expired and clear it if so.
        Returns True if cooldown was cleared, False otherwise.
        Caller is responsible for persisting the change to config.json.
        """
        if self.cooldown_until is not None and not self.is_cooled:
            self.cooldown_until = None
            self.cooldown_reason = None
            return True
        return False
