"""
src/provider/base.py — Provider Base Class
==========================================
Runtime dataclass representing a single configured AI provider.

Mirrors `ProviderConfig` from src/config.py but serves as the
base for the runtime provider system (registry + forwarding).

Port of: models.rs AiProvider struct L58-L66
"""

from dataclasses import dataclass, field


@dataclass
class Provider:
    """
    Base dataclass for all AI providers.

    Fields match the config.json ``providers[]`` entry exactly.
    Subclasses (e.g. OpenAICompatProvider) add protocol-specific
    behaviour (forward_request, etc.).

    Attributes:
        name:       Human-readable identifier, e.g. "opencode-zen".
        base_url:   Base URL of the external API, e.g.
                    "https://opencode.ai/zen/v1".
        api_key:    User's API key for this provider.
        protocol:   Wire protocol — "openai" is the only supported value in v1.
                    Future: "gemini" | "claude".
        enabled:    If False the registry skips this provider during routing.
        model_map:  IDE model key → external provider model name.
                    Keys are the exact internal model key strings returned by
                    ``fetchAvailableModels`` (confirmed in Phase 4):
                    e.g. {"gpt-oss-120b-medium": "minimax-m2.5-free"}

    Reference: models.rs AiProvider struct L58-L66
    """

    name: str
    base_url: str
    api_key: str
    protocol: str                              # "openai" | "gemini" | "claude"
    enabled: bool
    model_map: dict[str, str] = field(default_factory=dict)
