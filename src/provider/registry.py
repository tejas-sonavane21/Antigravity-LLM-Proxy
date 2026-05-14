"""
src/provider/registry.py — Provider Registry
=============================================
Loads Provider instances from config and matches incoming IDE model
names to the correct provider + external target model.

Port of:
  - save/load_providers()        → simplified: config-driven, no persistence file
  - find_provider_for_model()    → provider.rs L153-L186

Matching strategy (provider.rs L153-L186):
  1. Skip disabled providers.
  2. Normalize both the incoming model name and each model_map key using
     normalize_model_name() (Phase 4, model_extractor.py).
  3. Try exact match (case-insensitive raw OR normalized).
  4. Try prefix match: normalized model starts with key + "-" or key + "@"
     e.g. "gpt-oss-120b-medium-latest" matches key "gpt-oss-120b-medium"
  5. First match wins (config order = priority order).
"""

import logging
from typing import TYPE_CHECKING

from .base import Provider
from src.converter.model_extractor import normalize_model_name

if TYPE_CHECKING:
    from src.config import ProviderConfig

log = logging.getLogger("provider.registry")


class ProviderRegistry:
    """
    Manages loaded providers and routes model names to providers.

    Usage::

        registry = ProviderRegistry()
        registry.load_from_config(config.providers)

        result = registry.find_provider_for_model("gpt-oss-120b-medium")
        if result:
            provider, target_model = result
            # -> (Provider(name="opencode-zen", ...), "minimax-m2.5-free")

    Port of: provider.rs find_provider_for_model() L153-L186
    """

    def __init__(self) -> None:
        self._providers: list[Provider] = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_from_config(self, provider_configs: list) -> None:
        """
        Build ``Provider`` instances from parsed ``ProviderConfig`` objects.

        Called once at startup after config is loaded.
        Replaces any previously loaded providers.

        Args:
            provider_configs: List of ``ProviderConfig`` dataclass instances
                              from ``src.config.load_config()``.
        """
        self._providers = []
        for pc in provider_configs:
            provider = Provider(
                name=pc.name,
                base_url=pc.base_url,
                api_key=pc.api_key,
                protocol=pc.protocol,
                enabled=pc.enabled,
                model_map=dict(pc.model_map),  # shallow copy — immutable strings
            )
            self._providers.append(provider)
            status = "ENABLED" if provider.enabled else "DISABLED"
            log.debug(
                f"  Loaded provider [{status}] {provider.name!r} "
                f"| protocol={provider.protocol} "
                f"| {len(provider.model_map)} model mapping(s)"
            )

        log.info(
            f"Provider registry ready: "
            f"{len(self.enabled_providers)} enabled, "
            f"{len(self.disabled_providers)} disabled"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def providers(self) -> list[Provider]:
        """All loaded providers (enabled + disabled)."""
        return list(self._providers)

    @property
    def enabled_providers(self) -> list[Provider]:
        """Only providers with ``enabled=True``."""
        return [p for p in self._providers if p.enabled]

    @property
    def disabled_providers(self) -> list[Provider]:
        """Only providers with ``enabled=False``."""
        return [p for p in self._providers if not p.enabled]

    # ------------------------------------------------------------------
    # Model matching
    # ------------------------------------------------------------------

    def find_provider_for_model(
        self, model_name: str
    ) -> tuple[Provider, str] | None:
        """
        Find a provider that has a ``model_map`` entry matching ``model_name``.

        Args:
            model_name: Raw model name as received from the IDE request.
                        E.g. "gpt-oss-120b-medium" or
                        "models/gpt-oss-120b-medium:streamGenerateContent"

        Returns:
            ``(provider, target_model)`` on success, where ``target_model``
            is the external model name to send in the OpenAI request.
            ``None`` if no enabled provider has a matching mapping.

        Matching logic (port of provider.rs L153-L186):
          1. Skip disabled providers.
          2. Normalize both incoming model and each model_map key.
          3. Exact match: case-insensitive raw OR normalized strings equal.
          4. Prefix match: normalized model starts with key + "-" or key + "@".
          5. First match across providers wins.
        """
        model_trimmed = model_name.strip()
        if not model_trimmed:
            return None

        model_norm = normalize_model_name(model_trimmed)

        for provider in self._providers:
            if not provider.enabled:
                continue

            for from_model, target_model in provider.model_map.items():
                from_trimmed = from_model.strip()
                if not from_trimmed:
                    continue

                from_norm = normalize_model_name(from_trimmed)
                if not from_norm:
                    continue

                # Exact match (case-insensitive) — provider.rs L176-L177
                exact = (
                    model_trimmed.lower() == from_trimmed.lower()
                    or model_norm == from_norm
                )

                # Prefix match — provider.rs L178-L179
                # e.g. "gpt-oss-120b-medium-2026" matches key "gpt-oss-120b-medium"
                prefix = model_norm.startswith(from_norm + "-") or model_norm.startswith(
                    from_norm + "@"
                )

                if exact or prefix:
                    log.debug(
                        f"  Model match: [{model_name!r}] → "
                        f"provider [{provider.name!r}] target [{target_model!r}] "
                        f"({'exact' if exact else 'prefix'})"
                    )
                    return (provider, target_model)

        # No provider matched
        log.debug(f"  No provider match for model [{model_name!r}] → pass-through")
        return None
