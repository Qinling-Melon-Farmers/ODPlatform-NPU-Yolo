"""Merge runtime configuration from defaults, YAML and CLI sources."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ValidationError


class ConfigSource(str, Enum):
    """Known runtime config sources, ordered by precedence in callers."""

    DEFAULT = "DEFAULT"
    YAML = "YAML"
    CLI = "CLI"


@dataclass
class ConfigMetadata:
    """Source metadata for one effective field value."""

    key: str
    value: Any
    source: ConfigSource | str
    timestamp: datetime
    overridden_from: ConfigMetadata | None = None

    @property
    def source_label(self) -> str:
        return self.source.value if isinstance(self.source, ConfigSource) else str(self.source)

    def chain(self) -> list[ConfigMetadata]:
        items = [self]
        current = self.overridden_from
        while current is not None:
            items.append(current)
            current = current.overridden_from
        return items

    def chain_str(self) -> str:
        return " <- ".join(f"{item.value}({item.source_label})" for item in self.chain())


class ConfigMerger:
    """Merge config dictionaries and retain source lineage for audit logs."""

    def __init__(self, *, track_sources: bool = True) -> None:
        self.track_sources = track_sources
        self._metadata: dict[str, ConfigMetadata] = {}
        self._overridden_keys: list[str] = []

    def preview(
        self,
        config_class: type[BaseModel],
        *,
        sources: list[tuple[ConfigSource | str, dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        """Merge sources without validating into a Pydantic model."""
        return self._do_merge(config_class, sources or [])

    def merge(
        self,
        config_class: type[BaseModel],
        *,
        sources: list[tuple[ConfigSource | str, dict[str, Any]]] | None = None,
    ) -> BaseModel:
        """Merge sources and validate the result using config_class."""
        merged = self._do_merge(config_class, sources or [])
        try:
            return config_class.model_validate(merged)
        except ValidationError as exc:
            self._add_source_notes(exc)
            raise

    def _do_merge(
        self,
        config_class: type[BaseModel],
        sources: list[tuple[ConfigSource | str, dict[str, Any]]],
    ) -> dict[str, Any]:
        self._metadata.clear()
        self._overridden_keys.clear()
        merged: dict[str, Any] = {}
        self._apply_source(merged, self._extract_defaults(config_class), ConfigSource.DEFAULT)
        for source, data in sources:
            self._apply_source(merged, data or {}, source)
        return merged

    @staticmethod
    def _extract_defaults(config_class: type[BaseModel]) -> dict[str, Any]:
        try:
            model = config_class()
            return model.model_dump(exclude_none=True)
        except Exception:
            return {}

    def _apply_source(self, merged: dict[str, Any], data: dict[str, Any], source: ConfigSource | str) -> None:
        for key, value in data.items():
            if value is None:
                continue
            previous = self._metadata.get(key)
            if key in merged and previous is not None:
                self._overridden_keys.append(key)
            merged[key] = value
            if self.track_sources:
                self._metadata[key] = ConfigMetadata(
                    key=key,
                    value=value,
                    source=source,
                    timestamp=datetime.now(),
                    overridden_from=previous,
                )

    def get_metadata(self, key: str) -> ConfigMetadata | None:
        """Return source metadata for one field."""
        return self._metadata.get(key)

    def get_source_report(self) -> str:
        """Return a human-readable summary grouped by effective source."""
        if not self.track_sources:
            return "source tracking disabled"
        by_source: dict[str, list[str]] = {}
        for key, meta in self._metadata.items():
            by_source.setdefault(meta.source_label, []).append(key)
        lines = ["runtime config sources"]
        for label in (ConfigSource.CLI.value, ConfigSource.YAML.value, ConfigSource.DEFAULT.value):
            keys = sorted(by_source.get(label, []))
            if keys:
                lines.append(f"{label}: {', '.join(keys)}")
        custom = sorted(source for source in by_source if source not in {item.value for item in ConfigSource})
        for label in custom:
            lines.append(f"{label}: {', '.join(sorted(by_source[label]))}")
        return "\n".join(lines)

    def get_override_report(self) -> str:
        """Return fields whose values were overridden by higher-priority sources."""
        unique_keys = list(dict.fromkeys(self._overridden_keys))
        if not unique_keys:
            return "runtime config overrides: none"
        lines = ["runtime config overrides"]
        for key in unique_keys:
            meta = self._metadata.get(key)
            if meta is not None:
                lines.append(f"{key}: {meta.chain_str()}")
        return "\n".join(lines)

    def to_audit_log(self) -> dict[str, Any]:
        """Return machine-readable source metadata."""
        return {
            "fields": {
                key: {
                    "value": meta.value,
                    "source": meta.source_label,
                    "chain": [
                        {"value": item.value, "source": item.source_label}
                        for item in meta.chain()
                    ],
                }
                for key, meta in self._metadata.items()
            },
            "overridden": list(dict.fromkeys(self._overridden_keys)),
        }

    def _add_source_notes(self, exc: ValidationError) -> None:
        for error in exc.errors():
            loc = error.get("loc") or ()
            if not loc:
                continue
            key = str(loc[0])
            meta = self._metadata.get(key)
            if meta is not None:
                exc.add_note(f"runtime config source for {key}: {meta.chain_str()}")
