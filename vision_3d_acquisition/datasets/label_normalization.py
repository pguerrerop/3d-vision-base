from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LabelNormalizationResult:
    semantic_labels: list[str]
    superclass_labels: list[str]
    normalized_class: str | None
    normalization_version: str
    unknown_tags: list[str]
    repaired_split_tags: list[str]


class LabelNormalizationService:
    def __init__(self, *, taxonomy_name: str = "mining_balls_v1", config_dir: Path | None = None) -> None:
        base = config_dir or Path("config") / "label_normalization"
        self._taxonomy_path = base / f"{taxonomy_name}.json"
        if not self._taxonomy_path.is_file():
            raise FileNotFoundError(f"Normalization taxonomy not found: {self._taxonomy_path}")
        payload = json.loads(self._taxonomy_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid normalization taxonomy payload: {self._taxonomy_path}")
        raw_version = payload.get("version")
        self.version = str(raw_version or taxonomy_name)
        mappings = payload.get("mappings") if isinstance(payload.get("mappings"), dict) else payload
        self._map: dict[str, dict[str, list[str]]] = {}
        self._alias_to_key: dict[str, str] = {}
        for raw_key, raw_value in mappings.items():
            if not isinstance(raw_value, dict):
                continue
            key = self._norm_key(str(raw_key))
            self._map[key] = {
                "semantic_labels": self._norm_list(raw_value.get("semantic_labels")),
                "superclass_labels": self._norm_list(raw_value.get("superclass_labels")),
            }
            self._alias_to_key[key] = key
            for alias in self._norm_list_raw(raw_value.get("aliases")):
                self._alias_to_key[self._norm_key(alias)] = key

    def normalize_tags(self, tags: list[str] | None) -> LabelNormalizationResult:
        resolved_tags, repaired = self._repair_split_tags(tags or [])
        semantic: list[str] = []
        superclass: list[str] = []
        unknown: list[str] = []
        for raw_tag in resolved_tags:
            tag = str(raw_tag).strip()
            if not tag:
                continue
            key = self._norm_key(tag)
            canonical_key = self._alias_to_key.get(key, key)
            mapped = self._map.get(canonical_key)
            if mapped is None:
                if tag not in unknown:
                    unknown.append(tag)
                continue
            for item in mapped["semantic_labels"]:
                if item not in semantic:
                    semantic.append(item)
            for item in mapped["superclass_labels"]:
                if item not in superclass:
                    superclass.append(item)
        normalized_class = semantic[0] if semantic else None
        return LabelNormalizationResult(
            semantic_labels=semantic,
            superclass_labels=superclass,
            normalized_class=normalized_class,
            normalization_version=self.version,
            unknown_tags=unknown,
            repaired_split_tags=repaired,
        )

    def apply_to_take_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        tags = metadata.get("tags") if isinstance(metadata.get("tags"), list) else []
        result = self.normalize_tags([str(item) for item in tags])
        next_payload = dict(metadata)
        next_payload["semantic_labels"] = result.semantic_labels
        next_payload["superclass_labels"] = result.superclass_labels
        next_payload["normalized_class"] = result.normalized_class
        next_payload["normalization_version"] = result.normalization_version
        warnings = next_payload.get("normalization_warnings")
        existing = [str(item) for item in warnings] if isinstance(warnings, list) else []
        warning_list = [
            item
            for item in existing
            if not item.startswith("UNMAPPED_TAG:") and not item.startswith("REPAIRED_SPLIT_TAG:")
        ]
        for repaired in result.repaired_split_tags:
            marker = f"REPAIRED_SPLIT_TAG:{repaired}"
            if marker not in warning_list:
                warning_list.append(marker)
        if result.unknown_tags:
            for tag in result.unknown_tags:
                marker = f"UNMAPPED_TAG:{tag}"
                if marker not in warning_list:
                    warning_list.append(marker)
        next_payload["normalization_warnings"] = warning_list
        return next_payload

    def _repair_split_tags(self, tags: list[str]) -> tuple[list[str], list[str]]:
        repaired_tags: list[str] = []
        diagnostics: list[str] = []
        i = 0
        while i < len(tags):
            current = str(tags[i]).strip()
            if not current:
                i += 1
                continue
            if i + 1 < len(tags):
                nxt = str(tags[i + 1]).strip()
                merged = self._try_merge_split(current, nxt)
                if merged is not None:
                    repaired_tags.append(merged)
                    diagnostics.append(f"{current} + {nxt} -> {merged}")
                    i += 2
                    continue
            repaired_tags.append(current)
            i += 1
        return repaired_tags, diagnostics

    def _try_merge_split(self, left: str, right: str) -> str | None:
        # Known repair for split tokenization: ok 2 (2 + 5") -> ok 2 (2,5")
        left_norm = self._norm_key(left)
        right_norm = self._norm_key(right)
        if left_norm == "ok 2 (2" and right_norm == '5")':
            merged = 'ok 2 (2,5")'
            if self._norm_key(merged) in self._alias_to_key:
                return merged
        return None

    @staticmethod
    def _norm_key(value: str) -> str:
        return " ".join(value.strip().lower().split())

    @staticmethod
    def _norm_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            text = str(item).strip().upper()
            if text and text not in out:
                out.append(text)
        return out

    @staticmethod
    def _norm_list_raw(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in out:
                out.append(text)
        return out
