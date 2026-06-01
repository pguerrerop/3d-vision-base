from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MLStorage:
    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "ml"
        for name in ("datasets", "feature_sets", "experiments", "models", "evaluations", "deployments"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def _path(self, bucket: str, item_id: str) -> Path:
        return self.root / bucket / f"{item_id}.json"

    def save(self, bucket: str, item_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        path = self._path(bucket, item_id)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload

    def get(self, bucket: str, item_id: str) -> dict[str, Any] | None:
        path = self._path(bucket, item_id)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self, bucket: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for path in sorted((self.root / bucket).glob("*.json")):
            try:
                items.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return items

    def write_artifact(self, rel_path: str, payload: Any) -> str:
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return str(path)
