from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class ClassifierBackend(Protocol):
    backend_name: str
    backend_version: str
    supported_model_formats: list[str]
    feature_requirements: dict[str, Any]
    def train(self, x: list[list[float]], y: list[str], config: dict[str, Any]) -> None: ...
    def predict(self, x: list[list[float]]) -> list[str]: ...
    def predict_proba(self, x: list[list[float]]) -> list[dict[str, float]]: ...
    def export_model(self, model_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class SklearnBackend:
    classifier_type: str
    model: Any | None = None
    backend_name: str = "sklearn"
    backend_version: str = "1.x"
    supported_model_formats: list[str] = None  # type: ignore[assignment]
    feature_requirements: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.supported_model_formats is None:
            self.supported_model_formats = ["pickle"]
        if self.feature_requirements is None:
            self.feature_requirements = {"requires_mm_calibrated_features": True}

    def _build(self, config: dict[str, Any]) -> Any:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC

        if self.classifier_type == "logistic_regression":
            return LogisticRegression(max_iter=int(config.get("max_iter", 1000)))
        if self.classifier_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=int(config.get("n_estimators", 200)),
                random_state=int(config.get("seed", 42)),
            )
        if self.classifier_type == "gradient_boosting":
            return GradientBoostingClassifier(random_state=int(config.get("seed", 42)))
        return SVC(probability=True, random_state=int(config.get("seed", 42)))

    def train(self, x: list[list[float]], y: list[str], config: dict[str, Any]) -> None:
        model = self._build(config)
        model.fit(x, y)
        self.model = model

    def predict(self, x: list[list[float]]) -> list[str]:
        if self.model is None:
            return []
        return [str(v) for v in self.model.predict(x)]

    def predict_proba(self, x: list[list[float]]) -> list[dict[str, float]]:
        if self.model is None or not hasattr(self.model, "predict_proba"):
            return [{} for _ in x]
        probs = self.model.predict_proba(x)
        classes = [str(v) for v in self.model.classes_]
        return [{classes[idx]: float(val) for idx, val in enumerate(row)} for row in probs]

    def export_model(self, model_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pkl"
        with model_path.open("wb") as fh:
            pickle.dump(self.model, fh)
        metadata_path = model_dir / "model_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return {"model_path": str(model_path), "metadata_path": str(metadata_path), "backend_name": self.backend_name, "backend_version": self.backend_version}


class HeuristicBackend:
    backend_name: str = "heuristic"
    backend_version: str = "builtin-v1"
    supported_model_formats: list[str] = ["none"]
    feature_requirements: dict[str, Any] = {"requires_mm_calibrated_features": True}
    def train(self, x: list[list[float]], y: list[str], config: dict[str, Any]) -> None:
        return None

    def predict(self, x: list[list[float]]) -> list[str]:
        # Very simple threshold-compatible fallback; runtime heuristic remains canonical.
        out: list[str] = []
        for row in x:
            max_h = row[5] if len(row) > 5 else 0.0
            sph = row[1] if len(row) > 1 else 0.0
            out.append("BALL_GOOD" if max_h >= 8.0 and sph >= 0.8 else "BALL_SCRAP")
        return out

    def predict_proba(self, x: list[list[float]]) -> list[dict[str, float]]:
        return [{"BALL_GOOD": 0.75, "BALL_SCRAP": 0.25} for _ in x]

    def export_model(self, model_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
        model_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = model_dir / "model_metadata.json"
        metadata_path.write_text(json.dumps({**metadata, "backend": "heuristic"}, indent=2) + "\n", encoding="utf-8")
        return {"metadata_path": str(metadata_path), "backend_name": self.backend_name, "backend_version": self.backend_version}


def make_backend(classifier_type: str) -> ClassifierBackend:
    if classifier_type == "heuristic":
        return HeuristicBackend()
    return SklearnBackend(classifier_type=classifier_type)


def load_model(model_path: Path) -> Any:
    with model_path.open("rb") as fh:
        return pickle.load(fh)
