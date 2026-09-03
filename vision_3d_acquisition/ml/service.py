from __future__ import annotations

import json
import random
import time
from datetime import UTC, datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vision_3d_acquisition.ml.backend_registry import get_available_backends, load_backend
from vision_3d_acquisition.ml.backends import load_model
from vision_3d_acquisition.ml.evaluation import evaluate_predictions
from vision_3d_acquisition.ml.features.extractor import export_rows, features_from_object, rows_to_matrix
from vision_3d_acquisition.ml.features.schemas import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, feature_ordering_hash, schema_fingerprint
from vision_3d_acquisition.ml.models import MLExperiment, MLDataset, ModelDeployment, TrainedModel
from vision_3d_acquisition.ml.splits import build_split
from vision_3d_acquisition.ml.runtime import MLRuntimeResolver
from vision_3d_acquisition.ml.deployments import DeploymentService
from vision_3d_acquisition.ml.storage import MLStorage
from vision_3d_acquisition.storage import db


@dataclass
class SplitResult:
    train: list[dict[str, Any]]
    validation: list[dict[str, Any]]
    test: list[dict[str, Any]]


class MLService:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.storage = MLStorage(data_dir)
        self.runtime = MLRuntimeResolver(data_dir)
        self.deployments = DeploymentService(data_dir)

    def list_datasets(self) -> list[dict[str, Any]]:
        return self.storage.list("datasets")

    def create_dataset(self, payload: dict[str, Any]) -> dict[str, Any]:
        ds = MLDataset.model_validate(payload)
        return self.storage.save("datasets", ds.id, ds.model_dump(mode="json"))

    def list_experiments(self) -> list[dict[str, Any]]:
        return self.storage.list("experiments")

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        return self.storage.get("models", model_id)

    def list_models(self) -> list[dict[str, Any]]:
        return self.storage.list("models")

    def build_rows_for_dataset(self, dataset_id: str) -> list[dict[str, Any]]:
        ds = self.storage.get("datasets", dataset_id)
        if not ds:
            raise ValueError(f"unknown dataset: {dataset_id}")
        filters = ds.get("filters") if isinstance(ds.get("filters"), dict) else {}
        include_datasets = set(str(x) for x in ds.get("source_dataset_ids", []))
        rows: list[dict[str, Any]] = []
        for metadata_path in (self.data_dir / "incoming").glob("*/metadata.json"):
            take_id = metadata_path.parent.name
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            take_mgmt = metadata.get("take_management") if isinstance(metadata.get("take_management"), dict) else {}
            dataset_id_value = str(take_mgmt.get("dataset_id") or metadata.get("dataset_id") or "")
            session_id = str(take_mgmt.get("session_id") or metadata.get("session_id") or "") or None
            if include_datasets and dataset_id_value not in include_datasets:
                continue
            if filters.get("exclude_archived", True) and bool(take_mgmt.get("is_archived")):
                continue
            if filters.get("validation_status") and str(take_mgmt.get("validation_status") or "") != str(filters.get("validation_status")):
                continue
            result_payload = self._load_result_for_take(take_id)
            if not isinstance(result_payload, dict):
                continue
            objects = result_payload.get("objects") if isinstance(result_payload.get("objects"), list) else []
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                labels = list(take_mgmt.get("labels") or [])
                row = features_from_object(
                    obj=obj,
                    take_id=take_id,
                    dataset_id=dataset_id_value or None,
                    session_id=session_id,
                    labels=[str(v) for v in labels],
                    validation_status=(str(take_mgmt.get("validation_status")) if take_mgmt.get("validation_status") else None),
                )
                row["expected_class"] = str(take_mgmt.get("expected_class") or metadata.get("expected_class") or obj.get("superclass") or "UNKNOWN")
                row["artifacts"] = {
                    "take": f"incoming/{take_id}",
                    "result": self._guess_result_path(take_id),
                }
                rows.append(row)
        return rows

    def run_experiment(self, payload: dict[str, Any]) -> dict[str, Any]:
        exp = MLExperiment.model_validate(payload).model_copy(update={"status": "running"})
        self.storage.save("experiments", exp.id, exp.model_dump(mode="json"))
        log_dir = self.data_dir / "ml" / "experiments" / exp.id / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._append_log(log_dir / "train.log", f"[start] experiment={exp.id} classifier={exp.classifier_type} split={exp.split_strategy}")

        rows = self.build_rows_for_dataset(exp.dataset_id)
        self._append_log(log_dir / "train.log", f"[features] rows={len(rows)}")
        paths = export_rows(rows, self.data_dir / "ml" / "evaluations" / exp.id, "features")
        split_payload = build_split(rows, strategy=exp.split_strategy, seed=int(exp.training_config.get("seed", 42)), output_dir=self.data_dir / "ml" / "evaluations" / exp.id)
        self._append_log(
            log_dir / "train.log",
            f"[split] train={len(split_payload['train'])} validation={len(split_payload['validation'])} test={len(split_payload['test'])}",
        )

        x_train, _ = rows_to_matrix(split_payload["train"])
        y_train = [str(r.get("expected_class") or "UNKNOWN") for r in split_payload["train"]]
        x_test, _ = rows_to_matrix(split_payload["test"])
        y_test = [str(r.get("expected_class") or "UNKNOWN") for r in split_payload["test"]]

        mode = str(payload.get("experiment_mode") or ("comparison_experiment" if payload.get("classifier_types") else "single"))
        backend_ids = [str(x) for x in payload.get("classifier_types", []) if str(x).strip()]
        if not backend_ids:
            backend_ids = [str(exp.classifier_type)]

        comparison_rows: list[dict[str, Any]] = []
        by_backend: dict[str, dict[str, Any]] = {}
        evaluation_backends: dict[str, str] = {}
        single_eval_artifacts: dict[str, str] = {}
        all_backend_meta = get_available_backends(include_placeholders=True)

        for backend_id in backend_ids:
            backend = load_backend(backend_id)
            t0 = time.perf_counter()
            backend.train(x_train, y_train, exp.training_config)
            train_ms = (time.perf_counter() - t0) * 1000.0
            p0 = time.perf_counter()
            y_pred = backend.predict(x_test)
            infer_ms = (time.perf_counter() - p0) * 1000.0

            failed_examples = []
            for row, yt, yp in zip(split_payload["test"], y_test, y_pred):
                if yt == yp:
                    continue
                failed_examples.append(
                    {
                        "take_id": row.get("take_id"),
                        "object_id": row.get("object_id"),
                        "truth": yt,
                        "predicted": yp,
                        "artifacts": row.get("artifacts"),
                    }
                )
            backend_eval_dir = (self.data_dir / "ml" / "evaluations" / exp.id) if len(backend_ids) == 1 else (self.data_dir / "ml" / "evaluations" / exp.id / backend_id)
            ev = evaluate_predictions(y_true=y_test, y_pred=y_pred, failed_examples=failed_examples, output_dir=backend_eval_dir)
            self._append_log(log_dir / "train.log", f"[evaluate] backend={backend_id} accuracy={ev['summary'].get('accuracy')} train_ms={train_ms:.3f} infer_ms={infer_ms:.3f}")

            model_id = f"model_{exp.id}" if len(backend_ids) == 1 else f"model_{exp.id}_{backend_id}"
            model_dir = self.data_dir / "ml" / "models" / model_id
            export_meta = {
                "split_strategy": exp.split_strategy,
                "model_id": model_id,
                "training_experiment_id": exp.id,
                "classifier_family": backend_id,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "feature_columns": FEATURE_COLUMNS,
                "training_config": exp.training_config,
                "metrics": ev["summary"],
                "feature_ordering_hash": feature_ordering_hash(FEATURE_COLUMNS),
                "schema_fingerprint": schema_fingerprint(FEATURE_SCHEMA_VERSION, FEATURE_COLUMNS),
                "training_duration_ms": train_ms,
                "inference_duration_ms": infer_ms,
            }
            model_paths = backend.export_model(model_dir, export_meta)

            model = TrainedModel(
                id=model_id,
                name=f"{exp.name} [{backend_id}]",
                classifier_family=backend_id,
                training_experiment_id=exp.id,
                feature_schema={"version": FEATURE_SCHEMA_VERSION, "columns": FEATURE_COLUMNS, "ordering_hash": feature_ordering_hash(FEATURE_COLUMNS), "schema_fingerprint": schema_fingerprint(FEATURE_SCHEMA_VERSION, FEATURE_COLUMNS), "feature_count": len(FEATURE_COLUMNS), "dtype": "float"},
                label_schema={"classes": sorted(set(y_train + y_test))},
                metrics_summary=ev["summary"],
            )
            backend_meta = next((row for row in all_backend_meta if str(row.get("id")) == backend_id), {})
            self.storage.save(
                "models",
                model.id,
                {
                    **model.model_dump(mode="json"),
                    "artifact_paths": model_paths,
                    "backend": {
                        "backend_name": model_paths.get("backend_name"),
                        "backend_version": model_paths.get("backend_version"),
                        "supported_model_formats": ["pickle"],
                        "feature_requirements": {
                            "requires_mm_calibrated_features": True,
                            "required_metric_fields": ["height_max_mm", "surface_sphere_fit_rmse_mm"],
                            "required_calibrated_fields": ["height_max_mm"],
                            "required_modalities": ["heightmap"],
                            "requires_mm_calibration": True,
                        },
                        "interpretability_level": backend_meta.get("interpretability_level"),
                        "training_speed": backend_meta.get("training_speed"),
                        "inference_speed": backend_meta.get("inference_speed"),
                        "calibration_requirements": backend_meta.get("calibration_requirements"),
                    },
                },
            )
            summary = dict(ev["summary"])
            summary["backend_id"] = backend_id
            summary["training_duration_ms"] = train_ms
            summary["inference_duration_ms"] = infer_ms
            comparison_rows.append(summary)
            by_backend[backend_id] = {
                "metrics_summary": summary,
                "evaluation_artifacts": ev["artifacts"],
                "model_id": model_id,
                "runtime_metadata": {
                    "backend_type": backend_id,
                    "inference_duration_ms": infer_ms,
                    "feature_count": len(FEATURE_COLUMNS),
                    "feature_schema_hash": feature_ordering_hash(FEATURE_COLUMNS),
                    "explanation_availability": "coefficients" if backend_id == "logistic_regression" else "feature_importance" if backend_id in {"random_forest", "gradient_boosting"} else "limited",
                },
            }
            evaluation_backends[backend_id] = str(backend_eval_dir)
            if len(backend_ids) == 1:
                single_eval_artifacts = dict(ev["artifacts"])

        ranked = sorted(
            comparison_rows,
            key=lambda r: (
                float(r.get("f1_macro", 0.0)),
                float(r.get("precision_macro", 0.0)),
                float(r.get("recall_macro", 0.0)),
                -float(r.get("inference_duration_ms", 0.0)),
                -float(r.get("training_duration_ms", 0.0)),
            ),
            reverse=True,
        )
        comparison_summary = {
            "experiment_id": exp.id,
            "mode": mode,
            "backend_ids": backend_ids,
            "ranking": ranked,
            "best_backend_candidate": str(ranked[0].get("backend_id")) if ranked else None,
            "ranking_policy": ["f1_macro", "precision_macro", "recall_macro", "inference_duration_ms", "training_duration_ms"],
            "by_backend": by_backend,
        }
        comparison_summary_path = self.data_dir / "ml" / "evaluations" / exp.id / "comparison_summary.json"
        comparison_summary_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_summary_path.write_text(json.dumps(comparison_summary, indent=2) + "\n", encoding="utf-8")

        completed = exp.model_copy(
            update={
                "status": "completed",
                "metrics_summary": ranked[0] if ranked else {},
                "artifact_paths": {
                    **single_eval_artifacts,
                    **paths,
                    "comparison_summary": str(comparison_summary_path),
                    "split_manifest": split_payload["split_manifest_path"],
                    "leakage_report": split_payload["leakage_report_path"],
                    "evaluation_backends": json.dumps(evaluation_backends),
                },
            }
        )
        # The files are useful artifacts, but this immutable DB row is the
        # decision record used to compare iterations and report n/recall later.
        conn = db.catalog_for_process(self.data_dir)
        now = datetime.now(UTC).isoformat()
        for backend_id, backend_result in by_backend.items():
            evaluation_id = f"evaluation_{exp.id}_{backend_id}"
            conn.execute(
                "INSERT OR REPLACE INTO evaluation_run(id,experiment_id,dataset_id,classifier_id,split_strategy,recipe_snapshot_json,metrics_json,artifacts_json,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (evaluation_id, exp.id, exp.dataset_id, backend_id, exp.split_strategy,
                 json.dumps(payload.get("recipe_snapshot") or {}, ensure_ascii=False),
                 json.dumps(backend_result.get("metrics_summary") or {}, ensure_ascii=False),
                 json.dumps(backend_result.get("evaluation_artifacts") or {}, ensure_ascii=False), now),
            )
        self._append_log(log_dir / "train.log", "[done] status=completed")
        return self.storage.save("experiments", completed.id, completed.model_dump(mode="json"))

    def list_backends(self) -> list[dict[str, Any]]:
        return get_available_backends(include_placeholders=True)

    def promote_model(self, *, model_id: str, notes: str | None = None) -> dict[str, Any]:
        model = self.storage.get("models", model_id)
        if not model:
            raise ValueError(f"model not found: {model_id}")
        model["promoted"] = True
        model["deployment_notes"] = notes
        return self.storage.save("models", model_id, model)

    def deploy_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.deployments.create_deployment(payload)

    def activate_deployment(self, deployment_id: str) -> dict[str, Any]:
        return self.deployments.activate(deployment_id)

    def deactivate_deployment(self, deployment_id: str) -> dict[str, Any]:
        return self.deployments.deactivate(deployment_id)

    def rollback_deployment(self, deployment_id: str) -> dict[str, Any]:
        return self.deployments.rollback(deployment_id)

    def list_deployments(self) -> list[dict[str, Any]]:
        return self.deployments.list_deployments()

    def list_active_deployments(self) -> list[dict[str, Any]]:
        return self.deployments.list_active()

    def resolve_active_deployment(self, *, pipeline_id: str, stage_id: str, pipeline_family: str = "25d") -> dict[str, Any]:
        return self.runtime.resolve_active_deployment(pipeline_id=pipeline_id, stage_id=stage_id, pipeline_family=pipeline_family)

    def predict_row(self, *, model_id: str, row: dict[str, Any]) -> tuple[str, dict[str, float]]:
        model_meta = self.storage.get("models", model_id)
        if not model_meta:
            raise ValueError(f"model not found: {model_id}")
        model_path = Path(str((model_meta.get("artifact_paths") or {}).get("model_path") or ""))
        if not model_path.is_file():
            raise ValueError("model file missing")
        model = load_model(model_path)
        x = [[float(row.get(col) or 0.0) for col in FEATURE_COLUMNS]]
        y = str(model.predict(x)[0])
        probs: dict[str, float] = {}
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(x)[0]
            cls = [str(v) for v in model.classes_]
            probs = {cls[idx]: float(v) for idx, v in enumerate(p)}
        return y, probs

    def _split_rows(self, rows: list[dict[str, Any]], strategy: str, seed: int) -> SplitResult:
        shuffled = list(rows)
        random.Random(seed).shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        return SplitResult(train=shuffled[:n_train], validation=shuffled[n_train : n_train + n_val], test=shuffled[n_train + n_val :])

    def _load_result_for_take(self, take_id: str) -> dict[str, Any] | None:
        guessed = self._guess_result_path(take_id)
        if guessed and Path(guessed).is_file():
            try:
                return json.loads(Path(guessed).read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _guess_result_path(self, take_id: str) -> str | None:
        candidates = [
            self.data_dir / "acquisition" / "trispector_ftp" / "processed" / take_id / "result.json",
            self.data_dir / "processed" / take_id / "result.json",
        ]
        for path in candidates:
            if path.is_file():
                return str(path)
        found = list(self.data_dir.glob(f"**/{take_id}/result.json"))
        return str(found[0]) if found else None

    def dataset_insights(self, dataset_id: str) -> dict[str, Any]:
        rows = self.build_rows_for_dataset(dataset_id)
        class_counts: dict[str, int] = {}
        session_counts: dict[str, int] = {}
        dataset_counts: dict[str, int] = {}
        validation_counts: dict[str, int] = {}
        for row in rows:
            klass = str(row.get("expected_class") or "UNKNOWN")
            class_counts[klass] = class_counts.get(klass, 0) + 1
            sid = str(row.get("session_id") or "unknown")
            session_counts[sid] = session_counts.get(sid, 0) + 1
            did = str(row.get("dataset_id") or "unknown")
            dataset_counts[did] = dataset_counts.get(did, 0) + 1
            validation = str(row.get("validation_status") or "unknown")
            validation_counts[validation] = validation_counts.get(validation, 0) + 1
        warnings: list[str] = []
        if len(class_counts) < 2:
            warnings.append("insufficient_class_diversity")
        if any(v < 3 for v in class_counts.values()):
            warnings.append("small_class_warning")
        if class_counts:
            max_class = max(class_counts.values())
            min_class = min(class_counts.values())
            if min_class > 0 and (max_class / float(min_class)) > 5.0:
                warnings.append("class_imbalance_warning")
        if validation_counts.get("validated", 0) < max(1, int(len(rows) * 0.2)):
            warnings.append("low_validation_coverage")
        return {
            "dataset_id": dataset_id,
            "object_count": len(rows),
            "class_counts": class_counts,
            "session_counts": session_counts,
            "source_dataset_counts": dataset_counts,
            "validation_counts": validation_counts,
            "warnings": warnings,
        }

    def labels_preview(self, dataset_id: str, *, limit: int = 200) -> dict[str, Any]:
        rows = self.build_rows_for_dataset(dataset_id)
        items = [
            {
                "take_id": row.get("take_id"),
                "object_id": row.get("object_id"),
                "dataset_id": row.get("dataset_id"),
                "session_id": row.get("session_id"),
                "expected_class": row.get("expected_class"),
                "validation_status": row.get("validation_status"),
                "labels": row.get("labels", []),
                "artifacts": row.get("artifacts", {}),
            }
            for row in rows[: max(1, limit)]
        ]
        return {"dataset_id": dataset_id, "count": len(rows), "items": items}

    def features_preview(self, dataset_id: str, *, limit: int = 200) -> dict[str, Any]:
        rows = self.build_rows_for_dataset(dataset_id)
        schema = {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_count": len(FEATURE_COLUMNS),
            "feature_columns": FEATURE_COLUMNS,
            "feature_ordering_hash": feature_ordering_hash(FEATURE_COLUMNS),
            "schema_fingerprint": schema_fingerprint(FEATURE_SCHEMA_VERSION, FEATURE_COLUMNS),
        }
        preview: list[dict[str, Any]] = []
        missing_counts = {col: 0 for col in FEATURE_COLUMNS}
        for row in rows:
            for col in FEATURE_COLUMNS:
                if row.get(col) is None:
                    missing_counts[col] += 1
        for row in rows[: max(1, limit)]:
            preview.append({k: row.get(k) for k in ("take_id", "object_id", "dataset_id", "session_id", "expected_class", *FEATURE_COLUMNS)})
        diagnostics = {
            "missing_counts": missing_counts,
            "constant_features": [col for col in FEATURE_COLUMNS if len({r.get(col) for r in rows}) <= 1] if rows else [],
            "sparse_features": [col for col, n in missing_counts.items() if rows and (n / float(len(rows))) > 0.3],
        }
        return {"dataset_id": dataset_id, "count": len(rows), "schema": schema, "diagnostics": diagnostics, "rows": preview}

    def split_preview(self, dataset_id: str, *, strategy: str, seed: int) -> dict[str, Any]:
        rows = self.build_rows_for_dataset(dataset_id)
        preview_dir = self.data_dir / "ml" / "evaluations" / "_split_preview" / f"{dataset_id}_{strategy}_{seed}"
        payload = build_split(rows, strategy=strategy, seed=seed, output_dir=preview_dir)
        report = payload.get("leakage_report", {})
        warnings = list(report.get("warnings", [])) if isinstance(report.get("warnings"), list) else []
        confidence = "low" if warnings else "high"
        return {
            "dataset_id": dataset_id,
            "strategy": strategy,
            "seed": seed,
            "split_manifest": payload.get("split_manifest", {}),
            "leakage_report": report,
            "confidence_level": confidence,
        }

    def training_logs(self, experiment_id: str) -> dict[str, Any]:
        log_dir = self.data_dir / "ml" / "experiments" / experiment_id / "logs"
        entries: list[dict[str, Any]] = []
        if log_dir.is_dir():
            for path in sorted(log_dir.glob("*.log")):
                lines = path.read_text(encoding="utf-8").splitlines()
                entries.append({"file": path.name, "line_count": len(lines), "tail": lines[-200:]})
        return {"experiment_id": experiment_id, "logs": entries}

    def evaluation_preview(self, experiment_id: str) -> dict[str, Any]:
        base = self.data_dir / "ml" / "evaluations" / experiment_id
        files = {
            "summary": base / "evaluation_summary.json",
            "failed_examples": base / "failed_examples.json",
            "confusion_matrix": base / "confusion_matrix.csv",
            "per_class_metrics": base / "per_class_metrics.csv",
        }
        out: dict[str, Any] = {"experiment_id": experiment_id, "artifacts": {}}
        for key, path in files.items():
            out["artifacts"][key] = str(path) if path.is_file() else None
        out["summary"] = json.loads(files["summary"].read_text(encoding="utf-8")) if files["summary"].is_file() else {}
        out["failed_examples"] = json.loads(files["failed_examples"].read_text(encoding="utf-8")) if files["failed_examples"].is_file() else []
        out["confusion_matrix_csv"] = files["confusion_matrix"].read_text(encoding="utf-8") if files["confusion_matrix"].is_file() else ""
        out["per_class_metrics_csv"] = files["per_class_metrics"].read_text(encoding="utf-8") if files["per_class_metrics"].is_file() else ""
        comparison_path = base / "comparison_summary.json"
        out["comparison_summary"] = json.loads(comparison_path.read_text(encoding="utf-8")) if comparison_path.is_file() else {}
        return out

    def runtime_health_history(self, *, days: int = 14) -> dict[str, Any]:
        stats_dir = self.data_dir / "ml" / "runtime_stats"
        rows: list[dict[str, Any]] = []
        if stats_dir.is_dir():
            for path in sorted(stats_dir.glob("*.json"), reverse=True)[: max(1, days)]:
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    payload = {}
                rows.append({"day": path.stem, "stats": payload})
        return {"items": rows}

    def _append_log(self, path: Path, line: str) -> None:
        stamp = datetime.now(UTC).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fp:
            fp.write(f"{stamp} {line}\n")
