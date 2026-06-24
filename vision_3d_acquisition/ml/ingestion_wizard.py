from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vision_3d_acquisition.datasets import DatasetService
from vision_3d_acquisition.datasets.physical_objects import PhysicalObjectRegistry
from vision_3d_acquisition.ml.label_manifest_builder import build_canonical_manifest
from vision_3d_acquisition.ml.label_normalization import CLASS_MAP, LABEL_SCHEMA_ID, normalize_label
from vision_3d_acquisition.ml.ml_set_materializer import materialize_ml_set
from vision_3d_acquisition.ml.object_grouping import build_physical_object_grouping
from vision_3d_acquisition.ml.range_expansion import expand_row_references
from vision_3d_acquisition.ml.take_reference_resolver import build_take_index, resolve_reference

ALLOWED_LABEL_VALUES = {
    "good",
    "damaged",
    "deformed",
    "small",
    "partial",
    "scrap",
    "planchuela",
    "cubo",
    "duda",
    "empty",
    "no_object",
    "unknown",
}

ALLOWED_NORMALIZED_CLASSES = {
    "buena",
    "buena_menos",
    "ahuevada",
    "mitad",
    "chica",
    "bola_con_chip",
    "cadena",
    "duda",
    "chatarra",
    "pedazo",
    "parece_planchuela",
    "planchuela",
    "planchuela_doblada",
    "perno",
    "tuerca",
    "cubo",
}

ALLOWED_SUPERCLASSES = {
    "BALL_GOOD",
    "BALL_SCRAP",
    "SCRAP_METAL",
    "UNKNOWN",
    "REVIEW",
}


class IngestionWizardService:
    MAX_UPLOAD_BYTES = 5 * 1024 * 1024

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.service = DatasetService(data_dir)
        self.registry = PhysicalObjectRegistry(self._make_settings())

    @property
    def runs_dir(self) -> Path:
        path = self.data_dir / "ml_ingestion_runs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def create_run(self, *, dataset_id: str, session_ids: list[str], name: str | None = None) -> dict[str, Any]:
        run_id = f"ingestion_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}"
        payload = {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "session_ids": session_ids,
            "name": name or run_id,
            "status": "created",
            "created_at": datetime.now(UTC).isoformat(),
            "policy": {
                "include_uncertain": False,
                "include_empty_scene": False,
                "include_calibration_cube": True,
                "include_review_required": False,
                "include_unlabeled": False,
                "split_strategy": "by_physical_object_id",
                "split_ratios": {"train": 0.7, "validation": 0.15, "test": 0.15},
                "tasks": ["mining_ball_condition_v1"],
                "seed": 42,
            },
        }
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self._write(run_dir / "state.json", payload)
        return payload

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._read(self.runs_dir / run_id / "state.json")

    def ingest_table(self, run_id: str, *, content: str, input_format: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        run_dir = self.runs_dir / run_id
        (run_dir / "raw_table.txt").write_text(content, encoding="utf-8")

        payload = self._build_parsed_payload(
            content=content,
            input_format=input_format,
            provenance={
                "source_type": "pasted_text",
                "filename": None,
                "detected_format": "tsv" if input_format.lower().strip() in {"tsv", "text", "paste"} else input_format.lower().strip(),
                "byte_size": len(content.encode("utf-8")),
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            },
        )
        self._write(run_dir / "parsed_rows.json", payload)
        run["status"] = "table_ingested"
        self._write(run_dir / "state.json", run)
        return {
            "rows": payload["rows"],
            "schema": payload["schema"],
            "warnings": payload["warnings"],
            "diagnostics": payload["diagnostics"],
            "provenance": payload["provenance"],
        }

    def ingest_uploaded_file(self, run_id: str, *, filename: str, content_bytes: bytes) -> dict[str, Any]:
        run = self.get_run(run_id)
        run_dir = self.runs_dir / run_id
        safe_name = Path(filename or "uploaded_table").name
        extension = Path(safe_name).suffix.lower()
        if extension not in {".tsv", ".csv", ".txt"}:
            raise ValueError("Unsupported file extension. Use .tsv, .csv, or .txt.")
        if not content_bytes:
            raise ValueError("Uploaded file is empty.")
        if len(content_bytes) > self.MAX_UPLOAD_BYTES:
            raise ValueError(f"Uploaded file exceeds max size of {self.MAX_UPLOAD_BYTES} bytes.")
        try:
            content = content_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Uploaded file is not readable as UTF-8.") from exc
        if not content.strip():
            raise ValueError("Uploaded file is empty.")
        detected_format = self._detect_delimiter_format(content)
        (run_dir / safe_name).write_text(content, encoding="utf-8")
        payload = self._build_parsed_payload(
            content=content,
            input_format=detected_format,
            provenance={
                "source_type": "uploaded_file",
                "filename": safe_name,
                "detected_format": detected_format,
                "byte_size": len(content_bytes),
                "sha256": hashlib.sha256(content_bytes).hexdigest(),
            },
        )
        self._write(run_dir / "parsed_rows.json", payload)
        run["status"] = "table_ingested"
        self._write(run_dir / "state.json", run)
        return {
            "rows": payload["rows"],
            "schema": payload["schema"],
            "warnings": payload["warnings"],
            "diagnostics": payload["diagnostics"],
            "provenance": payload["provenance"],
        }

    def reconcile(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        run_dir = self.runs_dir / run_id
        parsed = self._read(run_dir / "parsed_rows.json")
        rows = parsed.get("rows", [])
        take_index = build_take_index(self.service, run["dataset_id"], run.get("session_ids") or None)

        reconciled: list[dict[str, Any]] = []
        unresolved: list[dict[str, Any]] = []
        ambiguous: list[dict[str, Any]] = []
        duplicates: list[dict[str, Any]] = []
        seen: dict[str, int] = {}

        for row in rows:
            refs = expand_row_references(row)
            if not refs:
                unresolved.append({"source_row_index": row.get("source_row_index"), "reason": "no_refs"})
                continue
            for ref in refs:
                match = resolve_reference(ref, take_index)
                if match.status == "ambiguous":
                    ambiguous.append({"source_row_index": row.get("source_row_index"), "ref": ref, "candidates": match.candidates})
                    continue
                if match.status in {"unresolved", "empty"}:
                    unresolved.append({"source_row_index": row.get("source_row_index"), "ref": ref})
                    continue
                take_id = str(match.matched_take_id)
                seen[take_id] = seen.get(take_id, 0) + 1
                if seen[take_id] > 1:
                    duplicates.append({"source_row_index": row.get("source_row_index"), "take_id": take_id})
                semantics = self._resolve_row_semantics(row)
                meta = take_index.get(take_id, {})
                reconciled.append({
                    "source_row_index": row.get("source_row_index"),
                    "source_row": row.get("source_row") or row.get("source_row_index"),
                    "take_id": take_id,
                    "image_ref": ref,
                    "source_session_id": meta.get("session_id"),
                    "raw_operator_label": semantics["raw_operator_label"],
                    "raw_label": semantics["raw_operator_label"],
                    "label": row.get("label") or semantics["raw_operator_label"],
                    "normalized_class": semantics["normalized_class"],
                    "superclass": semantics["superclass"],
                    "annotation_confidence": semantics["annotation_confidence"],
                    "needs_review": semantics["needs_review"],
                    "review_required": semantics["review_required"],
                    "label_policy": semantics["label_policy"],
                    "include": semantics["include"],
                    "default_trainable": semantics["default_trainable"],
                    "normalization_source": semantics["normalization_source"],
                    "normalization_version": semantics["normalization_version"],
                    "d1_mm": row.get("d1_mm") or None,
                    "d2_mm": row.get("d2_mm") or None,
                    "d3_mm": row.get("d3_mm") or None,
                    "resolution_method": match.status,
                    "created_from_range": len(refs) > 1,
                    "label_schema_id": LABEL_SCHEMA_ID,
                    "physical_object_id": row.get("physical_object_id") or row.get("object_id") or None,
                    "extra_fields": row.get("extra_fields") if isinstance(row.get("extra_fields"), dict) else {},
                    "normalization_warnings": semantics["warnings"],
                    "normalization_warning_codes": semantics["warning_codes"],
                })

        grouping = build_physical_object_grouping(reconciled)
        grouped = grouping["rows"]
        normalization_preview = self._build_normalization_preview(grouped)
        payload = {
            "rows": grouped,
            "diagnostics": {
                "unresolved_take_refs": unresolved,
                "ambiguous_take_refs": ambiguous,
                "duplicate_take_assignments": duplicates,
                "unlabeled_pool": [row for row in grouped if not row.get("normalized_class")],
                "object_conflicts": grouping["conflicts"],
                "grouping_summary": grouping["summary"],
                "normalization_preview": normalization_preview,
            },
        }
        self.registry.sync_from_manifest_rows(run["dataset_id"], grouped)
        self._write(run_dir / "reconciled_rows.json", payload)
        run["status"] = "reconciled"
        self._write(run_dir / "state.json", run)
        return payload

    def configure_policy(self, run_id: str, policy_updates: dict[str, Any]) -> dict[str, Any]:
        run = self.get_run(run_id)
        run["policy"] = {**(run.get("policy") or {}), **policy_updates}
        self._write(self.runs_dir / run_id / "state.json", run)
        return run["policy"]

    def generate_manifest(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        run_dir = self.runs_dir / run_id
        reconciled = self._read(run_dir / "reconciled_rows.json")
        rows = list(reconciled.get("rows") or [])
        diagnostics = dict(reconciled.get("diagnostics") or {})
        self._ensure_manifest_ready(rows, diagnostics)
        parsed = self._read(run_dir / "parsed_rows.json")
        provenance = dict(parsed.get("provenance") or {})
        result = build_canonical_manifest(run_dir, rows, diagnostics, provenance)
        run["status"] = "manifest_generated"
        self._write(run_dir / "state.json", run)
        return result

    def materialize(self, run_id: str, ml_set_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        run_dir = self.runs_dir / run_id
        reconciled = self._read(run_dir / "reconciled_rows.json")
        rows = list(reconciled.get("rows") or [])
        diagnostics = dict(reconciled.get("diagnostics") or {})
        self._ensure_manifest_ready(rows, diagnostics)
        result = materialize_ml_set(
            data_dir=self.data_dir,
            ml_set_id=ml_set_id,
            dataset_id=run["dataset_id"],
            manifest_rows=rows,
            policy=run.get("policy") or {},
        )
        if not bool(result.get("ok", False)):
            raise ValueError("; ".join(str(item) for item in (result.get("errors") or ["Materialization persistence verification failed."])))
        run["status"] = "materialized"
        run["ml_set_id"] = ml_set_id
        self._write(run_dir / "state.json", run)
        return result

    def apply_labels_to_metadata(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        run_dir = self.runs_dir / run_id
        reconciled = self._read(run_dir / "reconciled_rows.json")
        rows = list(reconciled.get("rows") or [])
        updated = 0
        for row in rows:
            take_id = str(row.get("take_id") or "")
            sid = str(row.get("source_session_id") or "")
            if not take_id or not sid:
                continue
            self.service.upsert_take_metadata(
                take_id=take_id,
                dataset_id=run["dataset_id"],
                session_id=sid,
                updates={
                    "expected_class": row.get("normalized_class"),
                    "physical_object_id": row.get("physical_object_id"),
                    "validation_status": "needs_review" if row.get("needs_review") else "valid",
                    "semantic_labels": [row.get("normalized_class")] if row.get("normalized_class") else [],
                    "superclass_labels": [row.get("superclass")] if row.get("superclass") else [],
                    "normalization_version": row.get("normalization_version") or row.get("label_schema_id"),
                },
                source_metadata={"session_id": sid},
            )
            updated += 1
        return {"updated": updated}

    def _make_settings(self):
        from vision_3d_acquisition.api.settings import ApiSettings

        settings = ApiSettings(
            data_dir=self.data_dir,
            incoming_dir=self.data_dir / "incoming",
            processed_dir=self.data_dir / "processed",
            state_dir=self.data_dir / "state",
            events_dir=self.data_dir / "events",
            sessions_dir=self.data_dir / "sessions",
            datasets_dir=self.data_dir / "datasets",
        )
        settings.ensure_directories()
        return settings

    def _parse_table(self, content: str, input_format: str) -> list[dict[str, str]]:
        fmt = input_format.lower().strip()
        if fmt == "xlsx":
            raise ValueError("XLSX ingestion is planned; export as CSV/TSV for now.")
        delimiter = "\t" if fmt in {"tsv", "text", "paste"} else ";" if fmt == "csv_semicolon" else ","
        reader = csv.DictReader(io.StringIO(content), delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("Missing header row.")
        headers = [str(item or "").strip() for item in reader.fieldnames]
        if not any(headers):
            raise ValueError("Missing header row.")
        rows = []
        for row in reader:
            if None in row:
                raise ValueError("Malformed CSV/TSV row.")
            if not any(str(v or "").strip() for v in row.values()):
                continue
            rows.append({str(k or "").strip(): str(v or "").strip() for k, v in row.items()})
        return rows

    def _infer_schema(self, rows: list[dict[str, str]]) -> dict[str, str]:
        keys = [k for row in rows for k in row.keys()]
        lower = {k.lower(): k for k in set(keys)}
        def find(*cands: str) -> str:
            for cand in cands:
                if cand in lower:
                    return lower[cand]
            return ""
        return {
            "label_column": find("label", "etiqueta", "clase"),
            "image_ref_column": find("image_ref", "image", "take", "foto", "id"),
            "object_id_column": find("object_id", "physical_object_id", "objeto", "object"),
            "source_row_column": find("source_row", "row", "table_row"),
            "raw_label_column": find("raw_label", "etiqueta_original", "original_label"),
            "normalized_class_column": find("normalized_class", "normalized_label", "canonical_class"),
            "superclass_column": find("superclass", "super_class", "class_group"),
            "label_policy_column": find("label_policy", "policy"),
            "review_required_column": find("review_required", "needs_review"),
            "include_column": find("include", "included", "use_for_training"),
            "d1_column": find("d1", "d1_mm"),
            "d2_column": find("d2", "d2_mm"),
            "d3_column": find("d3", "d3_mm"),
            "from_column": find("from", "start", "first"),
            "to_column": find("to", "end", "last"),
        }

    def _schema_warnings(self, schema: dict[str, str]) -> list[str]:
        warnings = []
        if not schema.get("label_column"):
            warnings.append("Could not infer label column.")
        if not schema.get("image_ref_column") and not schema.get("from_column"):
            warnings.append("Could not infer image reference columns.")
        return warnings

    def _build_parsed_payload(self, *, content: str, input_format: str, provenance: dict[str, Any]) -> dict[str, Any]:
        rows = self._parse_table(content, input_format)
        if not rows:
            raise ValueError("No data rows found.")
        inferred = self._infer_schema(rows)
        parsed_rows = []
        known_columns = {value for value in inferred.values() if value}
        headers = list(rows[0].keys())
        for idx, row in enumerate(rows):
            label_value = row.get(inferred["label_column"], "") if inferred.get("label_column") else ""
            raw_label_value = row.get(inferred["raw_label_column"], "") if inferred.get("raw_label_column") else ""
            parsed_rows.append({
                "source_row_index": idx + 1,
                "raw": row,
                "source_row": row.get(inferred["source_row_column"], "") if inferred.get("source_row_column") else "",
                "raw_label": raw_label_value,
                "label": label_value or raw_label_value,
                "object_id": row.get(inferred["object_id_column"], "") if inferred.get("object_id_column") else "",
                "physical_object_id": row.get(inferred["object_id_column"], "") if inferred.get("object_id_column") else "",
                "normalized_class": row.get(inferred["normalized_class_column"], "") if inferred.get("normalized_class_column") else "",
                "superclass": row.get(inferred["superclass_column"], "") if inferred.get("superclass_column") else "",
                "label_policy": row.get(inferred["label_policy_column"], "") if inferred.get("label_policy_column") else "",
                "review_required": self._parse_optional_bool(row.get(inferred["review_required_column"], "")) if inferred.get("review_required_column") else None,
                "include": self._parse_optional_bool(row.get(inferred["include_column"], "")) if inferred.get("include_column") else None,
                "d1_mm": row.get(inferred["d1_column"], "") if inferred.get("d1_column") else "",
                "d2_mm": row.get(inferred["d2_column"], "") if inferred.get("d2_column") else "",
                "d3_mm": row.get(inferred["d3_column"], "") if inferred.get("d3_column") else "",
                "image_ref": row.get(inferred["image_ref_column"], "") if inferred.get("image_ref_column") else "",
                "from": row.get(inferred["from_column"], "") if inferred.get("from_column") else "",
                "to": row.get(inferred["to_column"], "") if inferred.get("to_column") else "",
                "extra_fields": {key: value for key, value in row.items() if key not in known_columns and str(value or "").strip()},
                "source_type": provenance.get("source_type"),
                "source_filename": provenance.get("filename"),
                "source_detected_format": provenance.get("detected_format"),
                "source_byte_size": provenance.get("byte_size"),
                "source_sha256": provenance.get("sha256"),
            })
        warnings = self._schema_warnings(inferred)
        if warnings:
            if any("label column" in item for item in warnings):
                raise ValueError("Missing label column.")
            if any("image reference columns" in item for item in warnings):
                raise ValueError("Missing image_ref or range columns.")
        recognized_columns = sorted([
            key.replace("_column", "")
            for key, value in inferred.items()
            if value
        ])
        diagnostics = {
            "detected_format": provenance.get("detected_format"),
            "row_count": len(parsed_rows),
            "column_count": len(headers),
            "required_field_inference": {
                "label_column": bool(inferred.get("label_column")),
                "image_ref_or_range": bool(inferred.get("image_ref_column") or inferred.get("from_column")),
            },
            "recognized_semantic_columns": {key: value for key, value in inferred.items() if value},
            "recognized_semantic_column_names": recognized_columns,
            "ignored_extra_columns_count": len([header for header in headers if header not in known_columns]),
            "ignored_extra_columns": [header for header in headers if header not in known_columns],
        }
        full_provenance = {
            **provenance,
            "row_count": len(parsed_rows),
            "column_count": len(headers),
        }
        return {
            "rows": parsed_rows,
            "schema": inferred,
            "warnings": warnings,
            "diagnostics": diagnostics,
            "provenance": full_provenance,
        }

    @staticmethod
    def _detect_delimiter_format(content: str) -> str:
        first_line = content.splitlines()[0] if content.splitlines() else ""
        if "\t" in first_line:
            return "tsv"
        if ";" in first_line:
            return "csv_semicolon"
        if "," in first_line:
            return "csv"
        return "unknown"

    @staticmethod
    def _parse_optional_bool(value: Any) -> bool | None:
        raw = str(value or "").strip().lower()
        if not raw:
            return None
        if raw in {"true", "1", "yes", "y"}:
            return True
        if raw in {"false", "0", "no", "n"}:
            return False
        return None

    def _resolve_row_semantics(self, row: dict[str, Any]) -> dict[str, Any]:
        label_value = str(row.get("label") or "").strip()
        raw_label_value = str(row.get("raw_label") or "").strip()
        taxonomy = normalize_label(label_value)
        provided_normalized = str(row.get("normalized_class") or "").strip()
        provided_superclass = str(row.get("superclass") or "").strip()
        raw_operator_label = str(raw_label_value or taxonomy["raw_operator_label"] or label_value).strip()
        needs_review_override = self._parse_optional_bool(row.get("review_required"))
        include_override = self._parse_optional_bool(row.get("include"))
        warnings: list[str] = []
        warning_codes: list[str] = []
        has_table_canonical = bool(provided_normalized or provided_superclass)

        normalized_class = provided_normalized or str(taxonomy["normalized_class"] or "")
        superclass = provided_superclass or str(taxonomy["superclass"] or "")
        normalization_source = "table" if has_table_canonical else "taxonomy"
        if normalized_class == "REVIEW_REQUIRED" and normalization_source != "table":
            normalization_source = "needs_review"

        if provided_normalized and provided_normalized not in ALLOWED_NORMALIZED_CLASSES:
            warnings.append(f"Unknown normalized_class: {provided_normalized}")
            warning_codes.append("invalid_normalized_class")
        if provided_superclass and provided_superclass not in ALLOWED_SUPERCLASSES:
            warnings.append(f"Unknown superclass: {provided_superclass}")
            warning_codes.append("invalid_superclass")
        if not has_table_canonical:
            normalized_by_taxonomy = str(taxonomy["normalized_class"] or "")
            normalized_key = normalized_by_taxonomy.lower()
            label_key = label_value.lower().replace(" ", "_")
            if label_value and label_key not in ALLOWED_LABEL_VALUES and normalized_key == "review_required":
                warnings.append(f"Unmapped raw label: {label_value}")
                warning_codes.append("unmapped_raw_label")

        review_required = needs_review_override if needs_review_override is not None else bool(taxonomy["needs_review"])
        label_policy = str(row.get("label_policy") or "").strip().lower() or ("review" if review_required else "include")
        if label_policy == "review":
            review_required = True
        include_value = include_override if include_override is not None else label_policy != "exclude"
        if label_policy == "exclude":
            include_value = False
        default_trainable = bool(include_value) and label_policy == "include" and not review_required

        confidence = float(taxonomy["annotation_confidence"])
        if normalization_source == "table":
            confidence = 1.0 if not review_required else min(confidence, 0.6)

        return {
            "raw_operator_label": raw_operator_label,
            "normalized_class": normalized_class,
            "superclass": superclass,
            "annotation_confidence": confidence,
            "needs_review": bool(review_required),
            "review_required": bool(review_required),
            "label_policy": label_policy,
            "include": bool(include_value),
            "default_trainable": bool(default_trainable),
            "normalization_source": normalization_source,
            "normalization_version": LABEL_SCHEMA_ID,
            "warnings": warnings,
            "warning_codes": warning_codes,
        }

    @staticmethod
    def _build_normalization_preview(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        buckets: dict[tuple[str, str, str, str, str, bool, str], dict[str, Any]] = {}
        for row in rows:
            key = (
                str(row.get("raw_label") or row.get("raw_operator_label") or ""),
                str(row.get("label") or ""),
                str(row.get("normalized_class") or ""),
                str(row.get("superclass") or ""),
                str(row.get("label_policy") or ""),
                bool(row.get("review_required", row.get("needs_review", False))),
                str(row.get("normalization_source") or "needs_review"),
            )
            bucket = buckets.setdefault(key, {
                "raw_label": key[0],
                "label": key[1],
                "normalized_class": key[2],
                "superclass": key[3],
                "policy": key[4],
                "review_required": key[5],
                "source": key[6],
                "count": 0,
                "warnings": set(),
                "warning_codes": set(),
            })
            bucket["count"] += 1
            for warning in row.get("normalization_warnings") or []:
                bucket["warnings"].add(str(warning))
            for code in row.get("normalization_warning_codes") or []:
                bucket["warning_codes"].add(str(code))
        return [
            {**value, "warnings": sorted(value["warnings"]), "warning_codes": sorted(value["warning_codes"])}
            for value in sorted(buckets.values(), key=lambda item: (str(item["normalized_class"]), str(item["raw_label"])))
        ]

    @staticmethod
    def _ensure_manifest_ready(rows: list[dict[str, Any]], diagnostics: dict[str, Any]) -> None:
        blocking_errors: list[str] = []
        unresolved = list(diagnostics.get("unresolved_take_refs") or [])
        ambiguous = list(diagnostics.get("ambiguous_take_refs") or [])
        object_conflicts = list(diagnostics.get("object_conflicts") or [])
        if unresolved:
            blocking_errors.append(f"{len(unresolved)} unresolved refs remain.")
        if ambiguous:
            blocking_errors.append(f"{len(ambiguous)} ambiguous refs remain.")
        if object_conflicts:
            blocking_errors.append(f"{len(object_conflicts)} physical object conflicts remain.")

        missing_canonical = 0
        missing_physical_object_id = 0
        invalid_canonical = 0
        for row in rows:
            if not str(row.get("take_id") or "").strip() or not str(row.get("normalized_class") or "").strip():
                missing_canonical += 1
            if not str(row.get("physical_object_id") or "").strip():
                missing_physical_object_id += 1
            warning_codes = row.get("normalization_warning_codes") or []
            if any(code in {"invalid_normalized_class", "invalid_superclass"} for code in warning_codes):
                invalid_canonical += 1

        if missing_canonical:
            blocking_errors.append(f"{missing_canonical} rows are missing required canonical fields.")
        if missing_physical_object_id:
            blocking_errors.append(f"{missing_physical_object_id} rows are missing physical_object_id coverage.")
        if invalid_canonical:
            blocking_errors.append(f"{invalid_canonical} rows contain invalid normalized_class/superclass values.")

        if blocking_errors:
            raise ValueError("Cannot generate canonical manifest: " + " ".join(blocking_errors))

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _write(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
