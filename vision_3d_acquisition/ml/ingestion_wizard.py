from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vision_3d_acquisition.datasets import DatasetService
from vision_3d_acquisition.ml.label_manifest_builder import build_canonical_manifest
from vision_3d_acquisition.ml.label_normalization import LABEL_SCHEMA_ID, normalize_label
from vision_3d_acquisition.ml.ml_set_materializer import materialize_ml_set
from vision_3d_acquisition.ml.object_grouping import assign_physical_object_groups
from vision_3d_acquisition.ml.range_expansion import expand_row_references
from vision_3d_acquisition.ml.take_reference_resolver import build_take_index, resolve_reference


class IngestionWizardService:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.service = DatasetService(data_dir)

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

        rows = self._parse_table(content, input_format)
        inferred = self._infer_schema(rows)
        parsed_rows = []
        for idx, row in enumerate(rows):
            parsed_rows.append({
                "source_row_index": idx + 1,
                "raw": row,
                "label": row.get(inferred["label_column"], "") if inferred.get("label_column") else "",
                "d1_mm": row.get(inferred["d1_column"], "") if inferred.get("d1_column") else "",
                "d2_mm": row.get(inferred["d2_column"], "") if inferred.get("d2_column") else "",
                "d3_mm": row.get(inferred["d3_column"], "") if inferred.get("d3_column") else "",
                "image_ref": row.get(inferred["image_ref_column"], "") if inferred.get("image_ref_column") else "",
                "from": row.get(inferred["from_column"], "") if inferred.get("from_column") else "",
                "to": row.get(inferred["to_column"], "") if inferred.get("to_column") else "",
            })

        self._write(run_dir / "parsed_rows.json", {"rows": parsed_rows, "schema": inferred})
        run["status"] = "table_ingested"
        self._write(run_dir / "state.json", run)
        return {"rows": parsed_rows, "schema": inferred, "warnings": self._schema_warnings(inferred)}

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
                norm = normalize_label(row.get("label"))
                meta = take_index.get(take_id, {})
                reconciled.append({
                    "source_row_index": row.get("source_row_index"),
                    "take_id": take_id,
                    "source_session_id": meta.get("session_id"),
                    "raw_operator_label": norm["raw_operator_label"],
                    "normalized_class": norm["normalized_class"],
                    "superclass": norm["superclass"],
                    "annotation_confidence": norm["annotation_confidence"],
                    "needs_review": norm["needs_review"],
                    "d1_mm": row.get("d1_mm") or None,
                    "d2_mm": row.get("d2_mm") or None,
                    "d3_mm": row.get("d3_mm") or None,
                    "resolution_method": match.status,
                    "created_from_range": len(refs) > 1,
                    "label_schema_id": LABEL_SCHEMA_ID,
                })

        grouped = assign_physical_object_groups(reconciled)
        payload = {
            "rows": grouped,
            "diagnostics": {
                "unresolved_take_refs": unresolved,
                "ambiguous_take_refs": ambiguous,
                "duplicate_take_assignments": duplicates,
                "unlabeled_pool": [row for row in grouped if not row.get("normalized_class")],
            },
        }
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
        result = build_canonical_manifest(run_dir, rows, diagnostics)
        run["status"] = "manifest_generated"
        self._write(run_dir / "state.json", run)
        return result

    def materialize(self, run_id: str, ml_set_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        run_dir = self.runs_dir / run_id
        reconciled = self._read(run_dir / "reconciled_rows.json")
        rows = list(reconciled.get("rows") or [])
        result = materialize_ml_set(
            data_dir=self.data_dir,
            ml_set_id=ml_set_id,
            dataset_id=run["dataset_id"],
            manifest_rows=rows,
            policy=run.get("policy") or {},
        )
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
                    "normalization_version": row.get("label_schema_id"),
                },
                source_metadata={"session_id": sid},
            )
            updated += 1
        return {"updated": updated}

    def _parse_table(self, content: str, input_format: str) -> list[dict[str, str]]:
        fmt = input_format.lower().strip()
        if fmt == "xlsx":
            raise ValueError("XLSX ingestion is planned; export as CSV/TSV for now.")
        delimiter = "\t" if fmt in {"tsv", "text", "paste"} else ","
        reader = csv.DictReader(io.StringIO(content), delimiter=delimiter)
        rows = []
        for row in reader:
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

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _write(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
