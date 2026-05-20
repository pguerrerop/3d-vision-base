#!/usr/bin/env python3
"""Small POC operations helpers for summaries, labels, validation, and exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vision_3d_acquisition.poc.exports import export_labeled_dataset_summary, export_object_metrics, write_rows  # noqa: E402
from vision_3d_acquisition.poc.labels import ALLOWED_LABELS, list_labeled_takes, load_labels, save_labels  # noqa: E402
from vision_3d_acquisition.poc.summary import build_poc_run_summary, validate_result_payload  # noqa: E402
from vision_3d_acquisition.contracts.modalities import infer_modalities  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="POC review, labeling, validation, and export helpers.")
    parser.add_argument("--data-dir", default="data", type=Path, help="Data root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser("summary", help="Print a POC run summary for a processed take")
    summary.add_argument("take_id")

    label = subparsers.add_parser("label", help="Write labels for a take")
    label.add_argument("take_id")
    label.add_argument("--label", action="append", required=True, choices=sorted(ALLOWED_LABELS), dest="labels")
    label.add_argument("--notes", default=None)
    label.add_argument("--reviewer", default=None)

    subparsers.add_parser("list-labels", help="List labeled takes")

    export_labels = subparsers.add_parser("export-labels", help="Export labeled dataset summary")
    export_labels.add_argument("--output", required=True, type=Path)

    export_objects = subparsers.add_parser("export-objects", help="Export object-level metrics")
    export_objects.add_argument("--take-id", default=None)
    export_objects.add_argument("--output", required=True, type=Path)

    validate = subparsers.add_parser("validate-result", help="Validate a take result.json contract")
    validate.add_argument("take_id")

    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    if args.command == "summary":
        payload = _result_payload(data_dir, args.take_id)
        metadata = _metadata(data_dir, args.take_id)
        summary_payload = build_poc_run_summary(
            payload,
            metadata=metadata,
            output_dir=data_dir / "processed" / args.take_id,
            engine=payload.get("processing_engine"),
        )
        summary_payload["input_modalities"] = payload.get("input_modalities") or infer_modalities(metadata, data_dir / "incoming" / args.take_id)
        print(json.dumps(summary_payload, indent=2))
        return 0
    if args.command == "label":
        payload = save_labels(data_dir, args.take_id, args.labels, notes=args.notes, reviewer=args.reviewer)
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "list-labels":
        print(json.dumps(list_labeled_takes(data_dir), indent=2))
        return 0
    if args.command == "export-labels":
        rows = export_labeled_dataset_summary(data_dir)
        write_rows(args.output, rows)
        print(f"wrote: {args.output}")
        return 0
    if args.command == "export-objects":
        rows = export_object_metrics(data_dir, take_id=args.take_id)
        write_rows(args.output, rows)
        print(f"wrote: {args.output}")
        return 0
    if args.command == "validate-result":
        validate_result_payload(_result_payload(data_dir, args.take_id))
        labels = load_labels(data_dir, args.take_id)
        print(json.dumps({"take_id": args.take_id, "valid": True, "labels_present": labels is not None}, indent=2))
        return 0
    return 1


def _result_payload(data_dir: Path, take_id: str) -> dict:
    path = data_dir / "processed" / take_id / "result.json"
    if not path.is_file():
        raise SystemExit(f"result not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata(data_dir: Path, take_id: str) -> dict | None:
    path = data_dir / "incoming" / take_id / "metadata.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
