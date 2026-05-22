#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vision_3d_acquisition.datasets import DatasetService, LabelNormalizationService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize take tags into canonical semantic labels.")
    parser.add_argument("--dataset", required=True, help="Dataset id")
    parser.add_argument("--taxonomy", default="mining_balls_v1", help="Taxonomy file name without extension")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--take-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-dir", default="data")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(str(args.data_dir)).expanduser().resolve()
    service = DatasetService(data_dir)
    if service.get_dataset(str(args.dataset)) is None:
        raise SystemExit(f"Dataset not found: {args.dataset}")
    normalizer = LabelNormalizationService(taxonomy_name=str(args.taxonomy))

    rows = service.iter_dataset_takes(str(args.dataset), session_id=(str(args.session_id) if args.session_id else None))
    normalized_count = 0
    unknown_counts: dict[str, int] = {}
    repaired_counts: dict[str, int] = {}
    semantic_counts: dict[str, int] = {}
    superclass_counts: dict[str, int] = {}

    for sid, take_id, payload in rows:
        if args.take_id and take_id != str(args.take_id):
            continue
        next_payload = normalizer.apply_to_take_metadata(payload)
        normalized_count += 1
        for tag in [item.split(":", 1)[1] for item in (next_payload.get("normalization_warnings") or []) if str(item).startswith("UNMAPPED_TAG:")]:
            unknown_counts[tag] = unknown_counts.get(tag, 0) + 1
        for item in [str(v).split(":", 1)[1] for v in (next_payload.get("normalization_warnings") or []) if str(v).startswith("REPAIRED_SPLIT_TAG:")]:
            repaired_counts[item] = repaired_counts.get(item, 0) + 1
        for label in (next_payload.get("semantic_labels") or []):
            key = str(label)
            semantic_counts[key] = semantic_counts.get(key, 0) + 1
        for label in (next_payload.get("superclass_labels") or []):
            key = str(label)
            superclass_counts[key] = superclass_counts.get(key, 0) + 1

        if not args.dry_run:
            service.upsert_take_metadata(
                take_id=take_id,
                dataset_id=str(args.dataset),
                session_id=sid,
                updates={
                    "semantic_labels": next_payload.get("semantic_labels") or [],
                    "superclass_labels": next_payload.get("superclass_labels") or [],
                    "normalized_class": next_payload.get("normalized_class"),
                    "normalization_version": next_payload.get("normalization_version"),
                    "normalization_warnings": next_payload.get("normalization_warnings") or [],
                },
                source_metadata={"session_id": sid},
            )

    summary = {
        "dataset_id": str(args.dataset),
        "taxonomy": str(args.taxonomy),
        "normalization_version": normalizer.version,
        "dry_run": bool(args.dry_run),
        "normalized_count": normalized_count,
        "unknown_tags": dict(sorted(unknown_counts.items(), key=lambda item: (-item[1], item[0]))),
        "repaired_split_tags": dict(sorted(repaired_counts.items(), key=lambda item: (-item[1], item[0]))),
        "superclass_distribution": dict(sorted(superclass_counts.items(), key=lambda item: (-item[1], item[0]))),
        "semantic_distribution": dict(sorted(semantic_counts.items(), key=lambda item: (-item[1], item[0]))),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
