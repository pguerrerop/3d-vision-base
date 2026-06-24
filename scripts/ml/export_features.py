#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from vision_3d_acquisition.ml import MLService
from vision_3d_acquisition.ml.features.extractor import export_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    svc = MLService(Path(args.data_dir))
    rows = svc.build_rows_for_dataset(args.dataset_id)
    output_dir = Path(args.output_dir) if args.output_dir else (Path(args.data_dir) / "ml" / "evaluations" / f"export_{args.dataset_id}")
    print(export_rows(rows, output_dir, "features"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
