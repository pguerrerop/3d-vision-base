#!/usr/bin/env python3
"""Run the real segmentation pipeline for the latest ready incoming take."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vision_3d_acquisition.apps.ball_inspection import run_ball_inspection_flow  # noqa: E402
from vision_3d_acquisition.calibration.calibration_storage import load_calibration  # noqa: E402
from vision_3d_acquisition.contracts.modalities import format_modalities, infer_modalities  # noqa: E402
from vision_3d_acquisition.processing.pipeline import load_processing_config, process_take  # noqa: E402
from vision_3d_acquisition.contracts.result import ProcessingProfileStage  # noqa: E402
from vision_3d_acquisition.poc.calibration_mode import AUTO_PLANE_WARNING, confirm_auto_mode  # noqa: E402
from vision_3d_acquisition.poc.summary import build_calibration_diagnostics, build_poc_run_summary  # noqa: E402
from vision_3d_acquisition.runtime_config import clear_default_calibration, resolve_calibration, set_default_calibration  # noqa: E402
from vision_3d_acquisition.vision_core.serialization import write_processing_outputs, write_result_json  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_unprocessed_take(data_dir: Path) -> Path | None:
    incoming = data_dir / "incoming"
    processed = data_dir / "processed"
    if not incoming.exists():
        return None
    candidates: list[tuple[str, Path]] = []
    for take_dir in incoming.iterdir():
        if not take_dir.is_dir() or not (take_dir / "READY").is_file():
            continue
        if (processed / take_dir.name / "DONE").exists():
            continue
        metadata = read_json(take_dir / "metadata.json") if (take_dir / "metadata.json").is_file() else {}
        candidates.append((metadata.get("created_at") or take_dir.name, take_dir))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Process the latest ready take with the real segmentation pipeline.")
    parser.add_argument("--data-dir", default="data", type=Path, help="Data root")
    parser.add_argument(
        "--engine",
        choices=("legacy", "native"),
        default="legacy",
        help="Processing engine: legacy compatibility path or stage-native ball inspection path",
    )
    parser.add_argument("--config", default=Path("config/processing.yaml"), type=Path, help="Processing config YAML")
    parser.add_argument("--calibration", default=None, type=Path, help="Calibration JSON to apply")
    parser.add_argument("--set-default-calibration", default=None, type=Path, help="Set runtime default calibration and exit")
    parser.add_argument("--clear-default-calibration", action="store_true", help="Clear runtime default calibration and exit")
    parser.add_argument("--require-calibration", action="store_true", help="Fail instead of using automatic plane estimation")
    parser.add_argument("--yes", action="store_true", help="Assume yes for interactive confirmation prompts")
    parser.add_argument("--non-interactive", action="store_true", help="Disable interactive prompts for automation/service runs")
    parser.add_argument("--allow-no-calibration", action="store_true", help="Allow automatic plane estimation without prompting")
    parser.add_argument("--profile", action="store_true", help="Print a detailed processing timing table")
    parser.add_argument("--skip-debug-images", action="store_true", help="Skip debug PNG rendering for production latency measurement")
    parser.add_argument(
        "--prefer-fast-cloud",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Prefer point_cloud.npz runtime input when present",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    for child in ("incoming", "processed", "state", "events"):
        (data_dir / child).mkdir(parents=True, exist_ok=True)

    if args.set_default_calibration is not None:
        status = set_default_calibration(args.set_default_calibration)
        print(f"default_calibration_file: {status.default_calibration_file}")
        return 0
    if args.clear_default_calibration:
        clear_default_calibration()
        print("default_calibration_file: none")
        return 0

    take_dir = latest_unprocessed_take(data_dir)
    if take_dir is None:
        print("No ready unprocessed take found.", file=sys.stderr)
        return 1

    output_dir = data_dir / "processed" / take_dir.name
    metadata = read_json(take_dir / "metadata.json") if (take_dir / "metadata.json").is_file() else {}
    modalities = infer_modalities(metadata, take_dir)
    print(f"Available modalities: {format_modalities(modalities)}")
    print(f"Pipeline: {'ball_inspection' if args.engine == 'native' else 'legacy_segmentation'}")
    print("Required modalities: point_cloud")
    if "point_cloud" not in modalities:
        print(f"Pipeline requires point_cloud, but take has {format_modalities(modalities)}.", file=sys.stderr)
        print("Processing aborted.", file=sys.stderr)
        return 1
    resolution = resolve_calibration(args.calibration)
    if resolution.warning:
        print(f"WARNING: {resolution.warning}", file=sys.stderr)
    calibration = resolution.calibration
    _print_calibration_selection(calibration, resolution.source)
    if not resolution.active:
        if args.require_calibration:
            print("error: calibration is required but no calibration file was specified.", file=sys.stderr)
            return 1
        print("No calibration selected. Using automatic plane estimation only.", file=sys.stderr)
        print(AUTO_PLANE_WARNING, file=sys.stderr)
        if args.engine != "legacy" and not args.allow_no_calibration and not confirm_auto_mode(yes=args.yes, non_interactive=args.non_interactive):
            print("Canceled.", file=sys.stderr)
            return 1
    try:
        if args.engine == "legacy":
            config = load_processing_config(args.config)
            result = process_take(
                take_dir,
                output_dir,
                config,
                calibration=calibration,
                calibration_resolution_source=resolution.source,
                skip_debug_images=args.skip_debug_images,
                prefer_fast_cloud=args.prefer_fast_cloud,
            )
            result_payload = result.model_dump(mode="json")
            event_start = time.perf_counter()
            try:
                write_processing_outputs(
                    data_dir,
                    output_dir,
                    result_payload,
                    runtime_message="Segmentation pipeline active; classification pending.",
                )
            finally:
                _append_profile_stage(result, "event/state_update", "output", event_start, time.perf_counter())
                metadata = read_json(take_dir / "metadata.json") if (take_dir / "metadata.json").is_file() else {}
                result.calibration_diagnostics = build_calibration_diagnostics(result.model_dump(mode="json"), metadata=metadata)
                result.poc_summary = build_poc_run_summary(
                    result.model_dump(mode="json"),
                    metadata=metadata,
                    output_dir=output_dir,
                    engine="legacy",
                )
                result_payload = result.model_dump(mode="json")
                write_result_json(output_dir / "result.json", result_payload)
            processed_take_id = result.take_id
        else:
            native_result = run_ball_inspection_flow(
                data_dir,
                take_id=take_dir.name,
                config_path=args.config,
                calibration=calibration,
                calibration_resolution_source=resolution.source,
                skip_debug_images=args.skip_debug_images,
                prefer_fast_cloud=args.prefer_fast_cloud,
            )
            output_dir = native_result.output_dir
            result_payload = native_result.result_payload
            processed_take_id = native_result.take_id
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"processed_take: {processed_take_id}")
    print(f"folder: {output_dir}")
    _print_poc_summary(result_payload.get("poc_summary"))
    if args.profile:
        _print_profile(result_payload.get("profiling"))
    return 0


def _append_profile_stage(result, name: str, category: str, started_at: float, ended_at: float) -> None:
    if result.profiling is None:
        return
    result.profiling.stages.append(
        ProcessingProfileStage(
            name=name,
            category=category,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=max(0.0, (ended_at - started_at) * 1000.0),
        )
    )
    result.profiling.total_ms += result.profiling.stages[-1].duration_ms
    result.profiling.debug_artifacts_ms = sum(stage.duration_ms for stage in result.profiling.stages if stage.category == "debug_artifacts")
    result.profiling.io_ms = sum(stage.duration_ms for stage in result.profiling.stages if stage.category == "io")
    result.profiling.production_ms = sum(
        stage.duration_ms for stage in result.profiling.stages if stage.category not in {"debug_artifacts", "output"}
    )
    result.timing_ms.total = int(round(result.profiling.total_ms))


def _print_profile(profiling: dict | None) -> None:
    if not profiling:
        print("profiling: unavailable")
        return
    print()
    print("Timing profile")
    print(f"  total_ms: {profiling['total_ms']:.1f}")
    print(f"  production_ms: {profiling['production_ms']:.1f}")
    print(f"  debug_artifacts_ms: {profiling['debug_artifacts_ms']:.1f}")
    print(f"  io_ms: {profiling['io_ms']:.1f}")
    print()
    print(f"{'stage':34} {'category':22} {'ms':>10} {'in pts':>10} {'out pts':>10} notes")
    print("-" * 104)
    for stage in profiling["stages"]:
        in_points = "-" if stage.get("input_points") is None else str(stage["input_points"])
        out_points = "-" if stage.get("output_points") is None else str(stage["output_points"])
        notes = stage.get("notes") or ""
        print(f"{stage['name'][:34]:34} {stage['category'][:22]:22} {stage['duration_ms']:10.1f} {in_points:>10} {out_points:>10} {notes}")


def _print_calibration_selection(calibration: Path | None, source: str) -> None:
    if calibration is None:
        print("Calibration: none")
        print("Status: OK")
        return
    try:
        calibration_model = load_calibration(calibration)
        calibration_type = calibration_model.calibration_type
    except Exception:
        calibration_type = "plane_3d"
    print(f"Calibration: {calibration_type} from {calibration} ({source})")
    print("Status: OK")


def _print_poc_summary(summary: dict | None) -> None:
    if not summary:
        return
    status = summary.get("status") or {}
    calibration = summary.get("calibration") or {}
    objects = summary.get("objects") or {}
    profiling = summary.get("profiling") or {}
    warnings = summary.get("warnings") or []
    print("poc_summary:")
    print(f"  ENGINE: {summary.get('engine')}")
    print(f"  CALIBRATION: {calibration.get('display')}")
    print(f"  STATUS: {calibration.get('status')}")
    print(f"  demo_ready: {status.get('demo_ready')}")
    print(f"  trustworthy: {status.get('trustworthy')}")
    print(f"  objects: {objects.get('found')} found, {objects.get('accepted')} accepted, {objects.get('rejected')} rejected")
    print(f"  total_ms: {profiling.get('total_processing_ms')}")
    print(f"  warnings: {', '.join(warnings) if warnings else 'none'}")


if __name__ == "__main__":
    raise SystemExit(main())
