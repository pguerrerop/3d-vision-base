#!/usr/bin/env python3
"""Run ball-inspection application pipeline against a capture source."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vision_3d_acquisition.apps.ball_inspection import run_ball_inspection_flow  # noqa: E402
from vision_3d_acquisition.calibration.calibration_storage import load_calibration  # noqa: E402
from vision_3d_acquisition.contracts.modalities import format_modalities, infer_modalities  # noqa: E402
from vision_3d_acquisition.poc.calibration_mode import AUTO_PLANE_WARNING, confirm_auto_mode  # noqa: E402
from vision_3d_acquisition.runtime_config import clear_default_calibration, resolve_calibration, set_default_calibration  # noqa: E402


@dataclass(frozen=True)
class TakeResolution:
    take_id: str
    source: str


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_take(data_dir: Path, explicit_take_id: str | None) -> TakeResolution:
    if explicit_take_id:
        return TakeResolution(take_id=explicit_take_id, source="explicit")

    latest_unprocessed = _latest_ready_take(data_dir, require_unprocessed=True)
    if latest_unprocessed is not None:
        return TakeResolution(take_id=latest_unprocessed.name, source="latest_unprocessed")

    latest = _latest_ready_take(data_dir, require_unprocessed=False)
    if latest is not None:
        return TakeResolution(take_id=latest.name, source="latest")

    raise FileNotFoundError("No ready takes found.")


def _latest_ready_take(data_dir: Path, *, require_unprocessed: bool) -> Path | None:
    incoming = data_dir / "incoming"
    processed = data_dir / "processed"
    if not incoming.exists():
        return None
    candidates: list[tuple[str, Path]] = []
    for take_dir in incoming.iterdir():
        if not take_dir.is_dir() or not (take_dir / "READY").is_file():
            continue
        if require_unprocessed and (processed / take_dir.name / "DONE").exists():
            continue
        metadata = read_json(take_dir / "metadata.json") if (take_dir / "metadata.json").is_file() else {}
        candidates.append((str(metadata.get("created_at") or take_dir.name), take_dir))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ball inspection pipeline.")
    parser.add_argument("--data-dir", default="data", type=Path, help="Data root")
    parser.add_argument("--take-id", default=None, help="Specific capture/take id override")
    parser.add_argument("--config", default=Path("config/processing.yaml"), type=Path, help="Processing config YAML")
    parser.add_argument("--calibration", default=None, type=Path, help="Calibration JSON to apply")
    parser.add_argument("--set-default-calibration", default=None, type=Path, help="Set runtime default calibration and exit")
    parser.add_argument("--clear-default-calibration", action="store_true", help="Clear runtime default calibration and exit")
    parser.add_argument("--require-calibration", action="store_true", help="Fail instead of using automatic plane estimation")
    parser.add_argument("--yes", action="store_true", help="Assume yes for interactive confirmation prompts")
    parser.add_argument("--non-interactive", action="store_true", help="Disable interactive prompts for automation/service runs")
    parser.add_argument("--allow-no-calibration", action="store_true", help="Allow automatic plane estimation without prompting")
    parser.add_argument("--skip-debug-images", action="store_true", help="Skip debug PNG rendering")
    parser.add_argument(
        "--prefer-fast-cloud",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Prefer point_cloud.npz runtime input when present",
    )
    parser.add_argument("--profile", action="store_true", help="Print app pipeline profiling table")
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

    try:
        take_resolution = resolve_take(data_dir, args.take_id)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    calibration_resolution = resolve_calibration(args.calibration)
    if calibration_resolution.warning:
        print(f"WARNING: {calibration_resolution.warning}", file=sys.stderr)
    calibration = calibration_resolution.calibration
    metadata = read_json(data_dir / "incoming" / take_resolution.take_id / "metadata.json")
    modalities = infer_modalities(metadata, data_dir / "incoming" / take_resolution.take_id)
    if "point_cloud" not in modalities:
        print(f"Pipeline requires point_cloud, but take has {format_modalities(modalities)}.", file=sys.stderr)
        print("Processing aborted.", file=sys.stderr)
        return 1
    if not calibration_resolution.active:
        if args.require_calibration:
            print("error: calibration is required but no calibration file was specified.", file=sys.stderr)
            return 1
        print("No calibration selected. Using automatic plane estimation only.", file=sys.stderr)
        print(AUTO_PLANE_WARNING, file=sys.stderr)
        if not args.allow_no_calibration and not confirm_auto_mode(yes=args.yes, non_interactive=args.non_interactive):
            print("Canceled.", file=sys.stderr)
            return 1
    _print_selection(take_resolution, calibration, args.calibration is not None, calibration_resolution.source, modalities)
    try:
        result = run_ball_inspection_flow(
            data_dir,
            take_id=take_resolution.take_id,
            config_path=args.config,
            calibration=calibration,
            calibration_resolution_source=calibration_resolution.source,
            skip_debug_images=args.skip_debug_images,
            prefer_fast_cloud=args.prefer_fast_cloud,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"processed_take: {result.take_id}")
    print(f"folder: {result.output_dir}")
    _print_poc_summary(result.result_payload.get("poc_summary"))
    if args.profile:
        print("pipeline_profile:")
        for stage in result.profiling.get("stages", []):
            print(f"  - {stage['name']}: {stage['duration_ms']:.2f} ms ({stage['category']})")
    return 0


def _print_selection(
    take_resolution: TakeResolution,
    calibration: Path | None,
    explicit_calibration: bool,
    calibration_source: str,
    modalities: list[str],
) -> None:
    if take_resolution.source == "explicit":
        print("Selected take:")
    else:
        print("Selected latest take:")
    print(take_resolution.take_id)
    print()
    print(f"Available modalities: {format_modalities(modalities)}")
    print("Pipeline: ball_inspection")
    print("Required modalities: point_cloud")
    print("Calibration:")
    if calibration is None:
        print("NONE")
    else:
        source_label = "explicit" if explicit_calibration else calibration_source
        try:
            calibration_model = load_calibration(calibration)
            calibration_type = calibration_model.calibration_type
        except Exception:
            calibration_type = "plane_3d"
        print(f"{calibration_type} from {calibration} ({source_label})")
    print("Status: OK")
    print()
    print("Engine:")
    print("native")
    print()
    print("Processing...")
    print()


def _print_poc_summary(summary: dict | None) -> None:
    if not summary:
        return
    status = summary.get("status") or {}
    calibration = summary.get("calibration") or {}
    objects = summary.get("objects") or {}
    warnings = summary.get("warnings") or []
    print("poc_summary:")
    print(f"  ENGINE: {summary.get('engine')}")
    print(f"  CALIBRATION: {calibration.get('display')}")
    print(f"  STATUS: {calibration.get('status')}")
    print(f"  demo_ready: {status.get('demo_ready')}")
    print(f"  calibration_valid: {status.get('calibration_valid')}")
    print(f"  objects: {objects.get('found')} found, {objects.get('accepted')} accepted, {objects.get('rejected')} rejected")
    print(f"  warnings: {', '.join(warnings) if warnings else 'none'}")


if __name__ == "__main__":
    raise SystemExit(main())
