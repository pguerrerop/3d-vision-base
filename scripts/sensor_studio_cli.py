#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

PIPELINE_ID_25D = "mining_steel_ball_classification_25d"


def _settings(data_dir: Path):
    from vision_3d_acquisition.api.settings import ApiSettings  # noqa: WPS433

    settings = ApiSettings(
        data_dir=data_dir,
        incoming_dir=data_dir / "incoming",
        processed_dir=data_dir / "processed",
        state_dir=data_dir / "state",
        events_dir=data_dir / "events",
        sessions_dir=data_dir / "sessions",
        datasets_dir=data_dir / "datasets",
    )
    settings.ensure_directories()
    return settings


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validation_service(data_dir: Path):
    from vision_3d_acquisition.validation import ValidationService  # noqa: WPS433
    return ValidationService(_settings(data_dir))


def cmd_validation_list_suites(args: argparse.Namespace) -> int:
    print(json.dumps(_validation_service(args.data_dir).list_suites(), indent=2))
    return 0


def cmd_validation_compare(args: argparse.Namespace) -> int:
    result = _validation_service(args.data_dir).compare(args.baseline_id, candidate_run_id=args.candidate_run_id)
    print(json.dumps(result, indent=2))
    return 1 if result["status"] == "regression" else 0


def cmd_validation_run_suite(args: argparse.Namespace) -> int:
    report = _validation_service(args.data_dir).execute_suite(args.suite_id, {"candidate_run_id": args.candidate_run_id, "recipe_id": args.recipe_id})
    print(json.dumps(report, indent=2))
    return 1 if report["status"] in {"failed", "completed_with_regressions"} else 0


def cmd_validation_inspect_execution(args: argparse.Namespace) -> int:
    print(json.dumps(_validation_service(args.data_dir).get_execution(args.execution_id), indent=2))
    return 0


def cmd_validation_rebuild_indexes(args: argparse.Namespace) -> int:
    print(json.dumps(_validation_service(args.data_dir).rebuild_indexes(), indent=2))
    return 0


def cmd_validation_verify_integrity(args: argparse.Namespace) -> int:
    result = _validation_service(args.data_dir).verify_integrity()
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 2


def cmd_validation_inspect_suite(args: argparse.Namespace) -> int:
    print(json.dumps(_validation_service(args.data_dir).get_suite(args.suite_id), indent=2)); return 0


def cmd_validation_coverage(args: argparse.Namespace) -> int:
    print(json.dumps(_validation_service(args.data_dir).coverage(args.suite_id), indent=2)); return 0


def cmd_validation_archive_restore(args: argparse.Namespace) -> int:
    service = _validation_service(args.data_dir); result = service.restore_suite(args.suite_id) if args.command == "restore-suite" else service.archive_suite(args.suite_id)
    print(json.dumps(result, indent=2)); return 0


def cmd_validation_baseline_history(args: argparse.Namespace) -> int:
    print(json.dumps(_validation_service(args.data_dir).baseline_history(take_id=args.take_id, pipeline_id=args.pipeline_id, artifact_id=args.artifact_id), indent=2)); return 0


def cmd_validation_set_baseline_active(args: argparse.Namespace) -> int:
    print(json.dumps(_validation_service(args.data_dir).set_active(args.baseline_id, args.command == "activate-baseline"), indent=2)); return 0


def cmd_validation_promote(args: argparse.Namespace) -> int:
    result = _validation_service(args.data_dir).promote_comparison(args.execution_id, args.case_id, args.comparison_id, {"activate": args.activate, "notes": args.notes, "carry_forward_policy": not args.reset_policy, "reviewed_by": args.reviewed_by})
    print(json.dumps(result, indent=2)); return 0


def cmd_validation_update_suite(args: argparse.Namespace) -> int:
    payload = {key: value for key, value in {"name":args.name,"description":args.description,"status":args.status}.items() if value is not None}
    print(json.dumps(_validation_service(args.data_dir).update_suite(args.suite_id, payload), indent=2)); return 0


def cmd_validation_duplicate_suite(args: argparse.Namespace) -> int:
    print(json.dumps(_validation_service(args.data_dir).duplicate_suite(args.suite_id), indent=2)); return 0


def cmd_validation_case(args: argparse.Namespace) -> int:
    service=_validation_service(args.data_dir)
    resolution={"mode":args.resolution_mode}
    if args.resolution_mode=="pinned": resolution["baseline_id"]=args.baseline_id
    if args.resolution_mode=="allowed_versions": resolution["allowed_baseline_ids"]=args.allowed_baseline_id
    payload={"enabled":args.enabled,"tags":args.tag,"notes":args.notes,"baseline_resolution":resolution}
    if args.command=="add-case":
        payload.update({"take_id":args.take_id,"baseline_ids":args.baseline_id_list or ([args.baseline_id] if args.baseline_id else [])}); result=service.add_case(args.suite_id,payload)
    elif args.command=="update-case": result=service.update_case(args.suite_id,args.case_id,payload)
    else: result=service.delete_case(args.suite_id,args.case_id)
    print(json.dumps(result,indent=2)); return 0


def cmd_validation_list_executions(args: argparse.Namespace) -> int:
    rows=_validation_service(args.data_dir).list_executions()
    if args.suite_id: rows=[row for row in rows if row.get("suite_id")==args.suite_id]
    print(json.dumps(rows,indent=2)); return 0


def cmd_validation_smoke(args: argparse.Namespace) -> int:
    """Deterministic local smoke: acquisition, processing, approval, and comparison."""
    from vision_3d_acquisition.apps.ball_inspection_25d import run_ball_inspection_25d_flow  # noqa: WPS433
    from vision_3d_acquisition.synthetic import create_synthetic_25d_take  # noqa: WPS433
    take_id, _ = create_synthetic_25d_take(args.data_dir, session_id="validation_smoke", seed=25)
    run_ball_inspection_25d_flow(args.data_dir, take_id=take_id)
    store = _validation_service(args.data_dir)
    resolved = store._resolve(take_id=take_id, pipeline_id=PIPELINE_ID_25D, run_id="latest")
    candidates = ["final_selected_support_mask", "final_object_mask", "normalized_heightmap", "measurement_diagnostics", "classification_explanation"]
    artifact_map = {str(item.get("artifact_id")): item for item in resolved.get("artifacts", []) if isinstance(item, dict)}
    baseline_ids: list[str] = []
    for artifact_id in candidates:
        artifact = artifact_map.get(artifact_id)
        if not artifact:
            continue
        policy = {"comparator": "binary_mask", "min_iou": 0.97, "max_removed_fraction": 0.01} if artifact_id.endswith("mask") else {}
        if artifact_id == "final_object_mask": policy["blocksDownstreamOnFailure"] = True
        approved = store.create_baselines({"take_id": take_id, "pipeline_id": PIPELINE_ID_25D, "run_id": "latest", "approval_scope": "artifact", "stage_id": artifact.get("stage_id"), "processing_unit_id": artifact.get("processing_unit_id"), "artifact_id": artifact_id, "notes": "deterministic validation smoke", "comparison_policy": policy})
        baseline_ids.extend(item["id"] for item in approved["baselines"])
    if not baseline_ids:
        print(json.dumps({"ok": False, "take_id": take_id, "reason": "no comparison-eligible artifacts"}, indent=2))
        return 2
    suite = store.create_suite({"name": "validation smoke", "pipeline_id": PIPELINE_ID_25D, "description": "Temporary deterministic smoke suite"})
    # Keep the smoke suite focused on the deterministic segmentation contract;
    # the other approved artifacts establish coverage without making expected
    # downstream semantic changes obscure the intended first divergence.
    store.add_case(suite["id"], {"take_id": take_id, "baseline_ids": [baseline_ids[candidates.index("final_object_mask")]]})
    unchanged = store.execute_suite(suite["id"], {"candidate_run_id": "latest"})
    # Deliberately empty segmentation foreground. Baselines remain snapshots of
    # the first run while the normal pipeline creates a new candidate output.
    run_ball_inspection_25d_flow(args.data_dir, take_id=take_id, stage_params={"remove_belt_segment_objects": {"min_height_mm": 100.0}})
    changed = store.execute_suite(suite["id"], {"candidate_run_id": "latest"})
    changed_case = changed["cases"][0] if changed.get("cases") else {}
    divergence = changed_case.get("first_divergence") or {}
    expected = changed_case.get("status") == "regression" and divergence.get("artifact_id") == "final_object_mask"
    # Promotion is explicit: version 2 is created from the reviewed candidate;
    # the service deactivates v1 only as part of this explicit promotion.
    changed_artifact = next((item for item in store._resolve(take_id=take_id, pipeline_id=PIPELINE_ID_25D, run_id="latest").get("artifacts", []) if isinstance(item, dict) and item.get("artifact_id") == "final_object_mask"), None)
    promoted = store.create_baselines({"take_id": take_id, "pipeline_id": PIPELINE_ID_25D, "run_id": "latest", "approval_scope": "artifact", "stage_id": (changed_artifact or {}).get("stage_id"), "artifact_id": "final_object_mask", "notes": "smoke promotion v2", "comparison_policy": {"comparator":"binary_mask","min_iou":0.97,"max_removed_fraction":0.01}})["baselines"][0]
    cases_path = store.root / "suites" / suite["id"] / "cases.json"
    case_rows = _read_json(cases_path)
    old_mask_baseline = baseline_ids[candidates.index("final_object_mask")]
    for case_row in case_rows:
        case_row["baseline_ids"] = [promoted["id"] if item == old_mask_baseline else item for item in case_row.get("baseline_ids", [])]
    cases_path.write_text(json.dumps(case_rows, indent=2) + "\n", encoding="utf-8")
    promoted_report = store.execute_suite(suite["id"], {"candidate_run_id": "latest"})
    ok = unchanged["cases"][0]["status"] == "pass" and expected and promoted["version"] == 2 and store.get_baseline(baseline_ids[candidates.index("final_object_mask")])["status"] == "inactive" and promoted_report["cases"][0]["status"] == "pass"
    print(json.dumps({"ok": ok, "take_id": take_id, "baseline_count": len(baseline_ids), "unchanged_status": unchanged["cases"][0]["status"], "changed_status": changed_case.get("status"), "first_divergence": divergence, "promoted_baseline": {"id":promoted["id"],"version":promoted["version"],"status":promoted["status"]}, "post_promotion_status": promoted_report["cases"][0]["status"]}, indent=2))
    return 0 if ok else 1


def _print_summary(header: str, rows: list[tuple[str, object]]) -> None:
    print(header)
    for key, value in rows:
        print(f"  {key}: {value}")


def _latest_take_id(data_dir: Path) -> str | None:
    incoming = data_dir / "incoming"
    if not incoming.is_dir():
        return None
    candidates = [p for p in incoming.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].name


def _list_takes(data_dir: Path, limit: int | None) -> list[str]:
    incoming = data_dir / "incoming"
    if not incoming.is_dir():
        return []
    candidates = [p for p in incoming.iterdir() if p.is_dir() and not p.name.startswith(".")]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if limit is None:
        return [p.name for p in candidates]
    return [p.name for p in candidates[: max(1, limit)]]


def cmd_25d_create_synthetic(args: argparse.Namespace) -> int:
    from vision_3d_acquisition.synthetic import create_synthetic_25d_take  # noqa: WPS433

    created: list[tuple[str, Path]] = []
    for i in range(max(1, args.count)):
        take_id, take_dir = create_synthetic_25d_take(
            args.data_dir,
            session_id=args.session_id,
            include_reflectance=not args.no_reflectance,
            seed=25 + i,
        )
        created.append((take_id, take_dir))

    for take_id, take_dir in created:
        _print_summary(
            "Created synthetic 25D take",
            [
                ("take_id", take_id),
                ("take_dir", take_dir),
                ("session_id", args.session_id),
            ],
        )
    return 0


def _print_result_overview(data_dir: Path, take_id: str) -> int:
    result_path = data_dir / "processed" / take_id / "result.json"
    if not result_path.is_file():
        print(f"result not found: {result_path}", file=sys.stderr)
        return 2

    payload = _read_json(result_path)
    pipeline = payload.get("processing_pipeline") or {}
    summary = payload.get("summary") or {}
    objects = payload.get("objects") or []
    artifacts = payload.get("artifacts") or []

    _print_summary(
        "25D result summary",
        [
            ("take_id", take_id),
            ("status", payload.get("status")),
            ("pipeline_id", pipeline.get("id")),
            ("pipeline_family", pipeline.get("pipeline_family")),
            ("object_count", summary.get("object_count", len(objects))),
            ("result_path", result_path),
        ],
    )

    print("Objects:")
    if not objects:
        print("  - none")
    for item in objects:
        heights = item.get("height_above_belt_mm") or {}
        print(
            "  - id={id} class={cls} label={label} max_h={max_h} mean_h={mean_h} p95_h={p95_h}".format(
                id=item.get("object_id"),
                cls=item.get("class_name"),
                label=item.get("label"),
                max_h=heights.get("max_height_mm"),
                mean_h=heights.get("mean_height_mm"),
                p95_h=heights.get("p95_height_mm"),
            )
        )

    print("Artifacts:")
    if not artifacts:
        print("  - none")
    for artifact in artifacts:
        artifact_id = artifact.get("artifact_id")
        rel_path = artifact.get("path")
        artifact_path = (result_path.parent / rel_path) if rel_path else "-"
        print(f"  - {artifact_id}: {artifact_path}")

    return 0


def cmd_25d_demo(args: argparse.Namespace) -> int:
    from vision_3d_acquisition.apps.ball_inspection_25d import run_ball_inspection_25d_flow  # noqa: WPS433
    from vision_3d_acquisition.synthetic import create_synthetic_25d_take  # noqa: WPS433

    take_id = args.take_id
    if not take_id:
        take_id, take_dir = create_synthetic_25d_take(args.data_dir, session_id=args.session_id)
        _print_summary(
            "Created synthetic input for demo",
            [
                ("take_id", take_id),
                ("take_dir", take_dir),
                ("session_id", args.session_id),
            ],
        )

    _print_summary("Running 25D demo pipeline", [("take_id", take_id), ("pipeline_id", PIPELINE_ID_25D)])
    run_ball_inspection_25d_flow(args.data_dir, take_id=take_id)
    return _print_result_overview(args.data_dir, take_id)


def cmd_25d_process(args: argparse.Namespace) -> int:
    from vision_3d_acquisition.apps.ball_inspection_25d import run_ball_inspection_25d_flow  # noqa: WPS433

    take_dir = args.data_dir / "incoming" / args.take_id
    if not take_dir.is_dir():
        print(f"take not found in incoming/: {take_dir}", file=sys.stderr)
        return 2
    if not (take_dir / "READY").is_file():
        print(f"take is missing READY marker: {take_dir / 'READY'}", file=sys.stderr)
        return 2

    _print_summary("Processing existing 25D take", [("take_id", args.take_id), ("pipeline_id", PIPELINE_ID_25D)])
    run_ball_inspection_25d_flow(args.data_dir, take_id=args.take_id)
    return _print_result_overview(args.data_dir, args.take_id)


def cmd_25d_validate_api(args: argparse.Namespace) -> int:
    from vision_3d_acquisition.api.main import ExecuteTakeRequest, process_take_for_pipeline  # noqa: WPS433

    take_dir = args.data_dir / "incoming" / args.take_id
    if not take_dir.is_dir():
        print(f"take not found in incoming/: {take_dir}", file=sys.stderr)
        return 2

    payload = ExecuteTakeRequest(pipeline_id=PIPELINE_ID_25D, reprocess=True)
    response = process_take_for_pipeline(args.take_id, payload, _settings(args.data_dir))

    _print_summary(
        "API contract validation complete",
        [
            ("take_id", args.take_id),
            ("pipeline_id", PIPELINE_ID_25D),
            ("status", response.get("status")),
        ],
    )
    print(json.dumps(response, indent=2, default=str))
    return 0


def cmd_25d_smoke_contract_recipe_compare(args: argparse.Namespace) -> int:
    from vision_3d_acquisition.debug.contract_recipe_compare_smoke import run_25d_contract_recipe_compare_smoke  # noqa: WPS433

    summary = run_25d_contract_recipe_compare_smoke(args.data_dir)
    _print_summary(
        "25D contract/recipe/compare smoke workflow complete",
        [
            ("pipeline_id", summary.get("pipeline_id")),
            ("contract_ok", summary.get("contract_validation", {}).get("ok")),
            ("recipe_compare_id", summary.get("recipe_compare_id")),
            ("run_compare_id", summary.get("run_compare_id")),
            ("comparison_index_count", summary.get("comparison_index_count")),
            ("mask_changed_percent", (summary.get("mask_pixel_diff") or {}).get("changed_percent")),
        ],
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_25d_demo_contract_workflow(args: argparse.Namespace) -> int:
    from vision_3d_acquisition.debug.demo_contract_workflow import run_25d_demo_contract_workflow  # noqa: WPS433

    summary = run_25d_demo_contract_workflow(args.data_dir)
    _print_summary(
        "25D demo contract/recipe/compare/graph workflow ready",
        [
            ("pipeline_id", summary.get("pipeline_id")),
            ("demo_take_id", summary.get("demo_take_id")),
            ("base_recipe_id", (summary.get("recipes") or {}).get("base_recipe_id")),
            ("run_compare_id", summary.get("run_compare_id")),
            ("studio_open_url", summary.get("studio_open_url")),
            ("comparison_json_path", summary.get("comparison_json_path")),
        ],
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_25d_demo_partial_rerun_plan(args: argparse.Namespace) -> int:
    from vision_3d_acquisition.debug.demo_partial_rerun_plan import run_25d_demo_partial_rerun_plan  # noqa: WPS433

    summary = run_25d_demo_partial_rerun_plan(args.data_dir)
    plans = summary.get("partial_rerun_plans") or {}
    _print_summary(
        "25D demo partial rerun planner workflow ready",
        [
            ("pipeline_id", summary.get("pipeline_id")),
            ("demo_take_id", summary.get("demo_take_id")),
            ("default_plan_id", summary.get("default_plan_id")),
            ("planner_graph_url", summary.get("planner_graph_url")),
            ("plan_count", len(plans)),
        ],
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_25d_demo_partial_rerun_execute(args: argparse.Namespace) -> int:
    from vision_3d_acquisition.debug.demo_partial_rerun_execute import run_25d_demo_partial_rerun_execute  # noqa: WPS433

    summary = run_25d_demo_partial_rerun_execute(args.data_dir)
    execution = summary.get("partial_rerun_execution") or {}
    verification = summary.get("verification") or {}
    _print_summary(
        "25D demo partial rerun execution workflow ready",
        [
            ("pipeline_id", summary.get("pipeline_id")),
            ("parent_run_id", summary.get("parent_run_id")),
            ("child_run_id", summary.get("child_run_id")),
            ("comparison_id", summary.get("comparison_id")),
            ("boundary_stage_id", execution.get("boundary_stage_id")),
            ("graph_workspace_url", summary.get("graph_workspace_url")),
            ("parent_run_url", summary.get("parent_run_url")),
            ("child_run_url", summary.get("child_run_url")),
            ("comparison_url", summary.get("comparison_url")),
            ("verify: child_run_in_listing", verification.get("child_run_in_listing")),
            ("verify: lineage_resolves_parent", verification.get("lineage_resolves_parent")),
            ("verify: lineage_resolves_children", verification.get("lineage_resolves_children")),
            ("verify: artifact_manifest_exists", verification.get("artifact_manifest_exists")),
            ("verify: comparison_available", verification.get("comparison_available")),
            ("verify: graph_workspace_url_includes_child_run_id", verification.get("graph_workspace_url_includes_child_run_id")),
        ],
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_25d_inspect_result(args: argparse.Namespace) -> int:
    return _print_result_overview(args.data_dir, args.take_id)


def cmd_25d_debug_reference(args: argparse.Namespace) -> int:
    from vision_3d_acquisition.apps.ball_inspection_25d import run_ball_inspection_25d_flow  # noqa: WPS433
    from vision_3d_acquisition.debug.reference_audit import build_reference_audit_bundle  # noqa: WPS433

    take_id = args.take_id
    pipeline_id = args.pipeline or PIPELINE_ID_25D
    result_path = args.data_dir / "processed" / take_id / "result.json"

    stage_params: dict | None = None
    if args.stage_params:
        stage_params_path = Path(args.stage_params)
        if not stage_params_path.is_file():
            print(f"stage params file not found: {stage_params_path}", file=sys.stderr)
            return 2
        stage_params = _read_json(stage_params_path)

    if args.rerun or not result_path.is_file():
        if not args.rerun and not result_path.is_file():
            print(
                f"No processed result found for take_id={take_id!r} at {result_path}. "
                "Re-run with --rerun to process the take first.",
                file=sys.stderr,
            )
            return 2
        take_dir = args.data_dir / "incoming" / take_id
        if not take_dir.is_dir():
            print(f"take not found in incoming/: {take_dir}", file=sys.stderr)
            return 2
        if not (take_dir / "READY").is_file():
            print(f"take is missing READY marker: {take_dir / 'READY'}", file=sys.stderr)
            return 2
        _print_summary(
            "Re-running 25D pipeline before audit bundle generation",
            [("take_id", take_id), ("pipeline_id", pipeline_id)],
        )
        run_ball_inspection_25d_flow(args.data_dir, take_id=take_id, stage_params=stage_params)

    try:
        bundle = build_reference_audit_bundle(
            args.data_dir,
            take_id=take_id,
            pipeline_id=pipeline_id,
            run_id=args.run_id,
            output_root=args.output_dir,
            include_overlays=args.include_overlays,
            include_diffs=args.include_diffs,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    support_rows = json.loads((bundle.output_dir / "10_summary" / "support_loss.json").read_text(encoding="utf-8"))
    top_fraction = sorted(support_rows, key=lambda item: float(item.get("removed_fraction") or 0.0), reverse=True)[:3]
    top_abs = sorted(support_rows, key=lambda item: int(item.get("removed_pixels") or 0), reverse=True)[:3]

    _print_summary(
        "25D reference audit bundle generated",
        [
            ("output_dir", bundle.output_dir),
            ("artifacts_copied", bundle.artifacts_copied),
            ("derived_masks_generated", bundle.derived_masks_generated),
            ("missing_artifacts", len(bundle.missing_artifacts)),
            ("index_html", bundle.output_dir / "index.html"),
        ],
    )
    print("Top support losses by removed fraction:")
    for row in top_fraction:
        print(
            f"  - {row.get('step')}: removed_fraction={float(row.get('removed_fraction') or 0.0) * 100:.2f}% "
            f"removed_pixels={row.get('removed_pixels')}"
        )
    print("Top support losses by absolute removed pixels:")
    for row in top_abs:
        print(
            f"  - {row.get('step')}: removed_pixels={row.get('removed_pixels')} "
            f"removed_fraction={float(row.get('removed_fraction') or 0.0) * 100:.2f}%"
        )

    if args.open:
        import webbrowser

        webbrowser.open((bundle.output_dir / "index.html").resolve().as_uri())
    return 0


def cmd_25d_inspect_feature(args: argparse.Namespace) -> int:
    from vision_3d_acquisition.ml.features.surface_sphere_fit_workflow import inspect_take_feature  # noqa: WPS433

    summary = inspect_take_feature(
        processed_dir=args.data_dir / "processed",
        take_id=args.take_id,
        feature_key=args.feature,
    )
    _print_summary(
        "25D feature inspection",
        [
            ("take_id", summary.get("take_id")),
            ("run_id", summary.get("run_id") or "-"),
            ("pipeline_id", summary.get("pipeline_id") or "-"),
            ("object_count", summary.get("object_count")),
            ("objects_with_feature_value", summary.get("objects_with_feature_value")),
            ("objects_missing_feature", summary.get("objects_missing_feature")),
            ("backfill_object_count", summary.get("backfill_object_count")),
            ("min", summary.get("min")),
            ("max", summary.get("max")),
            ("mean", summary.get("mean")),
            ("likely_reason", summary.get("likely_reason") or "-"),
        ],
    )
    print("Locations checked:")
    for location in summary.get("locations_checked") or []:
        print(f"  - {location}")
    if summary.get("feature_vector_path"):
        print(f"feature_vector.json value: {summary.get('feature_vector_value')}")
    if summary.get("backfill_sidecar_path"):
        print(f"backfill sidecar: {summary.get('backfill_sidecar_path')}")
    print("Per-object:")
    for item in summary.get("per_object") or []:
        print(
            "  - id={id} value={value} location={location} missing_reason={missing_reason}".format(
                id=item.get("object_id"),
                value=item.get("value"),
                location=item.get("location") or "-",
                missing_reason=item.get("missing_reason") or "-",
            )
        )
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    return 0 if summary.get("found") else 1


def cmd_25d_clean(args: argparse.Namespace) -> int:
    if not args.take_id and not args.session_id:
        print("provide --take-id or --session-id", file=sys.stderr)
        return 2

    removed: list[Path] = []
    if args.take_id:
        for base in ("incoming", "processed"):
            target = args.data_dir / base / args.take_id
            if target.exists():
                shutil.rmtree(target)
                removed.append(target)

    if args.session_id:
        incoming = args.data_dir / "incoming"
        for take_dir in incoming.iterdir() if incoming.is_dir() else []:
            if not take_dir.is_dir() or take_dir.name.startswith("."):
                continue
            metadata_path = take_dir / "metadata.json"
            if not metadata_path.is_file():
                continue
            metadata = _read_json(metadata_path)
            if str(metadata.get("session_id") or "") != args.session_id:
                continue
            for base in ("incoming", "processed"):
                target = args.data_dir / base / take_dir.name
                if target.exists() and target not in removed:
                    shutil.rmtree(target)
                    removed.append(target)

    _print_summary("Cleaned 25D data", [("removed_paths", len(removed))])
    for path in removed:
        print(f"  - {path}")
    return 0


def cmd_25d_interactive(args: argparse.Namespace) -> int:
    print("Sensor Studio 25D interactive mode")
    print("Commands: create-synthetic, demo, process, validate-api, inspect-result, latest, list, clean, quit")
    while True:
        try:
            raw = input("25d> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not raw:
            continue
        if raw in {"q", "quit", "exit"}:
            return 0
        if raw == "latest":
            latest = _latest_take_id(args.data_dir)
            print(latest or "no takes found")
            continue
        if raw == "list":
            for take_id in _list_takes(args.data_dir, limit=20):
                print(f"- {take_id}")
            continue

        parts = raw.split()
        cmd = parts[0]
        try:
            if cmd == "create-synthetic":
                local = argparse.Namespace(data_dir=args.data_dir, session_id="synthetic_25d_demo", count=1, no_reflectance=False)
                cmd_25d_create_synthetic(local)
            elif cmd == "demo":
                local = argparse.Namespace(data_dir=args.data_dir, take_id=None, session_id="synthetic_25d_demo")
                cmd_25d_demo(local)
            elif cmd == "process":
                if len(parts) < 2:
                    print("usage: process <take_id>")
                    continue
                local = argparse.Namespace(data_dir=args.data_dir, take_id=parts[1])
                cmd_25d_process(local)
            elif cmd == "validate-api":
                if len(parts) < 2:
                    print("usage: validate-api <take_id>")
                    continue
                local = argparse.Namespace(data_dir=args.data_dir, take_id=parts[1])
                cmd_25d_validate_api(local)
            elif cmd == "inspect-result":
                if len(parts) < 2:
                    print("usage: inspect-result <take_id>")
                    continue
                local = argparse.Namespace(data_dir=args.data_dir, take_id=parts[1])
                cmd_25d_inspect_result(local)
            elif cmd == "clean":
                if len(parts) < 3:
                    print("usage: clean take <take_id> | clean session <session_id>")
                    continue
                mode, value = parts[1], parts[2]
                if mode == "take":
                    local = argparse.Namespace(data_dir=args.data_dir, take_id=value, session_id=None)
                elif mode == "session":
                    local = argparse.Namespace(data_dir=args.data_dir, take_id=None, session_id=value)
                else:
                    print("usage: clean take <take_id> | clean session <session_id>")
                    continue
                cmd_25d_clean(local)
            else:
                print(f"unknown command: {cmd}")
        except SystemExit as exc:
            print(f"command failed: {exc}")


def cmd_studio_serve(_: argparse.Namespace) -> int:
    from vision_3d_acquisition.api.serve import main as serve_api_main  # noqa: WPS433

    serve_api_main()
    return 0


def _frontend_dir(frontend_dir: Path | None = None) -> Path:
    candidate = (frontend_dir or (_REPO_ROOT / "frontend")).resolve()
    if not candidate.is_dir():
        raise SystemExit(f"frontend directory not found: {candidate}")
    if not (candidate / "package.json").is_file():
        raise SystemExit(f"package.json not found in frontend directory: {candidate}")
    return candidate


def cmd_studio_web(args: argparse.Namespace) -> int:
    frontend_dir = _frontend_dir(args.frontend_dir)
    command = [args.package_manager, "run", "dev"]
    _print_summary(
        "Starting frontend dev server",
        [("frontend_dir", frontend_dir), ("command", " ".join(command))],
    )
    try:
        return subprocess.run(command, cwd=frontend_dir, check=False).returncode
    except FileNotFoundError as exc:
        raise SystemExit(f"command not found: {args.package_manager}") from exc


def _terminate_process(proc: subprocess.Popen[bytes] | None, label: str) -> None:
    if proc is None or proc.poll() is not None:
        return
    print(f"Stopping {label}...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def cmd_studio_dev(args: argparse.Namespace) -> int:
    frontend_dir = _frontend_dir(args.frontend_dir)
    web_command = [args.package_manager, "run", "dev"]
    api_command = [sys.executable, str(_REPO_ROOT / "scripts" / "run_api.py")]
    _print_summary(
        "Starting full studio dev stack",
        [
            ("api_command", " ".join(api_command)),
            ("web_command", " ".join(web_command)),
            ("frontend_dir", frontend_dir),
        ],
    )

    api_proc: subprocess.Popen[bytes] | None = None
    web_proc: subprocess.Popen[bytes] | None = None
    try:
        api_proc = subprocess.Popen(api_command, cwd=_REPO_ROOT)
        time.sleep(0.3)
        web_proc = subprocess.Popen(web_command, cwd=frontend_dir)
        while True:
            api_code = api_proc.poll()
            web_code = web_proc.poll()
            if api_code is not None:
                print(f"API process exited with code {api_code}.")
                _terminate_process(web_proc, "frontend")
                return int(api_code)
            if web_code is not None:
                print(f"Frontend process exited with code {web_code}.")
                _terminate_process(api_proc, "api")
                return int(web_code)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nReceived Ctrl+C, shutting down studio dev stack...")
        _terminate_process(web_proc, "frontend")
        _terminate_process(api_proc, "api")
        return 130


def cmd_acquisition_list_takes(args: argparse.Namespace) -> int:
    from vision_3d_acquisition.datasets.service import DatasetService  # noqa: WPS433

    limit = None if getattr(args, "no_limit", False) else args.limit
    show_db = bool(getattr(args, "show_db", False))
    show_session = bool(getattr(args, "show_session", False))
    show_headers = bool(getattr(args, "show_headers", False))
    dataset_id_filter = str(getattr(args, "dataset_id", "") or "").strip() or None
    dataset_service = DatasetService(args.data_dir) if (show_db or show_session or dataset_id_filter) else None

    if show_headers and (show_db or show_session):
        header_fields = ["take_id"]
        if show_db:
            header_fields.append("dataset_id")
        if show_session:
            header_fields.append("session_id")
        print("\t".join(header_fields))

    matched = 0
    for take_id in _list_takes(args.data_dir, None):
        take_meta: dict | None = None
        source_metadata = None
        if dataset_service:
            source_metadata = _read_json(args.data_dir / "incoming" / take_id / "metadata.json")
            take_meta = dataset_service.load_take_metadata(take_id=take_id, source_metadata=source_metadata)
        if dataset_id_filter and str((take_meta or {}).get("dataset_id") or "") != dataset_id_filter:
            continue
        if limit is not None and matched >= max(1, limit):
            break
        matched += 1

        if not (show_db or show_session):
            print(take_id)
            continue

        take_meta = take_meta or {}
        source_metadata = source_metadata or {}
        fields = [take_id]
        if show_db:
            fields.append(str(take_meta.get("dataset_id") or "-"))
        if show_session:
            fields.append(str(take_meta.get("session_id") or source_metadata.get("session_id") or "-"))
        print("\t".join(fields))
    return 0


def cmd_acquisition_latest(args: argparse.Namespace) -> int:
    latest = _latest_take_id(args.data_dir)
    if not latest:
        print("no takes found", file=sys.stderr)
        return 2
    print(latest)
    return 0


def _read_take_ids_file(path: Path) -> list[str]:
    rows: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        item = raw.strip()
        if not item:
            continue
        rows.append(item)
    return rows


def _collect_take_ids(explicit_take_ids: list[str] | None, take_file: Path | None) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in explicit_take_ids or []:
        cleaned = str(item).strip()
        if cleaned and cleaned not in seen:
            merged.append(cleaned)
            seen.add(cleaned)
    if take_file is not None:
        for item in _read_take_ids_file(take_file):
            if item not in seen:
                merged.append(item)
                seen.add(item)
    return merged


def _take_exists(data_dir: Path, take_id: str) -> bool:
    return (data_dir / "incoming" / take_id).is_dir()


def _parse_bool(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true|false")


def _resolve_ml_set(service, *, ml_set_id: str, dataset_id: str | None) -> tuple[dict[str, object], bool]:
    resolved = service.resolve_ml_set(ml_set_id=ml_set_id, dataset_id=dataset_id)
    auto_resolved = dataset_id is None
    return resolved, auto_resolved


def _write_update_take_session_log(data_dir: Path, payload: dict[str, object]) -> Path:
    logs_dir = data_dir / "runtime" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = logs_dir / f"update_take_session_{ts}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_ml_assign_splits_log(data_dir: Path, payload: dict[str, object]) -> Path:
    logs_dir = data_dir / "runtime" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = logs_dir / f"ml_assign_splits_{ts}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_ml_reprocess_set_log(data_dir: Path, payload: dict[str, object]) -> Path:
    logs_dir = data_dir / "runtime" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = logs_dir / f"ml_reprocess_set_{ts}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_ml_import_manifest_log(data_dir: Path, payload: dict[str, object]) -> Path:
    logs_dir = data_dir / "runtime" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = logs_dir / f"ml_import_manifest_{ts}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def cmd_acquisition_update_take_session(args: argparse.Namespace) -> int:
    from vision_3d_acquisition.datasets.service import DatasetService  # noqa: WPS433

    take_ids = _collect_take_ids(args.take_id, args.take_file)
    if not take_ids:
        print("no takes provided; use --take-id and/or --take-file", file=sys.stderr)
        return 2

    service = DatasetService(args.data_dir)
    try:
        result = service.batch_update_take_session(
            take_ids=take_ids,
            new_session_id=args.session_id,
            dataset_id=args.dataset_id,
            create_session=bool(args.create_session),
            session_name=args.session_name,
            allow_missing=bool(args.allow_missing),
            apply=bool(args.apply),
        )
    except ValueError as exc:
        msg = str(exc)
        if msg == "missing takes found":
            missing = [take_id for take_id in take_ids if not _take_exists(args.data_dir, take_id)]
            _print_summary(
                "Take validation",
                [("requested", len(take_ids)), ("valid", len(take_ids) - len(missing)), ("missing", len(missing))],
            )
            for take_id in missing:
                print(f"  missing: {take_id}")
            print("refusing to continue: missing takes found (use --allow-missing to continue)", file=sys.stderr)
        elif msg.startswith("takes belong to multiple datasets:"):
            print(msg, file=sys.stderr)
            print("provide --dataset-id to select the target dataset", file=sys.stderr)
        elif msg.startswith("destination session does not exist:"):
            print(f"{msg} (use --create-session)", file=sys.stderr)
        elif msg.startswith("could not infer dataset_id from takes"):
            print(f"{msg}; provide --dataset-id", file=sys.stderr)
        else:
            print(msg, file=sys.stderr)
        return 2

    valid_take_ids = result["valid_take_ids"]
    missing_take_ids = result["missing_take_ids"]
    rows = result["rows"]
    resolved_dataset_id = str(result["dataset_id"])
    destination_exists = bool(result["destination_session_exists"])
    created_session = result["created_session"]

    _print_summary(
        "Take validation",
        [("requested", result["requested"]), ("valid", len(valid_take_ids)), ("missing", len(missing_take_ids))],
    )
    for take_id in missing_take_ids:
        print(f"  missing: {take_id}")

    col_take = max(len("Take"), max(len(row["take_id"]) for row in rows))
    col_old = max(len("Old Session"), max(len(row["old_session_id"] or "-") for row in rows))
    col_new = max(len("New Session"), len(args.session_id))
    mode_label = "[APPLY]" if args.apply else "[DRY RUN]"
    print(mode_label)
    print()
    print(f"{'Take'.ljust(col_take)}  {'Old Session'.ljust(col_old)}  {'New Session'.ljust(col_new)}")
    print("-" * (col_take + col_old + col_new + 4))
    for row in rows:
        print(f"{row['take_id'].ljust(col_take)}  {(row['old_session_id'] or '-').ljust(col_old)}  {args.session_id.ljust(col_new)}")
    print()
    _print_summary(
        "Summary",
        [
            ("takes", len(rows)),
            ("dataset", resolved_dataset_id),
            ("destination session exists", "yes" if destination_exists else "no"),
            ("create session", "yes" if args.create_session else "no"),
        ],
    )

    if not args.apply:
        return 0

    log_payload: dict[str, object] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "args": {
            "data_dir": str(args.data_dir),
            "session_id": args.session_id,
            "session_name": args.session_name,
            "dataset_id": args.dataset_id,
            "create_session": bool(args.create_session),
            "allow_missing": bool(args.allow_missing),
            "apply": bool(args.apply),
            "take_ids": take_ids,
            "take_file": str(args.take_file) if args.take_file else None,
        },
        "dataset_id": resolved_dataset_id,
        "new_session_id": args.session_id,
        "affected_takes": result["updated_rows"],
        "created_session": created_session,
    }
    log_path = _write_update_take_session_log(args.data_dir, log_payload)
    print(f"\nWrote operation log: {log_path}")
    return 0


def cmd_ml_create_set(args: argparse.Namespace) -> int:
    from vision_3d_acquisition.datasets.service import DatasetService  # noqa: WPS433

    service = DatasetService(args.data_dir)
    try:
        payload = service.create_ml_set(
            dataset_id=args.dataset_id,
            ml_set_id=args.ml_set_id,
            name=args.name,
            description=args.description,
            task_type=args.task_type,
            notes=args.notes,
            overwrite=bool(args.overwrite),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    _print_summary(
        "Created ML set",
        [
            ("id", payload.get("id")),
            ("dataset_id", payload.get("dataset_id")),
            ("task_type", payload.get("task_type")),
            ("name", payload.get("name")),
            ("overwrite", "yes" if args.overwrite else "no"),
        ],
    )
    return 0


def cmd_ml_add_takes(args: argparse.Namespace) -> int:
    from vision_3d_acquisition.datasets.service import DatasetService  # noqa: WPS433

    take_ids = _collect_take_ids(args.take_id, args.take_file)
    if not take_ids:
        print("no takes provided; use --take-id and/or --take-file", file=sys.stderr)
        return 2

    service = DatasetService(args.data_dir)
    try:
        resolved_ml_set, auto_resolved = _resolve_ml_set(service, ml_set_id=args.ml_set_id, dataset_id=args.dataset_id)
        resolved_dataset_id = str(resolved_ml_set.get("dataset_id") or "")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if auto_resolved:
        _print_summary("Resolved MLSet", [("dataset_id", resolved_dataset_id), ("ml_set_id", args.ml_set_id)])

    statuses: dict[str, int] = {"added": 0, "updated": 0}
    failures: list[tuple[str, str]] = []
    for take_id in take_ids:
        try:
            result = service.add_take_to_ml_set(
                dataset_id=resolved_dataset_id,
                ml_set_id=args.ml_set_id,
                take_id=take_id,
                split=args.split,
                physical_object_id=args.physical_object_id,
                include=bool(args.include),
                notes=args.notes,
            )
            status = str(result.get("status") or "")
            if status in statuses:
                statuses[status] += 1
        except ValueError as exc:
            failures.append((take_id, str(exc)))

    if failures:
        for take_id, reason in failures:
            print(f"{take_id}: {reason}", file=sys.stderr)
        print("failed to add one or more takes", file=sys.stderr)
        return 2

    _print_summary(
        "ML set membership update",
        [
            ("dataset_id", resolved_dataset_id),
            ("ml_set_id", args.ml_set_id),
            ("requested", len(take_ids)),
            ("added", statuses["added"]),
            ("updated", statuses["updated"]),
            ("split", args.split),
        ],
    )
    return 0


def cmd_ml_list_set(args: argparse.Namespace) -> int:
    from vision_3d_acquisition.datasets.service import DatasetService  # noqa: WPS433

    service = DatasetService(args.data_dir)
    try:
        ml_set, auto_resolved = _resolve_ml_set(service, ml_set_id=args.ml_set_id, dataset_id=args.dataset_id)
        resolved_dataset_id = str(ml_set.get("dataset_id") or "")
        memberships = service.list_ml_set_memberships(dataset_id=resolved_dataset_id, ml_set_id=args.ml_set_id)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if auto_resolved:
        _print_summary("Resolved MLSet", [("dataset_id", resolved_dataset_id), ("ml_set_id", args.ml_set_id)])

    included = sum(1 for row in memberships if bool(row.get("include", True)))
    excluded = len(memberships) - included
    split_counts: dict[str, int] = {}
    for row in memberships:
        split = str(row.get("split") or "unassigned")
        split_counts[split] = split_counts.get(split, 0) + 1

    print("ML Set")
    print("------")
    print(f"id: {ml_set.get('id')}")
    print(f"dataset_id: {ml_set.get('dataset_id')}")
    print(f"task_type: {ml_set.get('task_type')}")
    print(f"name: {ml_set.get('name')}")
    print()
    print("Memberships")
    print("-----------")
    print(f"total: {len(memberships)}")
    print(f"included: {included}")
    print(f"excluded: {excluded}")
    print()
    print("Split distribution")
    print("------------------")
    if split_counts:
        for split, count in sorted(split_counts.items(), key=lambda item: item[0]):
            print(f"{split}: {count}")
    else:
        print("(empty)")

    if args.show_memberships and memberships:
        print()
        print("Sample memberships")
        print("------------------")
        limit = max(1, int(args.limit or 20))
        for row in memberships[:limit]:
            print(
                f"{row.get('take_id')} split={row.get('split')} include={row.get('include')} "
                f"physical_object_id={row.get('physical_object_id')}"
            )
    return 0


def cmd_ml_assign_splits(args: argparse.Namespace) -> int:
    from vision_3d_acquisition.datasets.service import DatasetService  # noqa: WPS433

    service = DatasetService(args.data_dir)
    try:
        ml_set, auto_resolved = _resolve_ml_set(service, ml_set_id=args.ml_set_id, dataset_id=args.dataset_id)
        resolved_dataset_id = str(ml_set.get("dataset_id") or "")
        result = service.assign_ml_set_splits(
            ml_set_id=args.ml_set_id,
            dataset_id=resolved_dataset_id,
            train_ratio=args.train,
            validation_ratio=args.validation,
            test_ratio=args.test,
            holdout_ratio=args.holdout,
            calibration_ratio=args.calibration,
            by_physical_object_id=bool(args.by_physical_object_id),
            seed=args.seed,
            dry_run=not bool(args.apply),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if auto_resolved:
        _print_summary("Resolved MLSet", [("dataset_id", resolved_dataset_id), ("ml_set_id", args.ml_set_id)])

    mode_label = "[APPLY]" if args.apply else "[DRY RUN]"
    print(mode_label)
    print()
    print("ML Set:")
    print(f"  {args.ml_set_id}")
    print()
    print("Assignment mode:")
    print(f"  by_physical_object_id: {'yes' if args.by_physical_object_id else 'no'}")
    print()
    print("Ratios:")
    print(f"  train: {args.train:.2f}")
    print(f"  validation: {args.validation:.2f}")
    print(f"  test: {args.test:.2f}")
    if args.holdout > 0:
        print(f"  holdout: {args.holdout:.2f}")
    if args.calibration > 0:
        print(f"  calibration: {args.calibration:.2f}")
    print()
    print("Grouped objects:")
    print(f"  total: {int(result['group_count'])}")
    print()
    print("Projected split distribution:" if not args.apply else "Final split distribution:")
    for split, counts in result["split_counts"].items():
        print(f"  {split}: {int(counts['groups'])} objects / {int(counts['takes'])} takes")

    if args.apply:
        log_payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "ml_set_id": args.ml_set_id,
            "dataset_id": resolved_dataset_id,
            "ratios": result["ratios"],
            "seed": args.seed,
            "assignment_mode": result["assignment_mode"],
            "group_count": result["group_count"],
            "take_count": result["take_count"],
            "final_split_counts": result["split_counts"],
        }
        log_path = _write_ml_assign_splits_log(args.data_dir, log_payload)
        print(f"\nWrote operation log: {log_path}")
    return 0


def cmd_ml_reprocess_set(args: argparse.Namespace) -> int:
    from vision_3d_acquisition.datasets.service import DatasetService  # noqa: WPS433

    service = DatasetService(args.data_dir)
    try:
        ml_set, auto_resolved = _resolve_ml_set(service, ml_set_id=args.ml_set_id, dataset_id=args.dataset_id)
        resolved_dataset_id = str(ml_set.get("dataset_id") or "")
        result = service.reprocess_ml_set(
            ml_set_id=args.ml_set_id,
            pipeline_id=args.pipeline_id,
            dataset_id=resolved_dataset_id,
            splits=args.split,
            include_only=True,
            reprocess=True,
            dry_run=not bool(args.apply),
            classifier_rules_path=args.classifier_rules,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if auto_resolved:
        _print_summary("Resolved MLSet", [("dataset_id", resolved_dataset_id), ("ml_set_id", args.ml_set_id)])

    mode_label = "[APPLY]" if args.apply else "[DRY RUN]"
    print(mode_label)
    print()
    print("ML Set:")
    print(f"  {args.ml_set_id}")
    print()
    print("Pipeline:")
    print(f"  {args.pipeline_id}")
    if args.classifier_rules:
        print(f"  classifier_rules: {args.classifier_rules}")
    print()
    print("Selected memberships:")
    print(f"  total: {int(result['selected_count'])}")
    for split, count in result["split_counts"].items():
        print(f"  {split}: {count}")
    print()
    print("Planned processing:" if not args.apply else "Processed takes:")
    for take_id in result["selected_take_ids"]:
        print(f"  {take_id}")

    if args.apply:
        print()
        print("Execution summary:")
        print(f"  success: {int(result['success_count'])}")
        print(f"  failures: {int(result['failure_count'])}")
        if result["failures"]:
            print("Failed takes:")
            for item in result["failures"]:
                print(f"  {item['take_id']}: {item['error']}")
        log_payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "ml_set_id": args.ml_set_id,
            "dataset_id": resolved_dataset_id,
            "pipeline_id": args.pipeline_id,
            "selected_splits": args.split or [],
            "classifier_rules_path": args.classifier_rules,
            "processed_takes": result["processed"],
            "failures": result["failures"],
        }
        log_path = _write_ml_reprocess_set_log(args.data_dir, log_payload)
        print(f"\nWrote operation log: {log_path}")
    return 0


def cmd_ml_export_features(args: argparse.Namespace) -> int:
    from vision_3d_acquisition.datasets.service import DatasetService  # noqa: WPS433

    service = DatasetService(args.data_dir)
    try:
        ml_set, auto_resolved = _resolve_ml_set(service, ml_set_id=args.ml_set_id, dataset_id=args.dataset_id)
        resolved_dataset_id = str(ml_set.get("dataset_id") or "")
        result = service.export_ml_set_features(
            ml_set_id=args.ml_set_id,
            pipeline_id=args.pipeline_id,
            output_path=args.output,
            dataset_id=resolved_dataset_id,
            splits=args.split,
            require_processed=bool(args.require_processed),
            include_diagnostics=bool(args.include_diagnostics),
            include_invalidity_flags=bool(args.include_invalidity_flags),
            include_provenance_summary=bool(args.include_provenance_summary),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if auto_resolved:
        _print_summary("Resolved MLSet", [("dataset_id", resolved_dataset_id), ("ml_set_id", args.ml_set_id)])
    _print_summary(
        "Exported ML features",
        [
            ("ml_set_id", args.ml_set_id),
            ("dataset_id", resolved_dataset_id),
            ("pipeline_id", args.pipeline_id),
            ("rows", result["row_count"]),
            ("columns", result["column_count"]),
            ("output", result["output_path"]),
            ("include_diagnostics", "yes" if args.include_diagnostics else "no"),
            ("include_invalidity_flags", "yes" if args.include_invalidity_flags else "no"),
            ("include_provenance_summary", "yes" if args.include_provenance_summary else "no"),
        ],
    )
    if result.get("missing_processed_take_ids"):
        print("Missing processed takes:")
        for take_id in result["missing_processed_take_ids"]:
            print(f"  {take_id}")
    return 0


def cmd_ml_import_manifest(args: argparse.Namespace) -> int:
    from vision_3d_acquisition.datasets.service import DatasetService  # noqa: WPS433

    service = DatasetService(args.data_dir)
    try:
        ml_set, auto_resolved = _resolve_ml_set(service, ml_set_id=args.ml_set_id, dataset_id=args.dataset_id)
        resolved_dataset_id = str(ml_set.get("dataset_id") or "")
        result = service.import_ml_manifest(
            ml_set_id=args.ml_set_id,
            manifest_path=args.manifest,
            dataset_id=resolved_dataset_id,
            dry_run=not bool(args.apply),
            allow_overwrite_object_id=bool(args.allow_overwrite_object_id),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if auto_resolved:
        _print_summary("Resolved MLSet", [("dataset_id", resolved_dataset_id), ("ml_set_id", args.ml_set_id)])
    mode_label = "[APPLY]" if args.apply else "[DRY RUN]"
    print(mode_label)
    print()
    print("Manifest:")
    print(f"  {args.manifest}")
    print()
    print("ML Set:")
    print(f"  {args.ml_set_id}")
    print()
    print("Rows:")
    print(f"  total: {int(result['rows_total'])}")
    print()
    print("Takes:")
    print(f"  total referenced: {int(result['takes_total_referenced'])}")
    print(f"  valid: {int(result['takes_valid'])}")
    print(f"  missing: {int(result['takes_missing'])}")
    print()
    print("Membership updates:")
    print(f"  added: {int(result['membership_added'])}")
    print(f"  updated: {int(result['membership_updated'])}")
    print()
    print("Physical objects:")
    print(f"  total: {int(result['physical_objects_total'])}")
    print()
    print("Labels:")
    labels = result.get("labels_summary") or {}
    if isinstance(labels, dict) and labels:
        for label, count in labels.items():
            print(f"  {label}: {count}")
    else:
        print("  (none)")

    if args.apply:
        log_payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "manifest_path": str(args.manifest),
            "ml_set_id": args.ml_set_id,
            "dataset_id": resolved_dataset_id,
            "rows_processed": result["rows_total"],
            "takes_processed": result["takes_total_referenced"],
            "labels_summary": result["labels_summary"],
            "object_count": result["physical_objects_total"],
            "conflicts": result["conflicts"],
        }
        log_path = _write_ml_import_manifest_log(args.data_dir, log_payload)
        print(f"\nWrote operation log: {log_path}")
    return 0


def cmd_ml_compute_feature(args: argparse.Namespace) -> int:
    from vision_3d_acquisition.ml.features.surface_sphere_fit_workflow import compute_ml_set_feature  # noqa: WPS433

    try:
        result = compute_ml_set_feature(
            data_dir=args.data_dir,
            ml_set_id=args.ml_set_id,
            feature_key=args.feature,
            mode=args.mode,
            dataset_id=args.dataset_id,
            pipeline_id=args.pipeline_id,
            dry_run=not bool(args.apply),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    mode_label = "[APPLY]" if args.apply else "[DRY RUN]"
    print(mode_label)
    _print_summary(
        "ML set feature compute",
        [
            ("ml_set_id", result.get("ml_set_id")),
            ("dataset_id", result.get("dataset_id")),
            ("feature_key", result.get("feature_key")),
            ("mode", result.get("mode")),
            ("take_count", result.get("take_count")),
        ],
    )
    for item in result.get("results") or []:
        print(
            "  - {take_id}: status={status} objects_with_value={objects_with_value} entries={entries}".format(
                take_id=item.get("take_id"),
                status=item.get("status"),
                objects_with_value=item.get("objects_with_feature_value", item.get("objects_with_value")),
                entries=item.get("entries_written"),
            )
        )
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_ml_list_rule_sets(args: argparse.Namespace) -> int:
    from vision_3d_acquisition.classifiers.mining_ball_rules import list_available_rule_sets  # noqa: WPS433

    rows = list_available_rule_sets()
    if args.classifier_id:
        rows = [row for row in rows if str(row.get("classifier_id") or "") == args.classifier_id]

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    if not rows:
        print("No classifier rule sets found in configs/classifiers.")
        return 0

    print("Available classifier rule sets")
    print("------------------------------")
    print()
    for row in rows:
        print(f"id: {row.get('rule_set_id')}")
        print(f"version: {row.get('version')}")
        print(f"classifier_id: {row.get('classifier_id')}")
        print(f"path: {row.get('path')}")
        if row.get("description"):
            print(f"description: {row.get('description')}")
        if row.get("dataset_id"):
            print(f"dataset_id: {row.get('dataset_id')}")
        if row.get("ml_set_id"):
            print(f"ml_set_id: {row.get('ml_set_id')}")
        if row.get("optimized_metric"):
            print(f"optimized_metric: {row.get('optimized_metric')}")
        if row.get("validation_macro_f1") is not None:
            print(f"validation_macro_f1: {row.get('validation_macro_f1')}")
        if row.get("source_type"):
            print(f"source_type: {row.get('source_type')}")
        print()
    return 0


def cmd_ml_show_active_rule_set(args: argparse.Namespace) -> int:
    from vision_3d_acquisition.classifiers.mining_ball_rules import resolve_classifier_rule_set  # noqa: WPS433
    from vision_3d_acquisition.pipelines.registry import get_pipeline  # noqa: WPS433

    pipeline_id = args.pipeline_id or "mining_steel_ball_classification_25d"
    pipeline = get_pipeline(pipeline_id)
    if pipeline is None:
        print(f"unknown pipeline: {pipeline_id}", file=sys.stderr)
        return 2
    classifier = pipeline.get("classifier") if isinstance(pipeline.get("classifier"), dict) else {}
    rule_set = classifier.get("rule_set") if isinstance(classifier.get("rule_set"), dict) else {}
    pipeline_config_path = rule_set.get("path")
    resolved = resolve_classifier_rule_set(
        runtime_override_path=args.classifier_rules,
        pipeline_config_path=str(pipeline_config_path) if pipeline_config_path else None,
    )
    payload = {
        "pipeline_id": pipeline_id,
        "classifier_engine": resolved.classifier_engine,
        "rule_set_id": resolved.rule_set_id,
        "rule_set_version": resolved.rule_set_version,
        "rule_set_path": resolved.rule_set_path,
        "rule_set_source": resolved.rule_set_source,
        "metadata": resolved.metadata,
        "warnings": resolved.warnings,
        "dataset_id": args.dataset_id,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print("Active classifier rule set")
    print("--------------------------")
    print(f"pipeline_id: {pipeline_id}")
    print(f"classifier_engine: {resolved.classifier_engine}")
    print(f"rule_set_id: {resolved.rule_set_id}")
    print(f"rule_set_version: {resolved.rule_set_version}")
    print(f"rule_set_source: {resolved.rule_set_source}")
    print(f"rule_set_path: {resolved.rule_set_path or '-'}")
    if resolved.metadata:
        for key in ("dataset_id", "ml_set_id", "optimized_metric", "validation_macro_f1", "created_at", "notes"):
            if key in resolved.metadata:
                print(f"{key}: {resolved.metadata.get(key)}")
    if resolved.warnings:
        print("warnings:")
        for warning in resolved.warnings:
            print(f"  - {warning}")
    return 0


def _catalog_indexer(data_dir: Path):
    from vision_3d_acquisition.storage.catalog_index import CatalogIndexer  # noqa: WPS433
    return CatalogIndexer(_settings(data_dir))


def cmd_index_rebuild(args: argparse.Namespace) -> int:
    report = _catalog_indexer(args.data_dir).rebuild(full=args.full)
    print(json.dumps(report.as_dict(), indent=2))
    return 1 if report.take_failures else 0


def cmd_index_status(args: argparse.Namespace) -> int:
    status = _catalog_indexer(args.data_dir).status()
    print(json.dumps(status, indent=2))
    return 0 if status["drift"] == 0 else 2


def cmd_index_export(args: argparse.Namespace) -> int:
    from vision_3d_acquisition.storage.catalog_export import export_catalog  # noqa: WPS433

    destination = args.to or (args.data_dir / f"export_{datetime.now(UTC):%Y%m%dT%H%M%SZ}")
    report = export_catalog(args.data_dir, destination)
    print(json.dumps(report.as_dict(), indent=2))
    return 0


def cmd_index_verify(args: argparse.Namespace) -> int:
    result = _catalog_indexer(args.data_dir).verify(limit=args.limit)
    if not args.show_all:
        result = {**result, "mismatched": result["mismatched"][:20]}
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sensor-studio",
        description="Unified Sensor Studio CLI wrapper for 25D runtime scripts and operations.",
    )
    top = parser.add_subparsers(dest="domain", required=False)

    p25d = top.add_parser("25d", help="Native 2.5D runtime and processing operations")
    p25d_sub = p25d.add_subparsers(dest="command", required=False)
    p25d.set_defaults(domain="25d")

    c = p25d_sub.add_parser("create-synthetic", help="Create synthetic 25D take(s)")
    c.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    c.add_argument("--session-id", default="synthetic_25d_demo", help="Synthetic session id")
    c.add_argument("--count", type=int, default=1, help="Number of takes to create")
    c.add_argument("--no-reflectance", action="store_true", help="Disable synthetic reflectance output")
    c.set_defaults(handler=cmd_25d_create_synthetic)

    d = p25d_sub.add_parser("demo", help="Run 25D demo pipeline on synthetic or existing take")
    d.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    d.add_argument("--take-id", default=None, help="Existing take id to process")
    d.add_argument("--session-id", default="synthetic_25d_demo", help="Session id when creating synthetic input")
    d.set_defaults(handler=cmd_25d_demo)

    p = p25d_sub.add_parser("process", help="Process an existing incoming 25D take")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    p.add_argument("--take-id", required=True, help="Existing take id in data/incoming")
    p.set_defaults(handler=cmd_25d_process)

    v = p25d_sub.add_parser("validate-api", help="Validate API processing contract for a specific 25D take")
    v.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    v.add_argument("--take-id", required=True, help="Existing take id in data/incoming")
    v.set_defaults(handler=cmd_25d_validate_api)

    smoke = p25d_sub.add_parser(
        "smoke-contract-recipe-compare",
        help="Run synthetic 25D contract validation, recipe, and comparison smoke workflow",
    )
    smoke.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    smoke.set_defaults(handler=cmd_25d_smoke_contract_recipe_compare)

    demo = p25d_sub.add_parser(
        "demo-contract-workflow",
        help="Create a known-good demo scenario for contract/recipe/compare/graph workflow",
    )
    demo.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    demo.set_defaults(handler=cmd_25d_demo_contract_workflow)

    demo_plan = p25d_sub.add_parser(
        "demo-partial-rerun-plan",
        help="Create a known-good demo scenario for partial rerun planning",
    )
    demo_plan.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    demo_plan.set_defaults(handler=cmd_25d_demo_partial_rerun_plan)

    demo_execute = p25d_sub.add_parser(
        "demo-partial-rerun-execute",
        help="Create a known-good demo scenario for safe partial rerun execution",
    )
    demo_execute.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    demo_execute.set_defaults(handler=cmd_25d_demo_partial_rerun_execute)

    i = p25d_sub.add_parser("inspect-result", help="Inspect and summarize processed result.json for a take")
    i.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    i.add_argument("--take-id", required=True, help="Processed take id in data/processed")
    i.set_defaults(handler=cmd_25d_inspect_result)

    debug_ref = p25d_sub.add_parser(
        "debug-reference",
        help="Generate a reproducible 25D reference-surface audit bundle for one take/run",
    )
    debug_ref.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    debug_ref.add_argument("--take-id", required=True, help="Take id to audit")
    debug_ref.add_argument("--run-id", default="latest", help="Processed run id (default: latest)")
    debug_ref.add_argument("--pipeline", default=PIPELINE_ID_25D, help="Pipeline id to audit")
    debug_ref.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Audit output root (default: data/debug/25d_reference_audit)",
    )
    debug_ref.add_argument("--rerun", action="store_true", help="Re-run the 25D pipeline before generating the bundle")
    debug_ref.add_argument("--stage-params", default=None, help="Optional stage params JSON used with --rerun")
    debug_ref.add_argument("--include-overlays", action=argparse.BooleanOptionalAction, default=True)
    debug_ref.add_argument("--include-diffs", action=argparse.BooleanOptionalAction, default=True)
    debug_ref.add_argument("--open", action="store_true", help="Open index.html after generation")
    debug_ref.set_defaults(handler=cmd_25d_debug_reference)

    inspect_feature = p25d_sub.add_parser("inspect-feature", help="Inspect whether a processed take contains a registered feature")
    inspect_feature.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    inspect_feature.add_argument("--take-id", required=True, help="Processed take id in data/processed")
    inspect_feature.add_argument("--feature", default="surface_sphere_fit_rmse_mm", help="Feature key to inspect")
    inspect_feature.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    inspect_feature.set_defaults(handler=cmd_25d_inspect_feature)

    cl = p25d_sub.add_parser("clean", help="Remove incoming/processed folders for a take or synthetic session")
    cl.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    cl.add_argument("--take-id", default=None, help="Take id to remove")
    cl.add_argument("--session-id", default=None, help="Session id to remove all matching takes")
    cl.set_defaults(handler=cmd_25d_clean)

    it = p25d_sub.add_parser("interactive", help="Start interactive 25D command prompt")
    it.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    it.set_defaults(handler=cmd_25d_interactive)

    studio = top.add_parser("studio", help="Studio services")
    studio_sub = studio.add_subparsers(dest="command", required=False)
    studio.set_defaults(domain="studio")
    serve = studio_sub.add_parser("serve", help="Run the API service")
    serve.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    serve.set_defaults(handler=cmd_studio_serve)

    web = studio_sub.add_parser("web", help="Run frontend dev server (frontend/package.json -> dev script)")
    web.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    web.add_argument("--package-manager", default="npm", choices=["npm", "pnpm"], help="Package manager for frontend dev server")
    web.add_argument("--frontend-dir", type=Path, default=_REPO_ROOT / "frontend", help="Frontend directory (default: repo/frontend)")
    web.set_defaults(handler=cmd_studio_web)

    dev = studio_sub.add_parser("dev", help="Run API + frontend dev servers together")
    dev.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    dev.add_argument("--package-manager", default="npm", choices=["npm", "pnpm"], help="Package manager for frontend dev server")
    dev.add_argument("--frontend-dir", type=Path, default=_REPO_ROOT / "frontend", help="Frontend directory (default: repo/frontend)")
    dev.set_defaults(handler=cmd_studio_dev)

    acq = top.add_parser("acquisition", help="Acquisition-level take discovery helpers")
    acq_sub = acq.add_subparsers(dest="command", required=False)
    acq.set_defaults(domain="acquisition")
    list_takes = acq_sub.add_parser("list-takes", help="List latest acquired take ids")
    list_takes.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    list_takes_group = list_takes.add_mutually_exclusive_group()
    list_takes_group.add_argument("--limit", type=int, default=20, help="Max take ids to print")
    list_takes_group.add_argument("--no-limit", action="store_true", help="Print all take ids")
    list_takes.add_argument("--dataset-id", default=None, help="Only include takes associated with this dataset id")
    list_takes.add_argument("--show-db", action="store_true", help="Include associated dataset id per take")
    list_takes.add_argument("--show-session", action="store_true", help="Include associated session id per take")
    list_takes.add_argument("--show-headers", action="store_true", help="Print header row when showing extra columns")
    list_takes.set_defaults(handler=cmd_acquisition_list_takes)

    latest = acq_sub.add_parser("latest", help="Print latest acquired take id")
    latest.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    latest.set_defaults(handler=cmd_acquisition_latest)

    update_take_session = acq_sub.add_parser("update-take-session", help="Batch update take session metadata (dry-run by default)")
    update_take_session.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    update_take_session.add_argument("--session-id", required=True, help="Destination session id")
    update_take_session.add_argument("--session-name", default=None, help="Session display name when creating destination session")
    update_take_session.add_argument("--dataset-id", default=None, help="Explicit dataset id when takes span multiple datasets")
    update_take_session.add_argument("--create-session", action="store_true", help="Create destination session if missing")
    update_take_session.add_argument("--take-id", action="append", default=[], help="Take id to update (repeatable)")
    update_take_session.add_argument("--take-file", type=Path, default=None, help="Newline-separated take id file")
    update_take_session.add_argument("--allow-missing", action="store_true", help="Continue when some requested takes are missing")
    mode = update_take_session.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Apply changes (default is dry-run)")
    mode.add_argument("--dry-run", action="store_true", help="Preview changes without writing (default)")
    update_take_session.set_defaults(handler=cmd_acquisition_update_take_session)

    validation = top.add_parser("validation", help="Approved-reference regression suites")
    validation_sub = validation.add_subparsers(dest="command", required=True)
    validation_list = validation_sub.add_parser("list-suites", help="List regression suites")
    validation_list.add_argument("--data-dir", type=Path, default=Path("data"))
    validation_list.set_defaults(handler=cmd_validation_list_suites)
    validation_run = validation_sub.add_parser("run-suite", help="Compare suite baselines with candidate runs")
    validation_run.add_argument("--data-dir", type=Path, default=Path("data"))
    validation_run.add_argument("--suite-id", required=True)
    validation_run.add_argument("--recipe-id", default=None)
    validation_run.add_argument("--candidate-run-id", default="latest")
    validation_run.set_defaults(handler=cmd_validation_run_suite)
    validation_compare = validation_sub.add_parser("compare", help="Compare one baseline to a candidate run")
    validation_compare.add_argument("--data-dir", type=Path, default=Path("data"))
    validation_compare.add_argument("--baseline-id", required=True)
    validation_compare.add_argument("--candidate-run-id", required=True)
    validation_compare.set_defaults(handler=cmd_validation_compare)
    validation_inspect = validation_sub.add_parser("inspect-execution", help="Inspect a persisted validation execution")
    validation_inspect.add_argument("--data-dir", type=Path, default=Path("data"))
    validation_inspect.add_argument("--execution-id", required=True)
    validation_inspect.set_defaults(handler=cmd_validation_inspect_execution)
    validation_rebuild = validation_sub.add_parser("rebuild-indexes", help="Rebuild validation indexes from canonical records")
    validation_rebuild.add_argument("--data-dir", type=Path, default=Path("data"))
    validation_rebuild.set_defaults(handler=cmd_validation_rebuild_indexes)
    validation_verify = validation_sub.add_parser("verify-integrity", help="Verify baseline snapshots, checksums, and suite references")
    validation_verify.add_argument("--data-dir", type=Path, default=Path("data"))
    validation_verify.set_defaults(handler=cmd_validation_verify_integrity)
    validation_smoke = validation_sub.add_parser("smoke", help="Run deterministic 2.5D validation baseline smoke workflow")
    validation_smoke.add_argument("--data-dir", type=Path, default=Path("data"))
    validation_smoke.set_defaults(handler=cmd_validation_smoke)
    for command, help_text in (("inspect-suite","Inspect a suite"),("coverage","Show suite coverage"),("archive-suite","Archive a suite"),("restore-suite","Restore an archived suite")):
        item=validation_sub.add_parser(command,help=help_text); item.add_argument("--data-dir",type=Path,default=Path("data")); item.add_argument("--suite-id",required=True)
        item.set_defaults(handler=cmd_validation_inspect_suite if command=="inspect-suite" else cmd_validation_coverage if command=="coverage" else cmd_validation_archive_restore)
    history=validation_sub.add_parser("baseline-history",help="List baseline version history")
    history.add_argument("--data-dir",type=Path,default=Path("data")); history.add_argument("--take-id"); history.add_argument("--pipeline-id"); history.add_argument("--artifact-id"); history.set_defaults(handler=cmd_validation_baseline_history)
    for command,help_text in (("activate-baseline","Activate a baseline version"),("deactivate-baseline","Deactivate a baseline version")):
        item=validation_sub.add_parser(command,help=help_text); item.add_argument("--data-dir",type=Path,default=Path("data")); item.add_argument("--baseline-id",required=True); item.set_defaults(handler=cmd_validation_set_baseline_active)
    promote=validation_sub.add_parser("promote-comparison",help="Promote an execution comparison candidate to a baseline")
    promote.add_argument("--data-dir",type=Path,default=Path("data")); promote.add_argument("--execution-id",required=True); promote.add_argument("--case-id",required=True); promote.add_argument("--comparison-id",required=True); promote.add_argument("--activate",action="store_true"); promote.add_argument("--notes",default=""); promote.add_argument("--reviewed-by",default=None); promote.add_argument("--reset-policy",action="store_true"); promote.set_defaults(handler=cmd_validation_promote)
    update_suite=validation_sub.add_parser("update-suite",help="Update suite name, description, or state")
    update_suite.add_argument("--data-dir",type=Path,default=Path("data")); update_suite.add_argument("--suite-id",required=True); update_suite.add_argument("--name"); update_suite.add_argument("--description"); update_suite.add_argument("--status",choices=["active","inactive","archived"]); update_suite.set_defaults(handler=cmd_validation_update_suite)
    duplicate_suite=validation_sub.add_parser("duplicate-suite",help="Duplicate suite and cases")
    duplicate_suite.add_argument("--data-dir",type=Path,default=Path("data")); duplicate_suite.add_argument("--suite-id",required=True); duplicate_suite.set_defaults(handler=cmd_validation_duplicate_suite)
    list_exec=validation_sub.add_parser("list-executions",help="List persisted validation executions")
    list_exec.add_argument("--data-dir",type=Path,default=Path("data")); list_exec.add_argument("--suite-id"); list_exec.set_defaults(handler=cmd_validation_list_executions)
    add_case=validation_sub.add_parser("add-case",help="Add a baseline-backed regression case")
    add_case.add_argument("--data-dir",type=Path,default=Path("data")); add_case.add_argument("--suite-id",required=True); add_case.add_argument("--take-id",required=True); add_case.add_argument("--baseline-id",dest="baseline_id_list",action="append",default=[]); add_case.add_argument("--resolution-mode",default="active",choices=["active","pinned","allowed_versions"]); add_case.add_argument("--pinned-baseline-id",dest="baseline_id"); add_case.add_argument("--allowed-baseline-id",action="append",default=[]); add_case.add_argument("--tag",action="append",default=[]); add_case.add_argument("--notes",default=""); add_case.add_argument("--enabled",action=argparse.BooleanOptionalAction,default=True); add_case.set_defaults(handler=cmd_validation_case)
    update_case=validation_sub.add_parser("update-case",help="Edit a regression case")
    update_case.add_argument("--data-dir",type=Path,default=Path("data")); update_case.add_argument("--suite-id",required=True); update_case.add_argument("--case-id",required=True); update_case.add_argument("--resolution-mode",required=True,choices=["active","pinned","allowed_versions"]); update_case.add_argument("--baseline-id"); update_case.add_argument("--allowed-baseline-id",action="append",default=[]); update_case.add_argument("--tag",action="append",default=[]); update_case.add_argument("--notes",default=""); update_case.add_argument("--enabled",action=argparse.BooleanOptionalAction,default=True); update_case.set_defaults(handler=cmd_validation_case)
    remove_case=validation_sub.add_parser("remove-case",help="Remove a case without deleting baselines")
    remove_case.add_argument("--data-dir",type=Path,default=Path("data")); remove_case.add_argument("--suite-id",required=True); remove_case.add_argument("--case-id",required=True); remove_case.set_defaults(handler=cmd_validation_case, resolution_mode="active", enabled=True, tag=[], notes="", baseline_id=None, allowed_baseline_id=[], baseline_id_list=[])

    ml = top.add_parser("ml", help="ML set persistence and membership management")
    ml_sub = ml.add_subparsers(dest="command", required=False)
    ml.set_defaults(domain="ml")

    ml_create = ml_sub.add_parser("create-set", help="Create or overwrite an ML set")
    ml_create.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    ml_create.add_argument("--dataset-id", required=True, help="Dataset id owning this ML set")
    ml_create.add_argument("--ml-set-id", required=True, help="ML set id")
    ml_create.add_argument("--name", required=True, help="Display name")
    ml_create.add_argument("--description", default=None, help="Optional description")
    ml_create.add_argument(
        "--task-type",
        required=True,
        choices=["classification", "regression", "detection", "segmentation", "clustering", "benchmark"],
        help="ML task type",
    )
    ml_create.add_argument("--notes", default=None, help="Optional notes")
    ml_create.add_argument("--overwrite", action="store_true", help="Overwrite if ML set already exists")
    ml_create.set_defaults(handler=cmd_ml_create_set)

    ml_add = ml_sub.add_parser("add-takes", help="Add or update take memberships in an ML set")
    ml_add.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    ml_add.add_argument("--ml-set-id", required=True, help="ML set id")
    ml_add.add_argument("--dataset-id", default=None, help="Dataset id to disambiguate ML set lookup")
    ml_add.add_argument("--take-id", action="append", default=[], help="Take id to add (repeatable)")
    ml_add.add_argument("--take-file", type=Path, default=None, help="Newline-separated take id file")
    ml_add.add_argument(
        "--split",
        default="unassigned",
        choices=["train", "validation", "test", "holdout", "calibration", "unassigned"],
        help="Membership split label",
    )
    ml_add.add_argument("--physical-object-id", default=None, help="Optional physical object id for leakage-safe grouping")
    ml_add.add_argument("--include", type=_parse_bool, default=True, help="Include membership (true|false)")
    ml_add.add_argument("--notes", default=None, help="Optional membership notes")
    ml_add.set_defaults(handler=cmd_ml_add_takes)

    ml_list = ml_sub.add_parser("list-set", help="Show ML set metadata and membership summary")
    ml_list.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    ml_list.add_argument("--ml-set-id", required=True, help="ML set id")
    ml_list.add_argument("--dataset-id", default=None, help="Dataset id to disambiguate ML set lookup")
    ml_list.add_argument("--show-memberships", action="store_true", help="Show sample membership rows")
    ml_list.add_argument("--limit", type=int, default=20, help="Max membership rows when --show-memberships is set")
    ml_list.set_defaults(handler=cmd_ml_list_set)

    ml_assign = ml_sub.add_parser("assign-splits", help="Assign deterministic leakage-safe splits to ML set memberships")
    ml_assign.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    ml_assign.add_argument("--ml-set-id", required=True, help="ML set id")
    ml_assign.add_argument("--dataset-id", default=None, help="Dataset id to disambiguate ML set lookup")
    ml_assign.add_argument("--train", type=float, default=0.7, help="Train split ratio")
    ml_assign.add_argument("--validation", type=float, default=0.15, help="Validation split ratio")
    ml_assign.add_argument("--test", type=float, default=0.15, help="Test split ratio")
    ml_assign.add_argument("--holdout", type=float, default=0.0, help="Optional holdout split ratio")
    ml_assign.add_argument("--calibration", type=float, default=0.0, help="Optional calibration split ratio")
    ml_assign.add_argument("--seed", type=int, default=42, help="Deterministic assignment seed")
    ml_assign.add_argument("--by-physical-object-id", action=argparse.BooleanOptionalAction, default=True, help="Group by physical_object_id (default: true)")
    mode_assign = ml_assign.add_mutually_exclusive_group()
    mode_assign.add_argument("--apply", action="store_true", help="Apply split changes")
    mode_assign.add_argument("--dry-run", action="store_true", help="Preview only (default)")
    ml_assign.set_defaults(handler=cmd_ml_assign_splits)

    ml_reprocess = ml_sub.add_parser("reprocess-set", help="Batch reprocess takes from an ML set through a pipeline")
    ml_reprocess.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    ml_reprocess.add_argument("--ml-set-id", required=True, help="ML set id")
    ml_reprocess.add_argument("--dataset-id", default=None, help="Dataset id to disambiguate ML set lookup")
    ml_reprocess.add_argument("--pipeline-id", required=True, help="Pipeline id to execute for each selected take")
    ml_reprocess.add_argument("--split", action="append", default=[], help="Split filter (repeatable)")
    ml_reprocess.add_argument("--classifier-rules", default=None, help="Optional classifier rule config path for 25D classification stage")
    mode_reprocess = ml_reprocess.add_mutually_exclusive_group()
    mode_reprocess.add_argument("--apply", action="store_true", help="Execute processing")
    mode_reprocess.add_argument("--dry-run", action="store_true", help="Preview only (default)")
    ml_reprocess.set_defaults(handler=cmd_ml_reprocess_set)

    ml_compute_feature = ml_sub.add_parser("compute-feature", help="Reprocess or backfill a registered feature across an ML set")
    ml_compute_feature.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    ml_compute_feature.add_argument("--ml-set-id", required=True, help="ML set id")
    ml_compute_feature.add_argument("--dataset-id", default=None, help="Dataset id to disambiguate ML set lookup")
    ml_compute_feature.add_argument("--feature", default="surface_sphere_fit_rmse_mm", help="Feature key to compute")
    ml_compute_feature.add_argument(
        "--mode",
        required=True,
        choices=["reprocess", "backfill"],
        help="reprocess: rerun pipeline; backfill: compute sidecar from existing artifacts",
    )
    ml_compute_feature.add_argument("--pipeline-id", default=PIPELINE_ID_25D, help="Pipeline id for reprocess mode")
    mode_compute = ml_compute_feature.add_mutually_exclusive_group()
    mode_compute.add_argument("--apply", action="store_true", help="Execute compute/backfill")
    mode_compute.add_argument("--dry-run", action="store_true", help="Preview only (default)")
    ml_compute_feature.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    ml_compute_feature.set_defaults(handler=cmd_ml_compute_feature)

    ml_export = ml_sub.add_parser("export-features", help="Export ML set features from pipeline processing outputs")
    ml_export.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    ml_export.add_argument("--ml-set-id", required=True, help="ML set id")
    ml_export.add_argument("--dataset-id", default=None, help="Dataset id to disambiguate ML set lookup")
    ml_export.add_argument("--pipeline-id", required=True, help="Pipeline id used for feature extraction")
    ml_export.add_argument("--output", type=Path, required=True, help="Output CSV path")
    ml_export.add_argument("--split", action="append", default=[], help="Split filter (repeatable)")
    ml_export.add_argument("--require-processed", action="store_true", help="Fail when selected takes are missing compatible processing outputs")
    ml_export.add_argument("--include-diagnostics", action="store_true", help="Include persisted diagnostics feature vector columns")
    ml_export.add_argument("--include-invalidity-flags", action="store_true", help="Include quality flag summary columns")
    ml_export.add_argument("--include-provenance-summary", action="store_true", help="Include compact provenance summary columns")
    ml_export.set_defaults(handler=cmd_ml_export_features)

    ml_import = ml_sub.add_parser("import-manifest", help="Import object-aware ML curation manifest into an ML set")
    ml_import.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    ml_import.add_argument("--ml-set-id", required=True, help="ML set id")
    ml_import.add_argument("--dataset-id", default=None, help="Dataset id to disambiguate ML set lookup")
    ml_import.add_argument("--manifest", type=Path, required=True, help="Manifest path (.csv or .json)")
    ml_import.add_argument("--allow-overwrite-object-id", action="store_true", help="Allow take object-id reassignment conflicts")
    mode_import = ml_import.add_mutually_exclusive_group()
    mode_import.add_argument("--apply", action="store_true", help="Apply manifest import")
    mode_import.add_argument("--dry-run", action="store_true", help="Preview only (default)")
    ml_import.set_defaults(handler=cmd_ml_import_manifest)

    ml_rule_sets = ml_sub.add_parser("list-rule-sets", help="List available classifier rule set configs")
    ml_rule_sets.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    ml_rule_sets.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    ml_rule_sets.add_argument("--classifier-id", default=None, help="Optional classifier_id filter")
    ml_rule_sets.set_defaults(handler=cmd_ml_list_rule_sets)

    ml_active_rules = ml_sub.add_parser("show-active-rule-set", help="Show resolved active classifier rule set")
    ml_active_rules.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    ml_active_rules.add_argument("--pipeline-id", default="mining_steel_ball_classification_25d", help="Pipeline id for pipeline-level rule set config lookup")
    ml_active_rules.add_argument("--dataset-id", default=None, help="Optional dataset context metadata")
    ml_active_rules.add_argument("--classifier-rules", default=None, help="Optional runtime override path")
    ml_active_rules.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    ml_active_rules.set_defaults(handler=cmd_ml_show_active_rule_set)

    index = top.add_parser("index", help="SQLite catalog index (data/index.db)")
    index_sub = index.add_subparsers(dest="command", required=False)
    index.set_defaults(domain="index")

    index_rebuild = index_sub.add_parser("rebuild", help="Build or refresh the catalog index from the filesystem")
    index_rebuild.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    index_rebuild.add_argument("--full", action="store_true", help="Reproject every take instead of only the stale ones")
    index_rebuild.set_defaults(handler=cmd_index_rebuild)

    index_status = index_sub.add_parser("status", help="Show schema version, row counts, and drift against disk")
    index_status.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    index_status.set_defaults(handler=cmd_index_status)

    index_export = index_sub.add_parser(
        "export", help="Write the authoritative documents back out as JSON directories"
    )
    index_export.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    index_export.add_argument("--to", type=Path, default=None, help="Destination (default: data/export_<timestamp>)")
    index_export.set_defaults(handler=cmd_index_export)

    index_verify = index_sub.add_parser("verify", help="Diff every indexed summary against a fresh filesystem read")
    index_verify.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    index_verify.add_argument("--limit", type=int, default=None, help="Check only the first N indexed takes")
    index_verify.add_argument("--show-all", action="store_true", help="List every mismatch instead of the first 20")
    index_verify.set_defaults(handler=cmd_index_verify)

    return parser


def main() -> int:
    parser = build_parser()
    if len(sys.argv) == 1:
        print("Sensor Studio CLI")
        print("Unified wrapper for 25D runtime operations, studio services, and acquisition helpers.\n")
        parser.print_help()
        print("\nQuick start:")
        print("  python scripts/sensor_studio_cli.py 25d demo --data-dir data")
        print("  python scripts/sensor_studio_cli.py 25d create-synthetic --data-dir data --session-id synthetic_25d_demo")
        print("  python scripts/sensor_studio_cli.py 25d process --data-dir data --take-id <take_id>")
        print("  python scripts/sensor_studio_cli.py 25d validate-api --data-dir data --take-id <take_id>")
        print("  python scripts/sensor_studio_cli.py 25d inspect-result --data-dir data --take-id <take_id>")
        print("  python scripts/sensor_studio_cli.py 25d interactive --data-dir data")
        print("  python scripts/sensor_studio_cli.py acquisition latest --data-dir data")
        print("\nTip: use --help at any level, for example:")
        print("  python scripts/sensor_studio_cli.py 25d --help")
        return 2
    args = parser.parse_args()
    if not getattr(args, "domain", None):
        parser.print_help()
        return 2
    if not getattr(args, "handler", None):
        if args.domain == "25d":
            print("Sensor Studio CLI - 25D")
            print("Run native 2.5D creation, processing, validation, inspection, and interactive workflows.\n")
            print("Examples:")
            print("  python scripts/sensor_studio_cli.py 25d demo --data-dir data")
            print("  python scripts/sensor_studio_cli.py 25d process --data-dir data --take-id <take_id>")
            print("  python scripts/sensor_studio_cli.py 25d validate-api --data-dir data --take-id <take_id>")
            print("  python scripts/sensor_studio_cli.py 25d interactive --data-dir data\n")
            print("Use --help to see all 25D commands:")
            print("  python scripts/sensor_studio_cli.py 25d --help")
            return 2
        if args.domain == "studio":
            print("Sensor Studio CLI - Studio")
            print("Start and manage studio-facing services.\n")
            print("Examples:")
            print("  python scripts/sensor_studio_cli.py studio serve --data-dir data\n")
            print("  python scripts/sensor_studio_cli.py studio web")
            print("  python scripts/sensor_studio_cli.py studio dev\n")
            print("Use --help to see all studio commands:")
            print("  python scripts/sensor_studio_cli.py studio --help")
            return 2
        if args.domain == "acquisition":
            print("Sensor Studio CLI - Acquisition")
            print("Inspect incoming takes and quick acquisition status.\n")
            print("Examples:")
            print("  python scripts/sensor_studio_cli.py acquisition list-takes --data-dir data --limit 20")
            print("  python scripts/sensor_studio_cli.py acquisition latest --data-dir data\n")
            print("  python scripts/sensor_studio_cli.py acquisition update-take-session --data-dir data --session-id s1 --take-id <take_id>")
            print("Use --help to see all acquisition commands:")
            print("  python scripts/sensor_studio_cli.py acquisition --help")
        if args.domain == "ml":
            print("Sensor Studio CLI - ML Sets")
            print("Create and curate reusable ML set groupings without changing acquisition sessions.\n")
            print("Examples:")
            print(
                "  python scripts/sensor_studio_cli.py ml create-set --data-dir data --dataset-id <id> --ml-set-id <id> "
                "--name \"My set\" --task-type classification"
            )
            print("  python scripts/sensor_studio_cli.py ml add-takes --data-dir data --ml-set-id <id> --take-file labels/takes.txt")
            print("  python scripts/sensor_studio_cli.py ml list-set --data-dir data --ml-set-id <id> --show-memberships")
            print("  python scripts/sensor_studio_cli.py ml assign-splits --data-dir data --ml-set-id <id> --seed 42 --apply")
            print("  python scripts/sensor_studio_cli.py ml reprocess-set --data-dir data --ml-set-id <id> --pipeline-id <pipeline> --apply")
            print("  python scripts/sensor_studio_cli.py ml export-features --data-dir data --ml-set-id <id> --pipeline-id <pipeline> --output data/ml_exports/features.csv")
            print("  python scripts/sensor_studio_cli.py ml import-manifest --data-dir data --ml-set-id <id> --manifest labels/manifest.csv --apply")
            print("  python scripts/sensor_studio_cli.py ml list-rule-sets --data-dir data")
            print("  python scripts/sensor_studio_cli.py ml show-active-rule-set --pipeline-id mining_steel_ball_classification_25d")
            print("\nUse --help to see all ML commands:")
            print("  python scripts/sensor_studio_cli.py ml --help")
        return 2
        parser.print_help()
        return 2
    args.data_dir = args.data_dir.resolve()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
