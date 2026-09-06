from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from vision_3d_acquisition.api.settings import ApiSettings, get_settings
from vision_3d_acquisition.validation import ValidationService

router = APIRouter(prefix="/api/validation", tags=["validation"])

class BaselineRequest(BaseModel):
    take_id: str; pipeline_id: str; run_id: str
    approval_scope: str = "artifact"
    stage_id: str | None = None; processing_unit_id: str | None = None; substage_id: str | None = None
    view_id: str | None = None; artifact_id: str | None = None
    notes: str | None = None; reviewed_by: str | None = None
    comparison_policy: dict[str, Any] = Field(default_factory=dict)

class SuiteRequest(BaseModel):
    name: str; pipeline_id: str; description: str | None = None
    default_comparison_policy: dict[str, Any] = Field(default_factory=dict)

def service(settings: ApiSettings = Depends(get_settings)) -> ValidationService: return ValidationService(settings)
def _error(fn):
    try: return fn()
    except KeyError as exc: raise HTTPException(404, f"Unknown validation record: {exc}")
    except ValueError as exc: raise HTTPException(422, str(exc))

@router.post("/baselines")
def create_baseline(payload: BaselineRequest, store: ValidationService = Depends(service)): return _error(lambda: store.create_baselines(payload.model_dump()))
@router.get("/baselines")
def list_baselines(take_id: str | None = None, pipeline_id: str | None = None, store: ValidationService = Depends(service)): return store.list_baselines(take_id=take_id, pipeline_id=pipeline_id)
@router.get("/baselines/history")
def baseline_history(take_id: str | None = None, pipeline_id: str | None = None, artifact_id: str | None = None, store: ValidationService = Depends(service)): return store.baseline_history(take_id=take_id,pipeline_id=pipeline_id,artifact_id=artifact_id)
@router.get("/baselines/{baseline_id}")
def get_baseline(baseline_id: str, store: ValidationService = Depends(service)): return _error(lambda: store.get_baseline(baseline_id) or (_ for _ in ()).throw(KeyError(baseline_id)))
@router.post("/baselines/{baseline_id}/activate")
def activate(baseline_id: str, store: ValidationService = Depends(service)): return _error(lambda: store.set_active(baseline_id, True))
@router.post("/baselines/{baseline_id}/deactivate")
def deactivate(baseline_id: str, store: ValidationService = Depends(service)): return _error(lambda: store.set_active(baseline_id, False))
@router.get("/baselines/{baseline_id}/impact")
def resolution_impact(baseline_id: str, store: ValidationService = Depends(service)):
    """Which cases would follow this version if it were activated, and which would not."""
    return _error(lambda: store.resolution_impact(store.get_baseline(baseline_id) or (_ for _ in ()).throw(KeyError(baseline_id))))
@router.post("/baselines/{baseline_id}/compare")
def compare(baseline_id: str, candidate_run_id: str, store: ValidationService = Depends(service)): return _error(lambda: store.compare(baseline_id, candidate_run_id=candidate_run_id))
@router.get("/files/{path:path}")
def get_file(path: str, store: ValidationService = Depends(service)):
    """Serve a file stored under the validation root: baseline snapshots and diff artifacts.

    Every path the service hands out (``baseline.snapshot_artifact_path``,
    entries in ``diff_artifacts``) is already relative to this root, so this is
    the one place that turns those references into bytes. The traversal guard
    mirrors ``safe_take_file``: reject any segment that could escape the root,
    then require the resolved path to still be inside it.
    """
    normalized = path.replace("\\", "/").strip("/")
    if not normalized or any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise HTTPException(404, "Unknown validation file")
    root = (store.root).resolve()
    candidate = (store.root / normalized).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise HTTPException(404, "Unknown validation file")
    if not candidate.is_file():
        raise HTTPException(404, "Unknown validation file")
    return FileResponse(candidate)
@router.post("/indexes/rebuild")
def rebuild_indexes(store: ValidationService = Depends(service)): return store.rebuild_indexes()
@router.get("/integrity")
def verify_integrity(store: ValidationService = Depends(service)): return store.verify_integrity()

@router.post("/suites")
def create_suite(payload: SuiteRequest, store: ValidationService = Depends(service)): return store.create_suite(payload.model_dump())
@router.get("/suites")
def list_suites(store: ValidationService = Depends(service)): return store.list_suites()
@router.get("/suites/{suite_id}")
def get_suite(suite_id: str, store: ValidationService = Depends(service)): return _error(lambda: store.get_suite(suite_id))
@router.put("/suites/{suite_id}")
def update_suite(suite_id: str, payload: dict[str, Any], store: ValidationService = Depends(service)): return _error(lambda: store.update_suite(suite_id, payload))
@router.post("/suites/{suite_id}/duplicate")
def duplicate_suite(suite_id: str, store: ValidationService = Depends(service)): return _error(lambda: store.duplicate_suite(suite_id))
@router.post("/suites/{suite_id}/archive")
def archive_suite(suite_id: str, store: ValidationService = Depends(service)): return _error(lambda: store.archive_suite(suite_id))
@router.post("/suites/{suite_id}/restore")
def restore_suite(suite_id: str, store: ValidationService = Depends(service)): return _error(lambda: store.restore_suite(suite_id))
@router.get("/suites/{suite_id}/coverage")
def suite_coverage(suite_id: str, store: ValidationService = Depends(service)): return _error(lambda: store.coverage(suite_id))
@router.get("/suites/{suite_id}/executions")
def suite_executions(suite_id: str, store: ValidationService = Depends(service)): return store.list_executions(suite_id=suite_id)
@router.post("/suites/{suite_id}/cases")
def add_case(suite_id: str, payload: dict[str, Any], store: ValidationService = Depends(service)): return _error(lambda: store.add_case(suite_id, payload))
@router.patch("/suites/{suite_id}/cases/{case_id}")
def update_case(suite_id: str, case_id: str, payload: dict[str, Any], store: ValidationService = Depends(service)): return _error(lambda: store.update_case(suite_id, case_id, payload))
@router.delete("/suites/{suite_id}/cases/{case_id}")
def delete_case(suite_id: str, case_id: str, store: ValidationService = Depends(service)): return _error(lambda: store.delete_case(suite_id, case_id))
@router.post("/suites/{suite_id}/execute")
def execute_suite(suite_id: str, payload: dict[str, Any] | None = None, store: ValidationService = Depends(service)): return _error(lambda: store.execute_suite(suite_id, payload or {}))
@router.get("/executions")
def list_executions(store: ValidationService = Depends(service)): return store.list_executions()
@router.get("/executions/{execution_id}")
def get_execution(execution_id: str, store: ValidationService = Depends(service)): return _error(lambda: store.get_execution(execution_id))
@router.get("/executions/{execution_id}/progress")
def progress(execution_id: str, store: ValidationService = Depends(service)): return _error(lambda: store.get_execution(execution_id).get("progress"))
@router.get("/executions/{execution_id}/matrix")
def matrix(execution_id: str, store: ValidationService = Depends(service)): return _error(lambda: store.matrix(execution_id))
@router.get("/executions/{execution_id}/stage-summary")
def stage_summary(execution_id: str, store: ValidationService = Depends(service)): return _error(lambda: store.stage_summary(execution_id))
@router.get("/suites/{suite_id}/stage-trend")
def stage_trend(suite_id: str, limit: int = 5, store: ValidationService = Depends(service)): return _error(lambda: store.stage_trend(suite_id, limit=limit))
@router.post("/executions/{execution_id}/comparisons/{comparison_id}/promote")
def promote(execution_id: str, comparison_id: str, payload: dict[str, Any], store: ValidationService = Depends(service)): return _error(lambda: store.promote_comparison(execution_id, str(payload.get("case_id") or ""), comparison_id, payload))
