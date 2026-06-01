from __future__ import annotations

import copy
from datetime import UTC, datetime
from typing import Any

from vision_3d_acquisition.ml.models import ModelDeployment
from vision_3d_acquisition.ml.runtime import MLRuntimeResolver
from vision_3d_acquisition.ml.storage import MLStorage


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


class DeploymentService:
    def __init__(self, data_dir):
        self.storage = MLStorage(data_dir)
        self.runtime = MLRuntimeResolver(data_dir)

    def list_deployments(self) -> list[dict[str, Any]]:
        rows = sorted(self.storage.list("deployments"), key=lambda d: str(d.get("deployed_at") or ""), reverse=True)
        return [self._enrich(row) for row in rows]

    def create_deployment(self, payload: dict[str, Any]) -> dict[str, Any]:
        base = {
            "status": "inactive",
            "activated_at": None,
            "deactivated_at": None,
            "superseded_by": None,
            "rollback_from": None,
            "runtime_validation": {},
            "compatibility_snapshot": {},
            "pipeline_family": payload.get("pipeline_family") or "25d",
        }
        dep = ModelDeployment.model_validate({**base, **payload})
        stored = dep.model_dump(mode="json")
        return self._enrich(self.storage.save("deployments", dep.id, stored))

    def activate(self, deployment_id: str) -> dict[str, Any]:
        dep = self.storage.get("deployments", deployment_id)
        if not dep:
            raise ValueError("deployment not found")
        model_id = str(dep.get("model_id") or "")
        model = self.storage.get("models", model_id)
        if not model:
            raise ValueError("model not found")
        if not bool(model.get("promoted")):
            raise ValueError("model must be promoted")

        active = self.get_active(
            pipeline_id=str(dep.get("target_pipeline_id") or ""),
            stage_id=str(dep.get("target_stage_id") or ""),
            pipeline_family=str(dep.get("pipeline_family") or "25d"),
        )

        compat = self.runtime.validate_compatibility(
            deployment=dep,
            model=model,
            pipeline_id=str(dep.get("target_pipeline_id") or ""),
            stage_id=str(dep.get("target_stage_id") or ""),
            pipeline_family=str(dep.get("pipeline_family") or "25d"),
        )
        activated = copy.deepcopy(dep)
        activated["id"] = f"{deployment_id}__active__{int(datetime.now(UTC).timestamp()*1000)}"
        activated["status"] = "active" if compat.get("compatible") else "invalid"
        activated["activated_at"] = now_iso()
        activated["runtime_validation"] = compat
        activated["compatibility_snapshot"] = compat.get("checks", {})
        self.storage.save("deployments", activated["id"], activated)

        if activated.get("status") == "active" and active and str(active.get("id")) != str(activated.get("id")):
            superseded = copy.deepcopy(active)
            superseded["id"] = f"{active.get('id')}__superseded__{int(datetime.now(UTC).timestamp()*1000)}"
            superseded["status"] = "superseded"
            superseded["deactivated_at"] = now_iso()
            superseded["superseded_by"] = activated["id"]
            self.storage.save("deployments", superseded["id"], superseded)

        return self._enrich(activated)

    def deactivate(self, deployment_id: str) -> dict[str, Any]:
        dep = self.storage.get("deployments", deployment_id)
        if not dep:
            raise ValueError("deployment not found")
        next_dep = copy.deepcopy(dep)
        next_dep["id"] = f"{deployment_id}__deactivated__{int(datetime.now(UTC).timestamp()*1000)}"
        next_dep["status"] = "inactive"
        next_dep["deactivated_at"] = now_iso()
        return self._enrich(self.storage.save("deployments", next_dep["id"], next_dep))

    def rollback(self, deployment_id: str) -> dict[str, Any]:
        dep = self.storage.get("deployments", deployment_id)
        if not dep:
            raise ValueError("deployment not found")
        src = str(dep.get("rollback_from") or dep.get("id") or "")
        rolled = copy.deepcopy(dep)
        rolled["id"] = f"{deployment_id}__rollback__{int(datetime.now(UTC).timestamp()*1000)}"
        rolled["status"] = "rolled_back"
        rolled["rollback_from"] = src
        rolled["activated_at"] = now_iso()
        self.storage.save("deployments", rolled["id"], rolled)
        return self.activate(rolled["id"])

    def get_active(self, *, pipeline_id: str, stage_id: str, pipeline_family: str) -> dict[str, Any] | None:
        candidates = [
            d for d in self.storage.list("deployments")
            if str(d.get("target_pipeline_id") or "") == pipeline_id
            and str(d.get("target_stage_id") or "") == stage_id
            and str(d.get("pipeline_family") or "") == pipeline_family
            and str(d.get("status") or "") == "active"
        ]
        candidates.sort(key=lambda d: str(d.get("activated_at") or d.get("deployed_at") or ""), reverse=True)
        return candidates[0] if candidates else None

    def list_active(self) -> list[dict[str, Any]]:
        by_key: dict[str, dict[str, Any]] = {}
        for d in self.storage.list("deployments"):
            if str(d.get("status") or "") != "active":
                continue
            key = "|".join([
                str(d.get("target_pipeline_id") or ""),
                str(d.get("target_stage_id") or ""),
                str(d.get("pipeline_family") or ""),
            ])
            current = by_key.get(key)
            if current is None or str(d.get("activated_at") or "") > str(current.get("activated_at") or ""):
                by_key[key] = d
        return [self._enrich(v) for v in list(by_key.values())]

    def _enrich(self, dep: dict[str, Any]) -> dict[str, Any]:
        model = self.storage.get("models", str(dep.get("model_id") or ""))
        model_version = str((model or {}).get("version") or "")
        return {
            **dep,
            "model_version": model_version or None,
            "rollback_lineage": {
                "rollback_from": dep.get("rollback_from"),
                "superseded_by": dep.get("superseded_by"),
            },
        }
