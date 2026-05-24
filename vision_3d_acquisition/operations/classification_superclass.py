from __future__ import annotations

from collections import Counter
from typing import Final
from typing import Any

SUPERCLASS_BALL_GOOD: Final[str] = "BALL_GOOD"
SUPERCLASS_BALL_SCRAP: Final[str] = "BALL_SCRAP"
SUPERCLASS_SCRAP_METAL: Final[str] = "SCRAP_METAL"
SUPERCLASS_UNKNOWN: Final[str] = "UNKNOWN"

SPHERICITY_3D_THRESHOLD_SCRAP_METAL: Final[float] = 0.30
SPHERICITY_3D_THRESHOLD_BALL_SCRAP: Final[float] = 0.75

_LABEL_TO_SUPERCLASS: dict[str, str] = {
    "bola_buena": SUPERCLASS_BALL_GOOD,
    "good_ball": SUPERCLASS_BALL_GOOD,
    "ball_good": SUPERCLASS_BALL_GOOD,
    "bola buena": SUPERCLASS_BALL_GOOD,
    "bola_con_chip": SUPERCLASS_BALL_SCRAP,
    "bola_partida": SUPERCLASS_BALL_SCRAP,
    "bola_deformada": SUPERCLASS_BALL_SCRAP,
    "worn_ball": SUPERCLASS_BALL_SCRAP,
    "broken_ball": SUPERCLASS_BALL_SCRAP,
    "deformed_ball": SUPERCLASS_BALL_SCRAP,
    "scrap de bola": SUPERCLASS_BALL_SCRAP,
    "perno": SUPERCLASS_SCRAP_METAL,
    "planchuela": SUPERCLASS_SCRAP_METAL,
    "malla": SUPERCLASS_SCRAP_METAL,
    "scrap": SUPERCLASS_SCRAP_METAL,
    "non_ball": SUPERCLASS_SCRAP_METAL,
    "metal_scrap": SUPERCLASS_SCRAP_METAL,
    "chatarra": SUPERCLASS_SCRAP_METAL,
}


def _sphericity_3d_from_object(item: dict[str, Any]) -> float | None:
    value = item.get("feature_sphericity_3d")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def apply_sphericity_3d_fallback_to_object(
    item: dict[str, Any],
    *,
    label: str,
    display_label: str,
    class_name: str,
    confidence: float,
    superclass: str,
) -> tuple[str, str, str, float, str]:
    """Secondary sph3d fallback after primary classification.

    BALL_GOOD and SCRAP_METAL objects are returned unchanged. All other objects
    receive debug metadata; classification is overridden only when sph3d
    falls below the BALL_SCRAP threshold.
    """
    sph3d = _sphericity_3d_from_object(item)
    debug = dict(item["debug"]) if isinstance(item.get("debug"), dict) else {}
    debug["sph3d_rule"] = {
        "sph3d": sph3d,
        "threshold_scrap_metal": SPHERICITY_3D_THRESHOLD_SCRAP_METAL,
        "threshold_ball_scrap": SPHERICITY_3D_THRESHOLD_BALL_SCRAP,
    }
    item["debug"] = debug

    if superclass == SUPERCLASS_BALL_GOOD:
        return label, display_label, class_name, confidence, superclass

    if superclass == SUPERCLASS_SCRAP_METAL:
        return label, display_label, class_name, confidence, superclass

    if sph3d is None:
        return label, display_label, class_name, confidence, superclass

    if sph3d < SPHERICITY_3D_THRESHOLD_SCRAP_METAL:
        item["classification_reason"] = (
            f"SCRAP_METAL because sph3d={sph3d:.3f} < {SPHERICITY_3D_THRESHOLD_SCRAP_METAL:.2f}"
        )
        return "chatarra", "Chatarra - Chatarra", "non_ball", confidence, SUPERCLASS_SCRAP_METAL

    if sph3d < SPHERICITY_3D_THRESHOLD_BALL_SCRAP:
        item["classification_reason"] = (
            f"BALL_SCRAP because {SPHERICITY_3D_THRESHOLD_SCRAP_METAL:.2f} <= "
            f"sph3d={sph3d:.3f} < {SPHERICITY_3D_THRESHOLD_BALL_SCRAP:.2f}"
        )
        return (
            "bola_con_chip",
            "Scrap de Bola - Bola con chip",
            "non_ball",
            confidence,
            SUPERCLASS_BALL_SCRAP,
        )

    return label, display_label, class_name, confidence, superclass


def map_label_to_superclass(label: str | None) -> str:
    normalized = str(label or "").strip().lower()
    if not normalized:
        return SUPERCLASS_UNKNOWN
    for key, superclass in _LABEL_TO_SUPERCLASS.items():
        if key in normalized:
            return superclass
    return SUPERCLASS_UNKNOWN


def classification_label_from_object(obj: dict[str, Any]) -> str:
    for key in ("label", "class_label", "class_name"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    cls = obj.get("classification")
    if isinstance(cls, dict):
        value = cls.get("label") or cls.get("class_label")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def select_dominant_classification(result_payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = result_payload if isinstance(result_payload, dict) else {}
    raw_objects = payload.get("objects") if isinstance(payload.get("objects"), list) else []
    objects = [item for item in raw_objects if isinstance(item, dict)]
    if not objects:
        return {"label": "no_object", "superclass": SUPERCLASS_UNKNOWN, "confidence": None, "object_count": 0, "objects": []}

    enriched: list[dict[str, Any]] = []
    for item in objects:
        label = classification_label_from_object(item)
        superclass = str(item.get("superclass") or "").strip() or map_label_to_superclass(label)
        confidence = item.get("confidence") if isinstance(item.get("confidence"), (int, float)) else None
        area = item.get("footprint_area_mm2")
        volume = item.get("feature_volume_proxy_mm3")
        weight = float(volume) if isinstance(volume, (int, float)) and float(volume) > 0 else (
            float(area) if isinstance(area, (int, float)) and float(area) > 0 else 0.0
        )
        enriched.append({"object": item, "label": label, "superclass": superclass, "confidence": confidence, "weight": weight})

    if len(enriched) == 1:
        first = enriched[0]
        return {
            "label": first["label"],
            "superclass": first["superclass"],
            "confidence": first["confidence"],
            "object_count": 1,
            "objects": objects,
        }

    by_superclass: dict[str, float] = {}
    for item in enriched:
        by_superclass[item["superclass"]] = by_superclass.get(item["superclass"], 0.0) + float(item["weight"])
    has_weight = any(float(item["weight"]) > 0.0 for item in enriched)
    if has_weight:
        dominant_superclass = max(by_superclass.items(), key=lambda pair: pair[1])[0]
    else:
        counts = Counter(item["superclass"] for item in enriched)
        dominant_superclass = counts.most_common(1)[0][0]
    dominant_objects = [item for item in enriched if item["superclass"] == dominant_superclass]
    dominant_object = max(dominant_objects, key=lambda item: (float(item["weight"]), float(item["confidence"] or 0.0)))
    return {
        "label": dominant_object["label"],
        "superclass": dominant_superclass,
        "confidence": dominant_object["confidence"],
        "object_count": len(objects),
        "objects": objects,
    }
