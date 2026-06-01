from __future__ import annotations

from typing import Any

LABEL_SCHEMA_ID = "mining_balls_labels_v1"

CLASS_MAP = {
    "cubo": ("CALIBRATION_CUBE", "REFERENCE_OBJECT", 0.98, False),
    "empty": ("EMPTY_SCENE", "BACKGROUND", 0.95, False),
    "no object": ("EMPTY_SCENE", "BACKGROUND", 0.95, False),
    "duda": ("REVIEW_REQUIRED", "UNKNOWN", 0.3, True),
    "planchuela": ("SCRAP_METAL", "SCRAP", 0.8, False),
    "scrap": ("SCRAP_METAL", "SCRAP", 0.85, False),
    "good": ("BALL_GOOD", "BALL", 0.9, False),
    "damaged": ("BALL_DAMAGED", "BALL", 0.85, False),
    "deformed": ("BALL_DEFORMED", "BALL", 0.85, False),
    "small": ("BALL_SMALL", "BALL", 0.9, False),
    "partial": ("BALL_PARTIAL", "BALL", 0.8, False),
}


def normalize_label(raw_label: Any) -> dict[str, Any]:
    raw = str(raw_label or "").strip()
    low = raw.lower()
    if not raw:
        return {
            "raw_operator_label": raw,
            "normalized_class": "REVIEW_REQUIRED",
            "superclass": "UNKNOWN",
            "annotation_confidence": 0.0,
            "needs_review": True,
            "label_schema_id": LABEL_SCHEMA_ID,
        }
    uncertain = low.endswith("?")
    key = low[:-1].strip() if uncertain else low
    mapped = CLASS_MAP.get(key)
    if mapped is None:
        mapped = ("REVIEW_REQUIRED", "UNKNOWN", 0.2, True)
    normalized_class, superclass, confidence, needs_review = mapped
    if uncertain and normalized_class == "BALL_DEFORMED":
        normalized_class = "BALL_DEFORMED_UNCERTAIN"
        needs_review = True
        confidence = min(confidence, 0.55)
    if uncertain:
        needs_review = True
        confidence = min(confidence, 0.6)
    return {
        "raw_operator_label": raw,
        "normalized_class": normalized_class,
        "superclass": superclass,
        "annotation_confidence": confidence,
        "needs_review": bool(needs_review),
        "label_schema_id": LABEL_SCHEMA_ID,
    }
