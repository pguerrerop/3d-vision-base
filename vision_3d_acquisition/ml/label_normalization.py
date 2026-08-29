from __future__ import annotations

from typing import Any

LABEL_SCHEMA_ID = "mining_balls_labels_v1"

AUDIT_SUPERCLASS_UNKNOWN = "UNKNOWN"
AUDIT_SUPERCLASS_MAP = {
    "ball_good": "BALL_GOOD",
    "ball_good_standard": "BALL_GOOD",
    "ball_good_small": "BALL_GOOD",
    "ball_good_medium": "BALL_GOOD",
    "ball_good_large": "BALL_GOOD",
    "buena": "BALL_GOOD",
    "buena_menos": "BALL_GOOD",
    "buena_minus": "BALL_GOOD",
    "bola_buena": "BALL_GOOD",
    "good": "BALL_GOOD",
    "good_ball": "BALL_GOOD",
    "ok": "BALL_GOOD",
    "ball_scrap": "BALL_SCRAP",
    "ball_defect": "BALL_SCRAP",
    "ball_defect_generic": "BALL_SCRAP",
    "ball_broken": "BALL_SCRAP",
    "ball_damaged": "BALL_SCRAP",
    "ball_deformed": "BALL_SCRAP",
    "ball_deformed_uncertain": "BALL_SCRAP",
    "ball_partial": "BALL_SCRAP",
    "ball_small": "BALL_SCRAP",
    "ahuevada": "BALL_SCRAP",
    "mitad": "BALL_SCRAP",
    "chica": "BALL_SCRAP",
    "bola_con_chip": "BALL_SCRAP",
    "cadena": "BALL_SCRAP",
    "duda": "BALL_SCRAP",
    "review_required": "BALL_SCRAP",
    "scrap_metal": "SCRAP_METAL",
    "scrap": "SCRAP_METAL",
    "scrap_generic": "SCRAP_METAL",
    "scrap_nut": "SCRAP_METAL",
    "scrap_bolt": "SCRAP_METAL",
    "scrap_flat_bar": "SCRAP_METAL",
    "scrap_unknown_block": "SCRAP_METAL",
    "chatarra": "SCRAP_METAL",
    "perno": "SCRAP_METAL",
    "tuerca": "SCRAP_METAL",
    "planchuela": "SCRAP_METAL",
    "planchuela_doblada": "SCRAP_METAL",
    "parece_planchuela": "SCRAP_METAL",
    "cubo": "SCRAP_METAL",
    "pedazo": "SCRAP_METAL",
    "non_ball": "SCRAP_METAL",
    "calibration_cube": "SCRAP_METAL",
}

CLASS_MAP = {
    "cubo": ("CALIBRATION_CUBE", "REFERENCE_OBJECT", 0.98, False),
    "empty": ("EMPTY_SCENE", "BACKGROUND", 0.95, False),
    "no object": ("EMPTY_SCENE", "BACKGROUND", 0.95, False),
    "duda": ("REVIEW_REQUIRED", "UNKNOWN", 0.3, True),
    "planchuela": ("SCRAP_METAL", "SCRAP", 0.8, False),
    "scrap": ("SCRAP_METAL", "SCRAP", 0.85, False),
    "good": ("BALL_GOOD", "BALL", 0.9, False),
    "buena": ("BALL_GOOD", "BALL", 0.9, False),
    "buena_menos": ("BALL_GOOD", "BALL", 0.9, False),
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


def resolve_audit_superclass(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return AUDIT_SUPERCLASS_UNKNOWN
    upper = raw.upper()
    if upper in {"BALL_GOOD", "BALL_SCRAP", "SCRAP_METAL", AUDIT_SUPERCLASS_UNKNOWN}:
        return upper
    normalized = raw.strip().lower().replace(" ", "_")
    return AUDIT_SUPERCLASS_MAP.get(normalized, AUDIT_SUPERCLASS_UNKNOWN)
