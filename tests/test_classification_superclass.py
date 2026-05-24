from __future__ import annotations

from pathlib import Path

from vision_3d_acquisition.operations.classification_superclass import (
    SPHERICITY_3D_THRESHOLD_BALL_SCRAP,
    SPHERICITY_3D_THRESHOLD_SCRAP_METAL,
    SUPERCLASS_BALL_GOOD,
    SUPERCLASS_BALL_SCRAP,
    SUPERCLASS_SCRAP_METAL,
    SUPERCLASS_UNKNOWN,
    apply_sphericity_3d_fallback_to_object,
    map_label_to_superclass,
    select_dominant_classification,
)
from vision_3d_acquisition.operations.summary import build_operations_card


def test_label_to_superclass_mapping() -> None:
    assert map_label_to_superclass("bola_buena") == SUPERCLASS_BALL_GOOD
    assert map_label_to_superclass("good_ball") == SUPERCLASS_BALL_GOOD
    assert map_label_to_superclass("bola_con_chip") == SUPERCLASS_BALL_SCRAP
    assert map_label_to_superclass("broken_ball") == SUPERCLASS_BALL_SCRAP
    assert map_label_to_superclass("non_ball") == SUPERCLASS_SCRAP_METAL
    assert map_label_to_superclass("metal_scrap") == SUPERCLASS_SCRAP_METAL


def test_unknown_fallback_for_unmapped_labels() -> None:
    assert map_label_to_superclass("mystery_object") == SUPERCLASS_UNKNOWN
    assert map_label_to_superclass(None) == SUPERCLASS_UNKNOWN


def test_multi_object_dominant_superclass_by_volume() -> None:
    result = {
        "objects": [
            {"object_id": 1, "label": "bola_buena", "confidence": 0.95, "feature_volume_proxy_mm3": 200.0},
            {"object_id": 2, "label": "non_ball", "confidence": 0.70, "feature_volume_proxy_mm3": 1200.0},
        ]
    }
    summary = select_dominant_classification(result)
    assert summary["superclass"] == SUPERCLASS_SCRAP_METAL
    assert summary["label"] == "non_ball"
    assert summary["object_count"] == 2


def test_operations_card_uses_result_superclass_and_label() -> None:
    result_payload = {
        "status": "ok",
        "processed_at": "2026-05-22T12:00:00Z",
        "summary": {"label": "bola_deformada", "superclass": SUPERCLASS_BALL_SCRAP, "confidence": 0.81},
        "objects": [
            {"object_id": 1, "label": "bola_deformada", "superclass": SUPERCLASS_BALL_SCRAP, "confidence": 0.81},
        ],
        "files": {},
    }
    card = build_operations_card(
        take_id="take_1",
        incoming_metadata={"created_at": "2026-05-22T11:59:00Z"},
        runtime_state={"state": "completed", "source": "trispector_ftp"},
        result_payload=result_payload,
        processed_dir=Path("."),
    )
    assert card["superclass"] == SUPERCLASS_BALL_SCRAP
    assert card["label"] == "bola_deformada"


def _apply_fallback(
    item: dict,
    *,
    label: str = "planchuela",
    display_label: str = "Chatarra - Planchuelas",
    class_name: str = "non_ball",
    confidence: float = 0.72,
    superclass: str = SUPERCLASS_SCRAP_METAL,
) -> tuple[str, str, str, float, str]:
    return apply_sphericity_3d_fallback_to_object(
        item,
        label=label,
        display_label=display_label,
        class_name=class_name,
        confidence=confidence,
        superclass=superclass,
    )


def test_sph3d_fallback_preserves_ball_good() -> None:
    item = {"feature_sphericity_3d": 0.10}
    label, display_label, class_name, confidence, superclass = _apply_fallback(
        item,
        label="bola_buena",
        display_label="Bola buena",
        class_name="ball",
        confidence=0.82,
        superclass=SUPERCLASS_BALL_GOOD,
    )
    assert label == "bola_buena"
    assert superclass == SUPERCLASS_BALL_GOOD
    assert class_name == "ball"
    assert display_label == "Bola buena"
    assert confidence == 0.82
    assert "classification_reason" not in item
    assert item["debug"]["sph3d_rule"]["sph3d"] == 0.10


def test_sph3d_fallback_preserves_primary_scrap_metal() -> None:
    item = {"feature_sphericity_3d": 0.497}
    label, display_label, class_name, _, superclass = _apply_fallback(item)
    assert label == "planchuela"
    assert superclass == SUPERCLASS_SCRAP_METAL
    assert class_name == "non_ball"
    assert display_label == "Chatarra - Planchuelas"
    assert "classification_reason" not in item


def test_sph3d_fallback_classifies_scrap_metal_below_threshold() -> None:
    item = {"feature_sphericity_3d": 0.20}
    label, _, class_name, _, superclass = _apply_fallback(
        item,
        label="unknown",
        display_label="Unknown",
        class_name="non_ball",
        confidence=0.72,
        superclass=SUPERCLASS_UNKNOWN,
    )
    assert label == "chatarra"
    assert superclass == SUPERCLASS_SCRAP_METAL
    assert class_name == "non_ball"
    assert item["classification_reason"] == "SCRAP_METAL because sph3d=0.200 < 0.30"
    assert item["debug"]["sph3d_rule"]["threshold_scrap_metal"] == SPHERICITY_3D_THRESHOLD_SCRAP_METAL


def test_sph3d_fallback_classifies_ball_scrap_in_mid_band() -> None:
    item = {"feature_sphericity_3d": 0.50}
    label, display_label, class_name, _, superclass = _apply_fallback(
        item,
        label="unknown",
        display_label="Unknown",
        class_name="non_ball",
        confidence=0.66,
        superclass=SUPERCLASS_UNKNOWN,
    )
    assert label == "bola_con_chip"
    assert superclass == SUPERCLASS_BALL_SCRAP
    assert class_name == "non_ball"
    assert display_label == "Scrap de Bola - Bola con chip"
    assert item["classification_reason"] == "BALL_SCRAP because 0.30 <= sph3d=0.500 < 0.75"
    assert item["debug"]["sph3d_rule"]["threshold_ball_scrap"] == SPHERICITY_3D_THRESHOLD_BALL_SCRAP


def test_sph3d_fallback_leaves_high_sph3d_unchanged() -> None:
    item = {"feature_sphericity_3d": 0.90}
    label, display_label, class_name, confidence, superclass = _apply_fallback(item)
    assert label == "planchuela"
    assert superclass == SUPERCLASS_SCRAP_METAL
    assert class_name == "non_ball"
    assert display_label == "Chatarra - Planchuelas"
    assert confidence == 0.72
    assert "classification_reason" not in item
