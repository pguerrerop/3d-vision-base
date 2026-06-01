from vision_3d_acquisition.ml.range_expansion import expand_row_references
from vision_3d_acquisition.ml.label_normalization import normalize_label


def test_range_expansion_inline() -> None:
    refs = expand_row_references({"image_ref": "139...143"})
    assert refs == ["139", "140", "141", "142", "143"]


def test_label_normalization_uncertain() -> None:
    payload = normalize_label("Deformed?")
    assert payload["normalized_class"] == "BALL_DEFORMED_UNCERTAIN"
    assert payload["needs_review"] is True
